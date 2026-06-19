(ns indrabrata.rag.services.file-rag-gemini
  "Gemini-powered File RAG.

   RAG flow:
     1. Send file inline to Gemini → extract educational topics as search query
     2. Embed search query via Gemini
     3. Vector search MongoDB container-vectors-gemini
     4. Send question + playlist context to Gemini LLM

   Naive RAG flow:
     1. Send file inline to Gemini (JSON mode) → answer + keywords
     2. Keyword search MongoDB containers by name
     3. Return answer + matching containers as sources"
  (:require
   [clojure.string :as str]
   [indrabrata.rag.services.gemini :as gemini]
   [indrabrata.rag.services.naive-rag-gemini :as naive-rag-svc]
   [indrabrata.rag.services.rag-gemini :as rag-gemini-svc]))

;; ---- System Prompts ----

(def ^:private topic-extract-prompt
  "Extract the main educational topics and key concepts from this document as a short search query (1-2 sentences). Output only the search query text, nothing else.")

(def ^:private file-rag-system-prompt
  "You are a friendly learning guide for an Indonesian educational platform.
Answer the user's question based on the uploaded document, then recommend relevant playlists from the provided context.
For each playlist: include its short ID (e.g. lp17073) and one sentence on why it fits.
If no playlists match, say so kindly.
Reply in the user's language (Bahasa Indonesia or English).")

(def ^:private naive-rag-keyword-prompt
  "You are a friendly assistant. Answer the user's question warmly, then extract search keywords.

Respond ONLY as valid JSON:
{\"answer\": \"<friendly answer>\", \"keywords\": [\"kw1\", \"kw2\"]}

- answer: concise, in the user's language
- keywords: 2-6 key nouns/topics; no filler words (how, what, is, etc.)")

;; ---- Inline File Query (no local extraction, no embeddings) ----

(defn files-api-query
  "Query an uploaded file via Gemini inline_data.
   File bytes are base64-encoded and sent directly in the request — no file upload needed.

   Returns {:answer \"...\" :usage {...}}"
  [gemini-config ^bytes file-bytes filename content-type user-question]
  (println (str "Gemini inline file query on \"" filename "\""))
  (let [api-key    (:api-key gemini-config)
        chat-model (:chat-model gemini-config)
        result     (gemini/inline-file-completion api-key chat-model
                                                  file-bytes filename content-type
                                                  user-question)]
    (println "  Gemini inline file call complete")
    {:answer (:content result)
     :usage  (:usage result)}))

;; ---- RAG with File ----

(defn rag-query
  "Execute RAG: extract topics from uploaded file inline, embed them,
   search MongoDB container-vectors-gemini, and answer with container context.

   Returns {:answer \"...\" :sources [...] :usage {...}}"
  [mongodb gemini-config ^bytes file-bytes filename content-type user-question
   & {:keys [top-k] :or {top-k 5}}]
  (println (str "Gemini File RAG (inline+vector) query on \"" filename "\""))
  (let [api-key         (:api-key gemini-config)
        chat-model      (:chat-model gemini-config)
        embedding-model (:embedding-model gemini-config)

        ;; Step 1: Extract topics from file inline
        _               (println "  Extracting topics from file...")
        extract-result  (gemini/inline-file-completion api-key chat-model
                                                       file-bytes filename content-type
                                                       topic-extract-prompt)
        search-query    (:content extract-result)
        extract-usage   (:usage extract-result)
        _               (println (str "  Search query: " search-query))

        ;; Step 2: Embed the extracted query
        embed-result    (gemini/create-embedding api-key embedding-model search-query)
        query-embedding (:embedding embed-result)
        _               (println (str "  Embedding done (" (count query-embedding) " dims)"))

        ;; Step 3: Vector search in MongoDB
        search-results  (rag-gemini-svc/vector-search mongodb query-embedding :limit top-k)
        _               (println (str "  Found " (count search-results) " relevant containers"))

        ;; Step 4: Build context + answer
        context-str     (->> search-results
                             (map-indexed (fn [idx r]
                                            (str (inc idx) ". " (:name r)
                                                 " (ID: " (:short_id r) ")\n"
                                                 "   " (:text r))))
                             (str/join "\n\n"))
        messages        [{:role "system" :content file-rag-system-prompt}
                         {:role "user"   :content (str "Relevant playlists:\n" context-str
                                                       "\n\nQuestion: " user-question)}]
        chat-result     (gemini/chat-completion api-key chat-model messages)
        chat-usage      (:usage chat-result)]

    {:answer  (:content chat-result)
     :sources (mapv (fn [r] {:name         (:name r)
                             :short_id     (:short_id r)
                             :container_id (:container_id r)
                             :score        (:score r)})
                    search-results)
     :usage   {:extract_tokens (:total_tokens extract-usage)
               :answer_tokens  (:total_tokens chat-usage)
               :total_tokens   (+ (or (:total_tokens extract-usage) 0)
                                  (or (:total_tokens chat-usage) 0))}}))

;; ---- Naive RAG with File (inline LLM → keyword search on MongoDB containers) ----

(defn naive-rag-query
  "Execute naive RAG against an uploaded file via Gemini inline_data.
   File bytes are sent inline for LLM answer + keyword extraction.
   Keywords are then used to search MongoDB containers by name.

   Returns {:answer \"...\" :keywords [...] :sources [...] :usage {...}}"
  [mongodb gemini-config ^bytes file-bytes filename content-type user-question
   & {:keys [top-k] :or {top-k 5}}]
  (println (str "Gemini File Naive RAG (inline) query on \"" filename "\""))
  (let [api-key       (:api-key gemini-config)
        chat-model    (:chat-model gemini-config)
        json-question (str naive-rag-keyword-prompt "\n\nQuestion: " user-question)
        llm-result    (gemini/inline-file-completion-json api-key chat-model
                                                          file-bytes filename content-type
                                                          json-question)
        answer        (get-in llm-result [:content :answer] "")
        keywords      (get-in llm-result [:content :keywords] [])
        _             (println (str "  Keywords: " keywords))
        sources       (naive-rag-svc/keyword-search mongodb keywords top-k)
        _             (println (str "  Found " (count sources) " matching containers"))]
    {:answer   answer
     :keywords keywords
     :sources  sources
     :usage    (:usage llm-result)}))
