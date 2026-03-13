(ns indrabrata.rag.services.rag-gemini
  "Gemini-powered RAG: query embedding → MongoDB vector search → LLM response.

   Requires a separate MongoDB vector search index named 'container_vector_index_gemini'
   configured for 768-dimensional Gemini embeddings (text-embedding-004)."
  (:require
   [clojure.string :as str]
   [indrabrata.rag.components.mongodb :as mongo]
   [indrabrata.rag.services.gemini :as gemini])
  (:import
   [java.util ArrayList]
   [org.bson Document]))

;; ---- Vector Search ----

(defn vector-search
  "Perform a $vectorSearch against the container-vectors-gemini collection.
   Requires index 'container_vector_index_gemini' with 768 dimensions."
  [mongodb query-embedding & {:keys [limit num-candidates]
                              :or   {limit 5 num-candidates 100}}]
  (let [coll             (mongo/get-collection mongodb "container-vectors-gemini")
        vector-search-doc (doto (Document.)
                            (.append "$vectorSearch"
                                     (doto (Document.)
                                       (.append "index" "container_vector_index_gemini")
                                       (.append "path" "embedding")
                                       (.append "queryVector" (ArrayList. (vec query-embedding)))
                                       (.append "numCandidates" (int num-candidates))
                                       (.append "limit" (int limit)))))
        project-doc      (doto (Document.)
                           (.append "$project"
                                    (doto (Document.)
                                      (.append "container_id" 1)
                                      (.append "short_id" 1)
                                      (.append "name" 1)
                                      (.append "text" 1)
                                      (.append "score" (doto (Document.)
                                                         (.append "$meta" "vectorSearchScore"))))))
        pipeline         (ArrayList. [vector-search-doc project-doc])
        results          (.aggregate coll pipeline)]
    (->> results
         (into [])
         (mapv (fn [^Document doc]
                 (into {}
                       (map (fn [^java.util.Map$Entry e]
                              [(keyword (.getKey e)) (.getValue e)]))
                       (.entrySet doc)))))))

;; ---- Context Building ----

(defn- build-context-string [results]
  (->> results
       (map-indexed
        (fn [idx result]
          (str (inc idx) ". " (:name result)
               " (ID: " (:short_id result) ", relevance: "
               (when-let [s (:score result)]
                 (format "%.3f" (double s)))
               ")\n"
               "   " (:text result))))
       (str/join "\n\n")))

(def ^:private system-prompt
  "You are a friendly learning guide for an Indonesian educational platform.

Answer the user's question warmly and concisely, then recommend relevant playlists from the context.
For each playlist: include its short ID (e.g. lp17073) and one sentence on why it fits.
If no playlists match, say so kindly.
Reply in the user's language (Bahasa Indonesia or English).")

;; ---- Main RAG Query ----

(defn query
  "Execute a full RAG query using Gemini:
   1. Embed the user's question (Gemini text-embedding-004)
   2. Vector search MongoDB container-vectors-gemini collection
   3. Build context from results
   4. Send to Gemini LLM with context

   Returns {:answer \"...\" :sources [...] :usage {...}}."
  [mongodb gemini-config user-question & {:keys [top-k] :or {top-k 5}}]
  (println (str "Gemini RAG query: \"" user-question "\""))

  (let [api-key    (:api-key gemini-config)
        emb-model  (:embedding-model gemini-config)
        chat-model (:chat-model gemini-config)

        embed-result    (gemini/create-embedding api-key emb-model user-question)
        query-embedding (:embedding embed-result)
        _               (println (str "  Generated query embedding (" (count query-embedding) " dims)"))

        search-results (vector-search mongodb query-embedding :limit top-k)
        _              (println (str "  Found " (count search-results) " relevant playlists"))

        context-str    (build-context-string search-results)
        messages       [{:role "system" :content system-prompt}
                        {:role "user"
                         :content (str "Playlists:\n" context-str
                                       "\n\nQuestion: " user-question)}]
        chat-result    (gemini/chat-completion api-key chat-model messages)
        chat-usage     (:usage chat-result)]

    {:answer  (:content chat-result)
     :sources (mapv (fn [r]
                      {:name         (:name r)
                       :short_id     (:short_id r)
                       :container_id (:container_id r)
                       :score        (:score r)})
                    search-results)
     :usage   {:prompt_tokens     (:prompt_tokens chat-usage)
               :completion_tokens (:completion_tokens chat-usage)
               :total_tokens      (:total_tokens chat-usage)}}))

;; ---- Conversation (multi-turn, no RAG) ----

(def ^:private conversation-system-prompt
  "You are a friendly and knowledgeable assistant for an Indonesian educational platform.
Answer the user's questions warmly and helpfully.
Reply in the user's language (Bahasa Indonesia or English).")

(defn conversation-query
  "Multi-turn chat without vector search.
   messages is a seq of {:role \"user\"|\"assistant\" :content \"...\"}.
   window-size trims history to the last N messages (default 10).
   Returns {:answer \"...\" :usage {...}}."
  [gemini-config messages & {:keys [window-size] :or {window-size 10}}]
  (let [windowed      (vec (take-last window-size messages))
        _             (println (str "Gemini conversation, " (count windowed) "/" (count messages) " messages (window=" window-size ")"))
        full-messages (concat [{:role "system" :content conversation-system-prompt}] windowed)
        result        (gemini/chat-completion
                       (:api-key gemini-config)
                       (:chat-model gemini-config)
                       (vec full-messages))]
    {:answer (:content result)
     :usage  (:usage result)}))
