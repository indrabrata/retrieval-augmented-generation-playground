(ns indrabrata.rag.services.image-rag-gemini
  "Gemini-powered Image RAG using Gemini Vision.

   RAG flow:
     1. Embed the text question using Gemini
     2. Vector search MongoDB (container-vectors-gemini) for relevant playlists
     3. Send image + question + context to Gemini vision LLM

   Naive RAG flow:
     1. Send image + question to Gemini vision LLM (JSON mode) → answer + keywords
     2. Keyword search MongoDB containers by name
     3. Return answer + sources"
  (:require
   [clojure.string :as str]
   [indrabrata.rag.components.mongodb :as mongo]
   [indrabrata.rag.services.gemini :as gemini]
   [indrabrata.rag.services.rag-gemini :as rag-gemini])
  (:import
   [com.mongodb.client MongoCollection]
   [java.util.regex Pattern]
   [org.bson Document]))

;; ---- Image Encoding ----

(defn- image->base64
  "Encode image bytes as raw base64 for Gemini inlineData."
  [^bytes image-bytes]
  (.encodeToString (java.util.Base64/getEncoder) image-bytes))

(defn- vision-parts
  "Build Gemini content parts with image + text."
  [^bytes image-bytes content-type text]
  [{:inlineData {:mimeType content-type
                 :data     (image->base64 image-bytes)}}
   {:text text}])

;; ---- System Prompts ----

(def ^:private rag-vision-system-prompt
  "You are a friendly learning guide for an Indonesian educational platform.

Analyze the image, answer the user's question warmly, then recommend relevant playlists from the context.
For each playlist: include its short ID (e.g. lp17073) and one sentence on why it fits.
If no playlists match, say so kindly.
Reply in the user's language (Bahasa Indonesia or English).")

(def ^:private naive-rag-vision-system-prompt
  "You are a friendly learning assistant for an Indonesian educational platform.
Analyze the image, answer the user's question warmly, then extract search keywords.

Respond ONLY as valid JSON:
{\"answer\": \"<friendly answer>\", \"keywords\": [\"kw1\", \"kw2\"]}

- answer: concise, in the user's language (Indonesian or English)
- keywords: 2–6 subject/topic terms from image + question; no filler words")

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

;; ---- Keyword Search (for naive RAG) ----

(defn- keywords->regex-pattern [keywords]
  (Pattern/compile (str/join "|" (map #(Pattern/quote %) keywords))
                   Pattern/CASE_INSENSITIVE))

(defn- score-document [^Document doc keywords]
  (let [name  (or (.getString doc "name") "")
        lower (str/lower-case name)]
    (->> keywords
         (filter #(str/includes? lower (str/lower-case %)))
         count)))

(defn- keyword-search [mongodb keywords top-k]
  (if (empty? keywords)
    []
    (let [coll    (mongo/get-collection mongodb "containers")
          pattern (keywords->regex-pattern keywords)
          filter  (doto (Document.)
                    (.append "name" (doto (Document.)
                                      (.append "$regex" pattern))))
          results (->> (.find ^MongoCollection coll ^org.bson.conversions.Bson filter)
                       (into []))]
      (->> results
           (map (fn [^Document doc]
                  {:name         (.getString doc "name")
                   :short_id     (.getString doc "short-id")
                   :container_id (let [raw (.get doc "_id")]
                                   (cond
                                     (instance? org.bson.types.ObjectId raw) (.toHexString ^org.bson.types.ObjectId raw)
                                     raw (.toString raw)
                                     :else nil))
                   :score        (score-document doc keywords)}))
           (sort-by :score >)
           (take top-k)
           vec))))

;; ---- RAG with Image ----

(defn image-rag-query
  "Execute RAG with an uploaded image + text question using Gemini:
   1. Embed the text question (Gemini)
   2. Vector search MongoDB container-vectors-gemini
   3. Send image + question + context to Gemini vision LLM

   Returns {:answer \"...\" :sources [...] :usage {...}}"
  [mongodb gemini-config ^bytes image-bytes image-content-type user-question
   & {:keys [top-k] :or {top-k 5}}]
  (println (str "Gemini Image RAG query: \"" user-question "\""))

  (let [api-key    (:api-key gemini-config)
        emb-model  (:embedding-model gemini-config)
        chat-model (:chat-model gemini-config)

        embed-result    (gemini/create-embedding api-key emb-model user-question)
        query-embedding (:embedding embed-result)
        _               (println (str "  Generated query embedding (" (count query-embedding) " dims)"))

        search-results (rag-gemini/vector-search mongodb query-embedding :limit top-k)
        _              (println (str "  Found " (count search-results) " relevant playlists"))

        context-str  (build-context-string search-results)
        user-content (vision-parts image-bytes image-content-type
                                   (str "Playlists:\n" context-str
                                        "\n\nQuestion: " user-question))
        messages     [{:role "system" :content rag-vision-system-prompt}
                      {:role "user"   :content user-content}]
        chat-result  (gemini/chat-completion api-key chat-model messages)
        chat-usage   (:usage chat-result)]

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

;; ---- Naive RAG with Image ----

(defn image-naive-rag-query
  "Execute naive RAG with an uploaded image + text question using Gemini Vision:
   1. Send image + question to Gemini vision LLM (JSON mode) → answer + keywords
   2. Keyword search MongoDB containers by name
   3. Return answer + sources

   Returns {:answer \"...\" :keywords [...] :sources [...] :usage {...}}"
  [mongodb gemini-config ^bytes image-bytes image-content-type user-question
   & {:keys [top-k] :or {top-k 5}}]
  (println (str "Gemini Image Naive RAG query: \"" user-question "\""))

  (let [api-key    (:api-key gemini-config)
        chat-model (:chat-model gemini-config)

        user-content (vision-parts image-bytes image-content-type user-question)
        messages     [{:role "system" :content naive-rag-vision-system-prompt}
                      {:role "user"   :content user-content}]
        llm-result   (gemini/chat-completion-json api-key chat-model messages)
        answer       (get-in llm-result [:content :answer] "")
        keywords     (get-in llm-result [:content :keywords] [])
        llm-usage    (:usage llm-result)
        _            (println (str "  Keywords: " keywords))

        sources (keyword-search mongodb keywords top-k)
        _       (println (str "  Found " (count sources) " matching containers"))]

    {:answer   answer
     :keywords keywords
     :sources  sources
     :usage    {:prompt_tokens     (:prompt_tokens llm-usage)
                :completion_tokens (:completion_tokens llm-usage)
                :total_tokens      (:total_tokens llm-usage)}}))
