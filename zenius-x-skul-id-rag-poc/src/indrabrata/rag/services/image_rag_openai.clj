(ns indrabrata.rag.services.image-rag-openai
  "Image-based RAG: use an uploaded image with a text prompt.

   RAG flow:
     1. Generate embedding for the text question
     2. Vector search MongoDB for relevant playlists
     3. Send image + question + context to OpenAI vision LLM

   Naive RAG flow:
     1. Send image + question to OpenAI vision LLM (JSON mode) → answer + keywords
     2. Keyword search MongoDB containers by name
     3. Return answer + sources"
  (:require
   [clojure.string :as str]
   [indrabrata.rag.components.mongodb :as mongo]
   [indrabrata.rag.services.openai :as openai]
   [indrabrata.rag.services.rag-openai :as rag])
  (:import
   [com.mongodb.client MongoCollection]
   [java.util.regex Pattern]
   [org.bson Document]))

;; ---- Image Encoding ----

(defn- image->data-url
  "Encode image bytes as a base64 data URL for OpenAI vision API."
  [^bytes image-bytes content-type]
  (let [b64 (.encodeToString (java.util.Base64/getEncoder) image-bytes)]
    (str "data:" content-type ";base64," b64)))

(defn- vision-user-content
  "Build a user message content array with image + text for the OpenAI vision API."
  [data-url text]
  [{:type "image_url" :image_url {:url data-url}}
   {:type "text" :text text}])

;; ---- System Prompts ----

(def ^:private rag-vision-system-prompt
  "You are a helpful learning assistant for an Indonesian educational platform.
Your job is to:
1. Analyze the provided image carefully.
2. Answer the user's question considering both the image content and the learning playlists context.
3. Recommend relevant learning playlists from the provided context.

When recommending playlists:
- Only recommend playlists from the provided context
- Explain briefly why each playlist is relevant
- Include the playlist short ID (e.g., lp17073) for reference
- If the context doesn't contain relevant playlists, say so honestly

Respond in the same language the user uses (Bahasa Indonesia or English).
Keep your response concise and helpful.")

(def ^:private naive-rag-vision-system-prompt
  "You are a helpful learning assistant for an Indonesian educational platform.

Your task:
1. Analyze the provided image carefully.
2. Answer the user's question based on the image content.
3. Extract 2–6 important search keywords from the question and image content to find relevant learning playlists. Focus on subject names, topic names, and educational concepts.

Respond ONLY with valid JSON in this exact shape:
{
  \"answer\": \"<your answer here>\",
  \"keywords\": [\"keyword1\", \"keyword2\", \"keyword3\"]
}

Rules:
- \"answer\" should be a helpful, concise response in the same language the user used (Bahasa Indonesia or English).
- \"keywords\" should be 2–6 short, specific educational terms.
- Do not include filler words like 'how', 'what', 'is', etc. in keywords.")

;; ---- Context Building ----

(defn- build-context-string
  [results]
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

(defn- keywords->regex-pattern
  [keywords]
  (let [escaped (map #(Pattern/quote %) keywords)
        joined  (str/join "|" escaped)]
    (Pattern/compile joined Pattern/CASE_INSENSITIVE)))

(defn- score-document
  [^Document doc keywords]
  (let [name  (or (.getString doc "name") "")
        lower (str/lower-case name)]
    (->> keywords
         (filter #(str/includes? lower (str/lower-case %)))
         count)))

(defn- keyword-search
  [mongodb keywords top-k]
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
  "Execute RAG with an uploaded image + text question:
   1. Embed the text question
   2. Vector search MongoDB for relevant playlists
   3. Send image + question + context to OpenAI vision LLM

   Returns {:answer \"...\" :sources [...] :usage {...}}"
  [mongodb openai-config ^bytes image-bytes image-content-type user-question
   & {:keys [top-k] :or {top-k 5}}]
  (println (str "Image RAG query: \"" user-question "\""))

  (let [api-key    (:api-key openai-config)
        emb-model  (:embedding-model openai-config)
        chat-model (:chat-model openai-config)

        ;; Step 1: Embed the question text
        embed-result    (openai/create-embedding api-key emb-model user-question)
        query-embedding (:embedding embed-result)
        embed-usage     (:usage embed-result)
        _               (println (str "  Generated query embedding (" (count query-embedding) " dims)"))

        ;; Step 2: Vector search
        search-results (rag/vector-search mongodb query-embedding :limit top-k)
        _              (println (str "  Found " (count search-results) " relevant playlists"))

        ;; Step 3: Build context string
        context-str (build-context-string search-results)

        ;; Step 4: Send image + question + context to vision LLM
        data-url    (image->data-url image-bytes image-content-type)
        messages    [{:role    "system"
                      :content rag-vision-system-prompt}
                     {:role    "user"
                      :content (vision-user-content
                                data-url
                                (str "Context - Available learning playlists:\n\n"
                                     context-str
                                     "\n\n---\n\n"
                                     "User question: " user-question))}]
        chat-result (openai/chat-completion api-key chat-model messages)
        chat-usage  (:usage chat-result)]

    {:answer  (:content chat-result)
     :sources (mapv (fn [r]
                      {:name         (:name r)
                       :short_id     (:short_id r)
                       :container_id (:container_id r)
                       :score        (:score r)})
                    search-results)
     :usage   {:embedding_tokens  (:total_tokens embed-usage)
               :prompt_tokens     (:prompt_tokens chat-usage)
               :completion_tokens (:completion_tokens chat-usage)
               :total_tokens      (+ (or (:total_tokens embed-usage) 0)
                                     (or (:total_tokens chat-usage) 0))}}))

;; ---- Naive RAG with Image ----

(defn image-naive-rag-query
  "Execute naive RAG with an uploaded image + text question:
   1. Send image + question to OpenAI vision LLM (JSON mode) → answer + keywords
   2. Keyword search MongoDB containers by name
   3. Return answer + sources

   Returns {:answer \"...\" :keywords [...] :sources [...] :usage {...}}"
  [mongodb openai-config ^bytes image-bytes image-content-type user-question
   & {:keys [top-k] :or {top-k 5}}]
  (println (str "Image Naive RAG query: \"" user-question "\""))

  (let [api-key    (:api-key openai-config)
        chat-model (:chat-model openai-config)

        ;; Step 1: Vision LLM extracts answer + keywords from image + question
        data-url   (image->data-url image-bytes image-content-type)
        messages   [{:role    "system"
                     :content naive-rag-vision-system-prompt}
                    {:role    "user"
                     :content (vision-user-content data-url user-question)}]
        llm-result (openai/chat-completion-json api-key chat-model messages)
        answer     (get-in llm-result [:content :answer] "")
        keywords   (get-in llm-result [:content :keywords] [])
        llm-usage  (:usage llm-result)
        _          (println (str "  Keywords: " keywords))

        ;; Step 2: Keyword search in MongoDB
        sources (keyword-search mongodb keywords top-k)
        _       (println (str "  Found " (count sources) " matching containers"))]

    {:answer   answer
     :keywords keywords
     :sources  sources
     :usage    {:prompt_tokens     (:prompt_tokens llm-usage)
                :completion_tokens (:completion_tokens llm-usage)
                :total_tokens      (:total_tokens llm-usage)}}))
