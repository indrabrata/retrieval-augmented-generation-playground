(ns indrabrata.rag.services.naive-rag
  "Naive RAG service.
   Flow:
   1. Send user question to LLM → get answer + extracted keywords (JSON)
   2. Use keywords to text-search MongoDB containers (name field)
   3. Return LLM answer + top-k matching containers as sources."
  (:require [clojure.string :as str]
            [indrabrata.rag.components.mongodb :as mongo]
            [indrabrata.rag.services.openai :as openai])
  (:import [com.mongodb.client MongoCollection]
           [org.bson Document]
           [java.util.regex Pattern]))

;; ---- Step 1: LLM — answer + keyword extraction ----

(def ^:private keyword-extraction-system-prompt
  "You are a helpful learning assistant for an Indonesian educational platform.

Your task:
1. Answer the user's question clearly and accurately.
2. Extract the most important search keywords from the question that would help find relevant learning playlists. Focus on subject names, topic names, and educational concepts.

Respond ONLY with valid JSON in this exact shape:
{
  \"answer\": \"<your answer here>\",
  \"keywords\": [\"keyword1\", \"keyword2\", \"keyword3\"]
}

Rules:
- \"answer\" should be a helpful, concise response in the same language the user used (Bahasa Indonesia or English).
- \"keywords\" should be 2–6 short, specific terms (single words or short phrases) relevant to finding educational playlists.
- Do not include filler words like 'how', 'what', 'is', etc. in keywords.")

(defn- extract-answer-and-keywords
  "Call the LLM to get an answer and keywords from the user question.
   Returns {:answer \"...\" :keywords [\"...\" ...] :usage {...}}."
  [api-key model user-question]
  (let [messages [{:role "system" :content keyword-extraction-system-prompt}
                  {:role "user" :content user-question}]
        result   (openai/chat-completion-json api-key model messages)]
    {:answer   (get-in result [:content :answer] "")
     :keywords (get-in result [:content :keywords] [])
     :usage    (:usage result)}))

;; ---- Step 2: MongoDB text search by keywords ----

(defn- keywords->regex-pattern
  "Build a case-insensitive regex that matches any of the given keywords."
  [keywords]
  (let [escaped (map #(Pattern/quote %) keywords)
        joined  (str/join "|" escaped)]
    (Pattern/compile joined Pattern/CASE_INSENSITIVE)))

(defn- score-document
  "Score a container document based on how many keywords appear in its name.
   Higher score = more keyword matches."
  [^Document doc keywords]
  (let [name  (or (.getString doc "name") "")
        lower (str/lower-case name)]
    (->> keywords
         (filter #(str/includes? lower (str/lower-case %)))
         count)))

(defn keyword-search
  "Search the containers collection for documents whose name matches any
   of the given keywords. Returns top-k results ordered by match score."
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
                  (let [name      (.getString doc "name")
                        short-id  (.getString doc "short-id")
                        container-id (let [raw (.get doc "_id")]
                                       (cond
                                         (instance? org.bson.types.ObjectId raw) (.toHexString ^org.bson.types.ObjectId raw)
                                         raw (.toString raw)
                                         :else nil))]
                    {:name         name
                     :short_id     short-id
                     :container_id container-id
                     :score        (score-document doc keywords)})))
           (sort-by :score >)
           (take top-k)
           vec))))

;; ---- Main Naive RAG Query ----

(defn query
  "Execute a naive RAG query:
   1. Ask LLM to answer the question and extract keywords
   2. Use keywords to search MongoDB containers by name similarity
   3. Return answer + top-k matched containers as sources

   Returns {:answer \"...\" :sources [...] :keywords [...] :usage {...}}."
  [mongodb openai-config user-question & {:keys [top-k] :or {top-k 5}}]
  (println (str "Naive RAG query: \"" user-question "\""))

  ;; Step 1: LLM answer + keyword extraction
  (let [{:keys [answer keywords usage]} (extract-answer-and-keywords
                                         (:api-key openai-config)
                                         (:chat-model openai-config)
                                         user-question)
        _  (println (str "  LLM answer extracted. Keywords: " keywords))

        ;; Step 2: Keyword search in MongoDB
        sources (keyword-search mongodb keywords top-k)
        _       (println (str "  Found " (count sources) " matching containers"))]

    {:answer   answer
     :keywords keywords
     :sources  sources
     :usage    {:prompt_tokens     (:prompt_tokens usage)
                :completion_tokens (:completion_tokens usage)
                :total_tokens      (:total_tokens usage)}}))
