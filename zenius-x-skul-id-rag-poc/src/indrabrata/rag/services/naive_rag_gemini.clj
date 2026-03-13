(ns indrabrata.rag.services.naive-rag-gemini
  "Gemini-powered Naive RAG.
   Flow:
   1. Send user question to Gemini → get answer + extracted keywords (JSON)
   2. Use keywords to text-search MongoDB containers (name field)
   3. Return Gemini answer + top-k matching containers as sources."
  (:require
   [clojure.string :as str]
   [indrabrata.rag.components.mongodb :as mongo]
   [indrabrata.rag.services.gemini :as gemini])
  (:import
   [com.mongodb.client MongoCollection]
   [java.util.regex Pattern]
   [org.bson Document]))

;; ---- Step 1: LLM — answer + keyword extraction ----

(def ^:private keyword-extraction-system-prompt
  "You are a friendly learning assistant for an Indonesian educational platform.
Answer the user's question warmly, then extract search keywords to find relevant playlists.

Respond ONLY as valid JSON:
{\"answer\": \"<friendly answer>\", \"keywords\": [\"kw1\", \"kw2\"]}

- answer: concise, in the user's language (Indonesian or English)
- keywords: 2–6 subject/topic terms; no filler words (how, what, is, etc.)")

(defn- extract-answer-and-keywords
  [api-key model user-question]
  (let [messages [{:role "system" :content keyword-extraction-system-prompt}
                  {:role "user"   :content user-question}]
        result   (gemini/chat-completion-json api-key model messages)]
    {:answer   (get-in result [:content :answer] "")
     :keywords (get-in result [:content :keywords] [])
     :usage    (:usage result)}))

;; ---- Step 2: MongoDB keyword search ----

(defn- keywords->regex-pattern [keywords]
  (Pattern/compile (str/join "|" (map #(Pattern/quote %) keywords))
                   Pattern/CASE_INSENSITIVE))

(defn- score-document [^Document doc keywords]
  (let [name  (or (.getString doc "name") "")
        lower (str/lower-case name)]
    (->> keywords
         (filter #(str/includes? lower (str/lower-case %)))
         count)))

(defn keyword-search [mongodb keywords top-k]
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

;; ---- Main Query ----

(defn query
  "Execute a naive RAG query using Gemini.
   Returns {:answer \"...\" :sources [...] :keywords [...] :usage {...}}."
  [mongodb gemini-config user-question & {:keys [top-k] :or {top-k 5}}]
  (println (str "Gemini Naive RAG query: \"" user-question "\""))

  (let [{:keys [answer keywords usage]} (extract-answer-and-keywords
                                         (:api-key gemini-config)
                                         (:chat-model gemini-config)
                                         user-question)
        _       (println (str "  Keywords: " keywords))
        sources (keyword-search mongodb keywords top-k)
        _       (println (str "  Found " (count sources) " matching containers"))]

    {:answer   answer
     :keywords keywords
     :sources  sources
     :usage    {:prompt_tokens     (:prompt_tokens usage)
                :completion_tokens (:completion_tokens usage)
                :total_tokens      (:total_tokens usage)}}))
