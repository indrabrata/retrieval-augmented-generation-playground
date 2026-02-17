(ns indrabrata.rag.services.rag
  "RAG (Retrieval-Augmented Generation) service.
   Handles: query embedding → vector search → context building → LLM response."
  (:require [indrabrata.rag.components.mongodb :as mongo]
            [indrabrata.rag.services.openai :as openai])
  (:import [com.mongodb.client MongoCollection]
           [org.bson Document]
           [java.util ArrayList]))

;; ---- Vector Search ----

(defn vector-search
  "Perform a vector search on the container-vectors collection using
   MongoDB Atlas $vectorSearch aggregation stage.

   Requires a vector search index named 'container_vector_index' on the
   container-vectors collection with the following configuration:
   - path: embedding
   - similarity: cosine
   - dimensions: 1536 (for text-embedding-3-small)

   Returns top-k matching container vectors with scores."
  [mongodb query-embedding & {:keys [limit num-candidates]
                              :or   {limit 5 num-candidates 100}}]
  (let [coll    (mongo/get-collection mongodb "container-vectors")
        ;; Build the $vectorSearch stage as a BSON Document
        ;; We must use raw Document construction because $vectorSearch
        ;; has specific structure requirements
        vector-search-doc (doto (Document.)
                            (.append "$vectorSearch"
                                     (doto (Document.)
                                       (.append "index" "container_vector_index")
                                       (.append "path" "embedding")
                                       (.append "queryVector" (ArrayList. (vec query-embedding)))
                                       (.append "numCandidates" (int num-candidates))
                                       (.append "limit" (int limit)))))
        ;; Project stage: exclude the embedding field, include score
        project-doc (doto (Document.)
                      (.append "$project"
                               (doto (Document.)
                                 (.append "container_id" 1)
                                 (.append "short_id" 1)
                                 (.append "name" 1)
                                 (.append "text" 1)
                                 (.append "score" (doto (Document.)
                                                    (.append "$meta" "vectorSearchScore"))))))
        pipeline (ArrayList. [vector-search-doc project-doc])
        results  (.aggregate coll pipeline)]
    (->> results
         (into [])
         (mapv (fn [^Document doc]
                 (let [m (into {}
                               (map (fn [^java.util.Map$Entry e]
                                      [(keyword (.getKey e)) (.getValue e)]))
                               (.entrySet doc))]
                   m))))))

;; ---- Context Building ----

(defn- build-context-string
  "Build a context string from vector search results for the LLM."
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
       (clojure.string/join "\n\n")))

(def ^:private system-prompt
  "You are a helpful learning assistant for an Indonesian educational platform.
Your job is to:
1. Answer the user's question clearly and accurately
2. Recommend relevant learning playlists based on the context provided

When recommending playlists:
- Only recommend playlists from the provided context
- Explain briefly why each playlist is relevant
- Include the playlist short ID (e.g., lp17073) for reference
- If the context doesn't contain relevant playlists, say so honestly

Respond in the same language the user uses (Bahasa Indonesia or English).
Keep your response concise and helpful.")

;; ---- Main RAG Query ----

(defn query
  "Execute a full RAG query:
   1. Embed the user's question
   2. Vector search for relevant playlists
   3. Build context from results
   4. Send to LLM with context

   Returns {:answer \"...\" :sources [...]}."
  [mongodb openai-config user-question & {:keys [top-k] :or {top-k 5}}]
  (println (str "RAG query: \"" user-question "\""))

  ;; Step 1: Generate embedding for the question
  (let [embed-result    (openai/create-embedding
                         (:api-key openai-config)
                         (:embedding-model openai-config)
                         user-question)
        query-embedding (:embedding embed-result)
        embed-usage     (:usage embed-result)
        _               (println (str "  Generated query embedding (" (count query-embedding) " dims)"))

        ;; Step 2: Vector search
        search-results  (vector-search mongodb query-embedding :limit top-k)
        _               (println (str "  Found " (count search-results) " relevant playlists"))

        ;; Step 3: Build context
        context-str     (build-context-string search-results)

        ;; Step 4: LLM completion
        messages        [{:role    "system"
                          :content system-prompt}
                         {:role    "user"
                          :content (str "Context - Available learning playlists:\n\n"
                                        context-str
                                        "\n\n---\n\n"
                                        "User question: " user-question)}]
        chat-result     (openai/chat-completion
                         (:api-key openai-config)
                         (:chat-model openai-config)
                         messages)
        chat-usage      (:usage chat-result)]

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
