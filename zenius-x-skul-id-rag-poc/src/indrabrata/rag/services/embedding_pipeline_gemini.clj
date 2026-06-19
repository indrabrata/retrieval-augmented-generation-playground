(ns indrabrata.rag.services.embedding-pipeline-gemini
  "Gemini embedding pipeline: generates enriched text from containers,
   embeds them using Gemini text-embedding-004 (768 dims), and stores
   results in the container-vectors-gemini collection.

   Requires a MongoDB Atlas vector search index named
   'container_vector_index_gemini' on container-vectors-gemini with:
     - path: embedding
     - similarity: cosine
     - dimensions: 768"
  (:require
   [indrabrata.rag.components.mongodb :as mongo]
   [indrabrata.rag.services.embedding-pipeline-openai :as base-pipeline]
   [indrabrata.rag.services.gemini :as gemini]))

;; ---- Batch Processing ----

(defn- process-batch
  "Process a batch of containers using Gemini embeddings.
   Returns vector documents ready for insertion into container-vectors-gemini."
  [mongodb gemini-config containers]
  (println "  Building enriched texts...")
  (let [texts      (mapv #(base-pipeline/build-enriched-text mongodb %) containers)
        _          (println (str "  Built " (count texts) " texts. Calling Gemini embeddings..."))
        embeddings (gemini/create-embeddings-batch
                    (:api-key gemini-config)
                    (:embedding-model gemini-config)
                    texts)
        _          (println (str "  Got " (count embeddings) " embeddings. Building vector docs..."))]
    (mapv (fn [container text embedding]
            (when (nil? embedding)
              (println (str "  WARNING: nil embedding for container " (:_id container))))
            {:container_id (:_id container)
             :short_id     (:short-id container)
             :name         (:name container)
             :text         text
             :embedding    (vec (or embedding []))
             :created_at   (java.util.Date.)
             :updated_at   (java.util.Date.)})
          containers texts embeddings)))

;; ---- Main Pipeline ----

(defn generate-and-store-embeddings!
  "Fetch all containers, generate Gemini embeddings, and store them
   in the container-vectors-gemini collection.

   Options:
     :batch-size - number of containers per batch (default 20)
     :clear?     - whether to clear existing vectors first (default true)

   Returns total number of vectors stored."
  [mongodb gemini-config & {:keys [batch-size clear?]
                             :or   {batch-size 5 clear? true}}]
  (println "Starting Gemini embedding pipeline...")
  (let [containers (mongo/find-all mongodb "containers")]
    (println (str "Found " (count containers) " containers to process."))

    (when clear?
      (println "Clearing existing container-vectors-gemini...")
      (mongo/delete-many! mongodb "container-vectors-gemini" {}))

    (let [batches (partition-all batch-size containers)
          total   (count batches)]
      (doseq [[idx batch] (map-indexed vector batches)]
        (println (str "Processing batch " (inc idx) "/" total
                      " (" (count batch) " containers)..."))
        (let [vector-docs (process-batch mongodb gemini-config batch)]
          (mongo/insert-many! mongodb "container-vectors-gemini" vector-docs)
          (println (str "  Inserted " (count vector-docs) " vector documents.")))
        (when (< (inc idx) total)
          (println "  Waiting 60s to respect Gemini rate limit...")
          (Thread/sleep 60000)))

      (let [total-vectors (mongo/count-documents mongodb "container-vectors-gemini")]
        (println (str "Gemini pipeline complete. Total vectors stored: " total-vectors))
        total-vectors))))
