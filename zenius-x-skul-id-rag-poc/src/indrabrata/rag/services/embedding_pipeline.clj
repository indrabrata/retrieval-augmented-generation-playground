(ns indrabrata.rag.services.embedding-pipeline
  "Pipeline to generate enriched text from containers and their related
   question/video instances, generate embeddings, and store them in
   the container-vectors collection."
  (:require
   [clojure.string :as str]
   [indrabrata.rag.components.mongodb :as mongo]
   [indrabrata.rag.services.openai :as openai]))

;; ---- Text Enrichment ----

(defn- strip-html
  "Remove HTML tags from a string."
  [s]
  (when s
    (-> s
        (str/replace #"<[^>]+>" " ")
        (str/replace #"&nbsp;" " ")
        (str/replace #"\s+" " ")
        clojure.string/trim)))

(defn- extract-latex
  "Extract LaTeX expressions from data-latex attributes in HTML."
  [s]
  (when s
    (->> (re-seq #"data-latex=\"([^\"]+)\"" s)
         (map second)
         (clojure.string/join " "))))

(defn- fetch-question-texts
  "Fetch question texts for a list of question instance IDs."
  [mongodb question-ids]
  (when (seq question-ids)
    (let [questions (mongo/find-by-ids mongodb "question-instances" question-ids)]
      (->> questions
           (map (fn [q]
                  (let [raw-text  (:question-text q)
                        clean     (strip-html raw-text)
                        latex     (extract-latex raw-text)
                        expl-raw  (:explanation-text q)
                        expl      (strip-html expl-raw)]
                    (str clean
                         (when latex (str " " latex))
                         (when expl (str " " expl))))))
           (clojure.string/join " | ")))))

(defn- fetch-video-titles
  "Fetch video titles for a list of video instance IDs."
  [mongodb video-ids]
  (when (seq video-ids)
    (let [videos (mongo/find-by-ids mongodb "video-instances" video-ids)]
      (->> videos
           (map :title)
           (filter some?)
           (clojure.string/join " | ")))))

(defn build-enriched-text
  "Build enriched text for a single container by combining its metadata
   with question and video content."
  [mongodb container]
  (try
    (let [name         (:name container "")
          description  (:description container "")
          instances    (:content-instances container [])
          question-ids (->> instances
                            (filter #(= "question" (:type %)))
                            (map :_id)
                            (filter some?))
          video-ids    (->> instances
                            (filter #(= "video" (:type %)))
                            (map :_id)
                            (filter some?))
          summary      (:instances-summary container)
          total-q      (or (:total-questions summary) 0)
          total-dur    (or (:total-duration-seconds summary) 0)
          question-txt (fetch-question-texts mongodb question-ids)
          video-txt    (fetch-video-titles mongodb video-ids)
          parts        (cond-> [(str "Playlist: " name)]
                         (not= name description)
                         (conj (str "Deskripsi: " description))

                         (pos? total-q)
                         (conj (str "Jumlah soal: " total-q))

                         (pos? total-dur)
                         (conj (str "Durasi video: " (int (/ total-dur 60)) " menit"))

                         (seq question-txt)
                         (conj (str "Soal-soal: " question-txt))

                         (seq video-txt)
                         (conj (str "Video: " video-txt)))]
      (clojure.string/join ". " parts))
    (catch Exception e
      (println (str "  ERROR in build-enriched-text for container _id=" (:_id container)
                    " name=" (:name container)))
      (println (str "  Exception: " (.getMessage e)))
      (.printStackTrace e)
      (throw e))))

;; ---- Batch Processing ----

(defn- process-batch
  "Process a batch of containers: build enriched text, generate embeddings,
   and return vector documents ready for insertion."
  [mongodb openai-config containers]
  (println "  Building enriched texts...")
  (let [texts      (mapv #(build-enriched-text mongodb %) containers)
        _          (println (str "  Built " (count texts) " texts. Calling OpenAI embeddings..."))
        embeddings (openai/create-embeddings-batch
                    (:api-key openai-config)
                    (:embedding-model openai-config)
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

(defn generate-and-store-embeddings!
  "Main pipeline: fetch all containers, generate enriched embeddings,
   and store them in the container-vectors collection.

   Options:
     :batch-size - number of containers to process at once (default 20)
     :clear?     - whether to clear existing vectors first (default true)"
  [mongodb openai-config & {:keys [batch-size clear?]
                            :or   {batch-size 20 clear? true}}]
  (println "Starting embedding pipeline...")
  (let [containers (mongo/find-all mongodb "containers")]
    (println (str "Found " (count containers) " containers to process."))

    (when clear?
      (println "Clearing existing container-vectors...")
      (mongo/delete-many! mongodb "container-vectors" {}))

    (let [batches (partition-all batch-size containers)
          total   (count batches)]
      (doseq [[idx batch] (map-indexed vector batches)]
        (println (str "Processing batch " (inc idx) "/" total
                      " (" (count batch) " containers)..."))
        (let [vector-docs (process-batch mongodb openai-config batch)]
          (mongo/insert-many! mongodb "container-vectors" vector-docs)
          (println (str "  Inserted " (count vector-docs) " vector documents."))))

      (let [total-vectors (mongo/count-documents mongodb "container-vectors")]
        (println (str "Pipeline complete. Total vectors stored: " total-vectors))
        total-vectors))))
