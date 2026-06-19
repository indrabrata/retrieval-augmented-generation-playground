(ns indrabrata.rag.routes
  "Pedestal routes and handlers for the RAG API."
  (:require
   [cheshire.core :as json]
   [clojure.java.io :as io]
   [clojure.string :as str]
   [indrabrata.rag.components.mongodb :as mongo]
   [indrabrata.rag.services.embedding-pipeline-openai :as pipeline]
   [indrabrata.rag.services.embedding-pipeline-gemini :as pipeline-gemini]
   [indrabrata.rag.services.file-rag-openai :as file-rag]
   [indrabrata.rag.services.file-rag-gemini :as file-rag-gemini]
   [indrabrata.rag.services.image-rag-openai :as image-rag]
   [indrabrata.rag.services.image-rag-gemini :as image-rag-gemini]
   [indrabrata.rag.services.naive-rag-openai :as naive-rag]
   [indrabrata.rag.services.naive-rag-gemini :as naive-rag-gemini]
   [indrabrata.rag.services.rag-openai :as rag]
   [indrabrata.rag.services.rag-gemini :as rag-gemini]
   [io.pedestal.http.ring-middlewares :as ring-mw])
  (:import
   [java.nio.file Files]))

;; ---- Response Helpers ----

(defn- json-response
  "Create a JSON response map."
  [status body]
  {:status  status
   :headers {"Content-Type" "application/json"}
   :body    (json/generate-string body)})

;; ---- Handlers ----

(defn health-handler
  "Health check endpoint."
  [_request]
  (json-response 200 {:status "ok" :service "rag-poc"}))

(defn chat-handler
  "POST /api/rag/chat - Main RAG chat endpoint.
   Body: {\"question\": \"...\", \"top_k\": 5}
   Returns: {\"answer\": \"...\", \"sources\": [...]}"
  [{:keys [components json-params]}]
  (let [{:keys [mongodb openai-config]} components
        question (:question json-params)
        top-k    (or (:top_k json-params) 5)]
    (if (str/blank? question)
      (json-response 400 {:error "question is required"})
      (try
        (let [result (rag/query mongodb openai-config question :top-k top-k)]
          (json-response 200 result))
        (catch Exception e
          (println (str "Error in chat handler: " (.getMessage e)))
          (json-response 500 {:error "Internal server error"
                              :message (.getMessage e)}))))))

(defn naive-chat-handler
  "POST /api/naive-rag/chat - Naive RAG chat endpoint.
   LLM answers the question and extracts keywords; keywords are used
   to search MongoDB containers by name similarity (no embeddings).
   Body: {\"question\": \"...\", \"top_k\": 5}
   Returns: {\"answer\": \"...\", \"keywords\": [...], \"sources\": [...]}"
  [{:keys [components json-params]}]
  (let [{:keys [mongodb openai-config]} components
        question (:question json-params)
        top-k    (or (:top_k json-params) 5)]
    (if (str/blank? question)
      (json-response 400 {:error "question is required"})
      (try
        (let [result (naive-rag/query mongodb openai-config question :top-k top-k)]
          (json-response 200 result))
        (catch Exception e
          (println (str "Error in naive chat handler: " (.getMessage e)))
          (json-response 500 {:error "Internal server error"
                              :message (.getMessage e)}))))))

(defn embed-handler
  "POST /api/embeddings/generate - Trigger embedding generation pipeline.
   Body: {\"batch_size\": 20, \"clear\": true}
   Returns: {\"total_vectors\": N}"
  [{:keys [components json-params]}]
  (let [{:keys [mongodb openai-config]} components
        batch-size (or (:batch_size json-params) 5)
        clear?     (if (contains? json-params :clear) (:clear json-params) true)]
    (cond
      (nil? mongodb)
      (json-response 500 {:error "MongoDB component is not available"})

      (nil? (:api-key openai-config))
      (json-response 500 {:error "OpenAI API key is not configured"})

      (nil? (:embedding-model openai-config))
      (json-response 500 {:error "OpenAI embedding model is not configured"})

      :else
      (try
        (let [total (pipeline/generate-and-store-embeddings!
                     mongodb openai-config
                     :batch-size batch-size
                     :clear? clear?)]
          (json-response 200 {:status        "completed"
                              :total_vectors total}))
        (catch Exception e
          (println (str "Error in embed handler: " (.getMessage e)))
          (json-response 500 {:error   "Internal server error"
                              :message (.getMessage e)}))))))

(defn stats-handler
  "GET /api/stats - Get collection statistics."
  [{:keys [components]}]
  (let [{:keys [mongodb]} components]
    (try
      (let [containers (mongo/count-documents mongodb "containers")
            vectors    (mongo/count-documents mongodb "container-vectors")
            questions  (mongo/count-documents mongodb "question-instances")
            videos     (mongo/count-documents mongodb "video-instances")]
        (json-response 200 {:containers       containers
                            :container_vectors vectors
                            :question_instances questions
                            :video_instances   videos}))
      (catch Exception e
        (json-response 500 {:error (.getMessage e)})))))

(defn- read-upload-file
  "Read the raw bytes from a Ring multipart file map (:tempfile key)."
  [file-info]
  (Files/readAllBytes (.toPath (:tempfile file-info))))

(defn upload-chat-handler
  "POST /api/rag/upload-and-chat - RAG on an uploaded file (in-memory vector search).
   Multipart fields: file (required), question (required), top_k (optional, default 5)
   Returns: {\"answer\": \"...\", \"sources\": [...], \"usage\": {...}}"
  [{:keys [components multipart-params]}]
  (let [{:keys [mongodb openai-config]} components
        file-info    (get multipart-params "file")
        question     (get multipart-params "question")
        top-k        (or (some-> (get multipart-params "top_k")
                                 str/trim
                                 (Integer/parseInt))
                         5)]
    (cond
      (str/blank? question)
      (json-response 400 {:error "question is required"})

      (nil? file-info)
      (json-response 400 {:error "file is required"})

      :else
      (try
        (let [file-bytes   (read-upload-file file-info)
              filename     (:filename file-info)
              content-type (:content-type file-info)
              result       (file-rag/rag-query mongodb openai-config file-bytes
                                               filename content-type
                                               question :top-k top-k)]
          (json-response 200 result))
        (catch Exception e
          (println (str "Error in upload chat handler: " (.getMessage e)))
          (json-response 500 {:error "Internal server error"
                              :message (.getMessage e)}))))))

(defn upload-naive-chat-handler
  "POST /api/naive-rag/upload-and-chat - Naive RAG on an uploaded file (keyword search).
   Multipart fields: file (required), question (required), top_k (optional, default 5)
   Returns: {\"answer\": \"...\", \"keywords\": [...], \"sources\": [...], \"usage\": {...}}"
  [{:keys [components multipart-params]}]
  (let [{:keys [mongodb openai-config]} components
        file-info    (get multipart-params "file")
        question     (get multipart-params "question")
        top-k        (or (some-> (get multipart-params "top_k")
                                 str/trim
                                 (Integer/parseInt))
                         5)]
    (cond
      (str/blank? question)
      (json-response 400 {:error "question is required"})

      (nil? file-info)
      (json-response 400 {:error "file is required"})

      :else
      (try
        (let [file-bytes   (read-upload-file file-info)
              filename     (:filename file-info)
              content-type (:content-type file-info)
              result       (file-rag/naive-rag-query mongodb openai-config file-bytes
                                                     filename content-type
                                                     question :top-k top-k)]
          (json-response 200 result))
        (catch Exception e
          (println (str "Error in upload naive chat handler: " (.getMessage e)))
          (json-response 500 {:error "Internal server error"
                              :message (.getMessage e)}))))))

(defn upload-files-api-chat-handler
  "POST /openai/files-api/upload-and-chat - Upload file to OpenAI Files API and query via Responses API.
   No local PDF extraction or embeddings. OpenAI handles the file natively.
   Multipart fields: file (required), question (required)
   Returns: {\"answer\": \"...\", \"usage\": {...}}"
  [{:keys [components multipart-params]}]
  (let [{:keys [openai-config]} components
        file-info (get multipart-params "file")
        question  (get multipart-params "question")]
    (cond
      (str/blank? question)
      (json-response 400 {:error "question is required"})

      (nil? file-info)
      (json-response 400 {:error "file is required"})

      :else
      (try
        (let [file-bytes   (read-upload-file file-info)
              filename     (:filename file-info)
              content-type (:content-type file-info)
              result       (file-rag/files-api-query openai-config file-bytes
                                                     filename content-type question)]
          (json-response 200 result))
        (catch Exception e
          (println (str "Error in files-api chat handler: " (.getMessage e)))
          (json-response 500 {:error "Internal server error"
                              :message (.getMessage e)}))))))

(defn upload-image-chat-handler
  "POST /api/rag/upload-image-and-chat - RAG on an uploaded image using OpenAI vision.
   Multipart fields: image (required), question (required), top_k (optional, default 5)
   Returns: {\"answer\": \"...\", \"sources\": [...], \"usage\": {...}}"
  [{:keys [components multipart-params]}]
  (let [{:keys [mongodb openai-config]} components
        image-info (get multipart-params "image")
        question   (get multipart-params "question")
        top-k      (or (some-> (get multipart-params "top_k")
                               str/trim
                               (Integer/parseInt))
                       5)]
    (cond
      (str/blank? question)
      (json-response 400 {:error "question is required"})

      (nil? image-info)
      (json-response 400 {:error "image is required"})

      :else
      (try
        (let [image-bytes  (read-upload-file image-info)
              content-type (or (:content-type image-info) "image/jpeg")
              result       (image-rag/image-rag-query
                            mongodb openai-config
                            image-bytes content-type
                            question :top-k top-k)]
          (json-response 200 result))
        (catch Exception e
          (println (str "Error in upload image chat handler: " (.getMessage e)))
          (json-response 500 {:error   "Internal server error"
                              :message (.getMessage e)}))))))

(defn upload-image-naive-chat-handler
  "POST /api/naive-rag/upload-image-and-chat - Naive RAG on an uploaded image using OpenAI vision.
   Vision LLM analyzes the image and extracts keywords; keywords are used to search MongoDB.
   Multipart fields: image (required), question (required), top_k (optional, default 5)
   Returns: {\"answer\": \"...\", \"keywords\": [...], \"sources\": [...], \"usage\": {...}}"
  [{:keys [components multipart-params]}]
  (let [{:keys [mongodb openai-config]} components
        image-info (get multipart-params "image")
        question   (get multipart-params "question")
        top-k      (or (some-> (get multipart-params "top_k")
                               str/trim
                               (Integer/parseInt))
                       5)]
    (cond
      (str/blank? question)
      (json-response 400 {:error "question is required"})

      (nil? image-info)
      (json-response 400 {:error "image is required"})

      :else
      (try
        (let [image-bytes  (read-upload-file image-info)
              content-type (or (:content-type image-info) "image/jpeg")
              result       (image-rag/image-naive-rag-query
                            mongodb openai-config
                            image-bytes content-type
                            question :top-k top-k)]
          (json-response 200 result))
        (catch Exception e
          (println (str "Error in upload image naive chat handler: " (.getMessage e)))
          (json-response 500 {:error   "Internal server error"
                              :message (.getMessage e)}))))))

;; ---- Gemini Handlers ----

(defn conversation-handler
  "POST /openai/conversation - Multi-turn chat without RAG.
   Body: {\"messages\": [{\"role\": \"user\", \"content\": \"...\"}, ...]}
   Returns: {\"answer\": \"...\", \"usage\": {...}}"
  [{:keys [components json-params]}]
  (let [{:keys [openai-config]} components
        messages    (:messages json-params)
        window-size (or (:window_size json-params) 10)]
    (cond
      (empty? messages)
      (json-response 400 {:error "messages is required and must not be empty"})

      (not= "user" (:role (last messages)))
      (json-response 400 {:error "last message must have role \"user\""})

      :else
      (try
        (json-response 200 (rag/conversation-query openai-config messages :window-size window-size))
        (catch Exception e
          (println (str "Error in conversation handler: " (.getMessage e)))
          (json-response 500 {:error "Internal server error" :message (.getMessage e)}))))))

(defn gemini-conversation-handler
  "POST /gemini/conversation - Multi-turn chat without RAG.
   Body: {\"messages\": [{\"role\": \"user\", \"content\": \"...\"}, ...]}
   Returns: {\"answer\": \"...\", \"usage\": {...}}"
  [{:keys [components json-params]}]
  (let [{:keys [gemini-config]} components
        messages    (:messages json-params)
        window-size (or (:window_size json-params) 10)]
    (cond
      (empty? messages)
      (json-response 400 {:error "messages is required and must not be empty"})

      (not= "user" (:role (last messages)))
      (json-response 400 {:error "last message must have role \"user\""})

      :else
      (try
        (json-response 200 (rag-gemini/conversation-query gemini-config messages :window-size window-size))
        (catch Exception e
          (println (str "Error in gemini conversation handler: " (.getMessage e)))
          (json-response 500 {:error "Internal server error" :message (.getMessage e)}))))))

(defn gemini-embed-handler
  "POST /gemini/embeddings/generate - Trigger Gemini embedding generation pipeline.
   Body: {\"batch_size\": 20, \"clear\": true}
   Returns: {\"total_vectors\": N}"
  [{:keys [components json-params]}]
  (let [{:keys [mongodb gemini-config]} components
        batch-size (or (:batch_size json-params) 20)
        clear?     (if (contains? json-params :clear) (:clear json-params) true)]
    (cond
      (nil? mongodb)
      (json-response 500 {:error "MongoDB component is not available"})

      (nil? (:api-key gemini-config))
      (json-response 500 {:error "Gemini API key is not configured"})

      :else
      (try
        (let [total (pipeline-gemini/generate-and-store-embeddings!
                     mongodb gemini-config
                     :batch-size batch-size
                     :clear? clear?)]
          (json-response 200 {:status "completed" :total_vectors total}))
        (catch Exception e
          (println (str "Error in gemini embed handler: " (.getMessage e)))
          (json-response 500 {:error "Internal server error" :message (.getMessage e)}))))))

(defn gemini-chat-handler
  "POST /gemini/rag/chat - Gemini vector RAG chat endpoint."
  [{:keys [components json-params]}]
  (let [{:keys [mongodb gemini-config]} components
        question (:question json-params)
        top-k    (or (:top_k json-params) 5)]
    (if (str/blank? question)
      (json-response 400 {:error "question is required"})
      (try
        (json-response 200 (rag-gemini/query mongodb gemini-config question :top-k top-k))
        (catch Exception e
          (println (str "Error in gemini chat handler: " (.getMessage e)))
          (json-response 500 {:error "Internal server error" :message (.getMessage e)}))))))

(defn gemini-naive-chat-handler
  "POST /gemini/naive-rag/chat - Gemini naive RAG chat endpoint."
  [{:keys [components json-params]}]
  (let [{:keys [mongodb gemini-config]} components
        question (:question json-params)
        top-k    (or (:top_k json-params) 5)]
    (if (str/blank? question)
      (json-response 400 {:error "question is required"})
      (try
        (json-response 200 (naive-rag-gemini/query mongodb gemini-config question :top-k top-k))
        (catch Exception e
          (println (str "Error in gemini naive chat handler: " (.getMessage e)))
          (json-response 500 {:error "Internal server error" :message (.getMessage e)}))))))

(defn gemini-files-api-chat-handler
  "POST /gemini/files-api/upload-and-chat - Query a file via Gemini inline_data.
   No local PDF extraction or embeddings. File is base64-encoded and sent inline.
   Multipart fields: file (required), question (required)
   Returns: {\"answer\": \"...\", \"usage\": {...}}"
  [{:keys [components multipart-params]}]
  (let [{:keys [gemini-config]} components
        file-info (get multipart-params "file")
        question  (get multipart-params "question")]
    (cond
      (str/blank? question) (json-response 400 {:error "question is required"})
      (nil? file-info)      (json-response 400 {:error "file is required"})
      :else
      (try
        (let [file-bytes   (read-upload-file file-info)
              filename     (:filename file-info)
              content-type (:content-type file-info)
              result       (file-rag-gemini/files-api-query gemini-config file-bytes
                                                            filename content-type question)]
          (json-response 200 result))
        (catch Exception e
          (println (str "Error in gemini files-api chat handler: " (.getMessage e)))
          (json-response 500 {:error "Internal server error" :message (.getMessage e)}))))))

(defn gemini-upload-chat-handler
  "POST /gemini/rag/upload-pdf-and-chat - Gemini file RAG endpoint."
  [{:keys [components multipart-params]}]
  (let [{:keys [mongodb gemini-config]} components
        file-info (get multipart-params "file")
        question  (get multipart-params "question")
        top-k     (or (some-> (get multipart-params "top_k") str/trim (Integer/parseInt)) 5)]
    (cond
      (str/blank? question) (json-response 400 {:error "question is required"})
      (nil? file-info)      (json-response 400 {:error "file is required"})
      :else
      (try
        (let [file-bytes   (read-upload-file file-info)
              filename     (:filename file-info)
              content-type (:content-type file-info)
              result       (file-rag-gemini/rag-query mongodb gemini-config file-bytes
                                                      filename content-type
                                                      question :top-k top-k)]
          (json-response 200 result))
        (catch Exception e
          (println (str "Error in gemini upload chat handler: " (.getMessage e)))
          (json-response 500 {:error "Internal server error" :message (.getMessage e)}))))))

(defn gemini-upload-naive-chat-handler
  "POST /gemini/naive-rag/upload-pdf-and-chat - Gemini file naive RAG endpoint."
  [{:keys [components multipart-params]}]
  (let [{:keys [mongodb gemini-config]} components
        file-info (get multipart-params "file")
        question  (get multipart-params "question")
        top-k     (or (some-> (get multipart-params "top_k") str/trim (Integer/parseInt)) 5)]
    (cond
      (str/blank? question) (json-response 400 {:error "question is required"})
      (nil? file-info)      (json-response 400 {:error "file is required"})
      :else
      (try
        (let [file-bytes   (read-upload-file file-info)
              filename     (:filename file-info)
              content-type (:content-type file-info)
              result       (file-rag-gemini/naive-rag-query mongodb gemini-config file-bytes
                                                            filename content-type
                                                            question :top-k top-k)]
          (json-response 200 result))
        (catch Exception e
          (println (str "Error in gemini upload naive chat handler: " (.getMessage e)))
          (json-response 500 {:error "Internal server error" :message (.getMessage e)}))))))

(defn gemini-upload-image-chat-handler
  "POST /gemini/rag/upload-image-and-chat - Gemini image RAG endpoint."
  [{:keys [components multipart-params]}]
  (let [{:keys [mongodb gemini-config]} components
        image-info (get multipart-params "image")
        question   (get multipart-params "question")
        top-k      (or (some-> (get multipart-params "top_k") str/trim (Integer/parseInt)) 5)]
    (cond
      (str/blank? question) (json-response 400 {:error "question is required"})
      (nil? image-info)     (json-response 400 {:error "image is required"})
      :else
      (try
        (let [image-bytes  (read-upload-file image-info)
              content-type (or (:content-type image-info) "image/jpeg")
              result       (image-rag-gemini/image-rag-query
                            mongodb gemini-config
                            image-bytes content-type
                            question :top-k top-k)]
          (json-response 200 result))
        (catch Exception e
          (println (str "Error in gemini upload image chat handler: " (.getMessage e)))
          (json-response 500 {:error "Internal server error" :message (.getMessage e)}))))))

(defn gemini-upload-image-naive-chat-handler
  "POST /gemini/naive-rag/upload-image-and-chat - Gemini image naive RAG endpoint."
  [{:keys [components multipart-params]}]
  (let [{:keys [mongodb gemini-config]} components
        image-info (get multipart-params "image")
        question   (get multipart-params "question")
        top-k      (or (some-> (get multipart-params "top_k") str/trim (Integer/parseInt)) 5)]
    (cond
      (str/blank? question) (json-response 400 {:error "question is required"})
      (nil? image-info)     (json-response 400 {:error "image is required"})
      :else
      (try
        (let [image-bytes  (read-upload-file image-info)
              content-type (or (:content-type image-info) "image/jpeg")
              result       (image-rag-gemini/image-naive-rag-query
                            mongodb gemini-config
                            image-bytes content-type
                            question :top-k top-k)]
          (json-response 200 result))
        (catch Exception e
          (println (str "Error in gemini upload image naive chat handler: " (.getMessage e)))
          (json-response 500 {:error "Internal server error" :message (.getMessage e)}))))))

;; ---- Static Resource Handlers ----

(defn openapi-json-handler
  "GET /openapi.json - Serve the OpenAPI specification."
  [_request]
  (if-let [resource (io/resource "public/openapi.json")]
    {:status  200
     :headers {"Content-Type" "application/json"}
     :body    (slurp resource)}
    {:status 404
     :headers {"Content-Type" "text/plain"}
     :body "Not found"}))

;; ---- Interceptor to inject components ----

(defn components-interceptor
  "Interceptor that injects system components into the request."
  [components]
  {:name  :components
   :enter (fn [context]
            (assoc-in context [:request :components] components))})

;; ---- Route Table ----

(defn routes
  "Build the route table. components-map is {:mongodb ... :openai-config ...}"
  [components-map]
  (let [inject (components-interceptor components-map)]
    #{["/health"                                :get  health-handler                    :route-name :health]
      ["/openai/rag/chat"                        :post [inject chat-handler]              :route-name :chat]
      ["/openai/rag/upload-pdf-and-chat"             :post [inject (ring-mw/multipart-params) upload-chat-handler]             :route-name :rag-upload-chat]
      ["/openai/files-api/upload-and-chat"           :post [inject (ring-mw/multipart-params) upload-files-api-chat-handler]    :route-name :files-api-upload-chat]
      ["/openai/rag/upload-image-and-chat"       :post [inject (ring-mw/multipart-params) upload-image-chat-handler]       :route-name :rag-upload-image-chat]
      ["/openai/naive-rag/chat"                  :post [inject naive-chat-handler]              :route-name :naive-rag-chat]
      ["/openai/naive-rag/upload-pdf-and-chat"       :post [inject (ring-mw/multipart-params) upload-naive-chat-handler]       :route-name :naive-rag-upload-chat]
      ["/openai/naive-rag/upload-image-and-chat" :post [inject (ring-mw/multipart-params) upload-image-naive-chat-handler] :route-name :naive-rag-upload-image-chat]
      ["/openai/conversation"              :post [inject conversation-handler]      :route-name :conversation]
      ["/openai/embeddings/generate"       :post [inject embed-handler]             :route-name :embed]
      ["/stats"                     :get  [inject stats-handler]             :route-name :stats]
      ["/openapi.json"                  :get  openapi-json-handler               :route-name :openapi-json]

      ;; Gemini endpoints
      ["/gemini/conversation"                    :post [inject gemini-conversation-handler]                                          :route-name :gemini-conversation]
      ["/gemini/embeddings/generate"             :post [inject gemini-embed-handler]                                             :route-name :gemini-embed]
      ["/gemini/rag/chat"                        :post [inject gemini-chat-handler]                                              :route-name :gemini-chat]
      ["/gemini/files-api/upload-and-chat"        :post [inject (ring-mw/multipart-params) gemini-files-api-chat-handler]        :route-name :gemini-files-api-upload-chat]
      ["/gemini/rag/upload-pdf-and-chat"         :post [inject (ring-mw/multipart-params) gemini-upload-chat-handler]           :route-name :gemini-rag-upload-chat]
      ["/gemini/rag/upload-image-and-chat"       :post [inject (ring-mw/multipart-params) gemini-upload-image-chat-handler]     :route-name :gemini-rag-upload-image-chat]
      ["/gemini/naive-rag/chat"                  :post [inject gemini-naive-chat-handler]                                       :route-name :gemini-naive-rag-chat]
      ["/gemini/naive-rag/upload-pdf-and-chat"   :post [inject (ring-mw/multipart-params) gemini-upload-naive-chat-handler]     :route-name :gemini-naive-rag-upload-chat]
      ["/gemini/naive-rag/upload-image-and-chat" :post [inject (ring-mw/multipart-params) gemini-upload-image-naive-chat-handler] :route-name :gemini-naive-rag-upload-image-chat]}))
