(ns indrabrata.rag.routes
  "Pedestal routes and handlers for the RAG API."
  (:require [cheshire.core :as json]
            [clojure.java.io :as io]
            [indrabrata.rag.services.rag :as rag]
            [indrabrata.rag.services.embedding-pipeline :as pipeline]
            [indrabrata.rag.components.mongodb :as mongo]))

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
  "POST /api/chat - Main RAG chat endpoint.
   Body: {\"question\": \"...\", \"top_k\": 5}
   Returns: {\"answer\": \"...\", \"sources\": [...]}"
  [{:keys [components json-params] :as request}]
  (let [{:keys [mongodb openai-config]} components
        question (:question json-params)
        top-k    (or (:top_k json-params) 5)]
    (if (clojure.string/blank? question)
      (json-response 400 {:error "question is required"})
      (try
        (let [result (rag/query mongodb openai-config question :top-k top-k)]
          (json-response 200 result))
        (catch Exception e
          (println (str "Error in chat handler: " (.getMessage e)))
          (json-response 500 {:error "Internal server error"
                              :message (.getMessage e)}))))))

(defn embed-handler
  "POST /api/embeddings/generate - Trigger embedding generation pipeline.
   Body: {\"batch_size\": 20, \"clear\": true}
   Returns: {\"total_vectors\": N}"
  [{:keys [components json-params] :as request}]
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
  [{:keys [components] :as request}]
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
    #{["/health"                  :get  health-handler       :route-name :health]
      ["/api/rag/chat"                :post [inject chat-handler]  :route-name :chat]
      ["/api/embeddings/generate" :post [inject embed-handler] :route-name :embed]
      ["/api/stats"               :get  [inject stats-handler] :route-name :stats]
      ["/openapi.json"            :get  openapi-json-handler   :route-name :openapi-json]}))
