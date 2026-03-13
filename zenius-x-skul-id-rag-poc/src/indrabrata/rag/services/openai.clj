(ns indrabrata.rag.services.openai
  "OpenAI API client for embeddings and chat completions."
  (:require
   [cheshire.core :as json]
   [clj-http.client :as http]
   [clojure.string :as str]))

(def ^:private base-url "https://api.openai.com/v1")

(defn- api-request
  "Make an authenticated request to the OpenAI API."
  [api-key method endpoint body]
  (let [url  (str base-url endpoint)
        opts {:headers          {"Authorization" (str "Bearer " api-key)
                                 "Content-Type"  "application/json"}
              :body             (json/generate-string body)
              :as               :json
              :content-type     :json
              :throw-exceptions false}
        resp (case method
               :post (http/post url opts))]
    (when-not (<= 200 (:status resp) 299)
      (println (str "OpenAI API error [" (:status resp) "]: " (:body resp)))
      (throw (ex-info (str "OpenAI API error: " (:status resp))
                      {:status (:status resp) :body (:body resp)})))
    (:body resp)))

;; ---- Embeddings ----

(defn create-embedding
  "Generate an embedding vector for the given text.
   Returns {:embedding [...] :usage {:prompt_tokens N :total_tokens N}}."
  [api-key model text]
  (let [resp (api-request api-key :post "/embeddings"
                          {:model model
                           :input text})]
    {:embedding (-> resp :data first :embedding)
     :usage     (:usage resp)}))

(defn create-embeddings-batch
  "Generate embeddings for a batch of texts.
   Returns a list of {:index N :embedding [...]} maps."
  [api-key model texts]
  (let [resp (api-request api-key :post "/embeddings"
                          {:model model
                           :input (vec texts)})]
    (->> (:data resp)
         (sort-by :index)
         (mapv :embedding))))

;; ---- Files API ----

(defn- infer-content-type
  "Infer MIME type from filename extension, falling back to provided content-type."
  [filename content-type]
  (let [lower (clojure.string/lower-case (or filename ""))]
    (cond
      (clojure.string/ends-with? lower ".pdf")  "application/pdf"
      (clojure.string/ends-with? lower ".txt")  "text/plain"
      (clojure.string/ends-with? lower ".md")   "text/markdown"
      (clojure.string/ends-with? lower ".csv")  "text/csv"
      (clojure.string/ends-with? lower ".json") "application/json"
      (clojure.string/ends-with? lower ".docx") "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
      (clojure.string/ends-with? lower ".pptx") "application/vnd.openxmlformats-officedocument.presentationml.presentation"
      (clojure.string/ends-with? lower ".xlsx") "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
      (and content-type (not= content-type "application/octet-stream")) content-type
      :else "application/octet-stream")))

(defn upload-file
  "Upload raw file bytes to OpenAI Files API with purpose=user_data.
   Returns the full file object map (includes :id)."
  [api-key ^bytes file-bytes filename content-type]
  (let [url           (str base-url "/files")
        resolved-type (infer-content-type filename content-type)
        _             (println (str "  Uploading as content-type: " resolved-type))
        resp (http/post url
                        {:headers      {"Authorization" (str "Bearer " api-key)}
                         :multipart    [{:name "purpose"  :content "user_data"}
                                        {:name         "file"
                                         :content      (java.io.ByteArrayInputStream. file-bytes)
                                         :filename     filename
                                         :content-type resolved-type}]
                         :as           :json
                         :throw-exceptions false})]
    (when-not (<= 200 (:status resp) 299)
      (println (str "OpenAI Files API upload error [" (:status resp) "]: " (:body resp)))
      (throw (ex-info (str "OpenAI Files API error: " (:status resp))
                      {:status (:status resp) :body (:body resp)})))
    (:body resp)))

(defn delete-file
  "Delete a file from OpenAI Files API. Silently ignores errors."
  [api-key file-id]
  (try
    (http/delete (str base-url "/files/" file-id)
                 {:headers          {"Authorization" (str "Bearer " api-key)}
                  :as               :json
                  :throw-exceptions false})
    (catch Exception _ nil)))

;; ---- Responses API ----

(defn responses-completion-with-file-id
  "Send a request to the OpenAI Responses API using a pre-uploaded file_id.
   Returns {:content \"...\" :usage {...}}."
  [api-key model file-id question]
  (let [resp (api-request api-key :post "/responses"
                          {:model model
                           :input [{:role    "user"
                                    :content [{:type    "input_file"
                                               :file_id file-id}
                                              {:type "input_text"
                                               :text question}]}]})]
    {:content (-> resp :output first :content first :text)
     :usage   {:prompt_tokens     (-> resp :usage :input_tokens)
               :completion_tokens (-> resp :usage :output_tokens)
               :total_tokens      (-> resp :usage :total_tokens)}}))

(defn responses-completion
  "Send a request to the OpenAI Responses API with an inline base64-encoded file.
   No prior upload needed — file bytes are sent directly as file_data.
   Returns {:content \"...\" :usage {:prompt_tokens N :completion_tokens N :total_tokens N}}."
  [api-key model ^bytes file-bytes filename content-type question]
  (let [mime-type  (infer-content-type filename content-type)
        b64-data   (str "data:" mime-type ";base64,"
                        (.encodeToString (java.util.Base64/getEncoder) file-bytes))
        resp (api-request api-key :post "/responses"
                          {:model model
                           :input [{:role    "user"
                                    :content [{:type      "input_file"
                                               :filename  filename
                                               :file_data b64-data}
                                              {:type "input_text"
                                               :text question}]}]})
        ;; output may contain reasoning steps before the actual message
        message-item (->> (:output resp)
                          (filter #(= (:type %) "message"))
                          first)
        text         (-> message-item :content first :text)]
    {:content text
     :usage   {:prompt_tokens     (-> resp :usage :input_tokens)
               :completion_tokens (-> resp :usage :output_tokens)
               :total_tokens      (-> resp :usage :total_tokens)}}))

(defn responses-completion-json
  "Like responses-completion but requests JSON-formatted output.
   Returns {:content <parsed-map> :usage {...}}."
  [api-key model ^bytes file-bytes filename content-type question]
  (let [mime-type  (infer-content-type filename content-type)
        b64-data   (str "data:" mime-type ";base64,"
                        (.encodeToString (java.util.Base64/getEncoder) file-bytes))
        resp (api-request api-key :post "/responses"
                          {:model model
                           :input [{:role    "user"
                                    :content [{:type      "input_file"
                                               :filename  filename
                                               :file_data b64-data}
                                              {:type "input_text"
                                               :text question}]}]
                           :text  {:format {:type "json_object"}}})
        message-item (->> (:output resp)
                          (filter #(= (:type %) "message"))
                          first)
        raw          (-> message-item :content first :text)]
    {:content (json/parse-string raw true)
     :usage   {:prompt_tokens     (-> resp :usage :input_tokens)
               :completion_tokens (-> resp :usage :output_tokens)
               :total_tokens      (-> resp :usage :total_tokens)}}))

;; ---- Chat Completions ----

(defn chat-completion
  "Send a chat completion request. messages is a seq of {:role \"...\" :content \"...\"}
   Returns {:content \"...\" :usage {:prompt_tokens N :completion_tokens N :total_tokens N}}."
  [api-key model messages]
  (let [resp (api-request api-key :post "/chat/completions"
                          {:model    model
                           :messages messages})]
    {:content (-> resp :choices first :message :content)
     :usage   (:usage resp)}))

(defn chat-completion-json
  "Send a chat completion request with JSON response format.
   The LLM is instructed to return valid JSON. messages is a seq of {:role \"...\" :content \"...\"}.
   Returns {:content <parsed-json> :raw-content \"...\" :usage {...}}."
  [api-key model messages]
  (let [resp (api-request api-key :post "/chat/completions"
                          {:model           model
                           :messages        messages
                           :response_format {:type "json_object"}})
        raw  (-> resp :choices first :message :content)]
    {:content     (json/parse-string raw true)
     :raw-content raw
     :usage       (:usage resp)}))
