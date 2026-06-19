(ns indrabrata.rag.services.gemini
  "Google Gemini API client for embeddings and chat completions."
  (:require
   [cheshire.core :as json]
   [clj-http.client :as http]
   [clojure.string :as str]))

(def ^:private base-url "https://generativelanguage.googleapis.com/v1beta")

(defn- api-request
  "Make an authenticated request to the Gemini API."
  [api-key model-endpoint body]
  (let [url  (str base-url "/models/" model-endpoint "?key=" api-key)
        opts {:headers          {"Content-Type" "application/json"}
              :body             (json/generate-string body)
              :as               :json
              :throw-exceptions false}
        resp (http/post url opts)]
    (when-not (<= 200 (:status resp) 299)
      (println (str "Gemini API error [" (:status resp) "]: " (:body resp)))
      (throw (ex-info (str "Gemini API error: " (:status resp))
                      {:status (:status resp) :body (:body resp)})))
    (:body resp)))

;; ---- Embeddings ----

(defn create-embedding
  "Generate an embedding vector for the given text using Gemini.
   Returns {:embedding [...] :usage {}}."
  [api-key model text]
  (let [resp (api-request api-key
                          (str model ":embedContent")
                          {:model   (str "models/" model)
                           :content {:parts [{:text text}]}})]
    {:embedding (-> resp :embedding :values)
     :usage     {}}))

(defn create-embeddings-batch
  "Generate embeddings for a batch of texts using Gemini batchEmbedContents.
   Returns a vector of embedding vectors, in the same order as texts."
  [api-key model texts]
  (let [requests (mapv (fn [t]
                         {:model   (str "models/" model)
                          :content {:parts [{:text t}]}})
                       texts)
        resp     (api-request api-key
                              (str model ":batchEmbedContents")
                              {:requests requests})]
    (mapv #(-> % :values) (:embeddings resp))))

;; ---- Inline File Completion ----

(defn- infer-mime-type
  "Infer MIME type from filename extension."
  [filename content-type]
  (let [lower (clojure.string/lower-case (or filename ""))]
    (cond
      (str/ends-with? lower ".pdf")  "application/pdf"
      (str/ends-with? lower ".txt")  "text/plain"
      (str/ends-with? lower ".md")   "text/markdown"
      (str/ends-with? lower ".csv")  "text/csv"
      (str/ends-with? lower ".json") "application/json"
      (str/ends-with? lower ".docx") "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
      (str/ends-with? lower ".pptx") "application/vnd.openxmlformats-officedocument.presentationml.presentation"
      (str/ends-with? lower ".xlsx") "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
      (and content-type (not= content-type "application/octet-stream")) content-type
      :else "application/octet-stream")))

(defn inline-file-completion
  "Send a generateContent request to Gemini with an inline base64-encoded file.
   Uses inline_data part — no file upload needed.
   Returns {:content \"...\" :usage {:prompt_tokens N :completion_tokens N :total_tokens N}}."
  [api-key model ^bytes file-bytes filename content-type question]
  (let [mime-type (infer-mime-type filename content-type)
        b64-data  (.encodeToString (java.util.Base64/getEncoder) file-bytes)
        _         (println (str "  Sending inline file as mime-type: " mime-type))
        body      {:contents [{:parts [{:inline_data {:mime_type mime-type
                                                      :data      b64-data}}
                                       {:text question}]}]}
        resp      (api-request api-key (str model ":generateContent") body)
        usage     (:usageMetadata resp)]
    {:content (-> resp :candidates first :content :parts first :text)
     :usage   {:prompt_tokens     (:promptTokenCount usage)
               :completion_tokens (:candidatesTokenCount usage)
               :total_tokens      (:totalTokenCount usage)}}))

(defn inline-file-completion-json
  "Like inline-file-completion but requests JSON-formatted output.
   Returns {:content <parsed-map> :usage {...}}."
  [api-key model ^bytes file-bytes filename content-type question]
  (let [mime-type (infer-mime-type filename content-type)
        b64-data  (.encodeToString (java.util.Base64/getEncoder) file-bytes)
        body      {:contents        [{:parts [{:inline_data {:mime_type mime-type
                                                             :data      b64-data}}
                                              {:text question}]}]
                   :generationConfig {:responseMimeType "application/json"}}
        resp      (api-request api-key (str model ":generateContent") body)
        raw       (-> resp :candidates first :content :parts first :text)
        usage     (:usageMetadata resp)]
    {:content (json/parse-string raw true)
     :usage   {:prompt_tokens     (:promptTokenCount usage)
               :completion_tokens (:candidatesTokenCount usage)
               :total_tokens      (:totalTokenCount usage)}}))

;; ---- Chat Completions ----

(defn- build-request
  "Convert OpenAI-style messages to a Gemini generateContent request body.
   Separates system messages from user/model turns.
   content can be a string (text) or a vector of parts (vision)."
  [messages extra-config]
  (let [system-msg (first (filter #(= (:role %) "system") messages))
        chat-msgs  (remove #(= (:role %) "system") messages)
        contents   (mapv (fn [{:keys [role content]}]
                           {:role  (if (= role "assistant") "model" role)
                            :parts (if (vector? content)
                                     content
                                     [{:text content}])})
                         chat-msgs)]
    (cond-> (merge {:contents contents} extra-config)
      system-msg (assoc :systemInstruction
                        {:parts [{:text (:content system-msg)}]}))))

(defn chat-completion
  "Send a chat completion request to Gemini.
   messages is a seq of {:role \"...\" :content \"...\"}.
   Returns {:content \"...\" :usage {:prompt_tokens N :completion_tokens N :total_tokens N}}."
  [api-key model messages]
  (let [resp  (api-request api-key (str model ":generateContent")
                           (build-request messages {}))
        usage (:usageMetadata resp)]
    {:content (-> resp :candidates first :content :parts first :text)
     :usage   {:prompt_tokens     (:promptTokenCount usage)
               :completion_tokens (:candidatesTokenCount usage)
               :total_tokens      (:totalTokenCount usage)}}))

(defn chat-completion-json
  "Send a chat completion request expecting JSON output from Gemini.
   Returns {:content <parsed-map> :raw-content \"...\" :usage {...}}."
  [api-key model messages]
  (let [resp  (api-request api-key (str model ":generateContent")
                           (build-request messages
                                          {:generationConfig
                                           {:responseMimeType "application/json"}}))
        raw   (-> resp :candidates first :content :parts first :text)
        usage (:usageMetadata resp)]
    {:content     (json/parse-string raw true)
     :raw-content raw
     :usage       {:prompt_tokens     (:promptTokenCount usage)
                   :completion_tokens (:candidatesTokenCount usage)
                   :total_tokens      (:totalTokenCount usage)}}))
