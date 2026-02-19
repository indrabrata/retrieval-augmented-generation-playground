(ns indrabrata.rag.services.openai
  "OpenAI API client for embeddings and chat completions."
  (:require [clj-http.client :as http]
            [cheshire.core :as json]))

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
