(ns indrabrata.rag.config
  "Configuration loading from config.edn with environment variable overrides."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]))

(defn load-config
  "Load configuration from resources/config.edn and override with environment variables."
  []
  (let [base-config (-> (io/resource "config.edn")
                        slurp
                        edn/read-string)
        env-overrides {:mongodb {:uri      (or (System/getenv "MONGODB_URI")
                                               (get-in base-config [:mongodb :uri]))
                                 :database (or (System/getenv "MONGODB_DATABASE")
                                               (get-in base-config [:mongodb :database]))}
                       :openai  {:api-key         (or (System/getenv "OPENAI_API_KEY")
                                                      (get-in base-config [:openai :api-key]))
                                 :embedding-model  (or (System/getenv "OPENAI_EMBEDDING_MODEL")
                                                       (get-in base-config [:openai :embedding-model]))
                                 :chat-model       (or (System/getenv "OPENAI_CHAT_MODEL")
                                                       (get-in base-config [:openai :chat-model]))}
                       :server  {:port (Integer/parseInt
                                        (or (System/getenv "SERVER_PORT")
                                            (str (get-in base-config [:server :port]))))}}]
    env-overrides))
