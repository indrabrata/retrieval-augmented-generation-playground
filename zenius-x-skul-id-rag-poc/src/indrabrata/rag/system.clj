(ns indrabrata.rag.system
  "System assembly using com.stuartsierra/component."
  (:require [com.stuartsierra.component :as component]
            [indrabrata.rag.config :as config]
            [indrabrata.rag.components.mongodb :as mongodb]
            [indrabrata.rag.components.pedestal :as pedestal]))

;; OpenAI config is not a stateful component, just a plain map.
;; We wrap it in a record so it can participate in the component system.

(defrecord OpenAIConfig [api-key embedding-model chat-model]
  component/Lifecycle
  (start [this] this)
  (stop [this] this))

(defn new-openai-config [config-map]
  (map->OpenAIConfig config-map))

(defrecord GeminiConfig [api-key embedding-model chat-model]
  component/Lifecycle
  (start [this] this)
  (stop [this] this))

(defn new-gemini-config [config-map]
  (map->GeminiConfig config-map))

(defn new-system
  "Create a new system map from configuration."
  ([]
   (new-system (config/load-config)))
  ([config]
   (component/system-map
     :mongodb
     (mongodb/new-mongodb
       (get-in config [:mongodb :uri])
       (get-in config [:mongodb :database]))

     :openai-config
     (new-openai-config (get config :openai))

     :gemini-config
     (new-gemini-config (get config :gemini))

     :pedestal
     (component/using
       (pedestal/new-pedestal (get-in config [:server :port]))
       [:mongodb :openai-config :gemini-config]))))
