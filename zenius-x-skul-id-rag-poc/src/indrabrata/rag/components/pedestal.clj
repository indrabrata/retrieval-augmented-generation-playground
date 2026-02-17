(ns indrabrata.rag.components.pedestal
  "Pedestal HTTP server component."
  (:require [com.stuartsierra.component :as component]
            [io.pedestal.connector :as conn]
            [io.pedestal.http.http-kit :as hk]
            [indrabrata.rag.routes :as routes]))

(defrecord Pedestal [port mongodb openai-config connector]
  component/Lifecycle

  (start [this]
    (println (str "Starting HTTP server on port " port "..."))
    (let [components-map {:mongodb       mongodb
                          :openai-config openai-config}
          connector      (-> (conn/default-connector-map port)
                             (conn/optionally-with-dev-mode-interceptors)
                             (conn/with-default-interceptors)
                             (conn/with-routes (routes/routes components-map))
                             (hk/create-connector nil)
                             (conn/start!))]
      (println (str "HTTP server started on port " port))
      (assoc this :connector connector)))

  (stop [this]
    (println "Stopping HTTP server...")
    (when connector
      (conn/stop! connector))
    (assoc this :connector nil)))

(defn new-pedestal
  "Create a new Pedestal component."
  [port]
  (map->Pedestal {:port port}))
