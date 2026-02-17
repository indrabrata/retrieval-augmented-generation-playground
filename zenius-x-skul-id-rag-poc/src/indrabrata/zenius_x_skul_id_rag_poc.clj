(ns indrabrata.zenius-x-skul-id-rag-poc
  "Main entry point for the RAG POC application."
  (:require [com.stuartsierra.component :as component]
            [indrabrata.rag.system :as system])
  (:gen-class))

(defonce ^:private system (atom nil))

(defn start!
  "Start the application system."
  []
  (println "============================================")
  (println "  Zenius x Skul.id RAG POC")
  (println "  Starting system...")
  (println "============================================")
  (let [sys (component/start (system/new-system))]
    (reset! system sys)
    (println "============================================")
    (println "  System started successfully!")
    (println "  API available at http://localhost:"
             (get-in sys [:pedestal :port]))
    (println "============================================")
    sys))

(defn stop!
  "Stop the application system."
  []
  (when-let [sys @system]
    (component/stop sys)
    (reset! system nil)
    (println "System stopped.")))

(defn -main
  "Application entry point."
  [& _args]
  (start!)
  ;; Add shutdown hook for graceful shutdown
  (.addShutdownHook
    (Runtime/getRuntime)
    (Thread. ^Runnable stop!))
  ;; Block main thread
  @(promise))
