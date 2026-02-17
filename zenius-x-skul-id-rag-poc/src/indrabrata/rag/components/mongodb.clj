(ns indrabrata.rag.components.mongodb
  "MongoDB component using the official Java driver.
   Manages connection lifecycle and provides database access."
  (:require [com.stuartsierra.component :as component])
  (:import [com.mongodb.client MongoClients MongoClient MongoDatabase MongoCollection]
           [com.mongodb ConnectionString MongoClientSettings]
           [org.bson Document]
           [org.bson.conversions Bson]
           [java.util ArrayList List]))

;; ---- Helper functions for converting between Clojure maps and BSON Documents ----

(defn- clj->document
  "Convert a Clojure map to a BSON Document."
  [m]
  (let [doc (Document.)]
    (doseq [[k v] m]
      (let [key-str (name k)]
        (cond
          (nil? v)        doc ;; skip nil values
          (map? v)        (.append doc key-str (clj->document v))
          (sequential? v) (.append doc key-str (java.util.ArrayList. (mapv #(cond (nil? %) nil (map? %) (clj->document %) :else %) v)))
          (keyword? v)    (.append doc key-str (name v))
          :else           (.append doc key-str v))))
    doc))

(defn- document->clj
  "Convert a BSON Document to a Clojure map."
  [^Document doc]
  (when doc
    (into {}
          (map (fn [^java.util.Map$Entry entry]
                 (let [k (.getKey entry)
                       v (.getValue entry)]
                   [(keyword k)
                    (cond
                      (instance? Document v) (document->clj v)
                      (instance? List v)     (mapv #(if (instance? Document %) (document->clj %) %) v)
                      :else                  v)])))
          (.entrySet doc))))

;; ---- MongoDB Component ----

(defrecord MongoDB [uri database-name ^MongoClient client ^MongoDatabase db]
  component/Lifecycle

  (start [this]
    (println (str "Connecting to MongoDB: " database-name))
    (let [settings (-> (MongoClientSettings/builder)
                       (.applyConnectionString (ConnectionString. uri))
                       (.build))
          client   (MongoClients/create settings)
          db       (.getDatabase client database-name)]
      (println "MongoDB connected successfully.")
      (assoc this :client client :db db)))

  (stop [this]
    (println "Disconnecting from MongoDB...")
    (when client
      (.close client))
    (assoc this :client nil :db nil)))

(defn new-mongodb
  "Create a new MongoDB component."
  [uri database-name]
  (map->MongoDB {:uri uri :database-name database-name}))

;; ---- Collection Operations ----

(defn get-collection
  "Get a MongoCollection from the database."
  ^MongoCollection [mongodb collection-name]
  (.getCollection ^MongoDatabase (:db mongodb) collection-name))

(defn find-all
  "Find all documents in a collection. Returns a seq of Clojure maps."
  [mongodb collection-name]
  (->> (get-collection mongodb collection-name)
       (.find)
       (into [])
       (mapv document->clj)))

(defn find-by-ids
  "Find documents by a list of IDs."
  [mongodb collection-name ids]
  (let [non-nil-ids (vec (filter some? ids))]
    (if (empty? non-nil-ids)
      []
      (let [coll   (get-collection mongodb collection-name)
            filter (Document. "_id" (Document. "$in" (java.util.ArrayList. non-nil-ids)))]
        (->> (.find coll filter)
             (into [])
             (mapv document->clj))))))

(defn find-one
  "Find a single document by ID."
  [mongodb collection-name id]
  (let [coll   (get-collection mongodb collection-name)
        filter (Document. "_id" id)]
    (document->clj (.first (.find coll filter)))))

(defn insert-one!
  "Insert a single document. Takes a Clojure map."
  [mongodb collection-name doc-map]
  (let [coll (get-collection mongodb collection-name)
        doc  (clj->document doc-map)]
    (.insertOne coll doc)))

(defn insert-many!
  "Insert multiple documents. Takes a seq of Clojure maps."
  [mongodb collection-name doc-maps]
  (when (seq doc-maps)
    (let [coll (get-collection mongodb collection-name)
          docs (java.util.ArrayList. (mapv clj->document doc-maps))]
      (.insertMany coll docs))))

(defn delete-many!
  "Delete all documents matching a filter."
  [mongodb collection-name filter-map]
  (let [coll   (get-collection mongodb collection-name)
        filter (clj->document filter-map)]
    (.deleteMany coll ^Bson filter)))

(defn aggregate
  "Run an aggregation pipeline. pipeline is a seq of Clojure maps (stages)."
  [mongodb collection-name pipeline]
  (let [coll     (get-collection mongodb collection-name)
        bson-pipeline (java.util.ArrayList. (mapv clj->document pipeline))]
    (->> (.aggregate coll bson-pipeline)
         (into [])
         (mapv document->clj))))

(defn count-documents
  "Count documents in a collection."
  [mongodb collection-name]
  (.countDocuments (get-collection mongodb collection-name)))
