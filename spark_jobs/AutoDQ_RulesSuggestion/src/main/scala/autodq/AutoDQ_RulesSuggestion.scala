package autodq

import org.apache.spark.sql.SparkSession

object AutoDQ_RulesSuggestion {
  private final val JOB_NAME = "AutoDQ_RulesSuggestion"
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName(JOB_NAME)
      .master("spark://spark-master:7077")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.hudi.catalog.HoodieCatalog")
      .config("spark.sql.extensions", "org.apache.spark.sql.hudi.HoodieSparkSessionExtension")
      .config("spark.kryo.registrator", "org.apache.spark.HoodieSparkKryoRegistrar")
      .config("hive.metastore.uris", "thrift://hive-metastore:9083")
      .enableHiveSupport()
      .getOrCreate()
  }
}
