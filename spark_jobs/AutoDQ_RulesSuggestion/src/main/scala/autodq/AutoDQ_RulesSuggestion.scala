package autodq

import com.amazon.deequ.analyzers.runners.AnalyzerContext.successMetricsAsDataFrame
import com.amazon.deequ.analyzers.runners.{AnalysisRunner, AnalyzerContext}
import com.amazon.deequ.analyzers.{ApproxCountDistinct, Completeness, MaxLength, Maximum, Mean, MinLength, Minimum, Size, Sum}
import com.amazon.deequ.suggestions.{ConstraintSuggestionRunner, Rules}
import io.prometheus.client.exporter.PushGateway
import io.prometheus.client.{CollectorRegistry, Gauge}
import org.apache.spark.sql.types.{NumericType, StringType, TimestampType}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.json4s.jackson.Serialization.writePretty
import org.json4s.DefaultFormats

import java.io.PrintWriter

object AutoDQ_RulesSuggestion {
  private final val JOB_NAME = "AutoDQ_RulesSuggestion"
  private final val OUTPUT_JSON_DIR = "/home/pknguyen2704/spark/deequ/rulessuggestion"
  private final val PUSHGATEWAY_URL = "localhost:9091"

  private def saveJsonToFile(jsonStr: String, path: String): Unit = {
    val writer = new PrintWriter(path)
    try {
      writer.write(jsonStr)
    } finally {
      writer.close()
    }
  }


  private def DeequAnalyzer(df: DataFrame): AnalyzerContext = {
    val sizeAnalyzer = Seq(Size())
    val analyzers = df.schema.fields.flatMap { field =>
      val colName = field.name
      val dataType = field.dataType
      val baseAnalyzers = Seq(
        Completeness(colName),
        ApproxCountDistinct(colName)
      )
      val typeSpecificAnalyzers = dataType match {
        case _: NumericType =>
          Seq(Minimum(colName), Maximum(colName), Mean(colName), Sum(colName))
        case StringType =>
          Seq(MaxLength(colName), MinLength(colName))
        case _ => Seq.empty
      }
      baseAnalyzers ++ typeSpecificAnalyzers
    }

    AnalysisRunner
      .onData(df)
      .addAnalyzers(sizeAnalyzer ++ analyzers)
      .run()
  }

  private def PushAnalyzerMetric(analyzerResultDF: DataFrame, tableName: String, pushGateWayUrl: String): Unit = {
    val registry = new CollectorRegistry()

    val gaugeDqAnalyzerResult = Gauge.build()
      .name("AutoDQ_Analyzer_Result")
      .help("AutoDQ analyzer result")
      .labelNames("table", "entity", "instance", "name")
      .register(registry)
    analyzerResultDF.collect().foreach { row =>
      val entity = row.getAs[String]("entity")
      val instance = row.getAs[String]("instance")
      val name = row.getAs[String]("name")
      val value = row.getAs[Double]("value")

      gaugeDqAnalyzerResult.labels(tableName, entity, instance, name).set(value)
    }
    val pushGatewayAnalyzer = new PushGateway(pushGateWayUrl)
    pushGatewayAnalyzer.pushAdd(registry, JOB_NAME)
  }

  def main(args: Array[String]): Unit = {
    implicit val formats: DefaultFormats.type = DefaultFormats
    val spark = SparkSession.builder()
      .appName(JOB_NAME)
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .getOrCreate()
    import spark.implicits._


    val jdbcUrl = "jdbc:mysql://localhost:3306/telecom_db"
    val jdbcUser = "root"
    val jdbcPassword = "Mysqlroot123@"
    val tableName = "cdr"

    val df = spark.read
      .format("jdbc")
      .option("url", jdbcUrl)
      .option("dbtable", tableName)
      .option("user", jdbcUser)
      .option("password", jdbcPassword)
      .option("driver", "com.mysql.cj.jdbc.Driver")
      .load()


    df.show(20, truncate = false)
    println("___START_DEEQU_RULES_SUGGESTION___")

    val suggestionResult = ConstraintSuggestionRunner()
      .onData(df)
      .addConstraintRules(Rules.DEFAULT)
      .run()

    val suggestionJson = writePretty(
      suggestionResult.constraintSuggestions.map { case (col, suggestions) =>
        Map(
          "column" -> col,
          "suggestions" -> suggestions.map { s =>
            Map(
              "description" -> s.description,
              "codeForConstraint" -> s.codeForConstraint,
              "constraint" -> s.constraint.toString
            )
          }
        )
      }
    )
    saveJsonToFile(suggestionJson, s"$OUTPUT_JSON_DIR/suggestions_${tableName}.json")

    println("___START_DEEQU_ANALYZER___")
    PushAnalyzerMetric(successMetricsAsDataFrame(spark, DeequAnalyzer(df)), tableName, PUSHGATEWAY_URL)

    spark.stop()
  }
}
