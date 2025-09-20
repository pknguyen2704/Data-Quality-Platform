import json
import os
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from pyspark.sql import SparkSession
from pydeequ.analyzers import *
from pydeequ.anomalyDetection import *
from pydeequ.suggestions import ConstraintSuggestionRunner, Rules
from pydeequ.repository import FileSystemMetricsRepository, ResultKey
from pydeequ.analyzers import AnalysisRunner

JOB_NAME = "AutoDQ_RulesSuggestion"
OUTPUT_JSON_DIR = "/home/pknguyen2704/spark/deequ/rulessuggestion"
PUSHGATEWAY_URL = "localhost:9091"


def save_json_to_file(json_obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)


def deequ_analyzer(spark, df):
    analyzers = [Size()]

    for field in df.schema.fields:
        col_name = field.name
        data_type = field.dataType.simpleString()

        base = [Completeness(col_name), ApproxCountDistinct(col_name)]
        type_specific = []
        if data_type in ["int", "bigint", "double", "float", "decimal"]:
            type_specific = [
                Minimum(col_name),
                Maximum(col_name),
                Mean(col_name),
                Sum(col_name),
            ]
        elif data_type == "string":
            type_specific = [
                MaxLength(col_name),
                MinLength(col_name),
            ]
        analyzers.extend(base + type_specific)

    result = (
        AnalysisRunner(spark)
        .onData(df)
        .addAnalyzers(analyzers)
        .run()
    )
    return result


def push_analyzer_metric(spark, analyzer_result, table_name, pushgateway_url):
    registry = CollectorRegistry()
    gauge = Gauge(
        "AutoDQ_Analyzer_Result",
        "AutoDQ analyzer result",
        ["table", "entity", "instance", "name"],
        registry=registry,
    )

    df_metrics = AnalyzerContext.successMetricsAsDataFrame(spark, analyzer_result)
    for row in df_metrics.collect():
        entity = row["entity"]
        instance = row["instance"]
        name = row["name"]
        value = row["value"]
        gauge.labels(table_name, entity, instance, name).set(value)

    push_to_gateway(pushgateway_url, job=JOB_NAME, registry=registry)


def main():
    spark = (
        SparkSession.builder.appName(JOB_NAME)
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )

    jdbc_url = "jdbc:mysql://localhost:3306/telecom_db"
    jdbc_user = "root"
    jdbc_password = "Mysqlroot123@"
    table_name = "cdr"

    df = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("dbtable", table_name)
        .option("user", jdbc_user)
        .option("password", jdbc_password)
        .option("driver", "com.mysql.cj.jdbc.Driver")
        .load()
    )

    df.show(20, truncate=False)

    print("___START_DEEQU_RULES_SUGGESTION___")
    suggestion_result = (
        ConstraintSuggestionRunner(spark)
        .onData(df)
        .addConstraintRules(Rules.DEFAULT())
        .run()
    )

    suggestions = []
    for col, suggs in suggestion_result.items():
        suggestions.append({
            "column": col,
            "suggestions": [
                {
                    "description": s.description,
                    "codeForConstraint": s.code_for_constraint,
                    "constraint": str(s.constraint),
                }
                for s in suggs
            ],
        })

    os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
    save_json_to_file(
        suggestions, f"{OUTPUT_JSON_DIR}/suggestions_{table_name}.json"
    )

    print("___START_DEEQU_ANALYZER___")
    analyzer_result = deequ_analyzer(spark, df)
    push_analyzer_metric(spark, analyzer_result, table_name, PUSHGATEWAY_URL)

    spark.stop()


if __name__ == "__main__":
    main()
