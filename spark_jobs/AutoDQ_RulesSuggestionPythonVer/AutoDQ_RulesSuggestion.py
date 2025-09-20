import os
os.environ["SPARK_VERSION"] = os.environ.get("SPARK_VERSION", "3.5")
import json
import gc
import time
from datetime import timedelta, datetime
import numpy as np
import pandas as pd
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

# pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import TimestampType

# pydeequ
from pydeequ.analyzers import *
from pydeequ.suggestions import *

# ML libs
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score

# constants / defaults
JOB_NAME = "AutoDQ_RulesSuggestionPython"
OUTPUT_JSON_DIR = "/home/pknguyen2704/spark/deequ/rulessuggestion"
PUSHGATEWAY_URL = "localhost:9091"

# ML / Hudi defaults
METRICS_DB = os.environ.get("METRICS_DB", "autoDQ_DB")
METRICS_TABLE = os.environ.get("METRICS_TABLE", "autoDQ_ml_metrics")
METRICS_PATH = os.environ.get("METRICS_PATH", f"hdfs://localhost:9000/hudi/{METRICS_TABLE}")
SPARK_MASTER = os.environ.get("SPARK_MASTER", "spark://bigdataplatform.asia-southeast1-a.c.data-quality-project-470101.internal:7077")
SAMPLE_FRAC = float(os.environ.get("SAMPLE_FRAC", 0.5))
RANDOM_STATE = int(os.environ.get("RANDOM_STATE", 42))
TEST_SIZE = float(os.environ.get("TEST_SIZE", 0.2))
N_ESTIMATORS = int(os.environ.get("N_ESTIMATORS", 500))
MAX_DEPTH = int(os.environ.get("MAX_DEPTH", 6))
EARLY_STOPPING_ROUNDS = int(os.environ.get("EARLY_STOPPING_ROUNDS", 20))

UNUSE_FEATURES_DEFAULT = [
    "record_id",
    "call_start",
    "call_end",
    "date_hour"
]

# helper: write json
def save_json_to_file(json_obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)

# ---------- Deequ analyzer ----------
def deequ_analyzer(spark, df):
    size_analyzer = Size()
    analyzers = []
    for field in df.schema.fields:
        col_name = field.name
        data_type = field.dataType

        # base analyzers
        analyzers.extend([Completeness(col_name), ApproxCountDistinct(col_name)])

        # type specific
        from pyspark.sql.types import NumericType, StringType
        if isinstance(data_type, NumericType):
            analyzers.extend([Minimum(col_name), Maximum(col_name), Mean(col_name), Sum(col_name)])
        elif isinstance(data_type, StringType):
            analyzers.extend([MaxLength(col_name), MinLength(col_name)])

    # run analysis (pydeequ chỉ có addAnalyzer)
    builder = AnalysisRunner(spark).onData(df).addAnalyzer(size_analyzer)
    for analyzer in analyzers:
        builder = builder.addAnalyzer(analyzer)

    analysis_result = builder.run()
    return analysis_result

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

def read_record(spark, jdbc_url, jdbc_user, jdbc_password, table_name, timestamp):
  df_all = (
    spark.read.format("jdbc")
    .option("dbtable", table_name)
    .option("url", jdbc_url)
    .option("user", jdbc_user)
    .option("password", jdbc_password)
    .option("driver", "com.mysql.cj.jdbc.Driver")
    .load()
  )
  date = timestamp.date()
  hhmm = timestamp.strftime("%H:%M")
  df_filtered = (
    df_all
    .withColumn("date_hour", F.col("date_hour").cast("timestamp"))
    .withColumn("hhmm", F.date_format("date_hour", "HH:mm"))
    .filter(
        (F.to_date("date_hour") == F.lit(date)) &
        (F.col("hhmm") == hhmm)
    )
    .drop("hhmm")
  )
  print(f"[INFO] Loaded latest day {date} at {hhmm}, count={df_filtered.count()}")
  return df_filtered

def read_last_three_day(spark, jdbc_url, jdbc_user, jdbc_password, table_name, timestamp):
  df_all = (
    spark.read.format("jdbc")
    .option("dbtable", table_name)
    .option("url", jdbc_url)
    .option("user", jdbc_user)
    .option("password", jdbc_password)
    .option("driver", "com.mysql.cj.jdbc.Driver")
    .load()
  )
  date = timestamp.date()
  hhmm = timestamp.strftime("%H:%M")
  yday = date - timedelta(days=1)
  day_before = date - timedelta(days=2)

  df_filtered = (
    df_all
    .withColumn("date_hour", F.col("date_hour").cast("timestamp"))
    .withColumn("hhmm", F.date_format("date_hour", "HH:mm"))
    .filter(
      (F.col("hhmm") == hhmm) &
      (F.to_date("date_hour").isin([date, yday, day_before]))
    )
    .withColumn(
      "label",
      F.when(F.to_date("date_hour") == F.lit(date), F.lit(1)).otherwise(F.lit(0))
    )
    .drop("hhmm")
  )

  cnt_total = df_filtered.count()
  cnt_today = df_filtered.filter(F.col("label") == 1).count()
  cnt_prev = df_filtered.filter(F.col("label") == 0).count()

  print(f"[INFO] Loaded 3 days ({day_before}, {yday}, {date}) at {hhmm} "
    f"=> total={cnt_total}, today={cnt_today}, prev2days={cnt_prev}")

  return df_filtered

def sample_balanced_df(df, sample_frac, seed):
  df_today = df.filter(F.col("label") == 1)
  df_other = df.filter(F.col("label") == 0)

  if sample_frac < 1.0:
    df_today = df_today.sample(fraction=sample_frac, seed=seed)
    df_other = df_other.sample(fraction=sample_frac, seed=seed)

  print(f"[INFO] Sampled Today records: {df_today.count()}, Others records: {df_other.count()}")
  return df_today.unionByName(df_other)

def to_pandas_for_ml(df_spark):
  df_pd = df_spark.toPandas()

  list_cols = [c for c in df_pd.columns if df_pd[c].apply(lambda v: isinstance(v, (list, tuple))).any()]
  for c in list_cols:
    df_pd[c] = df_pd[c].apply(lambda v: len(v) if isinstance(v, (list,tuple)) else np.nan)

  # datetime → string (naive)
  datetime_cols = df_pd.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns
  for c in datetime_cols:
    df_pd[c] = df_pd[c].dt.tz_localize(None).dt.strftime('%Y-%m-%d %H:%M:%S')

  return df_pd

def extract_features(X, unuse_features):
  use_features = [c for c in X.columns if c not in unuse_features and c not in ["label"]]
  final_features = []
  for c in use_features:
    if pd.api.types.is_numeric_dtype(X[c]):
      final_features.append(c)
    elif pd.api.types.is_object_dtype(X[c]) or pd.api.types.is_categorical_dtype(X[c]):
      X[c] = X[c].astype("category")
      final_features.append(c)
  return X[final_features]

def train_model_and_collect_metrics(spark, X_all, y_all, table_name):
  X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_all
  )

  clf = xgb.XGBClassifier(
    tree_method='hist',
    device='cpu',
    random_state=RANDOM_STATE,
    enable_categorical=True,
    eval_metric='logloss',
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    grow_policy='lossguide'
  )

  print("___START_TRAINING___")
  clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
  print("___END_TRAINING___")

  # --- ML metrics ---
  proba = clf.predict_proba(X_test)[:,1]
  y_pred = clf.predict(X_test)
  metrics_dict = {
    "roc_auc": roc_auc_score(y_test, proba),
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, zero_division=0),
    "recall": recall_score(y_test, y_pred, zero_division=0),
    "f1_score": f1_score(y_test, y_pred, zero_division=0)
  }

  metrics_df = pd.DataFrame([
    {
      "table": table_name,
      "type": "MLMetric",
      "column": None,
      "metric": k,
      "value": float(v)
    }
    for k,v in metrics_dict.items()
  ])

  # --- SHAP metrics ---
  sample_size = min(5000, len(X_test))
  X_test_sample = X_test.sample(n=sample_size, random_state=RANDOM_STATE) if sample_size > 0 else X_test

  explainer = shap.TreeExplainer(clf)
  shap_values = explainer.shap_values(X_test_sample)
  shap_mean_abs = np.abs(shap_values).mean(axis=0)

  shap_df = pd.DataFrame({
    "table": [table_name]*len(X_test_sample.columns),
    "type": ["SHAP"]*len(X_test_sample.columns),
    "column": X_test_sample.columns.astype(str),
    "metric": ["shap_mean_abs"]*len(X_test_sample.columns),
    "value": shap_mean_abs.astype(float)
  })

  all_metrics_df = pd.concat([metrics_df, shap_df], ignore_index=True)
  return spark.createDataFrame(all_metrics_df)

def push_ml_metric(spark_metrics_df, pushgateway_url, job_name):
  print(f"___PUSHING_ML_METRICS_TO_PUSHGATEWAY___")
  registry = CollectorRegistry()
  g = Gauge('AutoDQ_ML_and_Deequ_result', 'Data quality metrics result per run',
            ['table', 'metric_type', 'column', 'metric'], registry=registry)

  rows = spark_metrics_df.collect()
  for row in rows:
    d = row.asDict()
    table = d.get("table")
    metric_type = d.get("type")
    column = d.get("column")
    metric = d.get("metric")
    value = d.get("value")
    g.labels(table or "", metric_type or "", str(column) if column is not None else "", metric or "").set(value)

  push_to_gateway(pushgateway_url, job=job_name, registry=registry)

# ---------- Main ----------
def main():
  spark = (SparkSession.builder
    .appName(JOB_NAME)
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .getOrCreate()
  )

  table_name = spark.conf.get("spark.tableName", spark.conf.get("spark.table", "cdr"))
  sample_frac = float(spark.conf.get("spark.sampleFrac", str(SAMPLE_FRAC)))
  jdbc_url = spark.conf.get("spark.jdbc.url", "jdbc:mysql://localhost:3306/telecom_db")
  jdbc_user = spark.conf.get("spark.jdbc.user", "root")
  jdbc_password = spark.conf.get("spark.jdbc.password", "Mysqlroot123@")
  pushgateway_url = spark.conf.get("spark.pushgateway.url", PUSHGATEWAY_URL)

  print(f"[INFO] Starting job {JOB_NAME} for table {table_name}")

  timestamp_str = "2025-9-20 10:00:00"
  timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")

  print("__START_READ_TODAY_TIMESTAMP__")
  df = read_record(spark, jdbc_url, jdbc_user, jdbc_password, table_name, timestamp)
  df.show(20, truncate=False)

  print("___START_DEEQU_RULES_SUGGESTION___")
  suggestion_result = (ConstraintSuggestionRunner(spark)
    .onData(df)
    .addConstraintRule(DEFAULT())
    .run()
  )
  suggestions = [
    {
      "constraint_name": s.get("constraint_name"),
      "column_name": s.get("column_name"),
      "current_value": s.get("current_value"),
      "description": s.get("description"),
      "suggesting_rule": s.get("suggesting_rule"),
      "rule_description": s.get("rule_description"),
      "code_for_constraint": s.get("code_for_constraint"),
    }
    for s in suggestion_result.get("constraint_suggestions", [])
  ]
  os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
  save_json_to_file(
    {"table": table_name, "suggestions": suggestions},
    f"{OUTPUT_JSON_DIR}/suggestions_{table_name}.json",
  )
  print(f"[INFO] Suggestion file saved to {OUTPUT_JSON_DIR}/suggestions_{table_name}.json")

  print("___START_DEEQU_ANALYZER___")
  analyzer_result = deequ_analyzer(spark, df)
  push_analyzer_metric(spark, analyzer_result, table_name, PUSHGATEWAY_URL)
  del analyzer_result
  gc.collect()

  df_filtered = read_last_three_day(spark, jdbc_url, jdbc_user, jdbc_password, table_name, timestamp)
  df_filtered.show(20, truncate=False)

  sampled_df = sample_balanced_df(df_filtered, sample_frac, RANDOM_STATE).cache()

  print("___START_ANOMALLY_DETECTION_USING_ML___")
  df_xy_pd = to_pandas_for_ml(sampled_df)
  X_all = extract_features(df_xy_pd, UNUSE_FEATURES_DEFAULT)
  y_all = df_xy_pd["label"].values
  # cleanup pandas temp
  del df_xy_pd
  gc.collect()

  ml_metrics_spark_df = train_model_and_collect_metrics(
      spark, X_all, y_all, table_name
  )
  push_ml_metric(ml_metrics_spark_df, pushgateway_url, job_name=f"{JOB_NAME}_ml")
  # cleanup
  sampled_df.unpersist()
  del sampled_df, ml_metrics_spark_df, X_all, y_all
  gc.collect()

  spark.stop()
  print("___JOB_COMPLETED___")

if __name__ == "__main__":
  main()
