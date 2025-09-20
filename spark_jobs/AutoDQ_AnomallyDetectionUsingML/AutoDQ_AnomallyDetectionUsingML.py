import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import gc
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import TimestampType
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from datetime import timedelta
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
import socket
import os
os.environ["SPARK_VERSION"] = "3.5" 
import time
# pydeequ imports
from pydeequ.analyzers import *
JOB_NAME = "autoDQ_anomally_detection_using_ML_and_Deequ"
METRICS_DB = "autoDQ_DB"
METRICS_TABLE = "autoDQ_ml_metrics"
METRICS_PATH = f"hdfs://localhost:9000/hudi/{METRICS_TABLE}"
SPARK_MASTER = "spark://bigdataplatform.asia-southeast1-a.c.data-quality-project-470101.internal:7077"
PUSHGATEWAY_URL = "localhost:9091"
SAMPLE_FRAC = 0.5
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 500
MAX_DEPTH = 6
EARLY_STOPPING_ROUNDS = 20

UNUSE_FEATURES_DEFAULT = [
    "call_uid","call_start_time","call_response_time","call_ringing_time",
    "call_answer_time","call_end_time","call_date","call_hour",
    "session_uid","start_ts_epoch","response_ts_epoch","ringing_ts_epoch",
    "answer_ts_epoch","end_ts_epoch"
]

def push_metrics_to_pushgateway(spark_metrics_df, pushgateway_url, job_name=JOB_NAME):
    print(f"___PUSHING_METRICS_TO_PUSHGATEWAY___ {pushgateway_url}, job: {job_name}")
    registry = CollectorRegistry()
    # label names match scala version: entity, instance, metric_name
    g = Gauge('AutoDQ_ML_and_Deequ_result', 'Data quality metrics result per run',
              ['path', 'table', 'metric_type', 'column', 'metric'], registry=registry)

    rows = spark_metrics_df.collect()
    for row in rows:
        # try to be robust to different schemas
        path = row.asDict().get("path")
        table = row.asDict().get("table")
        metric_type = row.asDict().get("type")
        column = row.asDict().get("column")
        metric = row.asDict().get("metric")
        value = row.asDict().get("value")
        try:
            numeric_value = float(value)
        except Exception:
            continue
        g.labels(path, table, metric_type, column, metric).set(numeric_value)

    push_to_gateway(pushgateway_url, job=job_name, registry=registry)


def read_latest_records(spark, storage_path):
    df_all = spark.read.format("hudi").load(storage_path).select("processing_time")
    df_all = df_all.withColumn("processing_datetime", F.col("processing_time").cast("timestamp"))

    max_processing = df_all.agg(F.max("processing_datetime")).collect()[0][0]
    today_dt = max_processing.date()
    hour_minute = max_processing.strftime("%H:%M")
    yesterday_dt = today_dt - timedelta(days=1)

    df_filtered = (spark.read.format("hudi").load(storage_path)
                   .withColumn("processing_datetime", F.col("processing_time").cast("timestamp"))
                   .withColumn("hour_minute", F.date_format("processing_datetime", "HH:mm"))
                   .filter(
                       ((F.to_date("processing_datetime") == F.lit(today_dt)) &
                        (F.col("hour_minute") == hour_minute)) |
                       ((F.to_date("processing_datetime") == F.lit(yesterday_dt)) &
                        (F.col("hour_minute") == hour_minute))
                   ))

    df_filtered = df_filtered.withColumn(
        "label",
        F.when(F.to_date("processing_datetime") == F.lit(today_dt), F.lit(1)).otherwise(F.lit(0))
    )

    count_total = df_filtered.count()
    count_today = df_filtered.filter(F.col("label")==1).count()
    count_yday = df_filtered.filter(F.col("label")==0).count()
    print(f"[INFO] Total records: {count_total}, Today: {count_today}, Yesterday: {count_yday}")
    print(f"Latest processing_time today: {max_processing}, hour_minute: {hour_minute}")
    print(f"Today date: {today_dt}, Yesterday date: {yesterday_dt}")

    return df_filtered

def sample_balanced_df(df, sample_frac, seed):
    df_today = df.filter(F.col("label") == 1)
    df_yday = df.filter(F.col("label") == 0)

    if sample_frac < 1.0:
        df_today = df_today.sample(fraction=sample_frac, seed=seed)
        df_yday = df_yday.sample(fraction=sample_frac, seed=seed)
    
    print(f"[INFO] Sampled Today records: {df_today.count()}, Yesterday records: {df_yday.count()}")
    return df_today.unionByName(df_yday)


def to_pandas_for_ml(df_spark):
    """Convert Spark DF đã sampling sang Pandas + xử lý cột list, datetime"""
    df_pd = df_spark.toPandas()

    # list/tuple → length
    list_cols = [c for c in df_pd.columns if df_pd[c].apply(lambda v: isinstance(v, (list, tuple))).any()]
    for c in list_cols:
        df_pd[c] = df_pd[c].apply(lambda v: len(v) if isinstance(v, (list,tuple)) else np.nan)

    # datetime → string
    datetime_cols = df_pd.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns
    for c in datetime_cols:
        df_pd[c] = df_pd[c].dt.tz_localize(None).dt.strftime('%Y-%m-%d %H:%M:%S')

    return df_pd


def extract_features(X, unuse_features):
    use_features = [c for c in X.columns if c not in unuse_features and c not in ["label","processing_datetime","hour_minute","processing_time"]]
    hudi_meta = [c for c in X.columns if c.startswith("_hoodie_")]
    use_features = [c for c in use_features if c not in hudi_meta]

    final_features = []
    for c in use_features:
        if pd.api.types.is_numeric_dtype(X[c]):
            final_features.append(c)
        elif pd.api.types.is_object_dtype(X[c]):
            X[c] = X[c].astype("category")
            final_features.append(c)
    return X[final_features]

def train_model_and_collect_metrics(spark, X_all, y_all, storage_path, table_name, ts_value):
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

    ts = ts_value.to_pydatetime() if ts_value is not None else pd.Timestamp.now().to_pydatetime()

    # --- ML metrics ---
    proba = clf.predict_proba(X_test)[:,1]
    y_pred = clf.predict(X_test)
    metrics_dict = {
        "roc_auc": roc_auc_score(y_test, proba),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }

    metrics_df = pd.DataFrame([
        {
            "timestamp": ts,
            "path": storage_path,
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
    X_test_sample = X_test.sample(n=sample_size, random_state=RANDOM_STATE)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test_sample)
    shap_mean_abs = np.abs(shap_values).mean(axis=0)

    shap_df = pd.DataFrame({
        "timestamp": [ts]*len(X_test_sample.columns),
        "path": [storage_path]*len(X_test_sample.columns),
        "table": [table_name]*len(X_test_sample.columns),
        "type": ["SHAP"]*len(X_test_sample.columns),
        "column": X_test_sample.columns.astype(str),
        "metric": ["shap_mean_abs"]*len(X_test_sample.columns),
        "value": shap_mean_abs.astype(float)
    })

    all_metrics_df = pd.concat([metrics_df, shap_df], ignore_index=True)
    return spark.createDataFrame(all_metrics_df)


def main():
    spark = (SparkSession.builder
             .appName(JOB_NAME)
             .master(SPARK_MASTER)
             .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.hudi.catalog.HoodieCatalog")
             .config("spark.sql.extensions", "org.apache.spark.sql.hudi.HoodieSparkSessionExtension")
             .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
             .getOrCreate())

    table_name = spark.conf.get("spark.tableName", "schema1")
    sample_frac = float(spark.conf.get("spark.sampleFrac", str(SAMPLE_FRAC)))
    ts_value_str = spark.conf.get("spark.ts.value", None)
    ts_value = pd.Timestamp(ts_value_str.replace("T", " ")) if ts_value_str else None

    storage_path = f"hdfs://localhost:9000/hudi/src/{table_name}"

    df_filtered = read_latest_records(spark, storage_path)

    sampled_df = sample_balanced_df(df_filtered, sample_frac, RANDOM_STATE).cache()
    del df_filtered
    gc.collect()

    deequ_metrics_spark_df = run_deequ_analyzer(
        spark, sampled_df, storage_path, table_name, ts_value
    )
    try:
        push_metrics_to_pushgateway(deequ_metrics_spark_df, PUSHGATEWAY_URL, job_name=f"{JOB_NAME}_deequ")
    except Exception as e:
        print(f"[WARN] Failed to push dq metrics: {e}")

    # cleanup sau Deequ
    del deequ_metrics_spark_df
    gc.collect()

    # 4) ML
    df_xy_pd = to_pandas_for_ml(sampled_df)
    X_all = extract_features(df_xy_pd, UNUSE_FEATURES_DEFAULT)
    y_all = df_xy_pd["label"].values
    del df_xy_pd
    gc.collect()

    ml_metrics_spark_df = train_model_and_collect_metrics(
        spark, X_all, y_all, storage_path, table_name, ts_value
    )
    try:
        push_metrics_to_pushgateway(ml_metrics_spark_df, PUSHGATEWAY_URL, job_name=f"{JOB_NAME}_ml")
    except Exception as e:
        print(f"[WARN] Failed to push ml metrics: {e}")

    # cleanup sau ML
    del X_all, y_all, ml_metrics_spark_df
    gc.collect()

    # 5) cleanup sample
    sampled_df.unpersist()
    del sampled_df
    gc.collect()

    spark.stop()
    print("___JOB_COMPLETED___")


if __name__ == "__main__":
    main()