# -*- coding: utf-8 -*-
import argparse
import re
from datetime import datetime, timezone

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    concat_ws,
    current_timestamp,
    first,
    length,
    lit,
    lower,
    monotonically_increasing_id,
    regexp_extract,
    regexp_replace,
    row_number,
    sha2,
    trim,
    when,
)
from pyspark.sql.types import StringType, StructField, StructType, TimestampType
from pyspark.sql.window import Window


def build_spark(app_name):
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000")
        .config("spark.hadoop.fs.s3a.access.key", "admin")
        .config("spark.hadoop.fs.s3a.secret.key", "password")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.endpoint.region", "us-east-1")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.catalogImplementation", "hive")
        .config("spark.hadoop.hive.metastore.uris", "thrift://hive-metastore:9083")
        .enableHiveSupport()
        .getOrCreate()
    )


def sanitize_identifier(value, fallback):
    text = str(value or "").strip().lower()
    text = re.sub(r"[^0-9a-zA-Z_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = fallback
    if text[0].isdigit():
        text = f"c_{text}"
    return text


def list_bronze_datasets(spark, bronze_root):
    jvm = spark.sparkContext._jvm
    conf = spark.sparkContext._jsc.hadoopConfiguration()
    path = jvm.org.apache.hadoop.fs.Path(bronze_root.rstrip("/"))
    fs = path.getFileSystem(conf)
    if not fs.exists(path):
        return []

    datasets = []
    for status in fs.listStatus(path):
        if status.isDirectory():
            name = status.getPath().getName()
            if not name.startswith("_"):
                datasets.append(name)
    return sorted(datasets)


def cleaned_string(column_name):
    cleaned = regexp_replace(trim(col(column_name)), r"\s+", " ")
    return when(cleaned == "", None).otherwise(cleaned)


def register_delta_table(spark, database, table_name, location):
    database_location = location.rsplit("/", 1)[0]
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {database} LOCATION '{database_location}'")
    spark.sql(f"DROP TABLE IF EXISTS {database}.{table_name}")
    spark.sql(f"CREATE TABLE {database}.{table_name} USING DELTA LOCATION '{location}'")


def get_processed_datasets(spark, checkpoint_location):
    try:
        df = spark.read.format("delta").load(checkpoint_location)
        return set(row["dataset_name"] for row in df.select("dataset_name").collect())
    except Exception:
        return set()


def mark_dataset_processed(spark, dataset_name, checkpoint_location):
    schema = StructType(
        [
            StructField("dataset_name", StringType(), True),
            StructField("processed_at", TimestampType(), True),
        ]
    )
    df = spark.createDataFrame([(dataset_name, datetime.now(timezone.utc))], schema)
    df.write.format("delta").mode("append").save(checkpoint_location)


def get_max_silver_id(spark, table_location):
    try:
        df_existing = spark.read.format("delta").load(table_location)
        max_id = df_existing.agg({"cd_silver_id": "max"}).collect()[0][0]
        return int(max_id) if max_id is not None else 0
    except Exception:
        return 0


def normalize_column_names(df):
    renamed = df
    data_columns = [column for column in df.columns if not column.startswith("_")]
    for index, column in enumerate(data_columns, start=1):
        normalized_name = sanitize_identifier(column, f"col_{index}")
        if normalized_name != column:
            renamed = renamed.withColumnRenamed(column, normalized_name)
    return renamed


def cleanse_text_columns(df):
    data_columns = [column for column in df.columns if not column.startswith("_")]
    metadata_columns = [column for column in df.columns if column.startswith("_")]
    return df.select(
        *[cleaned_string(column).alias(column) for column in data_columns],
        *[col(column) for column in metadata_columns],
    )


def find_medicine_name_column(df):
    for column in ("ten_cay_thuoc", "ten_thuoc", "name"):
        if column in df.columns:
            return column
    return None


def normalize_medicine_name(df, name_col):
    return (
        df.filter(col(name_col).isNotNull() & (length(trim(col(name_col))) > 0))
        .withColumn(name_col, regexp_replace(lower(trim(col(name_col))), r"\s+", " "))
    )


def add_semantic_columns(df):
    if "tinh_vi_quy_kinh" not in df.columns:
        return df

    enriched = (
        df.withColumn("vi", regexp_extract(col("tinh_vi_quy_kinh"), r"(?i)vị\s*([^,;.\n]+)", 1))
        .withColumn("tinh", regexp_extract(col("tinh_vi_quy_kinh"), r"(?i)tính\s*([^,;.\n]+)", 1))
        .withColumn("quy_kinh", regexp_extract(col("tinh_vi_quy_kinh"), r"(?i)vào\s*([^.]+)", 1))
    )

    return enriched.select(
        *[
            when(trim(col(column)) == "", None).otherwise(regexp_replace(trim(col(column)), r"\s+", " ")).alias(column)
            if column in {"vi", "tinh", "quy_kinh"}
            else col(column)
            for column in enriched.columns
        ]
    )


def add_normalized_dosage_columns(df):
    if "lieu_dung" not in df.columns:
        return df

    text_col = regexp_replace(lower(col("lieu_dung")), ",", ".")
    range_pattern = r"(\d+(?:\.\d+)?)\s*(?:-|–|—|đến|den|to)\s*(\d+(?:\.\d+)?)"
    min_raw = regexp_extract(text_col, range_pattern, 1)
    max_raw = regexp_extract(text_col, range_pattern, 2)
    single_raw = regexp_extract(text_col, r"(\d+(?:\.\d+)?)", 1)
    multiplier = when(text_col.rlike(r"\bkg\b"), lit(1000.0)).otherwise(lit(1.0))

    min_text = when(min_raw != "", min_raw).when(single_raw != "", single_raw)
    max_text = (
        when(max_raw != "", max_raw)
        .when(min_raw != "", min_raw)
        .when(single_raw != "", single_raw)
    )
    min_value = min_text.cast("double") * multiplier
    max_value = max_text.cast("double") * multiplier

    return (
        df.withColumn("lieu_dung_min_g", min_value)
        .withColumn("lieu_dung_max_g", max_value)
    )


def drop_empty_rows_without_name(df):
    data_columns = [column for column in df.columns if not column.startswith("_")]
    non_empty_filter = None
    for column in data_columns:
        condition = col(column).isNotNull()
        non_empty_filter = condition if non_empty_filter is None else (non_empty_filter | condition)
    return df.filter(non_empty_filter) if non_empty_filter is not None else df


def merge_rows_by_medicine_name(df, name_col):
    merge_columns = [column for column in df.columns if column != name_col]
    return df.groupBy(name_col).agg(
        *[first(col(column), True).alias(column) for column in merge_columns]
    )


def write_silver_dataset(spark, bronze_dataset, silver_dataset, bronze_root, silver_root, database):
    bronze_location = f"{bronze_root.rstrip('/')}/{bronze_dataset}"
    silver_location = f"{silver_root.rstrip('/')}/{silver_dataset}"

    bronze_df = spark.read.format("delta").load(bronze_location)
    cleaned = normalize_column_names(bronze_df)
    cleaned = cleanse_text_columns(cleaned)

    if "stt" in cleaned.columns:
        stt_text = regexp_extract(col("stt"), r"\d+", 0)
        cleaned = cleaned.withColumn("stt", when(stt_text != "", stt_text).cast("int"))

    name_col = find_medicine_name_column(cleaned)
    if name_col:
        cleaned = normalize_medicine_name(cleaned, name_col)
    else:
        cleaned = drop_empty_rows_without_name(cleaned)

    cleaned = add_semantic_columns(cleaned)
    cleaned = add_normalized_dosage_columns(cleaned)
    cleaned = cleaned.dropDuplicates()

    if name_col:
        cleaned = merge_rows_by_medicine_name(cleaned, name_col)

    final_data_columns = [
        column
        for column in cleaned.columns
        if not column.startswith("_")
        and column not in ("row_hash", "silver_processed_at", "cd_silver_id", "dt_ingest_silver")
    ]
    hash_columns = [col(column) for column in final_data_columns]
    current_max_id = get_max_silver_id(spark, silver_location)
    id_window_spec = Window.orderBy(monotonically_increasing_id())

    silver_df = (
        cleaned.withColumn("row_hash", sha2(concat_ws("||", *hash_columns), 256))
        .withColumn("silver_processed_at", current_timestamp())
        .withColumn("temp_row_num", row_number().over(id_window_spec))
        .withColumn("cd_silver_id", (col("temp_row_num") + lit(current_max_id)).cast("int"))
        .drop("temp_row_num")
        .withColumn("dt_ingest_silver", current_timestamp())
    )

    silver_df.write.format("delta").mode("append").option("mergeSchema", "true").save(silver_location)
    register_delta_table(spark, database, silver_dataset, silver_location)

    return {"dataset": silver_dataset, "rows": silver_df.count(), "location": silver_location}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lakehouse-bucket", default="yhct-lakehouse")
    parser.add_argument("--silver-database", default="yhct_silver")
    parser.add_argument("--bronze-table-name", default="bronze_layer")
    parser.add_argument("--silver-table-name", default="silver_layer")
    args = parser.parse_args()

    spark = build_spark("yhct_bronze_to_silver")
    bronze_root = f"s3a://{args.lakehouse_bucket}/bronze"
    silver_root = f"s3a://{args.lakehouse_bucket}/silver"
    bronze_dataset = sanitize_identifier(args.bronze_table_name, "bronze_layer")
    silver_dataset = sanitize_identifier(args.silver_table_name, "silver_layer")
    checkpoint_key = f"{bronze_dataset}->{silver_dataset}"
    checkpoint_location = f"s3a://{args.lakehouse_bucket}/silver/_system_checkpoints/processed_datasets_log"

    datasets = list_bronze_datasets(spark, bronze_root)
    if bronze_dataset not in datasets:
        print(f"No bronze dataset named {bronze_dataset} found at {bronze_root}")
        return

    processed_datasets = get_processed_datasets(spark, checkpoint_location)
    if checkpoint_key in processed_datasets:
        print(f"SKIP: Dataset '{checkpoint_key}' was already processed")
        return

    print(f"Processing bronze dataset: {bronze_dataset}")
    result = write_silver_dataset(
        spark,
        bronze_dataset,
        silver_dataset,
        bronze_root,
        silver_root,
        args.silver_database,
    )
    mark_dataset_processed(spark, checkpoint_key, checkpoint_location)
    print(f"Wrote silver dataset {result['dataset']} with {result['rows']} rows at {result['location']}")
    print("Silver processing complete. Datasets processed this run: 1")


if __name__ == "__main__":
    main()
