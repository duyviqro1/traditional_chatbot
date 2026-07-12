# -*- coding: utf-8 -*-
import argparse
import re

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    concat_ws,
    count,
    countDistinct,
    current_timestamp,
    length,
    lit,
    monotonically_increasing_id,
    row_number,
    sha2,
    trim,
    when,
)
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


def register_delta_table(spark, database, table_name, location):
    database_location = location.rsplit("/", 1)[0]
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {database} LOCATION '{database_location}'")
    spark.sql(f"DROP TABLE IF EXISTS {database}.{table_name}")
    spark.sql(f"CREATE TABLE {database}.{table_name} USING DELTA LOCATION '{location}'")


def has_column(df, column_name):
    return column_name in df.columns


def column_or_null(df, column_name, alias_name=None):
    output_name = alias_name or column_name
    if has_column(df, column_name):
        return col(column_name).alias(output_name)
    return lit(None).cast("string").alias(output_name)


def non_empty(column_name):
    return col(column_name).isNotNull() & (length(trim(col(column_name))) > 0)


def find_medicine_name_column(df):
    for column_name in ("ten_cay_thuoc", "ten_thuoc", "name"):
        if column_name in df.columns:
            return column_name
    return None


def with_gold_id(df, id_column):
    window_spec = Window.orderBy(monotonically_increasing_id())
    return (
        df.withColumn("temp_row_num", row_number().over(window_spec))
        .withColumn(id_column, col("temp_row_num").cast("int"))
        .drop("temp_row_num")
    )


def write_gold_table(spark, df, database, table_name, location):
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(location)
    register_delta_table(spark, database, table_name, location)
    return {"table": table_name, "rows": df.count(), "location": location}


def build_medicine_catalog(silver_df):
    name_col = find_medicine_name_column(silver_df)
    source_file_col = "_source_file" if "_source_file" in silver_df.columns else "source_file"

    selected = silver_df.select(
        column_or_null(silver_df, name_col, "medicine_name") if name_col else lit(None).cast("string").alias("medicine_name"),
        column_or_null(silver_df, "ten_thuoc", "medicine_alt_name"),
        column_or_null(silver_df, "bo_phan_dung"),
        column_or_null(silver_df, "cong_dung"),
        column_or_null(silver_df, "lieu_dung"),
        column_or_null(silver_df, "lieu_dung_min_g"),
        column_or_null(silver_df, "lieu_dung_max_g"),
        column_or_null(silver_df, "vi"),
        column_or_null(silver_df, "tinh"),
        column_or_null(silver_df, "quy_kinh"),
        column_or_null(silver_df, "tinh_vi_quy_kinh"),
        column_or_null(silver_df, source_file_col, "source_file"),
        column_or_null(silver_df, "row_hash", "silver_row_hash"),
    )

    if name_col:
        selected = selected.filter(non_empty("medicine_name"))

    hash_columns = [
        "medicine_name",
        "medicine_alt_name",
        "bo_phan_dung",
        "cong_dung",
        "lieu_dung",
        "vi",
        "tinh",
        "quy_kinh",
    ]

    catalog_df = (
        selected.dropDuplicates(["medicine_name"])
        .withColumn("gold_row_hash", sha2(concat_ws("||", *[col(column) for column in hash_columns]), 256))
        .withColumn("gold_processed_at", current_timestamp())
    )
    return with_gold_id(catalog_df, "cd_gold_medicine_id")


def build_medicine_search(catalog_df):
    search_columns = [
        "medicine_name",
        "medicine_alt_name",
        "bo_phan_dung",
        "cong_dung",
        "lieu_dung",
        "vi",
        "tinh",
        "quy_kinh",
        "tinh_vi_quy_kinh",
    ]
    return catalog_df.select(
        col("cd_gold_medicine_id"),
        col("medicine_name"),
        concat_ws(" | ", *[col(column) for column in search_columns]).alias("search_text"),
        col("source_file"),
        col("gold_row_hash"),
        current_timestamp().alias("gold_processed_at"),
    )


def build_quality_report(silver_df, catalog_df):
    total_rows = silver_df.count()
    total_catalog_rows = catalog_df.count()
    name_col = find_medicine_name_column(silver_df)

    metrics = [
        ("silver_total_rows", total_rows),
        ("gold_catalog_rows", total_catalog_rows),
        ("duplicate_medicine_rows_removed", max(total_rows - total_catalog_rows, 0)),
    ]

    if name_col:
        missing_names = silver_df.filter(~non_empty(name_col)).count()
        distinct_names = silver_df.filter(non_empty(name_col)).agg(countDistinct(col(name_col))).collect()[0][0]
        metrics.extend(
            [
                ("missing_medicine_name_rows", missing_names),
                ("distinct_medicine_names", int(distinct_names or 0)),
            ]
        )

    for column_name in ("cong_dung", "lieu_dung", "bo_phan_dung", "tinh_vi_quy_kinh"):
        if column_name in silver_df.columns:
            metrics.append((f"missing_{column_name}_rows", silver_df.filter(~non_empty(column_name)).count()))

    if "lieu_dung" in silver_df.columns and "lieu_dung_min_g" in silver_df.columns:
        unparsable_dosage = silver_df.filter(non_empty("lieu_dung") & col("lieu_dung_min_g").isNull()).count()
        metrics.append(("unparsable_lieu_dung_rows", unparsable_dosage))

    return spark_create_metrics_df(silver_df.sparkSession, metrics)


def spark_create_metrics_df(spark, metrics):
    return (
        spark.createDataFrame(metrics, ["metric_name", "metric_value"])
        .withColumn("metric_value", col("metric_value").cast("long"))
        .withColumn("gold_processed_at", current_timestamp())
    )


def build_dimension_count(spark, df, column_name, statistic_name):
    if column_name not in df.columns:
        return spark.createDataFrame([], "statistic_name string, dimension_name string, dimension_value string, record_count long")

    return (
        df.filter(non_empty(column_name))
        .groupBy(col(column_name).alias("dimension_value"))
        .agg(count(lit(1)).cast("long").alias("record_count"))
        .withColumn("statistic_name", lit(statistic_name))
        .withColumn("dimension_name", lit(column_name))
        .select("statistic_name", "dimension_name", "dimension_value", "record_count")
    )


def build_medicine_stats(spark, catalog_df):
    empty_schema = "statistic_name string, dimension_name string, dimension_value string, record_count long"
    stats_df = spark.createDataFrame([], empty_schema)

    dimensions = [
        ("tinh", "medicine_count_by_tinh"),
        ("vi", "medicine_count_by_vi"),
        ("quy_kinh", "medicine_count_by_quy_kinh"),
        ("source_file", "medicine_count_by_source_file"),
    ]
    for column_name, statistic_name in dimensions:
        stats_df = stats_df.unionByName(build_dimension_count(spark, catalog_df, column_name, statistic_name))

    total_df = spark.createDataFrame(
        [("medicine_total", "all", "all", catalog_df.count())],
        empty_schema,
    )

    return stats_df.unionByName(total_df).withColumn("gold_processed_at", current_timestamp())


def write_gold_datasets(spark, silver_dataset, silver_root, gold_root, database):
    silver_location = f"{silver_root.rstrip('/')}/{silver_dataset}"
    silver_df = spark.read.format("delta").load(silver_location).cache()
    silver_df.count()

    catalog_df = build_medicine_catalog(silver_df).cache()
    catalog_df.count()
    search_df = build_medicine_search(catalog_df)
    stats_df = build_medicine_stats(spark, catalog_df)
    quality_df = build_quality_report(silver_df, catalog_df)

    tables = [
        ("medicine_catalog", catalog_df),
        ("medicine_search", search_df),
        ("medicine_stats", stats_df),
        ("quality_report", quality_df),
    ]

    results = []
    for table_name, df in tables:
        table_location = f"{gold_root.rstrip('/')}/{table_name}"
        results.append(write_gold_table(spark, df, database, table_name, table_location))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lakehouse-bucket", default="yhct-lakehouse")
    parser.add_argument("--silver-table-name", default="silver_layer")
    parser.add_argument("--gold-database", default="yhct_gold")
    args = parser.parse_args()

    spark = build_spark("yhct_silver_to_gold")
    silver_root = f"s3a://{args.lakehouse_bucket}/silver"
    gold_root = f"s3a://{args.lakehouse_bucket}/gold"
    silver_dataset = sanitize_identifier(args.silver_table_name, "silver_layer")

    print(f"Processing silver dataset: {silver_dataset}")
    results = write_gold_datasets(
        spark,
        silver_dataset,
        silver_root,
        gold_root,
        args.gold_database,
    )
    for result in results:
        print(f"Wrote gold table {result['table']} with {result['rows']} rows at {result['location']}")
    print(f"Gold processing complete. Tables written: {len(results)}")


if __name__ == "__main__":
    main()
