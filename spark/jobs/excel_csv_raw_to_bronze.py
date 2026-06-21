import argparse
import posixpath
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from io import BytesIO
from urllib.parse import urlparse
from zipfile import ZipFile

from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.types import StringType, StructField, StructType, TimestampType
from pyspark.sql.functions import (
    current_timestamp, 
    lit, 
    col, 
    row_number, 
    monotonically_increasing_id
)

EXCEL_NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


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


def unique_columns(raw_headers):
    seen = {}
    columns = []
    for index, header in enumerate(raw_headers, start=1):
        base = sanitize_identifier(header, f"col_{index}")
        count = seen.get(base, 0)
        seen[base] = count + 1
        columns.append(base if count == 0 else f"{base}_{count + 1}")
    return columns


def s3_basename(uri):
    parsed = urlparse(uri)
    return posixpath.basename(parsed.path)


def dataset_name_from_uri(uri):
    name = posixpath.splitext(s3_basename(uri))[0]
    return sanitize_identifier(name, "dataset")


def list_input_files(spark, input_uri):
    jvm = spark.sparkContext._jvm
    conf = spark.sparkContext._jsc.hadoopConfiguration()
    path = jvm.org.apache.hadoop.fs.Path(input_uri.rstrip("/") + "/*")
    fs = path.getFileSystem(conf)
    statuses = fs.globStatus(path)
    if not statuses:
        return []
    files = []
    for status in statuses:
        if status.isFile():
            uri = status.getPath().toString()
            if uri.lower().endswith((".csv", ".xlsx", ".xlsm")):
                files.append(uri)
    return sorted(files)


def read_s3_binary(spark, uri):
    content = spark.sparkContext.binaryFiles(uri, minPartitions=1).first()[1]
    return content if isinstance(content, bytes) else content.read()


def workbook_first_sheet_path(zip_file):
    workbook = ET.fromstring(zip_file.read("xl/workbook.xml"))
    rels = ET.fromstring(zip_file.read("xl/_rels/workbook.xml.rels"))
    first_sheet = workbook.find(".//x:sheets/x:sheet", EXCEL_NS)
    if first_sheet is None:
        raise ValueError("Workbook has no sheets")
    rel_id = first_sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
    rel_ns = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}
    for rel in rels.findall("r:Relationship", rel_ns):
        if rel.attrib.get("Id") == rel_id:
            target = rel.attrib["Target"].lstrip("/")
            return target if target.startswith("xl/") else f"xl/{target}"
    raise ValueError(f"Cannot resolve first worksheet relationship {rel_id}")


def load_shared_strings(zip_file):
    try:
        root = ET.fromstring(zip_file.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    values = []
    for item in root.findall("x:si", EXCEL_NS):
        values.append("".join(node.text or "" for node in item.findall(".//x:t", EXCEL_NS)))
    return values


def column_index(cell_ref):
    letters = "".join(char for char in cell_ref if char.isalpha())
    result = 0
    for char in letters.upper():
        result = result * 26 + ord(char) - ord("A") + 1
    return result - 1


def cell_value(cell, shared_strings):
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.findall(".//x:t", EXCEL_NS))
    value_node = cell.find("x:v", EXCEL_NS)
    value = "" if value_node is None or value_node.text is None else value_node.text
    if cell_type == "s" and value:
        return shared_strings[int(value)]
    if cell_type == "b":
        return "true" if value == "1" else "false"
    return value


def rows_from_xlsx_bytes(content):
    with ZipFile(BytesIO(content)) as zip_file:
        shared_strings = load_shared_strings(zip_file)
        sheet_path = workbook_first_sheet_path(zip_file)
        root = ET.fromstring(zip_file.read(sheet_path))
        rows = []
        max_width = 0
        for row in root.findall(".//x:sheetData/x:row", EXCEL_NS):
            values = []
            for cell in row.findall("x:c", EXCEL_NS):
                index = column_index(cell.attrib["r"])
                while len(values) <= index:
                    values.append("")
                values[index] = cell_value(cell, shared_strings)
            max_width = max(max_width, len(values))
            rows.append(values)
        return [row + [""] * (max_width - len(row)) for row in rows]


def read_excel_as_df(spark, uri):
    content = read_s3_binary(spark, uri)
    rows = rows_from_xlsx_bytes(content)
    non_empty = [row for row in rows if any(str(value).strip() for value in row)]
    if not non_empty:
        return None
    headers = unique_columns(non_empty[0])
    data_rows = []
    for row in non_empty[1:]:
        padded = row + [""] * (len(headers) - len(row))
        data_rows.append({headers[i]: str(padded[i]).strip() for i in range(len(headers))})
    schema = StructType([StructField(column, StringType(), True) for column in headers])
    return spark.createDataFrame(data_rows, schema=schema)


def read_csv_as_df(spark, uri):
    return (
        spark.read.option("header", "true")
        .option("inferSchema", "false")
        .option("multiLine", "true")
        .option("escape", '"')
        .csv(uri)
    )


def register_delta_table(spark, database, table_name, location):
    database_location = location.rsplit("/", 1)[0]
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {database} LOCATION '{database_location}'")
    spark.sql(f"DROP TABLE IF EXISTS {database}.{table_name}")
    spark.sql(f"CREATE TABLE {database}.{table_name} USING DELTA LOCATION '{location}'")


def get_processed_files(spark, checkpoint_location):
    """Đọc bảng Delta checkpoint để lấy danh sách các file (URI) đã được xử lý trước đó."""
    try:
        df = spark.read.format("delta").load(checkpoint_location)
        return set(row["file_uri"] for row in df.select("file_uri").collect())
    except Exception:
        return set()


def mark_file_processed(spark, uri, checkpoint_location):
    """Ghi nhận một file (URI) đã xử lý thành công vào bảng Delta checkpoint."""
    schema = StructType([
        StructField("file_uri", StringType(), True),
        StructField("processed_at", TimestampType(), True)
    ])
    df = spark.createDataFrame([(uri, datetime.now(timezone.utc))], schema)
    df.write.format("delta").mode("append").save(checkpoint_location)


def get_max_bronze_id(spark, table_location):
    """Đọc bảng Delta hiện tại để lấy ID lớn nhất. Nếu bảng chưa tồn tại, trả về 0."""
    try:
        df_existing = spark.read.format("delta").load(table_location)
        max_id = df_existing.agg({"cd_bronze_id": "max"}).collect()[0][0]
        return int(max_id) if max_id is not None else 0
    except Exception:
        return 0


# ==========================================
# HÀM GHI DỮ LIỆU CHÍNH
# ==========================================
def write_bronze_file(spark, uri, output_bucket, database, table_name):
    dataset = sanitize_identifier(table_name, "bronze_layer")
    extension = posixpath.splitext(uri.lower())[1]
    
    if extension == ".csv":
        df = read_csv_as_df(spark, uri)
    elif extension in (".xlsx", ".xlsm"):
        df = read_excel_as_df(spark, uri)
    else:
        return None

    if df is None or not df.columns:
        return None

    table_location = f"s3a://{output_bucket}/bronze/{dataset}"
    clean_column_names = unique_columns(df.columns)
    
    # 1. Tìm ID lớn nhất của bảng hiện tại
    current_max_id = get_max_bronze_id(spark, table_location)

    # 2. Định nghĩa Window để đánh số thứ tự dòng cho DataFrame hiện tại
    window_spec = Window.orderBy(monotonically_increasing_id())

    # 3. Tạo dữ liệu Bronze với cột ID tự động tăng và các cột audit
    bronze_df = (
        df.select([df[column].cast("string").alias(clean_column_names[i]) for i, column in enumerate(df.columns)])
        .withColumn("temp_row_num", row_number().over(window_spec))
        .withColumn("cd_bronze_id", col("temp_row_num") + lit(current_max_id)) # Tăng ID dựa trên ID cũ lớn nhất
        .drop("temp_row_num")
        .withColumn("dt_ingest_bronze", current_timestamp())
        .withColumn("_source_file", lit(s3_basename(uri)))
        .withColumn("_source_uri", lit(uri))
        .withColumn("_ingested_at", current_timestamp())
        .withColumn("_ingestion_date", lit(datetime.now(timezone.utc).date().isoformat()))
    )
    
    # Ghi dạng append để không ghi đè dữ liệu cũ khi có file mới
    bronze_df.write.format("delta").mode("append").option("mergeSchema", "true").save(table_location)
    register_delta_table(spark, database, dataset, table_location)
    
    return {"dataset": dataset, "rows": bronze_df.count(), "location": table_location}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-bucket", default="yhct-client")
    parser.add_argument("--input-prefix", default="uploads_excel")
    parser.add_argument("--output-bucket", default="yhct-lakehouse")
    parser.add_argument("--bronze-database", default="yhct_bronze")
    parser.add_argument("--bronze-table-name", default="bronze_layer")
    args = parser.parse_args()

    spark = build_spark("yhct_raw_to_bronze")
    input_uri = f"s3a://{args.input_bucket}/{args.input_prefix.strip('/')}"
    
    # Định nghĩa vị trí lưu trữ bảng Checkpoint
    checkpoint_location = f"s3a://{args.output_bucket}/bronze/_system_checkpoints/processed_files_log"

    files = list_input_files(spark, input_uri)
    if not files:
        print(f"No CSV/XLSX files found at {input_uri}")
        return

    # Lấy danh sách file đã xử lý
    processed_files = get_processed_files(spark, checkpoint_location)

    results = []
    for uri in files:
        # Bỏ qua file nếu đã nằm trong checkpoint
        if uri in processed_files:
            print(f"BỎ QUA (Đã xử lý trước đó): {uri}")
            continue

        print(f"Processing raw file: {uri}")
        result = write_bronze_file(
            spark,
            uri,
            args.output_bucket,
            args.bronze_database,
            args.bronze_table_name,
        )
        
        if result:
            # Ghi nhận file đã xử lý thành công vào checkpoint
            mark_file_processed(spark, uri, checkpoint_location)
            results.append(result)
            print(f"Wrote bronze dataset {result['dataset']} with {result['rows']} rows at {result['location']}")
            
    print(f"Bronze ingestion complete. Datasets processed: {len(results)}")


if __name__ == "__main__":
    main()
