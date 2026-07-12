import http.client
import json
import socket
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator


RAW_BUCKET = "yhct-client"
RAW_PREFIX = "uploads_excel"
LAKEHOUSE_BUCKET = "yhct-lakehouse"
BRONZE_TABLE_NAME = "bronze_layer"
SILVER_TABLE_NAME = "silver_layer"
GOLD_DATABASE = "yhct_gold"
MINIO_CONTAINER = "minio"
SPARK_CONTAINER = "delta-spark"
SPARK_PACKAGES = "io.delta:delta-spark_2.13:4.0.0,org.apache.hadoop:hadoop-aws:3.4.1"
SPARK_IVY_CACHE_DIR = "$HOME/.ivy2"
SPARK_PACKAGE_OPTS = f"--packages {SPARK_PACKAGES} --conf spark.jars.ivy={SPARK_IVY_CACHE_DIR}"
PREPARE_IVY_CACHE = f"mkdir -p {SPARK_IVY_CACHE_DIR}"


default_args = {
    "owner": "data_engineer",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


def ensure_minio_buckets():
    run_in_container(
        MINIO_CONTAINER,
        [
            "sh",
            "-c",
            f"mkdir -p /data/{RAW_BUCKET} /data/{LAKEHOUSE_BUCKET}",
        ],
    )


class UnixSocketHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path):
        super().__init__("localhost")
        self.socket_path = socket_path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self.socket_path)


def docker_api(method, path, body=None):
    payload = None if body is None else json.dumps(body).encode("utf-8")
    headers = {}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    conn = UnixSocketHTTPConnection("/var/run/docker.sock")
    conn.request(method, path, body=payload, headers=headers)
    response = conn.getresponse()
    data = response.read()
    conn.close()
    if response.status >= 300:
        message = data.decode("utf-8", errors="replace")
        raise RuntimeError(f"Docker API {method} {path} failed: {response.status} {message}")
    return data


def decode_docker_exec_output(data):
    chunks = []
    index = 0
    while index + 8 <= len(data):
        size = int.from_bytes(data[index + 4 : index + 8], byteorder="big")
        index += 8
        chunks.append(data[index : index + size])
        index += size
    if not chunks:
        chunks.append(data)
    return b"".join(chunks).decode("utf-8", errors="replace")


def run_in_container(container_name, command, workdir=None):
    create_body = {
        "AttachStdout": True,
        "AttachStderr": True,
        "Tty": False,
        "Cmd": command,
    }
    if workdir:
        create_body["WorkingDir"] = workdir
    exec_data = docker_api("POST", f"/containers/{container_name}/exec", create_body)
    exec_id = json.loads(exec_data.decode("utf-8"))["Id"]
    output_data = docker_api("POST", f"/exec/{exec_id}/start", {"Detach": False, "Tty": False})
    output = decode_docker_exec_output(output_data)
    print(output)
    inspect_data = docker_api("GET", f"/exec/{exec_id}/json")
    exit_code = json.loads(inspect_data.decode("utf-8"))["ExitCode"]
    if exit_code != 0:
        raise RuntimeError(f"Command in {container_name} failed with exit code {exit_code}")


def run_in_delta_spark(command):
    run_in_container(SPARK_CONTAINER, command, workdir="/opt/jobs")


def raw_to_bronze():
    run_in_delta_spark(
        [
            "bash",
            "-lc",
            f"{PREPARE_IVY_CACHE} && "
            f"spark-submit $SPARK_OPTS {SPARK_PACKAGE_OPTS} /opt/jobs/excel_csv_raw_to_bronze.py "
            f"--input-bucket {RAW_BUCKET} "
            f"--input-prefix {RAW_PREFIX} "
            f"--output-bucket {LAKEHOUSE_BUCKET} "
            "--bronze-database yhct_bronze "
            f"--bronze-table-name {BRONZE_TABLE_NAME}",
        ]
    )


def bronze_to_silver():
    run_in_delta_spark(
        [
            "bash",
            "-lc",
            f"{PREPARE_IVY_CACHE} && "
            f"spark-submit $SPARK_OPTS {SPARK_PACKAGE_OPTS} /opt/jobs/bronze_to_silver.py "
            f"--lakehouse-bucket {LAKEHOUSE_BUCKET} "
            "--silver-database yhct_silver "
            f"--bronze-table-name {BRONZE_TABLE_NAME} "
            f"--silver-table-name {SILVER_TABLE_NAME}",
        ]
    )


def silver_to_gold():
    run_in_delta_spark(
        [
            "bash",
            "-lc",
            f"{PREPARE_IVY_CACHE} && "
            f"spark-submit $SPARK_OPTS {SPARK_PACKAGE_OPTS} /opt/jobs/silver_to_gold.py "
            f"--lakehouse-bucket {LAKEHOUSE_BUCKET} "
            f"--silver-table-name {SILVER_TABLE_NAME} "
            f"--gold-database {GOLD_DATABASE}",
        ]
    )


with DAG(
    dag_id="yhct_excel_lakehouse_pipeline",
    default_args=default_args,
    description="Ingest Excel/CSV raw files from MinIO into Delta bronze, silver, and gold layers",
    start_date=datetime(2026, 6, 18),
    catchup=False,
    tags=["yhct", "lakehouse", "delta", "minio"],
) as dag:
    # ensure_buckets_task = PythonOperator(
    #     task_id="ensure_minio_buckets",
    #     python_callable=ensure_minio_buckets,
    # )

    raw_to_bronze_task = PythonOperator(
        task_id="raw_to_bronze",
        python_callable=raw_to_bronze,
    )

    bronze_to_silver_task = PythonOperator(
        task_id="bronze_to_silver",
        python_callable=bronze_to_silver,
    )

    silver_to_gold_task = PythonOperator(
        task_id="silver_to_gold",
        python_callable=silver_to_gold,
    )

    # ensure_buckets_task >> raw_to_bronze_task >> bronze_to_silver_task >> silver_to_gold_task
    raw_to_bronze_task >> bronze_to_silver_task >> silver_to_gold_task
