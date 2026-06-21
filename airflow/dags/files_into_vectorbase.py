from airflow import DAG
from airflow.operators.python import PythonOperator





# --- ĐỊNH NGHĨA AIRFLOW DAG ---
default_args = {
    'owner': 'data_engineer',
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

with DAG(
    'rag_incremental_load',
    default_args=default_args,
    description='Quét file PDF mới từ MinIO và nạp vào Qdrant',
    schedule_interval=timedelta(minutes=10), # Cứ 10 phút tự động chạy 1 lần
    start_date=datetime(2026, 4, 1),
    catchup=False,
    tags=['rag', 'langchain', 'minio']
) as dag:
    
    # Task duy nhất chạy toàn bộ logic Python ở trên
    run_ingestion_task = PythonOperator(
        task_id='sync_minio_to_qdrant',
        python_callable=process_new_pdfs
    )