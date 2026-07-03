from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,              
    UnstructuredExcelLoader 
)
import tempfile 
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
import boto3
import psycopg2
import logging
import os
import sys
from pathlib import Path

# Load API key from key.py
env_path = Path(__file__).resolve().parent.parent.parent / '.env'
sys.path.insert(0, str(env_path))
from key import OPENAI_API_KEY
try:
    from key import QDRANT_LOCAL_URL as KEY_FILE_QDRANT_LOCAL_URL
except ImportError:
    KEY_FILE_QDRANT_LOCAL_URL = None
try:
    from key import QDRANT_LOCAL_API_KEY as KEY_FILE_QDRANT_LOCAL_API_KEY
except ImportError:
    KEY_FILE_QDRANT_LOCAL_API_KEY = None
try:
    from key import QDRANT_LOCAL_COLLECTION as KEY_FILE_QDRANT_LOCAL_COLLECTION
except ImportError:
    KEY_FILE_QDRANT_LOCAL_COLLECTION = None
try:
    from key import LOCAL_DATABASE_URL as KEY_FILE_LOCAL_DATABASE_URL
except ImportError:
    KEY_FILE_LOCAL_DATABASE_URL = None
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# --- CONFIG ----

MINIO_ENDPOINT = 'http://localhost:9000'
MINIO_ACCESS_KEY = 'admin'
MINIO_SECRET_KEY = 'password'
MINIO_BUCKET = 'yhct-client'
MINIO_PREFIX = 'uploads_type1/'

PG_CONNECTION = (
    os.getenv("LOCAL_DATABASE_URL")
    or os.getenv("POSTGRES_LOCAL_URL")
    or KEY_FILE_LOCAL_DATABASE_URL
    or "postgresql://admin:admin@postgres_shared:5432/rag_lakehouse"
)

QDRANT_URL = (
    os.getenv("QDRANT_LOCAL_URL")
    or os.getenv("QDRANT_URL")
    or KEY_FILE_QDRANT_LOCAL_URL
    or "http://qdrant:6333"
)
QDRANT_COLLECTION = (
    os.getenv("QDRANT_LOCAL_COLLECTION")
    or os.getenv("QDRANT_COLLECTION")
    or KEY_FILE_QDRANT_LOCAL_COLLECTION
    or "medical_docs"
)
QDRANT_API_KEY = (
    os.getenv("QDRANT_LOCAL_API_KEY")
    or KEY_FILE_QDRANT_LOCAL_API_KEY
    or "qdrant_api_key"
)
QDRANT_TIMEOUT = int(os.getenv("QDRANT_TIMEOUT", "120"))
QDRANT_BATCH_SIZE = int(os.getenv("QDRANT_BATCH_SIZE", "16"))

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
# 1. Khai báo mô hình embedding (OpenAI)
# --- LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_processed_etags(pg_conn):
    """Lấy danh sách ETag đã xử lý từ Postgres"""
    cur = pg_conn.cursor()
    cur.execute("SELECT etag FROM processed_files")
    etags = {row[0] for row in cur.fetchall()}
    cur.close()
    return etags


def get_s3_client():
    """Tạo S3 client với cấu hình MinIO"""
    return boto3.client(
        's3',
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY
    )


def get_minio_files(s3_client):
    """Lấy danh sách file từ MinIO"""
    response = s3_client.list_objects_v2(Bucket=MINIO_BUCKET, Prefix=MINIO_PREFIX)
    return response.get('Contents', [])


def process_single_file(file_obj, pg_conn, embeddings, text_splitter, s3_client):
    """Xử lý một file (PDF, TXT, DOCX, MD, CSV, XLSX)"""
    file_key = file_obj['Key']
    
    if file_key.endswith('/'):
        return False
    
    file_ext = os.path.splitext(file_key)[1].lower()
    
    # Kiểm tra file extension hỗ trợ
    if file_ext not in ['.pdf', '.txt', '.docx', '.md', '.csv', '.xlsx']:
        logger.warning(f"Định dạng không hỗ trợ: {file_ext} ({file_key})")
        return False
    
    file_etag = file_obj['ETag'].replace('"', '')
    tmp_file_path = None
    
    try:
        # A. Download file từ MinIO vào temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file_path = tmp_file.name
            s3_client.download_fileobj(MINIO_BUCKET, file_key, tmp_file)
        
        # B. Load document theo định dạng
        if file_ext == '.pdf':
            loader = PyPDFLoader(tmp_file_path)
        elif file_ext == '.txt':
            loader = TextLoader(tmp_file_path, encoding='utf-8')
        elif file_ext == '.docx':
            loader = Docx2txtLoader(tmp_file_path)
        elif file_ext == '.md':
            loader = UnstructuredMarkdownLoader(tmp_file_path)
        elif file_ext == '.csv':
            loader = CSVLoader(tmp_file_path, encoding='utf-8')
        elif file_ext == '.xlsx':
            loader = UnstructuredExcelLoader(tmp_file_path)
        
        docs = loader.load()

        # --- BỔ SUNG ĐOẠN NÀY ĐỂ ĐỔI SOURCE ---
        # Tạo đường dẫn chuẩn định dạng S3 (Ví dụ: s3://yhct-client/uploads_type1/file.pdf)
        s3_path = f"s3://{MINIO_BUCKET}/{file_key}"
        
        # Lặp qua tất cả các trang/tài liệu vừa đọc và ghi đè metadata
        for doc in docs:
            doc.metadata['source'] = s3_path
        
        # C. Chunking
        chunks = text_splitter.split_documents(docs)
        logger.info(f"Đã cắt file thành {len(chunks)} đoạn: {file_key}")
        
        # D. Nạp vào Qdrant
        if not QDRANT_API_KEY:
            raise RuntimeError("Missing local Qdrant API key.")
        QdrantVectorStore.from_documents(
            chunks,
            embeddings,
            url=QDRANT_URL,
            collection_name=QDRANT_COLLECTION,
            api_key=QDRANT_API_KEY,
            timeout=QDRANT_TIMEOUT,
            batch_size=QDRANT_BATCH_SIZE
        )
        
        # E. Ghi log thành công vào Postgres
        cur = pg_conn.cursor()
        cur.execute(
            "INSERT INTO processed_files (file_name, file_path, etag) VALUES (%s, %s, %s)",
            (file_key.split('/')[-1], file_key, file_etag)
        )
        pg_conn.commit()
        cur.close()
        
        logger.info(f"[✓] Hoàn tất: {file_key}")
        return True
        
    except Exception as e:
        logger.error(f"[✗] Lỗi khi xử lý {file_key}: {e}", exc_info=True)
        pg_conn.rollback()
        return False
        
    finally:
        # Cleanup temp file
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)


def process_new_pdfs():
    """Quét MinIO và đối chiếu dengan Postgres để xử lý file mới"""
    logger.info("Bắt đầu quét MinIO và đối chiếu Postgres...")
    
    # Kết nối Postgres
    pg_conn = psycopg2.connect(PG_CONNECTION)
    
    try:
        # Lấy danh sách ETag đã xử lý
        processed_etags = get_processed_etags(pg_conn)
        
        # Chuẩn bị LangChain & Qdrant
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")   
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        # Quét MinIO
        s3_client = get_s3_client()
        files = get_minio_files(s3_client)
        
        new_files_count = 0
        
        for obj in files:
            file_key = obj['Key']
            file_etag = obj['ETag'].replace('"', '')
            
            # Đối chiếu: Nếu file mới thì xử lý
            if file_etag not in processed_etags:
                logger.info(f"[FILE MỚI] Đang xử lý: {file_key}")
                if process_single_file(obj, pg_conn, embeddings, text_splitter, s3_client):
                    new_files_count += 1
        
        if new_files_count == 0:
            logger.info("Trạng thái: Không có file nào mới.")
        else:
            logger.info(f"Đã xử lý {new_files_count} file mới.")
            
    finally:
        pg_conn.close()

def main():
    process_new_pdfs()

if __name__ == "__main__":
    main()
