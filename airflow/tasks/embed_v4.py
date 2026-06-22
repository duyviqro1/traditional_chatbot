# embedding_v4.py
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,               
    UnstructuredExcelLoader 
)
import tempfile 
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
import boto3
import psycopg2
import logging
import os
import sys
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from pydantic import BaseModel, Field

# Load API key từ file .env
env_path = Path(__file__).resolve().parent.parent.parent / '.env'
sys.path.insert(0, str(env_path))
from key import OPENAI_API_KEY
try:
    from key import QDRANT_CLOUD_API_KEY as KEY_FILE_QDRANT_CLOUD_API_KEY
except ImportError:
    KEY_FILE_QDRANT_CLOUD_API_KEY = None
try:
    from key import DEPLOY_MODE as KEY_FILE_DEPLOY_MODE
except ImportError:
    KEY_FILE_DEPLOY_MODE = None
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# --- CONFIG ----
DEPLOY_MODE = (os.getenv("DEPLOY_MODE") or KEY_FILE_DEPLOY_MODE or "local").strip().lower()

MINIO_ENDPOINT = os.getenv(
    "MINIO_ENDPOINT",
    "http://minio:9000" if DEPLOY_MODE == "docker" else "http://localhost:9000",
)
MINIO_ACCESS_KEY = 'admin'
MINIO_SECRET_KEY = 'password'
MINIO_BUCKET = 'yhct-client'
MINIO_PREFIX = 'uploads_type1/'

LOCAL_DATABASE_URL = os.getenv(
    "LOCAL_DATABASE_URL",
    "postgresql://admin:admin@postgres_shared:5432/rag_lakehouse"
    if DEPLOY_MODE == "docker"
    else "postgresql://admin:admin@127.0.0.1:5433/rag_lakehouse",
)
if DEPLOY_MODE in {"local", "docker"}:
    PG_CONNECTION = LOCAL_DATABASE_URL
    QDRANT_URL = os.getenv(
        "QDRANT_LOCAL_URL",
        "http://qdrant:6333" if DEPLOY_MODE == "docker" else "http://localhost:6333",
    )
    QDRANT_COLLECTION = os.getenv("QDRANT_LOCAL_COLLECTION", os.getenv("QDRANT_COLLECTION", "medical_docs"))
    QDRANT_API_KEY = os.getenv("QDRANT_LOCAL_API_KEY") or os.getenv("QDRANT_API_KEY") or "qdrant_api_key"
    QDRANT_TIMEOUT = int(os.getenv("QDRANT_TIMEOUT", "120"))
    QDRANT_BATCH_SIZE = int(os.getenv("QDRANT_BATCH_SIZE", "16"))
else:
    PG_CONNECTION = os.getenv("DATABASE_URL", LOCAL_DATABASE_URL)
    QDRANT_URL = os.getenv(
        "QDRANT_CLOUD_URL",
        "https://1a2c93a3-63cc-4363-bcf0-ccf4f0640ed1.us-east-1-1.aws.cloud.qdrant.io",
    )
    QDRANT_COLLECTION = os.getenv("QDRANT_CLOUD_COLLECTION", os.getenv("QDRANT_COLLECTION", "medical_docs"))
    QDRANT_API_KEY = (
        os.getenv("QDRANT_CLOUD_API_KEY")
        or os.getenv("QDRANT_API_KEY")
        or KEY_FILE_QDRANT_CLOUD_API_KEY
    )
    QDRANT_TIMEOUT = int(os.getenv("QDRANT_CLOUD_TIMEOUT", "120"))
    QDRANT_BATCH_SIZE = int(os.getenv("QDRANT_CLOUD_BATCH_SIZE", "16"))

# Cấu hình kích thước Chunk cứng cho dữ liệu danh mục sách
CHUNK_SIZE = 1200   
CHUNK_OVERLAP = 350 

# --- LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

extraction_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ========================================================
# 1. CẬP NHẬT ĐỊNH NGHĨA METADATA: ÉP ĐẦU RA SẠCH TIỀN TỐ
# ========================================================
class MedicalMetadata(BaseModel):
    diseases: str = Field(description="Danh sách các bệnh hoặc triệu chứng được nhắc đến để điều trị, cách nhau bằng dấu phẩy, viết thường. Nếu không có, ghi 'không xác định'.")
    herbs: str = Field(description="Danh sách TÊN CỐT LÕI của các loại thảo dược/vị thuốc, cách nhau bằng dấu phẩy, viết thường. BẮT BUỘC loại bỏ các từ chỉ bộ phận/phân loại đứng đầu. Nếu không có, ghi 'không xác định'.")


def get_processed_etags(pg_conn):
    cur = pg_conn.cursor()
    cur.execute("SELECT etag FROM processed_files")
    etags = {row[0] for row in cur.fetchall()}
    cur.close()
    return etags


def get_s3_client():
    return boto3.client(
        's3',
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY
    )


def get_minio_files(s3_client):
    return s3_client.list_objects_v2(Bucket=MINIO_BUCKET, Prefix=MINIO_PREFIX).get('Contents', [])


# ========================================================
# 2. THIẾT LẬP PROMPT NGHIÊM NGẶT ĐỂ CHUẨN HÓA DƯỢC LIỆU
# ========================================================
def extract_medical_metadata_with_llm(text, llm):
    prompt = f"""
    Bạn là một chuyên gia y học cổ truyền và hệ thống trích xuất dữ liệu thô. Hãy đọc đoạn văn bản sau để trích xuất:
    1. TÊN BỆNH hoặc TRIỆU CHỨNG được điều trị trong văn bản.
    2. TÊN CÁC LOẠI THẢO DƯỢC / VỊ THUỐC được nhắc đến.
    
    QUY TẮC CHUẨN HÓA NGHIÊM NGẶT CHO TRƯỜNG TÊN THẢO DƯỢC (herbs):
    - Bạn PHẢI loại bỏ hoàn toàn các từ bổ trợ, từ chỉ phân loại hoặc chỉ bộ phận dùng của cây đứng ở trước tên riêng, bao gồm: "cây", "lá", "hạt", "quả", "củ", "rễ", "thân", "hoa", "nụ", "vỏ", "vị thuốc", "thảo dược", "dược liệu".
    - CHỈ giữ lại tên gọi danh từ cốt lõi nhất của vị thuốc đó.
    
    VÍ DỤ MẪU CHUẨN HÓA:
    - Văn bản có chứa: "cây bưởi", "lá bưởi", "hạt bưởi" -> herbs: "bưởi"
    - Văn bản có chứa: "rễ đinh lăng", "vỏ đinh lăng" -> herbs: "đinh lăng"
    - Văn bản có chứa: "vị thuốc ba kích" -> herbs: "ba kích"
    - Văn bản có chứa: "sắc lá khôi uống" -> herbs: "khôi"
    
    Lưu ý bổ sung: Một số văn bản được scan từ PDF nên bị lỗi chèn khoảng trắng sai quy tắc (Ví dụ: 'ch ữ a đ ầ y b ụ ng' thực chất là 'chữa đầy bụng', 'c a m t h ả o' là 'cam thảo'). Hãy tự sửa lỗi dính/rời chữ này trước khi chuẩn hóa.

    Văn bản cần trích xuất:
    "{text}"
    """
    try:
        structured_llm = llm.with_structured_output(MedicalMetadata)
        response = structured_llm.invoke(prompt)
        return response.diseases.strip().lower(), response.herbs.strip().lower()
    except Exception as e:
        logger.error(f"Lỗi khi LLM trích xuất metadata: {e}")
        return "không xác định", "không xác định"
    

def process_chunking_and_metadata(docs, s3_path, llm, embeddings):
    final_chunks = []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )

    for doc in docs:
        doc.metadata['source'] = s3_path
        chunks = text_splitter.split_documents([doc])
        
        for chunk in chunks:
            disease_tags, herb_tags = extract_medical_metadata_with_llm(chunk.page_content, llm)
            
            chunk.metadata['disease'] = disease_tags
            chunk.metadata['herbs'] = herb_tags
            
            final_chunks.append(chunk)
            
    return final_chunks


def setup_qdrant_hybrid(final_chunks, embeddings):
    if not QDRANT_API_KEY:
        raise RuntimeError("Missing QDRANT API key for the selected DEPLOY_MODE.")

    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25", cache_dir=".fastembed_cache")
    QdrantVectorStore.from_documents(
        final_chunks,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings, 
        retrieval_mode=RetrievalMode.HYBRID, 
        url=QDRANT_URL,
        collection_name=QDRANT_COLLECTION,
        api_key=QDRANT_API_KEY,
        timeout=QDRANT_TIMEOUT,
        batch_size=QDRANT_BATCH_SIZE
    )


def process_single_file(file_obj, pg_conn, embeddings, s3_client):
    file_key = file_obj['Key']
    if file_key.endswith('/'):
        return False
    
    file_ext = os.path.splitext(file_key)[1].lower()
    if file_ext not in ['.pdf', '.txt', '.docx', '.md', '.csv', '.xlsx']:
        logger.warning(f"Định dạng không hỗ trợ: {file_ext} ({file_key})")
        return False
    
    file_etag = file_obj['ETag'].replace('"', '')
    tmp_file_path = None
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file_path = tmp_file.name
            s3_client.download_fileobj(MINIO_BUCKET, file_key, tmp_file)
        
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
        s3_path = f"s3://{MINIO_BUCKET}/{file_key}"
        
        final_chunks = process_chunking_and_metadata(docs, s3_path, extraction_llm, embeddings)
        setup_qdrant_hybrid(final_chunks, embeddings)
        logger.info(f"Đã cắt và nạp {len(final_chunks)} đoạn vào Qdrant (Hybrid Mode): {file_key}")
        
        cur = pg_conn.cursor()
        cur.execute(
            "INSERT INTO processed_files (file_name, file_path, etag) VALUES (%s, %s, %s)",
            (file_key.split('/')[-1], file_key, file_etag)
        )
        pg_conn.commit()
        cur.close()
        
        return True
    except Exception as e:
        logger.error(f"Lỗi khi xử lý {file_key}: {e}", exc_info=True)
        pg_conn.rollback()
        return False
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)


def process_new_pdfs():
    logger.info("Bắt đầu quét MinIO và đối chiếu Postgres...")
    logger.info(
        "Embed config: DEPLOY_MODE=%s | MINIO_ENDPOINT=%s | QDRANT_URL=%s | "
        "QDRANT_COLLECTION=%s | PG_CONNECTION=%s",
        DEPLOY_MODE,
        MINIO_ENDPOINT,
        QDRANT_URL,
        QDRANT_COLLECTION,
        PG_CONNECTION,
    )
    pg_conn = psycopg2.connect(PG_CONNECTION)
    try:
        processed_etags = get_processed_etags(pg_conn)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")   
        s3_client = get_s3_client()
        files = get_minio_files(s3_client)
        
        new_files_count = 0
        for obj in files:
            file_key = obj['Key']
            file_etag = obj['ETag'].replace('"', '')
            if file_etag not in processed_etags:
                logger.info(f"[FILE MỚI] Đang xử lý: {file_key}")
                if process_single_file(obj, pg_conn, embeddings, s3_client):
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
