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
from langchain_experimental.text_splitter import SemanticChunker
from pydantic import BaseModel, Field # <-- THÊM IMPORT PYDANTIC

# Load API key từ file .env
env_path = Path(__file__).resolve().parent.parent.parent / '.env'
sys.path.insert(0, str(env_path))
from key import OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# --- CONFIG ----
MINIO_ENDPOINT = 'http://localhost:9000'
MINIO_ACCESS_KEY = 'admin'
MINIO_SECRET_KEY = 'password'
MINIO_BUCKET = 'yhct-client'
MINIO_PREFIX = 'uploads_type1/'

PG_CONNECTION = "postgresql://admin:admin@127.0.0.1:5433/rag_lakehouse"

QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "medical_docs"
QDRANT_API_KEY = "qdrant_api_key"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# --- LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

extraction_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ========================================================
# 1. ĐỊNH NGHĨA CẤU TRÚC METADATA ĐẦU RA MONG MUỐN
# ========================================================
class MedicalMetadata(BaseModel):
    diseases: str = Field(description="Danh sách các bệnh hoặc triệu chứng được nhắc đến để điều trị, cách nhau bằng dấu phẩy, viết thường. Nếu không có, ghi 'không xác định'.")
    herbs: str = Field(description="Danh sách các loại cây thuốc, vị thuốc, thảo dược được nhắc đến trong đoạn, cách nhau bằng dấu phẩy, viết thường. Nếu không có, ghi 'không xác định'.")


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
    return s3_client.list_objects_v2(Bucket=MINIO_BUCKET, Prefix=MINIO_PREFIX).get('Contents', [])


# ========================================================
# 2. CẬP NHẬT HÀM TRÍCH XUẤT ĐA NHIỆM (BỆNH + THẢO DƯỢC)
# ========================================================
def extract_medical_metadata_with_llm(text, llm):
    """
    Sử dụng LLM Structured Output để trích xuất cả tên bệnh và tên thảo dược cùng lúc,
    giúp tối ưu hóa chi phí API và tăng tốc độ xử lý pipeline.
    """
    prompt = f"""
    Bạn là một chuyên gia y học cổ truyền. Hãy đọc đoạn văn bản sau để trích xuất:
    1. TÊN BỆNH hoặc TRIỆU CHỨNG được điều trị trong văn bản.
    2. TÊN CÁC LOẠI CÂY THUỐC / THẢO DƯỢC / VỊ THUỐC được nhắc đến.
    
    Lưu ý quan trọng: Một số văn bản được scan từ PDF nên bị lỗi chèn khoảng trắng sai quy tắc (Ví dụ: 'ch ữ a đ ầ y b ụ ng' thực chất là 'chữa đầy bụng', hoặc 'c am t h ả o' thực chất là 'cam thảo'). Hãy tự hiểu và chuẩn hóa lại từ ngữ trước khi điền vào kết quả.
    
    Văn bản:
    "{text}"
    """
    try:
        # Ép mô hình trả về đúng định dạng class MedicalMetadata đã khai báo
        structured_llm = llm.with_structured_output(MedicalMetadata)
        response = structured_llm.invoke(prompt)
        return response.diseases.strip().lower(), response.herbs.strip().lower()
    except Exception as e:
        logger.error(f"Lỗi khi LLM trích xuất metadata: {e}")
        return "không xác định", "không xác định"
    

def process_chunking_and_metadata(docs, s3_path, llm, embeddings):
    final_chunks = []
    
    # Khởi tạo bộ cắt ngữ nghĩa bằng mô hình OpenAI
    text_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile"
    )

    for doc in docs:
        doc.metadata['source'] = s3_path
        
        # Cắt tài liệu dựa trên sự thay đổi ngữ nghĩa giữa các câu
        chunks = text_splitter.split_documents([doc])
        
        for chunk in chunks:
            # Gọi LLM một lần duy nhất để lấy cả 2 thông tin
            disease_tags, herb_tags = extract_medical_metadata_with_llm(chunk.page_content, llm)
            
            # Ghi cả 2 trường dữ liệu mới vào metadata của chunk
            chunk.metadata['disease'] = disease_tags
            chunk.metadata['herbs'] = herb_tags
            
            final_chunks.append(chunk)
            
    return final_chunks


def setup_qdrant_hybrid(final_chunks, embeddings):
    # Khởi tạo mô hình Sparse cho Keyword Search (BM25)
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    # Nạp vào Qdrant với chế độ HYBRID
    QdrantVectorStore.from_documents(
        final_chunks,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings, 
        retrieval_mode=RetrievalMode.HYBRID, 
        url=QDRANT_URL,
        collection_name=QDRANT_COLLECTION,
        api_key=QDRANT_API_KEY,
        batch_size=1000
    )


def process_single_file(file_obj, pg_conn, embeddings, s3_client):
    """Xử lý một file (PDF, TXT, DOCX, MD, CSV, XLSX)"""
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

        # Tạo đường dẫn chuẩn định dạng S3
        s3_path = f"s3://{MINIO_BUCKET}/{file_key}"
        
        # C & D. Chunking, Metadata Extraction & Hybrid Ingestion
        final_chunks = process_chunking_and_metadata(docs, s3_path, extraction_llm, embeddings)
        setup_qdrant_hybrid(final_chunks, embeddings)
        logger.info(f"Đã cắt và nạp {len(final_chunks)} đoạn vào Qdrant (Hybrid Mode): {file_key}")
        
        # E. Ghi log thành công vào Postgres
        cur = pg_conn.cursor()
        cur.execute(
            "INSERT INTO processed_files (file_name, file_path, etag) VALUES (%s, %s, %s)",
            (file_key.split('/')[-1], file_key, file_etag)
        )
        pg_conn.commit()
        cur.close()
        
        logger.info(f"Hoàn tất: {file_key}")
        return True
        
    except Exception as e:
        logger.error(f"Lỗi khi xử lý {file_key}: {e}", exc_info=True)
        pg_conn.rollback()
        return False
        
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)


def process_new_pdfs():
    """Quét MinIO và đối chiếu với Postgres để xử lý file mới"""
    logger.info("Bắt đầu quét MinIO và đối chiếu Postgres...")
    
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
                # Đã bỏ tham số text_splitter dư thừa ở đây
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