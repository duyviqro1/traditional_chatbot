import os
import sys
import string
import warnings
import re
import unicodedata
from pathlib import Path
from typing import List
from urllib.parse import urlparse
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_tavily import TavilySearch
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from langgraph.graph import END, StateGraph
from langchain_core.output_parsers import StrOutputParser

from database import init_db, save_chat_turn_to_db, load_history_from_db
from source_mapping import SOURCE_MAPPING
# Tắt cảnh báo phiền phức của Pydantic để Terminal sạch đẹp
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# ==========================================
# 1. LOAD API KEY & CONFIG
# ==========================================
def get_streamlit_secret(name, default=None):
    try:
        import streamlit as st
        return st.secrets.get(name, default)
    except Exception:
        return default


env_path = Path(__file__).resolve().parent / '.env'
sys.path.insert(0, str(env_path))
try:
    from key import OPENAI_API_KEY as KEY_FILE_OPENAI_API_KEY
except ImportError:
    KEY_FILE_OPENAI_API_KEY = None
try:
    from key import TAVILY_API_KEY as KEY_FILE_TAVILY_API_KEY
except ImportError:
    KEY_FILE_TAVILY_API_KEY = None
try:
    from key import QDRANT_CLOUD_API_KEY as KEY_FILE_QDRANT_CLOUD_API_KEY
except ImportError:
    KEY_FILE_QDRANT_CLOUD_API_KEY = None

OPENAI_API_KEY = (
    os.getenv("OPENAI_API_KEY")
    or get_streamlit_secret("OPENAI_API_KEY")
    or KEY_FILE_OPENAI_API_KEY
)
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment, Streamlit secrets, or .env/key.py.")
TAVILY_API_KEY = (
    os.getenv("TAVILY_API_KEY")
    or get_streamlit_secret("TAVILY_API_KEY")
    or KEY_FILE_TAVILY_API_KEY
)
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

QDRANT_URL = (
    os.getenv("QDRANT_CLOUD_URL")
    or get_streamlit_secret("QDRANT_CLOUD_URL")
    or os.getenv("QDRANT_URL")
    or get_streamlit_secret("QDRANT_URL")
    or "https://1a2c93a3-63cc-4363-bcf0-ccf4f0640ed1.us-east-1-1.aws.cloud.qdrant.io"
)
QDRANT_COLLECTION = (
    os.getenv("QDRANT_CLOUD_COLLECTION")
    or get_streamlit_secret("QDRANT_CLOUD_COLLECTION")
    or os.getenv("QDRANT_COLLECTION")
    or get_streamlit_secret("QDRANT_COLLECTION")
    or "medical_docs"
)
QDRANT_API_KEY = (
    os.getenv("QDRANT_CLOUD_API_KEY")
    or get_streamlit_secret("QDRANT_CLOUD_API_KEY")
    or os.getenv("QDRANT_API_KEY")
    or get_streamlit_secret("QDRANT_API_KEY")
    or KEY_FILE_QDRANT_CLOUD_API_KEY
)
QDRANT_TIMEOUT = int(
    os.getenv("QDRANT_CLOUD_TIMEOUT")
    or get_streamlit_secret("QDRANT_CLOUD_TIMEOUT")
    or "120"
)
SEARCH_K = 10
WEB_SEARCH_K = 3
ALLOWED_WEB_DOMAINS = [
    "vienduoclieu.org.vn",
    "tracuuduoclieu.vn",
    "suckhoedoisong.vn",
    "vienydhdt.gov.vn",
    "vienyduocvietnam.org",
    "chanduoc.vn",
    "moh.gov.vn",
    "dav.gov.vn",
    "yhoccotruyen.gov.vn",
    "who.int",
    "ncbi.nlm.nih.gov",
    "msdmanuals.com",
    "mayoclinic.org",
    "vinmec.com",
    "tamanhhospital.vn",
]
MEDICAL_ROUTE_KEYWORDS = [
    "benh",
    "trieu chung",
    "dau",
    "sot",
    "ho",
    "cam",
    "cum",
    "viem",
    "da day",
    "ta trang",
    "tieu chay",
    "tao bon",
    "day bung",
    "kho tieu",
    "huyet ap",
    "tieu duong",
    "dai thao duong",
    "tri",
    "mo mau",
    "dau lung",
    "moi goi",
    "thoai hoa",
    "mat ngu",
    "suy nhuoc",
    "met moi",
    "than kinh",
    "mun nhot",
    "di ung",
    "man ngua",
    "nhiet mieng",
    "viem loi",
    "sau rang",
    "cay",
    "thuoc",
    "duoc lieu",
    "thao duoc",
    "bai thuoc",
    "cach dung",
    "lieu luong",
]
KNOWN_DISEASE_TERMS = {
    "cam mao": "cảm mạo",
    "cam cum": "cảm cúm",
    "sot": "sốt",
    "viem hong": "viêm họng",
    "ho": "ho",
    "viem phe quan": "viêm phế quản",
    "dau da day": "đau dạ dày",
    "viem loet da day": "viêm loét dạ dày",
    "ta trang": "tá tràng",
    "roi loan tieu hoa": "rối loạn tiêu hóa",
    "tieu chay": "tiêu chảy",
    "day bung": "đầy bụng",
    "tao bon": "táo bón",
    "tang huyet ap": "tăng huyết áp",
    "dai thao duong": "đái tháo đường",
    "tieu duong": "tiểu đường",
    "benh tri": "bệnh trĩ",
    "mo mau": "mỡ máu",
    "dau lung": "đau lưng",
    "moi goi": "mỏi gối",
    "thoai hoa khop": "thoái hóa khớp",
    "suy nhuoc than kinh": "suy nhược thần kinh",
    "mat ngu": "mất ngủ",
    "met moi": "mệt mỏi",
    "mun nhot": "mụn nhọt",
    "di ung": "dị ứng",
    "man ngua": "mẩn ngứa",
    "nhiet mieng": "nhiệt miệng",
    "viem loi": "viêm lợi",
    "sau rang": "sâu răng",
    "dau dau": "đau đầu",
}

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
if not QDRANT_API_KEY:
    raise RuntimeError("Missing QDRANT_CLOUD_API_KEY in environment or .env/key.py.")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=QDRANT_TIMEOUT)
qdrant = QdrantVectorStore(client=client, collection_name=QDRANT_COLLECTION, embedding=embeddings)
try:
    tavily_search_tool = TavilySearch(
        max_results=WEB_SEARCH_K,
        include_domains=ALLOWED_WEB_DOMAINS,
        include_answer=False,
        include_raw_content=False,
    )
except Exception as exc:
    tavily_search_tool = None
    print(f"[WEB] Chua khoi tao duoc TavilySearch: {exc}")

parser = StrOutputParser()

# Khởi tạo công cụ sửa lỗi chính tả
spell_check_prompt = ChatPromptTemplate.from_messages([
    ("system", """Bạn là một công cụ tiền xử lý văn bản tiếng Việt. Nhiệm vụ duy nhất của bạn là khôi phục dấu tiếng Việt và sửa lỗi chính tả.

CÁC QUY TẮC NGHIÊM NGẶT:
1. Nếu đầu vào là một câu hỏi không dấu, hãy thêm dấu tiếng Việt cho chuẩn xác.
2. CHỈ ĐƯỢC SỬA CHÍNH TẢ VÀ THÊM DẤU. TUYỆT ĐỐI KHÔNG tự ý trả lời, suy luận, hay điền thêm thông tin.
3. Nếu câu gốc đã viết đúng chính tả và đủ dấu, BẮT BUỘC trả về y nguyên câu gốc.
4. Chỉ trả về kết quả, không giải thích.

--- VÍ DỤ MẪU ---
Input: "cam thao co tac dung j"
Output: "Cam thảo có tác dụng gì?"

Input: "tôi hay bị đâu đầu tróng mặt thì uống cây j"
Output: "Tôi hay bị đau đầu chóng mặt thì uống cây gì?"
"""),
    ("human", "{input}")
])
spell_check_chain = spell_check_prompt | llm | parser

# ==========================================
# 2. ĐỊNH NGHĨA TRẠNG THÁI (STATE) VÀ SCHEMAS
# ==========================================
class GraphState(TypedDict, total=False):
    question: str
    chat_history: list
    documents: List[Document]
    generation: str
    optimized_query: str
    web_searched: bool

class RouteQuery(BaseModel):
    datasource: str = Field(description="Định tuyến câu hỏi tới 'vectorstore' nếu hỏi về y học/bệnh lý, hoặc 'chat' nếu là giao tiếp bình thường.")

class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Tài liệu có chứa thông tin trả lời câu hỏi không? Trả lời 'yes' hoặc 'no'.")

class MedicalEntities(BaseModel):
    diseases: List[str] = Field(
        description=(
            "Mảng chứa từ vựng chỉ bệnh lý/triệu chứng. BẮT BUỘC CHỈ SAO CHÉP "
            "từ có thật trong chuỗi đầu vào. Nếu không nhắc đích danh, trả về []."
        )
    )
    herbs: List[str] = Field(
        description=(
            "Mảng chứa tên cốt lõi của thảo dược. BẮT BUỘC CHỈ SAO CHÉP "
            "từ có thật trong chuỗi đầu vào. Lược bỏ chữ 'cây', 'lá', 'quả'."
        )
    )

# ==========================================
# CÁC HÀM TIỆN ÍCH (UTILS)
# ==========================================
def clean_text(text):
    text = " ".join(text.lower().split())
    return text.translate(str.maketrans('', '', string.punctuation))

def normalize_text(text: str) -> str:
    text = " ".join(text.lower().split())
    return text.translate(str.maketrans('', '', string.punctuation))

def fold_vietnamese_text(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return text

def is_obvious_medical_query(question: str) -> bool:
    question_norm = normalize_text(fold_vietnamese_text(question))
    return any(keyword in question_norm for keyword in MEDICAL_ROUTE_KEYWORDS)

def extract_known_disease_terms(question: str):
    question_norm = normalize_text(fold_vietnamese_text(question))
    return [
        disease
        for keyword, disease in KNOWN_DISEASE_TERMS.items()
        if keyword in question_norm
    ]

def is_query_rewrite_safe(original_question: str, rewritten_query: str) -> bool:
    if not rewritten_query:
        return False
    rewritten_norm = normalize_text(fold_vietnamese_text(rewritten_query))
    unsafe_phrases = [
        "ban co the cho biet them",
        "vui long cho biet",
        "can them thong tin",
        "cu the hon",
        "vi du nhu",
    ]
    if any(phrase in rewritten_norm for phrase in unsafe_phrases):
        return False
    if rewritten_query.count("?") > 1:
        return False
    if len(rewritten_query) > max(len(original_question) * 3, 120):
        return False
    return True

def get_last_user_message(chat_history):
    for message in reversed(chat_history):
        if isinstance(message, HumanMessage):
            return message.content
    return ""

def parse_herb_field(value):
    if not value:
        return []
    if isinstance(value, list):
        return [item.strip().lower() for item in value if str(item).strip()]
    return [item.strip().lower() for item in str(value).split(",") if item.strip()]

def format_answer_sources(documents):
    if not documents:
        return ""

    source_lines = []
    seen_sources = set()

    for doc in documents:
        source_type = doc.metadata.get("source_type", "vectorstore")
        source = doc.metadata.get("source", "Khong ro nguon")
        url = doc.metadata.get("url", "")

        if source_type == "internet" and url:
            label = f"Internet: {url}"
            key = ("internet", url)
        else:
            label = f"Cơ sở dữ liệu nội bộ: {source}"
            key = ("vectorstore", source)

        if key in seen_sources:
            continue

        seen_sources.add(key)
        source_lines.append(f"- {label}")

    if not source_lines:
        return ""

    return "\n\nNguon tham khao:\n" + "\n".join(source_lines)

def strip_generated_sources(answer: str) -> str:
    source_heading_pattern = r"(?im)^\s*ngu[oồ]n\s+tham\s+kh[aả]o\s*:\s*$"
    match = re.search(source_heading_pattern, answer)
    if not match:
        return answer.strip()
    return answer[:match.start()].strip()

def build_herb_terms_from_docs(docs, herbs):
    herb_terms = {normalize_text(herb) for herb in herbs if herb.strip()}

    for doc in docs:
        content_norm = normalize_text(doc.page_content)
        if not any(herb in content_norm for herb in herb_terms):
            continue

        alias_match = re.search(
            r"(?i)tên\s+khác\s*:\s*(.+?)(?:\n\n|\n[A-ZÀ-Ỵ ]{3,}:|$)",
            doc.page_content,
            flags=re.DOTALL,
        )
        if not alias_match:
            continue

        aliases = re.split(r"[-,;()\n]+", alias_match.group(1))
        for alias in aliases:
            alias_norm = normalize_text(alias)
            if alias_norm and 1 <= len(alias_norm.split()) <= 4:
                herb_terms.add(alias_norm)

    return herb_terms

def filter_recipe_blocks_by_herbs(docs, herbs):
    if not herbs:
        return docs

    herb_terms = build_herb_terms_from_docs(docs, herbs)
    filtered_docs = []

    for doc in docs:
        parts = re.split(r"(?=(?:^|\n)Bài\s+\d+\s*[:.])", doc.page_content)
        selected_parts = []

        for idx, part in enumerate(parts):
            part_norm = normalize_text(part)
            has_target_herb = any(term in part_norm for term in herb_terms)

            if idx == 0:
                if has_target_herb:
                    selected_parts.append(part.strip())
                continue

            if has_target_herb:
                selected_parts.append(part.strip())

        if selected_parts:
            doc.page_content = "\n\n".join(selected_parts)
            filtered_docs.append(doc)

    if len(filtered_docs) != len(docs):
        print(
            f"   -> [FILTER] Lọc nội dung bài thuốc theo cây được hỏi: {len(docs)} -> {len(filtered_docs)} đoạn."
        )

    return filtered_docs

def build_condition_terms(diseases):
    terms = set()
    for disease in diseases or []:
        disease_norm = normalize_text(disease)
        if not disease_norm:
            continue
        terms.add(disease_norm)
        for token in disease_norm.split():
            if len(token) >= 3:
                terms.add(token)
    return terms

def has_condition_match(text_norm, condition_terms):
    if not condition_terms:
        return True
    return any(term in text_norm for term in condition_terms)

def filter_recipe_blocks_by_entities(docs, herbs, diseases):
    if not herbs and not diseases:
        return docs

    herb_terms = build_herb_terms_from_docs(docs, herbs) if herbs else set()
    condition_terms = build_condition_terms(diseases)
    filtered_docs = []

    for doc in docs:
        parts = re.split(r"(?=(?:^|\n)Bài\s+\d+\s*[:.])", doc.page_content)
        selected_parts = []

        for idx, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue

            part_norm = normalize_text(part)
            has_target_herb = True if not herb_terms else any(term in part_norm for term in herb_terms)
            has_target_condition = has_condition_match(part_norm, condition_terms)

            if has_target_herb and has_target_condition:
                selected_parts.append(part)

        if selected_parts:
            doc.page_content = "\n\n".join(selected_parts)
            filtered_docs.append(doc)

    if len(filtered_docs) != len(docs):
        print(f"   -> [FILTER] Lọc nội dung theo cây + bệnh/triệu chứng: {len(docs)} -> {len(filtered_docs)} đoạn.")

    return filtered_docs

def is_spellcheck_safe(original_text: str, corrected_text: str) -> bool:
    if not corrected_text:
        return False
    if "\n" in corrected_text:
        return False
    if corrected_text.count("?") > 1:
        return False
    if "?" in original_text and "?" not in corrected_text:
        return False
    max_len = max(int(len(original_text) * 1.2), len(original_text) + 15)
    if len(corrected_text) > max_len:
        return False
    original_words = original_text.split()
    corrected_words = corrected_text.split()
    if len(corrected_words) > len(original_words) + 4:
        return False
    original_sent_count = sum(original_text.count(p) for p in [".", "?", "!"])
    corrected_sent_count = sum(corrected_text.count(p) for p in [".", "?", "!"])
    if original_sent_count <= 1 and corrected_sent_count > 1:
        return False
    return True

def extract_entities_from_query(user_query: str, llm):
    rule_diseases = extract_known_disease_terms(user_query)
    prompt = f"""
    BẠN LÀ MỘT CỖ MÁY PHÂN TÍCH CÚ PHÁP VĂN BẢN (TEXT PARSER).
    BẠN KHÔNG PHẢI LÀ BÁC SĨ HAY CHUYÊN GIA Y TẾ.

    Nhiệm vụ duy nhất: Đọc chuỗi văn bản đầu vào và SAO CHÉP LẠI ĐÚNG NHỮNG TỪ NGỮ
    chỉ bệnh hoặc tên cây thuốc CÓ MẶT CHÍNH XÁC trong chuỗi đó.

    LỆNH CẤM TUYỆT ĐỐI:
    1. KHÔNG ĐƯỢC TỰ SUY LUẬN, không thêm bất kỳ tên bệnh/cây thuốc nào không xuất hiện.
    2. Nếu người dùng hỏi chung chung không nhắc bệnh, mảng diseases PHẢI LÀ [].

    QUY TẮC LÀM SẠCH TÊN CÂY (herbs):
    - Bỏ các từ: "cây", "lá", "củ", "quả", "rễ", "hoa", "vị thuốc".
    - Nếu có chữ trong ngoặc, tách thành phần tử riêng.

    CHUỖI ĐẦU VÀO: "{user_query}"
    """
    try:
        structured_llm = llm.with_structured_output(MedicalEntities)
        response = structured_llm.invoke(prompt)

        safe_diseases = [d.strip().lower() for d in response.diseases] if response.diseases else []
        safe_diseases = list(dict.fromkeys(safe_diseases + rule_diseases))
        safe_herbs = [h.strip().lower() for h in response.herbs] if response.herbs else []
        return safe_diseases, safe_herbs
    except Exception:
        return rule_diseases, []

def get_filtered_retriever(qdrant_vectorstore, primary_query, llm, fallback_query=None):
    diseases, herbs = extract_entities_from_query(primary_query, llm)

    if not diseases and not herbs and fallback_query:
        print("   [FILTER] -> Không có thực thể ở câu hiện tại, thử fallback câu hỏi trước đó.")
        diseases, herbs = extract_entities_from_query(fallback_query, llm)

    search_kwargs = {"k": SEARCH_K}
    must_conditions = []
    should_conditions = []

    if diseases:
        print(f"   [FILTER] -> Kích hoạt lọc nhóm Bệnh (Toán tử OR): {diseases}")
        disease_conditions = [
            models.FieldCondition(key="metadata.disease", match=models.MatchText(text=d))
            for d in diseases
        ]
        should_conditions.extend(disease_conditions)

    if herbs:
        print(f"   [FILTER] -> Kích hoạt lọc nhóm Thảo dược (Toán tử OR): {herbs}")
        herb_conditions = [
            models.FieldCondition(key="metadata.herbs", match=models.MatchText(text=h))
            for h in herbs
        ]
        must_conditions.append(models.Filter(should=herb_conditions))

    if must_conditions or should_conditions:
        search_kwargs["filter"] = models.Filter(
            must=must_conditions or None,
            should=should_conditions or None,
        )
    else:
        print("   [FILTER] -> Câu hỏi chung, tìm kiếm tự do không áp bộ lọc Metadata.")

    return qdrant_vectorstore.as_retriever(search_kwargs=search_kwargs)

# ==========================================
# 3. ĐỊNH NGHĨA CÁC NODES (HÀNH ĐỘNG)
# ==========================================
def spellcheck(state: GraphState):
    """Sửa lỗi chính tả và kiểm tra an toàn"""
    print("--- KIỂM TRA CHÍNH TẢ & THÊM DẤU ---")
    original_q = state["question"]
    
    corrected_q = spell_check_chain.invoke({"input": original_q})
    
    # Kích hoạt bộ lọc an toàn
    if not is_spellcheck_safe(original_q, corrected_q):
        corrected_q = original_q
        
    original_clean = clean_text(original_q)
    corrected_clean = clean_text(corrected_q)
    
    if original_clean != corrected_clean:
        print(f" -> Đã sửa: '{original_q}' thành '{corrected_q}'")
    else:
        print(" -> Câu đã chuẩn, không cần sửa.")
        
    return {"question": corrected_q}

# Dùng câu hoàn chỉnh thay vì ép LLM trích xuất keyword
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Viết lại câu hỏi độc lập dựa trên ngữ cảnh lịch sử chat. KHÔNG trả lời câu hỏi, chỉ định dạng lại để nó có ý nghĩa hoàn chỉnh."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])
query_rewrite_chain = contextualize_q_prompt | llm | StrOutputParser()

def retrieve(state: GraphState):
    """Tìm kiếm tài liệu từ Qdrant (Dùng câu hoàn chỉnh)"""
    print("--- 🔍 ĐANG TÌM KIẾM TÀI LIỆU (QDRANT) ---")
    original_question = state["question"]
    chat_history = state.get("chat_history", [])
    
    if chat_history:
        optimized_query = query_rewrite_chain.invoke({
            "input": original_question,
            "chat_history": chat_history
        })
    else:
        optimized_query = query_rewrite_chain.invoke({
            "input": original_question,
            "chat_history": []
        })

    print(f" -> Câu hỏi gốc: {original_question}")
    print(f" -> Câu hỏi đi tìm Qdrant: {optimized_query}")
    
    if not is_query_rewrite_safe(original_question, optimized_query):
        print(" -> Cau rewrite khong an toan cho retrieval, dung cau hoi goc.")
        optimized_query = original_question

    fallback_query = get_last_user_message(chat_history)
    dynamic_retriever = get_filtered_retriever(
        qdrant,
        original_question,
        llm,
        fallback_query=fallback_query,
    )

    docs = dynamic_retriever.invoke(optimized_query)
    docs = sorted(
        docs,
        key=lambda doc: (
            str(doc.metadata.get("source", "")),
            str(doc.metadata.get("chunk_id", "")),
            doc.page_content[:80],
        ),
    )

    diseases, herbs = extract_entities_from_query(original_question, llm)
    if not diseases and not herbs and fallback_query:
        diseases, herbs = extract_entities_from_query(fallback_query, llm)

    if herbs:
        filtered_docs = []
        for doc in docs:
            herb_candidates = parse_herb_field(doc.metadata.get("herbs", ""))
            content_norm = normalize_text(doc.page_content)
            if any(h in herb_candidates or h in content_norm for h in herbs):
                filtered_docs.append(doc)
        if len(filtered_docs) != len(docs):
            print(f"   -> [FILTER] Lọc theo cây thuốc: {len(docs)} -> {len(filtered_docs)} đoạn.")
        docs = sorted(
            filtered_docs,
            key=lambda doc: (
                str(doc.metadata.get("source", "")),
                str(doc.metadata.get("chunk_id", "")),
                doc.page_content[:80],
            ),
        )
        docs = filter_recipe_blocks_by_entities(docs, herbs, diseases)

    for doc in docs:
        raw_source = doc.metadata.get("source", "")
        file_name = raw_source.split("/")[-1]
        doc.metadata["source"] = SOURCE_MAPPING.get(file_name, file_name)
    
    # IN LOG ĐỂ DEBUG GIỐNG ADVANCED RAG
    print("\n--- 📚 CÁC CONTEXT NHẬN VỀ TỪ QDRANT ---")
    if not docs:
        print("❌ Không tìm thấy tài liệu nào.")
    else:
        for i, doc in enumerate(docs):
            source_file = doc.metadata.get("source", "Không rõ nguồn")
            print(f"\n[{i+1}] Source: {source_file}")
            print(f"Nội dung:\n{doc.page_content[:200]}...") # In 200 chữ đầu cho đỡ rác màn hình
    print("-" * 50 + "\n")
    
    return {"documents": docs, "question": original_question, "optimized_query": optimized_query, "web_searched": False}

def web_search(state: GraphState):
    """Tim bo sung tren internet nhung chi trong domain whitelist."""
    print("--- WEB SEARCH: TIM BO SUNG TREN NGUON DUOC PHEP ---")
    question = state["question"]
    search_query = state.get("optimized_query") or question

    docs = []
    if tavily_search_tool is None:
        print(" -> Tavily chua duoc cau hinh. Hay kiem tra TAVILY_API_KEY.")
        tavily_results = []
    else:
        try:
            tavily_results = tavily_search_tool.invoke({"query": search_query})
        except Exception as exc:
            print(f" -> Tavily search loi: {exc}")
            tavily_results = []

    if isinstance(tavily_results, dict):
        tavily_results = tavily_results.get("results", [])

    for result in tavily_results or []:
        if not isinstance(result, dict):
            continue

        url = result.get("url", "")
        content = result.get("content") or result.get("raw_content") or result.get("snippet") or ""
        if not url or not content:
            continue

        source_domain = urlparse(url).netloc.lower().split(":")[0]
        if source_domain.startswith("www."):
            source_domain = source_domain[4:]

        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": source_domain,
                    "url": url,
                    "source_type": "internet",
                },
            )
        )

    if not docs:
        print(" -> Khong tim thay tai lieu web phu hop trong whitelist.")
    else:
        print(f" -> Tim thay {len(docs)} tai lieu web trong whitelist.")
        for i, doc in enumerate(docs):
            print(f"    [{i+1}] {doc.metadata.get('url', doc.metadata.get('source', 'unknown'))}")

    return {
        "documents": docs,
        "question": question,
        "optimized_query": search_query,
        "web_searched": True,
    }

def direct_chat(state: GraphState):
    """Trả lời các câu hỏi giao tiếp cơ bản (Không dùng DB)"""
    print("--- GIAO TIẾP THÔNG THƯỜNG ---")
    question = state["question"]
    chat_history = state.get("chat_history", [])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Bạn là một trợ lý ảo thân thiện. Hãy tiếp nối cuộc trò chuyện dưới đây."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    chain = prompt | llm
    response = chain.invoke({"question": question, "chat_history": chat_history}).content
    return {"generation": response, "question": question}

def generate(state: GraphState):
    """Sinh câu trả lời dựa trên tài liệu (RAG)"""
    print("--- ĐANG SINH CÂU TRẢ LỜI ---")
    question = state["question"]
    documents = state["documents"]
    chat_history = state.get("chat_history", [])

    if not documents:
        return {
            "generation": "Không tìm thấy tài liệu phù hợp.",
            "question": question,
        }
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Bạn là một trợ lý ảo chuyên sâu về Y học cổ truyền, chẩn đoán bệnh và tư vấn dược liệu.

     Nhiệm vụ của bạn là trả lời câu hỏi dựa TRÊN DUY NHẤT tài liệu được cung cấp dưới đây.

    CÁC QUY TẮC NGHIÊM NGẶT ĐỂ TRÁNH ẢO TƯỞNG VÀ ĐẢM BẢO CHẤT LƯỢNG:
    1. ĐIỀU KIỆN TIÊN QUYẾT: Chỉ trả lời nếu nội dung câu hỏi hoặc triệu chứng của người dùng ĐÃ ĐƯỢC NHẮC ĐẾN hoặc CÓ DỮ LIỆU liên quan trực tiếp trong phần 'Tài liệu'.
    2. NẾU KHÔNG CÓ THÔNG TIN: Nếu phần 'Tài liệu' trống rỗng, hoặc hoàn toàn không chứa thông tin giúp trả lời câu hỏi, bạn BẮT BUỘC phải trả về câu sau và KHÔNG ĐƯỢC NÓI GÌ THÊM: "Xin lỗi, tôi chưa có thông tin về vấn đề này trong cơ sở dữ liệu hiện tại."
    3. TUYỆT ĐỐI KHÔNG tự bịa đặt, không suy diễn từ kiến thức y học cá nhân ngoài tài liệu.
    4. RÀNG BUỘC THEO CÂY THUỐC VÀ BỆNH/TRIỆU CHỨNG ĐƯỢC HỎI: Nếu câu hỏi nhắc tên cây thuốc/vị thuốc và bệnh/triệu chứng cụ thể, CHỈ được liệt kê công dụng/bài thuốc/cách dùng đồng thời liên quan trực tiếp đến đúng cây thuốc/vị thuốc đó và đúng bệnh/triệu chứng đó. KHÔNG được liệt kê bài thuốc chỉ có cây thuốc nhưng chữa bệnh khác, hoặc chỉ cùng bệnh nhưng không chứa cây thuốc được hỏi.
    5. MỨC ĐỘ CHI TIẾT (QUAN TRỌNG): Khi tài liệu có chứa các cách dùng, bài thuốc, liều lượng (gram), hay các loại cây phối hợp phù hợp trực tiếp với cây thuốc được hỏi, bạn PHẢI liệt kê ĐẦY ĐỦ TẤT CẢ các cách đó. TUYỆT ĐỐI KHÔNG ĐƯỢC tóm tắt qua loa hay bỏ sót bất kỳ một bài thuốc / liều lượng nào.
    6. Định dạng trả lời: NÊN SỬ DỤNG gạch đầu dòng (-) hoặc đánh số (1, 2, 3...) để phân tách các bài thuốc, các cách dùng khác nhau giúp người đọc dễ hiểu. Trình bày rõ ràng, rành mạch.
   
    Tài liệu:
    {context}"""),
        MessagesPlaceholder(variable_name="chat_history"), # Nhúng trí nhớ vào Prompt sinh văn bản
        ("human", "{question}")
    ])
    
    chain = prompt | llm
    context_blocks = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "Khong ro nguon")
        url = doc.metadata.get("url", "")
        source_type = doc.metadata.get("source_type", "vectorstore")
        source_line = f"[Nguon {i} | loai={source_type} | source={source}"
        if url:
            source_line += f" | url={url}"
        source_line += "]"
        context_blocks.append(f"{source_line}\n{doc.page_content}")
    context_str = "\n\n".join(context_blocks)
    response = chain.invoke({"context": context_str, "question": question, "chat_history": chat_history}).content
    response = strip_generated_sources(response) + format_answer_sources(documents)
    return {"generation": response, "question": question}

def fallback_answer(state: GraphState):
    """Kịch bản Tự sửa sai (Self-Correction) khi tài liệu sai"""
    print("--- KÍCH HOẠT FALLBACK ---")
    fallback_msg = (
        "Xin lỗi, tôi chưa tìm thấy thông tin phù hợp trong cơ sở dữ liệu hiện tại.\n"
        "Bạn có thể mô tả rõ hơn triệu chứng hoặc tên bệnh để tôi hỗ trợ tốt hơn."
    )
    return {"generation": fallback_msg, "question": state["question"]}

# ==========================================
# 4. ĐỊNH NGHĨA CÁC ĐIỀU KIỆN RẼ NHÁNH (EDGES)
# ==========================================
def route_question(state: GraphState):
    """Bộ định tuyến (Router): Quyết định có cần tìm Qdrant không"""
    print("--- ROUTER: PHÂN TÍCH CÂU HỎI ---")
    question = state["question"]
    if is_obvious_medical_query(question):
        print(" -> Rule guard: Cau hoi y te ro rang, kich hoat RAG.")
        return "retrieve"

    structured_llm_router = llm.with_structured_output(RouteQuery)
    
    system = """
Bạn là bộ định tuyến câu hỏi.

Chọn:
- 'vectorstore':
    + triệu chứng
    + bệnh
    + cây thuốc
    + y học cổ truyền
    + thuốc
    + sức khỏe
    + cơ thể
    + điều trị
    + đau nhức
    + thực phẩm chữa bệnh

- 'chat':
    + chào hỏi
    + cảm ơn
    + trò chuyện xã giao
    + hỏi AI là gì
"""
    route_prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{question}")])
    
    question_router = route_prompt | structured_llm_router
    source = question_router.invoke({"question": question})
    
    if source.datasource == "vectorstore":
        print(" -> Quyết định: Kích hoạt RAG (Tìm DB)")
        return "retrieve"
    else:
        print(" -> Quyết định: Trả lời trực tiếp")
        return "direct_chat"

def check_relevance(state: GraphState):
    """Grader (Giám khảo): Kiểm tra tài liệu có đúng ý câu hỏi không"""
    print("--- GRADER: KIỂM DUYỆT TÀI LIỆU ---")
    question = state["question"]
    documents = state["documents"]

    if not documents:
        if not state.get("web_searched", False):
            print(" -> Ket qua: Tai lieu rong, thu tim web whitelist.")
            return "web_search"
        print(" -> Kết quả: Tài liệu RỖNG (Kích hoạt Fallback)")
        return "fallback"
    
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    system = """Bạn là bộ kiểm duyệt tài liệu cho hệ thống RAG y học cổ truyền.

Chỉ trả lời 'yes' nếu tài liệu có thông tin trả lời trực tiếp câu hỏi.

Nếu câu hỏi có cả tên cây thuốc/vị thuốc và bệnh/triệu chứng, tài liệu phải đồng thời thỏa mãn:
1. Có nhắc đúng cây thuốc/vị thuốc hoặc tên đồng nghĩa.
2. Có nhắc đúng bệnh/triệu chứng được hỏi hoặc triệu chứng gần nghĩa trực tiếp.
3. Có công dụng, cách dùng, liều dùng hoặc bài thuốc liên quan trực tiếp.

Trả lời 'no' nếu tài liệu chỉ nhắc cây thuốc nhưng không nhắc bệnh/triệu chứng được hỏi, hoặc chỉ nhắc bệnh/triệu chứng nhưng không liên quan cây thuốc."""
    
    grade_prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "Tài liệu: {context}\n\nCâu hỏi: {question}")])
    retrieval_grader = grade_prompt | structured_llm_grader
    context_str = "\n".join([doc.page_content for doc in documents])
    score = retrieval_grader.invoke({"question": question, "context": context_str})
    
    if score.binary_score == "yes":
        print(" -> Kết quả: Tài liệu HỮU ÍCH (Cho phép trả lời)")
        return "generate"
    else:
        if not state.get("web_searched", False):
            print(" -> Ket qua: Tai lieu lac de, thu tim web whitelist.")
            return "web_search"
        print(" -> Kết quả: Tài liệu LẠC ĐỀ (Kích hoạt Fallback)")
        return "fallback"

# ==========================================
# 5. LẮP RÁP ĐỒ THỊ (BUILD GRAPH)
# ==========================================
workflow = StateGraph(GraphState)

# Khai báo các Nodes
workflow.add_node("spellcheck", spellcheck) 
workflow.add_node("retrieve", retrieve)
workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate)
workflow.add_node("direct_chat", direct_chat)
workflow.add_node("fallback_answer", fallback_answer)

# Khai báo đường đi (Edges)
workflow.set_entry_point("spellcheck")

workflow.add_conditional_edges(
    "spellcheck",
    route_question,
    {
        "retrieve": "retrieve",
        "direct_chat": "direct_chat",
    }
)

workflow.add_conditional_edges(
    "retrieve",
    check_relevance,
    {
        "generate": "generate",
        "web_search": "web_search",
        "fallback": "fallback_answer",
    }
)

workflow.add_conditional_edges(
    "web_search",
    check_relevance,
    {
        "generate": "generate",
        "web_search": "fallback_answer",
        "fallback": "fallback_answer",
    }
)

workflow.add_edge("generate", END)
workflow.add_edge("direct_chat", END)
workflow.add_edge("fallback_answer", END)

# Đóng gói Agent
app = workflow.compile()

# ==========================================
# 6. HÀM CHẠY THỬ
# ==========================================
if __name__ == "__main__":
    # 1. Khởi tạo Database
    init_db()
    
    # 2. Định nghĩa Session ID (Có thể thay đổi khi làm multi-user sau này)
    SESSION_ID = "agentic_user_01"
    
    # 3. Tải lịch sử từ DB lên RAM
    chat_history = load_history_from_db(SESSION_ID, limit=25)
    
    if chat_history:
        print(f"📦 Đã khôi phục {len(chat_history)//2} phiên hỏi-đáp từ Database!")
    else:
        chat_history = [] # Nếu user mới tinh thì khởi tạo rỗng
    
    while True:
        question = input("\n👤 BẠN HỎI: ")
        if question.lower() in ['quit', 'q', 'exit']:
            print("Tạm biệt!")
            break
            
        inputs = {"question": question, "chat_history": chat_history}

        # Chạy đồ thị LangGraph
        result = app.invoke(inputs)
        
        final_answer = result['generation']
        
        # Lấy context documents ra (nếu AI rẽ nhánh Direct Chat thì documents có thể không tồn tại)
        context_docs = result.get('documents', [])

        # In kết quả cuối cùng ra Terminal
        print("\n" + "="*50)
        print(f"🤖 BOT ĐÁP:\n{final_answer}")
        print("="*50 + "\n")
        
        # --- LƯU VÀO DATABASE ---
        # Đóng gói bằng chứng thành dạng Dictionary (chỉ lưu đoạn ngắn để nhẹ DB)
        sources_list = []
        if context_docs:
            for doc in context_docs:
                sources_list.append({
                    "content_snippet": doc.page_content[:200] + "...",
                    "metadata": doc.metadata
                })
                
        # Gọi hàm lưu vào PostgreSQL
        save_chat_turn_to_db(SESSION_ID, question, final_answer, sources_list)
        
        # --- CẬP NHẬT TRÍ NHỚ RAM ---
        # Giữ lại 10 tin nhắn gần nhất để làm Context cho câu hỏi tiếp theo
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=final_answer))
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]            
