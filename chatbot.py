import string
import os
import sys
import re
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http import models
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

# Call database functions from database.py
from database import init_db, save_chat_turn_to_db, load_history_from_db
from source_mapping import SOURCE_MAPPING

from typing import List
from pydantic import BaseModel, Field

# ==========================================
# LOAD API KEY
# ==========================================
def get_streamlit_secret(name, default=None):
    try:
        import streamlit as st
        return st.secrets.get(name, default)
    except Exception:
        return default


def is_running_in_container():
    return os.path.exists("/.dockerenv")


def normalize_qdrant_url(url):
    if not url:
        return url
    if not is_running_in_container() and url.rstrip("/") == "http://qdrant:6333":
        return "http://localhost:6333"
    return url


env_path = Path(__file__).resolve().parent / '.env'
sys.path.insert(0, str(env_path))
try:
    from key import OPENAI_API_KEY as KEY_FILE_OPENAI_API_KEY
except ImportError:
    KEY_FILE_OPENAI_API_KEY = None
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

OPENAI_API_KEY = (
    os.getenv("OPENAI_API_KEY")
    or get_streamlit_secret("OPENAI_API_KEY")
    or KEY_FILE_OPENAI_API_KEY
)
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment, Streamlit secrets, or .env/key.py.")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# CONFIG
QDRANT_URL = normalize_qdrant_url(
    os.getenv("QDRANT_LOCAL_URL")
    or get_streamlit_secret("QDRANT_LOCAL_URL")
    or os.getenv("QDRANT_URL")
    or get_streamlit_secret("QDRANT_URL")
    or KEY_FILE_QDRANT_LOCAL_URL
    or "http://qdrant:6333"
)
QDRANT_COLLECTION = (
    os.getenv("QDRANT_LOCAL_COLLECTION")
    or get_streamlit_secret("QDRANT_LOCAL_COLLECTION")
    or os.getenv("QDRANT_COLLECTION")
    or get_streamlit_secret("QDRANT_COLLECTION")
    or KEY_FILE_QDRANT_LOCAL_COLLECTION
    or "medical_docs"
)
QDRANT_API_KEY = (
    os.getenv("QDRANT_LOCAL_API_KEY")
    or get_streamlit_secret("QDRANT_LOCAL_API_KEY")
    or KEY_FILE_QDRANT_LOCAL_API_KEY
    or "qdrant_api_key"
)
QDRANT_TIMEOUT = int(
    os.getenv("QDRANT_TIMEOUT")
    or get_streamlit_secret("QDRANT_TIMEOUT")
    or "120"
)
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
SPARSE_EMBEDDING_MODEL = "Qdrant/bm25"
SEARCH_K = 30
ENTITY_SCORE_WEIGHT = 0.55
VECTOR_RANK_WEIGHT = 0.45

# Khởi tạo các biến toàn cục
if not QDRANT_API_KEY:
    raise RuntimeError("Missing local Qdrant API key.")
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
sparse_embeddings = FastEmbedSparse(model_name=SPARSE_EMBEDDING_MODEL, cache_dir=".fastembed_cache")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=QDRANT_TIMEOUT)
qdrant = QdrantVectorStore(
    client=client,
    collection_name=QDRANT_COLLECTION,
    embedding=embeddings,
    sparse_embedding=sparse_embeddings,
    retrieval_mode=RetrievalMode.HYBRID,
)
llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

# Chain tiền xử lý (Sửa lỗi chính tả & Thêm dấu)
spell_check_prompt = ChatPromptTemplate.from_messages([
    ("system", """Bạn là một công cụ tiền xử lý ngôn ngữ tiếng Việt. 
Nhiệm vụ của bạn:
1. Thêm dấu tiếng Việt nếu câu bị thiếu dấu (ví dụ: "cay actiso" -> "cây actisô").
2. TUYỆT ĐỐI GIỮ NGUYÊN CÁC DANH TỪ RIÊNG, TÊN CÂY THUỐC, THẢO DƯỢC, thuật ngữ y học dù nó được viết theo cách cũ hay phiên âm (ví dụ: actisô, atisô, sâm, quy...). KHÔNG ĐƯỢC tự ý sửa "actisô" thành "atisô" hoặc ngược lại.
3. KHÔNG ĐƯỢC thêm dấu chấm (.), dấu phẩy, hay dấu hỏi vào cuối kết quả.
4. KHÔNG đổi chữ hoa/chữ thường tùy tiện.
CHỈ trả về kết quả, KHÔNG giải thích."""),
    ("human", "{input}")
])
spell_check_chain = spell_check_prompt | llm | StrOutputParser()

# Prompt viết lại câu hỏi dựa trên lịch sử chat
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Viết lại câu hỏi độc lập dựa trên ngữ cảnh lịch sử chat. KHÔNG trả lời câu hỏi, chỉ định dạng lại để nó có ý nghĩa hoàn chỉnh."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
standalone_question_chain = contextualize_q_prompt | llm | StrOutputParser()

# Prompt trả lời y khoa nghiêm ngặt chống ảo tưởng
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """Bạn là một trợ lý ảo chuyên sâu về Y học cổ truyền, chẩn đoán bệnh và tư vấn dược liệu.

Nhiệm vụ của bạn là trả lời câu hỏi dựa TRÊN DUY NHẤT tài liệu được cung cấp dưới đây.

CÁC QUY TẮC NGHIÊM NGẶT ĐỂ TRÁNH ẢO TƯỞNG VÀ ĐẢM BẢO CHẤT LƯỢNG:
1. ĐIỀU KIỆN TIÊN QUYẾT: Chỉ trả lời nếu nội dung câu hỏi hoặc triệu chứng của người dùng ĐÃ ĐƯỢC NHẮC ĐẾN, hoặc có biến thể/hạng mục hẹp hơn liên quan trực tiếp trong phần 'Tài liệu'.
2. QUY TẮC BIẾN THỂ TÊN BỆNH / TRIỆU CHỨNG: Nếu câu hỏi chứa một cụm bệnh/triệu chứng lõi, thì các cụm trong tài liệu có chứa cụm lõi đó vẫn được xem là liên quan. Ví dụ: người dùng hỏi "mụn" hoặc "nổi mụn", tài liệu có "mụn nhọt", "mụn mủ", "mụn cóc" thì được phép trả lời dựa trên các đoạn đó; người dùng hỏi "mề đay" hoặc "nổi mề đay", tài liệu có "mày đay" hoặc "mề đay" thì xem là cùng vấn đề. Chỉ dùng các biến thể xuất hiện trong Tài liệu, không tự mở rộng sang bệnh khác.
3. CÂU HỎI YES/NO VỀ "CÂY A CÓ CHỮA BỆNH/CÔNG DỤNG B KHÔNG": Nếu Tài liệu có thông tin về đúng cây thuốc/vị thuốc A nhưng KHÔNG nhắc A dùng cho bệnh/công dụng B, KHÔNG được trả lời câu xin lỗi chung. Hãy trả lời theo hướng: "Theo tài liệu hiện có, tôi chưa thấy thông tin cho thấy A dùng để chữa/hỗ trợ B." Sau đó nêu ngắn gọn các công dụng của A thật sự có trong Tài liệu. Không được kết luận tuyệt đối rằng A không chữa B ngoài phạm vi tài liệu.
4. NẾU KHÔNG CÓ THÔNG TIN: Nếu phần 'Tài liệu' trống rỗng, hoặc hoàn toàn không chứa thông tin về cây thuốc/bệnh/công dụng được hỏi, bạn BẮT BUỘC phải trả về câu sau và KHÔNG ĐƯỢC NÓI GÌ THÊM: "Xin lỗi, tôi chưa có thông tin về vấn đề này trong cơ sở dữ liệu hiện tại."
5. TUYỆT ĐỐI KHÔNG tự bịa đặt, không suy diễn từ kiến thức y học cá nhân ngoài tài liệu.
6. RÀNG BUỘC THEO CÂY THUỐC ĐƯỢC HỎI: Nếu câu hỏi nhắc tên một cây thuốc hoặc vị thuốc cụ thể, CHỈ được liệt kê các công dụng/bài thuốc/cách dùng có chứa chính cây thuốc/vị thuốc đó hoặc tên đồng nghĩa của nó trong tài liệu. KHÔNG được liệt kê bài thuốc chỉ cùng bệnh/triệu chứng nhưng không chứa cây thuốc được hỏi.
7. MỨC ĐỘ CHI TIẾT (QUAN TRỌNG): Khi tài liệu có chứa các cách dùng, bài thuốc, liều lượng (gram), hay các loại cây phối hợp phù hợp trực tiếp với cây thuốc được hỏi, bạn PHẢI liệt kê ĐẦY ĐỦ TẤT CẢ các cách đó. TUYỆT ĐỐI KHÔNG ĐƯỢC tóm tắt qua loa hay bỏ sót bất kỳ một bài thuốc / liều lượng nào.
8. Định dạng trả lời: NÊN SỬ DỤNG gạch đầu dòng (-) hoặc đánh số (1, 2, 3...) để phân tách các bài thuốc, các cách dùng khác nhau giúp người đọc dễ hiểu. Trình bày rõ ràng, rành mạch.
9. Trích dẫn nguồn: Cuối câu trả lời (nếu tìm thấy), ghi rõ "Nguồn tham khảo: Tên các tài liệu".

Tài liệu:
{context}"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

document_prompt = PromptTemplate(
    input_variables=["page_content", "source"],
    template="Nội dung: {page_content}\nNguồn trích dẫn: {source}\n---"
)

question_answer_chain = create_stuff_documents_chain(
    llm=llm, 
    prompt=qa_prompt,
    document_prompt=document_prompt 
)


# ========================================================
# 1. NÂNG CẤP SANG PHIÊN BẢN MẢNG (LIST[STR]) ĐỂ CHỨA ĐA THỰC THỂ
# ========================================================
class MedicalEntities(BaseModel):
    diseases: List[str] = Field(description="Mảng chứa từ vựng chỉ bệnh lý/triệu chứng. BẮT BUỘC CHỈ SAO CHÉP từ có thật trong chuỗi đầu vào. Nếu câu hỏi không nhắc ĐÍCH DANH tên bệnh nào, BẮT BUỘC trả về mảng rỗng [].")
    herbs: List[str] = Field(description="Mảng chứa tên cốt lõi của thảo dược. BẮT BUỘC CHỈ SAO CHÉP từ có thật trong chuỗi đầu vào. Lược bỏ chữ 'cây', 'lá', 'quả'. Nếu có ngoặc đơn thì tách riêng. Trả về [] nếu không có.")


DISEASE_ENTITY_PREFIXES = (
    "toi bi ",
    "toi mac ",
    "em bi ",
    "em mac ",
    "minh bi ",
    "minh mac ",
    "chau bi ",
    "chau mac ",
    "be bi ",
    "be mac ",
    "nguoi benh bi ",
    "nguoi benh mac ",
    "dang bi ",
    "dang mac ",
    "bi ",
    "mac ",
    "co ",
    "noi ",
    "benh ",
    "chung ",
    "trieu chung ",
    "tôi bị ",
    "tôi mắc ",
    "em bị ",
    "em mắc ",
    "mình bị ",
    "mình mắc ",
    "cháu bị ",
    "cháu mắc ",
    "bé bị ",
    "bé mắc ",
    "người bệnh bị ",
    "người bệnh mắc ",
    "đang bị ",
    "đang mắc ",
    "bị ",
    "mắc ",
    "có ",
    "nổi ",
    "bệnh ",
    "chứng ",
    "triệu chứng ",
)


def clean_disease_entity(value):
    term = " ".join(str(value).strip().lower().split())
    term = term.strip(string.punctuation + " ")

    changed = True
    while changed:
        changed = False
        for prefix in DISEASE_ENTITY_PREFIXES:
            if term.startswith(prefix):
                term = term[len(prefix):].strip()
                changed = True
                break

    return term.strip(string.punctuation + " ")


def extract_entities_from_query(user_query, llm):
    """
    Trích xuất thực thể bằng persona Text Parser (Triệt tiêu hoàn toàn tính suy diễn Y học)
    """
    prompt = f"""
    BẠN LÀ MỘT CỖ MÁY PHÂN TÍCH CÚ PHÁP VĂN BẢN (TEXT PARSER). 
    BẠN KHÔNG PHẢI LÀ BÁC SĨ HAY CHUYÊN GIA Y TẾ. BẠN ĐÃ BỊ XÓA BỎ TOÀN BỘ KIẾN THỨC Y HỌC NỘI TẠI.
    
    Nhiệm vụ duy nhất của bạn là: Đọc chuỗi văn bản đầu vào và SAO CHÉP LẠI ĐÚNG NHỮNG TỪ NGỮ chỉ bệnh hoặc tên cây thuốc CÓ MẶT CHÍNH XÁC trong chuỗi đó.
    
    LỆNH CẤM TUYỆT ĐỐI (SẼ BỊ PHẠT NẾU VI PHẠM):
    1. KHÔNG ĐƯỢC TỰ SUY LUẬN. Tuyệt đối không tự điền thêm bất kỳ tên bệnh, tên cây thuốc, hoặc từ khóa nào không xuất hiện bằng chữ trong chuỗi đầu vào.
    2. Nếu người dùng hỏi chung chung như "tác dụng của cây A", "cây B chữa bệnh gì" (không nhắc đến một cái bệnh cụ thể nào), mảng diseases PHẢI LÀ MẢNG RỖNG [].
    
    QUY TẮC LÀM SẠCH TÊN BỆNH / TRIỆU CHỨNG (diseases):
    - Chỉ trả về CỤM BỆNH hoặc TRIỆU CHỨNG LÕI.
    - Được phép lược bỏ các từ chỉ trạng thái/ngữ cảnh như: "tôi bị", "bị", "mắc", "có", "đang bị", "nổi", "bệnh", "chứng", "triệu chứng".
    - Không được thêm bệnh mới không có trong câu; chỉ bỏ từ thừa ở đầu cụm.
    - Input: "tôi bị nổi mề đay" -> diseases: ["mề đay"], herbs: []
    - Input: "bị nổi mẩn đỏ ở tay" -> diseases: ["mẩn đỏ"], herbs: []
    - Input: "nổi mụn ở mặt" -> diseases: ["mụn"], herbs: []
    - Input: "mắc viêm xoang" -> diseases: ["viêm xoang"], herbs: []

    QUY TẮC LÀM SẠCH TÊN CÂY (herbs):
    - Bỏ các từ: "cây", "lá", "củ", "quả", "rễ", "hoa", "vị thuốc". Ví dụ: "lá khôi" -> "khôi".
    - Nếu có chữ trong ngoặc, tách thành phần tử riêng. Ví dụ: "cỏ xước (ngưu tất)" -> ["cỏ xước", "ngưu tất"].
    
    VÍ DỤ KIỂM THỬ:
    - Input: "công dụng của cây dạ cẩm là gì" -> diseases: [], herbs: ["dạ cẩm"]
    - Input: "cây bưởi và lá khôi chữa bệnh gì" -> diseases: [], herbs: ["bưởi", "khôi"]
    - Input: "dạ cẩm chữa nhiệt miệng" -> diseases: ["nhiệt miệng"], herbs: ["dạ cẩm"]
    - Input: "Cỏ xước (ngưu tất) chữa thoái hóa khớp" -> diseases: ["thoái hóa khớp"], herbs: ["cỏ xước", "ngưu tất"]
    - Input: "làm sao để chữa đau dạ dày" -> diseases: ["đau dạ dày"], herbs: []
    - Input: "làm sao để chữa đau đầu" -> diseases: ["đau đầu"], herbs: []
    
    CHUỖI ĐẦU VÀO CẦN PHÂN TÍCH: "{user_query}"
    """
    try:
        structured_llm = llm.with_structured_output(MedicalEntities)
        response = structured_llm.invoke(prompt)
        
        # CHỐT CHẶN BẰNG PYTHON: Ép về cụm bệnh/triệu chứng lõi.
        safe_diseases = []
        for disease in response.diseases or []:
            cleaned_disease = clean_disease_entity(disease)
            if cleaned_disease and cleaned_disease not in safe_diseases:
                safe_diseases.append(cleaned_disease)
        safe_herbs = [h.strip().lower() for h in response.herbs] if response.herbs else []
        
        return safe_diseases, safe_herbs
    except Exception as e:
        return [], []


def get_last_user_message(chat_history):
    for message in reversed(chat_history):
        if isinstance(message, HumanMessage):
            return message.content
    return ""


def normalize_text(text):
    text = " ".join(text.lower().split())
    return text.translate(str.maketrans('', '', string.punctuation))


def is_greeting_or_smalltalk(text):
    clean = normalize_text(text)
    greetings = {
        "chao",
        "chao ban",
        "xin chao",
        "xin chao ban",
        "hi",
        "hello",
        "hey",
        "chào",
        "chào bạn",
        "xin chào",
        "xin chào bạn",
    }
    return clean in greetings


def parse_herb_field(value):
    if not value:
        return []
    if isinstance(value, list):
        return [item.strip().lower() for item in value if str(item).strip()]
    return [item.strip().lower() for item in str(value).split(",") if item.strip()]


def entity_match_score(doc, diseases, herbs):
    metadata_herbs = parse_herb_field(doc.metadata.get('herbs', ''))
    metadata_disease = normalize_text(str(doc.metadata.get('disease', '')))
    content_norm = normalize_text(doc.page_content)

    score = 0
    for herb in herbs:
        herb_norm = normalize_text(herb)
        if herb_norm in metadata_herbs:
            score += 3
        elif herb_norm in content_norm:
            score += 2

    for disease in diseases:
        disease_norm = normalize_text(disease)
        if disease_norm and disease_norm in metadata_disease:
            score += 2
        elif disease_norm and disease_norm in content_norm:
            score += 1

    return score


def sort_docs_stably(docs):
    return sorted(
        docs,
        key=lambda doc: (
            str(doc.metadata.get("source", "")),
            str(doc.metadata.get("chunk_id", "")),
            doc.page_content[:80]
        )
    )


def weighted_rerank_docs(docs, diseases, herbs):
    if not docs:
        return []

    entity_scores = [entity_match_score(doc, diseases, herbs) for doc in docs]
    max_entity_score = max(entity_scores)
    max_rank = max(len(docs) - 1, 1)

    ranked_docs = []
    for rank, doc in enumerate(docs):
        entity_norm = entity_scores[rank] / max_entity_score if max_entity_score else 0
        vector_rank_norm = 1 - (rank / max_rank)
        final_score = (
            ENTITY_SCORE_WEIGHT * entity_norm
            + VECTOR_RANK_WEIGHT * vector_rank_norm
        )
        ranked_docs.append((final_score, entity_scores[rank], vector_rank_norm, doc))

    return [
        doc
        for final_score, entity_score, vector_rank_score, doc in sorted(
            ranked_docs,
            key=lambda item: (
                -item[0],
                -item[1],
                -item[2],
                str(item[3].metadata.get("source", "")),
                str(item[3].metadata.get("chunk_id", "")),
                item[3].page_content[:80],
            )
        )
    ]
    
# ========================================================
# 2. TRUY XUẤT HYBRID VÀ ƯU TIÊN ENTITY MỀM
# ========================================================
def get_filtered_retriever(qdrant_vectorstore, primary_query, llm, fallback_query=None):
    return get_hybrid_retriever(qdrant_vectorstore, primary_query, llm, fallback_query=fallback_query)

def get_hybrid_retriever(qdrant_vectorstore, primary_query, llm, fallback_query=None):
    search_kwargs = {"k": SEARCH_K}
    print("   [HYBRID] -> Dense+sparse retrieval without metadata should conditions.")
    return qdrant_vectorstore.as_retriever(search_kwargs=search_kwargs)


def chat_with_medical_bot(user_question: str, chat_history: list, qdrant_vectorstore):
    print("-" * 50)
    print("Đang suy nghĩ câu trả lời...")

    if is_greeting_or_smalltalk(user_question):
        print("   -> [SMALLTALK] Câu chào hỏi, trả lời trực tiếp không truy vấn Qdrant.")
        return "Chào bạn! Tôi có thể hỗ trợ bạn tra cứu thông tin về Y học cổ truyền, cây thuốc hoặc triệu chứng liên quan.", []
    
    corrected_question = spell_check_chain.invoke({"input": user_question})

    original_clean = normalize_text(user_question)
    corrected_clean = normalize_text(corrected_question)
    
    prefix_message = ""
    if original_clean != corrected_clean:
        prefix_message = f"*(Có phải ý bạn là: {corrected_question}?)*\n\n"
        print(f"   -> Đã phát hiện và sửa lỗi: {corrected_question}")
    
    print("Tớ đang đọc tài liệu, cậu đợi tí nhé...")
    
    standalone_query = standalone_question_chain.invoke({
        "chat_history": chat_history,
        "input": corrected_question
    })
    
    fallback_query = get_last_user_message(chat_history)
    dynamic_retriever = get_hybrid_retriever(
        qdrant_vectorstore,
        corrected_question,
        llm,
        fallback_query=fallback_query
    )
    
    context_docs = dynamic_retriever.invoke(standalone_query)
    print(f"   -> Đã lấy lên {len(context_docs)} đoạn tài liệu liên quan.")

    diseases, herbs = extract_entities_from_query(corrected_question, llm)
    if not diseases and not herbs and fallback_query:
        diseases, herbs = extract_entities_from_query(fallback_query, llm)

    if diseases or herbs:
        context_docs = weighted_rerank_docs(context_docs, diseases, herbs)
        print("   -> [HYBRID] Sap xep theo weighted score: 0.55 entity_norm + 0.45 vector_rank_norm.")
        
    TOP_K_TO_LLM = 10
    if len(context_docs) > TOP_K_TO_LLM:
        context_docs = context_docs[:TOP_K_TO_LLM]
        print(f"   -> [OPTIMIZE] Đã cắt giữ lại Top {TOP_K_TO_LLM} tài liệu tinh túy nhất để đưa cho LLM đọc.")

    if context_docs:
        print("   -> [TÀI LIỆU] Danh sách tài liệu đã lấy:")
        for idx, doc in enumerate(context_docs, start=1):
            raw_source = doc.metadata.get('source', 'không rõ nguồn')
            snippet = doc.page_content[:200].replace("\n", " ")
            print(f"      {idx}. Nguồn: {raw_source} | Trích đoạn: {snippet}...")
    
    # CHỐT CHẶN CỨNG: Chặn đứng LLM bịa đặt nếu Qdrant không trả ra dữ liệu phù hợp
    if not context_docs or len(context_docs) == 0:
        print("   -> [HỆ THỐNG] Không có tài liệu. Chặn LLM bịa đặt.")
        fallback_msg = prefix_message + "Xin lỗi, tôi chưa có thông tin về vấn đề này trong cơ sở dữ liệu hiện tại."
        return fallback_msg, []
        
    for doc in context_docs:
        raw_source = doc.metadata.get('source', '')
        file_name = raw_source.split('/')[-1]
        doc.metadata['source'] = SOURCE_MAPPING.get(file_name, file_name)

    response_text = question_answer_chain.invoke({
        "context": context_docs,
        "input": corrected_question,
        "chat_history": chat_history
    })
    
    final_answer = prefix_message + response_text
    
    print("\n" + "="*50)
    print(f"🤖 BOT ĐÁP:\n{final_answer}")
    print("="*50 + "\n")
    
    return final_answer, context_docs


if __name__ == "__main__":
    init_db()
    SESSION_ID = "user_01" 
    
    chat_history = load_history_from_db(SESSION_ID, limit=25)
    if chat_history:
        print(f"📦 Đã khôi phục {len(chat_history)//2} phiên hỏi-đáp từ Database!")

    while True:
        question = input("\n👤 BẠN HỎI: ")
        if question.lower() in ['quit', 'q', 'exit']:
            print("Tạm biệt!")
            break
            
        answer, context_docs = chat_with_medical_bot(question, chat_history, qdrant)
        
        sources_list = []
        for doc in context_docs:
            sources_list.append({
                "content_snippet": doc.page_content[:200] + "...", 
                "metadata": doc.metadata
            })
            
        save_chat_turn_to_db(SESSION_ID, question, answer, sources_list)
        
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]
