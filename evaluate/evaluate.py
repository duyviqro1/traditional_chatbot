import argparse
import json
import re
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import sys
from pathlib import Path

# ==========================================
# 0. CẤU HÌNH API KEY TỪ FILE KEY.PY
# ==========================================
env_path = Path(__file__).resolve().parent.parent / '.env'
sys.path.insert(0, str(env_path))
from key import OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ==========================================
# 1. KHỞI TẠO CÁC CÔNG CỤ
# ==========================================
print("Đang khởi tạo mô hình OpenAI Embeddings và GPT-4o-mini...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

# ==========================================
# 2. CÁC HÀM TÍNH TOÁN CHỈ SỐ
# ==========================================
def parse_id_selection(value: str | None) -> set[str] | None:
    if not value:
        return None

    selected_ids: set[str] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue

        if "-" in part:
            start_text, end_text = [x.strip() for x in part.split("-", 1)]
            if not start_text.isdigit() or not end_text.isdigit():
                raise argparse.ArgumentTypeError(f"Khoảng id không hợp lệ: {part}")
            start_id = int(start_text)
            end_id = int(end_text)
            if start_id > end_id:
                raise argparse.ArgumentTypeError(f"Khoảng id không hợp lệ: {part}")
            selected_ids.update(str(i) for i in range(start_id, end_id + 1))
        else:
            if not part.isdigit():
                raise argparse.ArgumentTypeError(f"Id không hợp lệ: {part}")
            selected_ids.add(part)

    if not selected_ids:
        raise argparse.ArgumentTypeError("Danh sách id không được rỗng.")
    return selected_ids


def filter_by_ids(items, selected_ids):
    if selected_ids is None:
        return items
    return [item for item in items if str(item.get("id")) in selected_ids]


def calculate_fact_score(question, reference, candidate, context_text=""):
    try:
        extract_prompt = ChatPromptTemplate.from_template(
            "Câu hỏi: {question}\n"
            "Câu trả lời: {candidate}\n\n"
            "Hãy liệt kê các tuyên bố sự thật cốt lõi dùng để trả lời TRỰC TIẾP câu hỏi trên.\n"
            "Chỉ lấy fact về cây thuốc đang được hỏi, bệnh/chứng đang được hỏi, thành phần bài thuốc, liều lượng, cách dùng, chống chỉ định hoặc lưu ý liên quan trực tiếp.\n"
            "Bỏ qua công dụng chung, bệnh khác, triệu chứng khác hoặc thông tin mở rộng nếu chúng không cần thiết để trả lời đúng câu hỏi.\n"
            "Ví dụ: nếu câu hỏi hỏi chữa đau lưng, mỏi gối thì bỏ qua fact về an thai, huyết áp tăng, bệnh khác nếu chúng không nằm trong cách chữa đau lưng, mỏi gối.\n"
            "Mỗi dòng là một fact ngắn gọn dưới dạng gạch đầu dòng.\n"
            "CHỈ liệt kê fact, không giải thích thêm."
        )
        extract_chain = extract_prompt | judge_llm | parser
        facts_raw = extract_chain.invoke({"question": question, "candidate": candidate})
        
        facts = [f.strip("- *") for f in facts_raw.split("\n") if f.strip()]
        if not facts: return 0.0

        verify_prompt = ChatPromptTemplate.from_template(
            "Câu hỏi: {question}\n"
            "Đáp án chuẩn:\n{reference}\n\n"
            "Ngữ cảnh truy xuất:\n{context_text}\n\n"
            "Hãy kiểm tra tuyên bố sau:\n"
            "{fact}\n\n"
            "Trả lời ĐÚNG chỉ khi tuyên bố vừa liên quan trực tiếp đến câu hỏi, vừa được hỗ trợ bởi Đáp án chuẩn hoặc Ngữ cảnh truy xuất.\n"
            "Trả lời SAI nếu tuyên bố nói về công dụng/bệnh khác không cần thiết cho câu hỏi, dù thông tin đó có xuất hiện trong ngữ cảnh.\n"
            "Trả lời SAI nếu tuyên bố mâu thuẫn hoặc không xuất hiện trong cả hai nguồn.\n"
            "CHỈ trả lời duy nhất 1 từ: ĐÚNG hoặc SAI."
        )
        verify_chain = verify_prompt | judge_llm | parser
        
        correct_count = 0
        for fact in facts:
            result = verify_chain.invoke({
                "question": question,
                "reference": reference,
                "context_text": context_text,
                "fact": fact,
            }).strip().upper()
            if result.startswith("ĐÚNG") or result.startswith("DUNG"):
                correct_count += 1
                
        return correct_count / len(facts)
    except Exception as e:
        print(f"Lỗi FactScore: {e}")
        return 0.0

def get_llm_score(prompt_template, **kwargs):
    try:
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | judge_llm | parser
        response = chain.invoke(kwargs)
        nums = re.findall(r"0\.\d+|1\.0|1|0", response)
        return float(nums[0]) if nums else 0.0
    except: 
        return 0.0

def calculate_llm_context_precision(question, contexts):
    hits = []
    prompt_template = (
        "Bạn là Giám khảo. Hãy đánh giá xem Đoạn văn này có chứa thông tin hữu ích để trả lời Câu hỏi không.\n\n"
        "Câu hỏi: {question}\n"
        "Đoạn văn: {ctx}\n\n"
        "QUY TRÌNH:\n"
        "1. Suy luận ngắn gọn xem thông tin trong đoạn văn có ăn khớp và giúp ích cho câu hỏi không.\n"
        "2. Ở cuối câu trả lời, BẮT BUỘC chốt lại bằng đúng 1 trong 2 cụm từ sau (phải có ngoặc vuông):\n"
        "[KẾT_LUẬN: CÓ]\n"
        "[KẾT_LUẬN: KHÔNG]"
    )
    for i, ctx in enumerate(contexts):
        try:
            prompt = ChatPromptTemplate.from_template(prompt_template)
            chain = prompt | judge_llm | parser
            res = chain.invoke({"question": question, "ctx": ctx})
            if "[KẾT_LUẬN: CÓ]" in res.upper():
                relevant_count = len(hits) + 1
                hits.append(relevant_count / (i + 1))
        except:
            pass
    return np.mean(hits) if hits else 0.0

# ==========================================
# 3. ĐỌC DỮ LIỆU JSON & MAP THEO ID
# ==========================================
base_dir = os.path.dirname(os.path.abspath(__file__))
cli_parser = argparse.ArgumentParser(description="Đánh giá answers.json, có thể chỉ định id câu hỏi.")
cli_parser.add_argument("--ids", type=parse_id_selection, help="Danh sách id cần đánh giá, ví dụ: 1 hoặc 1,3,5-7")
cli_parser.add_argument("--answers", default=os.path.join(base_dir, 'answers.json'), help="File câu trả lời cần đánh giá.")
cli_parser.add_argument("--ground-truth", default=os.path.join(base_dir, 'ground_truth.json'), help="File ground truth.")
args = cli_parser.parse_args()

answers_path = args.answers
gt_path = args.ground_truth

print(f"Đang đọc dữ liệu từ:\n- {answers_path}\n- {gt_path}...")

try:
    with open(answers_path, 'r', encoding='utf-8') as f:
        answers_data = json.load(f)
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
except FileNotFoundError as e:
    print(f"Không tìm thấy file JSON. Vui lòng kiểm tra lại đường dẫn! Lỗi: {e}")
    exit()

# TẠO TỪ ĐIỂN ĐỐI CHIẾU DỰA TRÊN TRƯỜNG 'id'
gt_dict = {item['id']: item['tra_loi'] for item in gt_data if 'id' in item}
answers_data = filter_by_ids(answers_data, args.ids)

if args.ids is not None:
    found_ids = {str(item.get("id")) for item in answers_data}
    missing_ids = sorted(args.ids - found_ids, key=lambda x: int(x) if x.isdigit() else x)
    print(f"Đang đánh giá {len(answers_data)} câu hỏi theo id chỉ định.")
    if missing_ids:
        print(f"Cảnh báo: Không tìm thấy kết quả cho id: {', '.join(missing_ids)}")

# ==========================================
# 4. CHẠY ĐÁNH GIÁ (EVALUATION LOOP)
# ==========================================
print(f"\n{'ID':<3} | {'Fact':<5} | {'Ans-Sim':<7} | {'Auto-Pre':<8} | {'Auto-Rec':<8} | {'LLM-Rel':<7} | {'LLM-Rec':<7} | {'Faithful':<8} | {'LLM-Pre':<7}")
print("-" * 85)

results_summary = []

for item in answers_data:
    q_id = item.get("id")
    
    if q_id is None:
        print("Cảnh báo: Bỏ qua một câu do file answers.json thiếu trường 'id'.")
        continue

    ground_truth = gt_dict.get(q_id)
    
    if not ground_truth:
        print(f"Bỏ qua câu {q_id} do không tìm thấy Ground Truth có ID tương ứng.")
        continue

    question = item["cau_hoi"].strip()
    bot_answer = item["tra_loi"]
    contexts = [doc["page_content"] for doc in item.get("context", [])]

    # --- TÍNH TOÁN SEMANTIC & AUTOMATIC CONTEXT ---
    v_bot = embeddings.embed_query(bot_answer)
    v_gt = embeddings.embed_query(ground_truth)
    sem_s = cosine_similarity([v_bot], [v_gt])[0][0]

    v_q = embeddings.embed_query(question)
    auto_precision = 0.0
    auto_recall = 0.0
    
    if contexts:
        v_ctx = [embeddings.embed_query(c) for c in contexts]
        
        # Auto Precision
        similarities = [cosine_similarity([v_q], [vc])[0][0] for vc in v_ctx]
        auto_pre_scores = []
        for i, sim in enumerate(similarities):
            if sim > 0.4: 
                relevant_count = len([s for s in similarities[:i+1] if s > 0.4])
                auto_pre_scores.append(relevant_count / (i + 1))
        auto_precision = np.mean(auto_pre_scores) if auto_pre_scores else 0.0
        
        # Auto Recall
        gt_ctx_similarities = [cosine_similarity([v_gt], [vc])[0][0] for vc in v_ctx]
        auto_recall = max(gt_ctx_similarities) if gt_ctx_similarities else 0.0

    context_text = "\n---\n".join(contexts) if contexts else ""

    # --- TÍNH TOÁN LLM JUDGED ---
    f_score = calculate_fact_score(question, ground_truth, bot_answer, context_text)
    
    rel_template = (
        "Bạn là một Giám khảo Y khoa. Nhiệm vụ của bạn là đánh giá Độ liên quan (Context Relevance) "
        "của Ngữ cảnh được truy xuất so với Câu hỏi của người dùng.\n\n"
        
        "THÔNG TIN ĐẦU VÀO:\n"
        "- Câu hỏi: {question}\n"
        "- Ngữ cảnh truy xuất: {context_text}\n\n"
        
        "QUY TRÌNH CHẤM ĐIỂM (Chain-of-Thought):\n"
        "Bước 1 (Phân tích Câu hỏi): Xác định cốt lõi người dùng đang muốn tìm thông tin gì (bệnh gì, cây gì, cách dùng ra sao).\n"
        "Bước 2 (Quét Ngữ cảnh): Kiểm tra xem ngữ cảnh có chứa thông tin thực sự giải quyết được mục tiêu ở Bước 1 hay không. "
        "Lưu ý: Chỉ trùng từ khóa là CHƯA ĐỦ, nội dung phải thực sự hữu ích.\n"
        "Bước 3 (Lập luận): Giải thích ngắn gọn mức độ hữu ích của ngữ cảnh (Giải quyết được toàn bộ, một phần, hay hoàn toàn lạc đề/vô dụng).\n"
        "Bước 4 (Chốt điểm): Ở DÒNG CUỐI CÙNG, đưa ra một con số từ 0.0 (hoàn toàn vô dụng) đến 1.0 (chứa đầy đủ thông tin để trả lời).\n"
    )
    recall_template = (
        "Bạn là một Giám khảo Y khoa. Nhiệm vụ của bạn là đánh giá Độ bao phủ (Context Recall), "
        "tức là kiểm tra xem Ngữ cảnh có mang về ĐỦ thông tin như Đáp án chuẩn hay không.\n\n"
        
        "THÔNG TIN ĐẦU VÀO:\n"
        "- Câu hỏi: {question}\n"
        "- Đáp án chuẩn (Ground Truth): {ground_truth}\n"
        "- Ngữ cảnh truy xuất: {context_text}\n\n"
        
        "QUY TRÌNH CHẤM ĐIỂM (Chain-of-Thought):\n"
        "Bước 1 (Trích xuất Đáp án chuẩn): Chia 'Đáp án chuẩn' thành các ý chính/sự thật quan trọng.\n"
        "Bước 2 (Đối chiếu Ngữ cảnh): Tìm kiếm xem mỗi ý chính ở Bước 1 có mặt trong 'Ngữ cảnh truy xuất' hay không.\n"
        "Bước 3 (Lập luận): Chỉ ra rõ ràng những ý nào Ngữ cảnh đã bao phủ được, và những ý nào Ngữ cảnh đã BỎ SÓT so với Đáp án chuẩn.\n"
        "Bước 4 (Chốt điểm): Ở DÒNG CUỐI CÙNG, đưa ra một con số từ 0.0 đến 1.0 "
        "(Điểm = Số ý có trong Ngữ cảnh / Tổng số ý của Đáp án chuẩn).\n"
    )
    faithfulness_template = (
        "Bạn là một Giám khảo Y khoa cực kỳ khắt khe. Nhiệm vụ của bạn là đánh giá Độ trung thực (Faithfulness) "
        "của Câu trả lời so với Ngữ cảnh truy xuất.\n\n"
        
        "Hãy thực hiện từng bước phân tích dưới đây. BẮT BUỘC phải viết ra suy luận của bạn trước khi cho điểm.\n\n"
        
        "THÔNG TIN ĐẦU VÀO:\n"
        "- Câu hỏi: {question}\n"
        "- Ngữ cảnh truy xuất: {context_text}\n"
        "- Câu trả lời của bot: {bot_answer}\n\n"
        
        "QUY TRÌNH CHẤM ĐIỂM (Chain-of-Thought):\n"
        "Bước 1 (Trích xuất): Bóc tách 'Câu trả lời của bot' thành các sự thật/mệnh đề độc lập (Ví dụ: 'Cây A chữa bệnh B', 'Liều dùng là C').\n"
        "Bước 2 (Kiểm chứng): Quét qua 'Ngữ cảnh truy xuất' để tìm bằng chứng cho TỪNG mệnh đề trên.\n"
        "   - Nếu mệnh đề có thông tin khớp hoàn toàn với ngữ cảnh -> Đánh giá: HỢP LỆ.\n"
        "   - Nếu mệnh đề chứa thông tin tự bịa, suy diễn, hoặc không xuất hiện trong ngữ cảnh -> Đánh giá: ẢO GIÁC.\n"
        "Bước 3 (Kết luận): Đưa ra nhận xét ngắn gọn về mức độ trung thực của câu trả lời.\n"
        "Bước 4 (Chốt điểm): Ở DÒNG CUỐI CÙNG của phản hồi, đưa ra một con số duy nhất từ 0.0 đến 1.0 "
        "(Tương đương với tỷ lệ mệnh đề HỢP LỆ trên tổng số mệnh đề).\n\n"
        
        "BẮT ĐẦU PHÂN TÍCH:\n"
    )
    
    c_relevance = get_llm_score(rel_template, question=question, context_text=context_text) if contexts else 0.0
    c_recall = get_llm_score(recall_template, question=question, ground_truth=ground_truth, context_text=context_text) if contexts else 0.0
    faithfulness = get_llm_score(faithfulness_template, question=question, context_text=context_text, bot_answer=bot_answer) if contexts else 0.0
    llm_precision = calculate_llm_context_precision(question, contexts) if contexts else 0.0

    # In kết quả
    print(f"{q_id:<3} | {f_score:<5.2f} | {sem_s:<7.2f} | {auto_precision:<8.2f} | {auto_recall:<8.2f} | {c_relevance:<7.2f} | {c_recall:<7.2f} | {faithfulness:<8.2f} | {llm_precision:<7.2f}")
    
    results_summary.append([f_score, sem_s, auto_precision, auto_recall, c_relevance, c_recall, faithfulness, llm_precision])

# ==========================================
# 5. TỔNG KẾT TRUNG BÌNH
# ==========================================
if results_summary:
    avgs = np.mean(results_summary, axis=0)
    print("-" * 85)
    print(f"{'AVG':<3} | {avgs[0]:<5.2f} | {avgs[1]:<7.2f} | {avgs[2]:<8.2f} | {avgs[3]:<8.2f} | {avgs[4]:<7.2f} | {avgs[5]:<7.2f} | {avgs[6]:<8.2f} | {avgs[7]:<7.2f}")
