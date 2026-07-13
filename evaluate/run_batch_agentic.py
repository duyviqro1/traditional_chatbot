import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Import đồ thị app từ file agentic_rag của bạn
from agentic_rag import app 
# Giả định bạn vẫn muốn lưu lịch sử chat vào DB để theo dõi
from database import init_db, load_history_from_db, save_chat_turn_to_db
from langchain_core.messages import AIMessage, HumanMessage

SESSION_ID = "agentic_user_01" # Dùng cùng Session ID với agentic_rag.py để kết quả batch đi cùng ngữ cảnh chat trực tiếp

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
                raise argparse.ArgumentTypeError(f"Khoang id khong hop le: {part}")
            start_id = int(start_text)
            end_id = int(end_text)
            if start_id > end_id:
                raise argparse.ArgumentTypeError(f"Khoang id khong hop le: {part}")
            selected_ids.update(str(i) for i in range(start_id, end_id + 1))
        else:
            selected_ids.add(part)

    if not selected_ids:
        raise argparse.ArgumentTypeError("Danh sach id khong duoc rong.")
    return selected_ids

def load_questions(file_path: Path) -> list:
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_existing_results(file_path: Path) -> list:
    if not file_path.exists():
        return []
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)

def serialize_context(docs: list) -> list:
    context_items = []
    if not docs:
        return context_items
    for doc in docs:
        context_items.append({
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        })
    return context_items

def filter_questions(questions: list, selected_ids: set[str] | None) -> list:
    if selected_ids is None:
        return questions
    return [item for item in questions if str(item.get("id")) in selected_ids]

def merge_results(existing_results: list, new_results: list, questions: list) -> list:
    by_id = {str(item.get("id")): item for item in existing_results if item.get("id") is not None}
    for item in new_results:
        if item.get("id") is not None:
            by_id[str(item.get("id"))] = item

    ordered_results = []
    seen_ids = set()
    for question in questions:
        q_id = str(question.get("id"))
        if q_id in by_id:
            ordered_results.append(by_id[q_id])
            seen_ids.add(q_id)

    for item in existing_results + new_results:
        item_id = str(item.get("id"))
        if item.get("id") is not None and item_id not in seen_ids:
            ordered_results.append(by_id[item_id])
            seen_ids.add(item_id)

    return ordered_results

def run_batch(
    input_path: Path,
    output_path: Path,
    selected_ids: set[str] | None = None,
    no_history: bool = False,
) -> None:
    init_db()

    questions = load_questions(input_path)
    selected_questions = filter_questions(questions, selected_ids)
    results = []

    if not selected_questions:
        requested = ", ".join(sorted(selected_ids)) if selected_ids else "tat ca"
        raise ValueError(f"Khong tim thay cau hoi nao voi id: {requested}")

    if selected_ids is not None:
        found_ids = {str(item.get("id")) for item in selected_questions}
        missing_ids = sorted(selected_ids - found_ids, key=lambda x: int(x) if x.isdigit() else x)
        if missing_ids:
            print(f"Cảnh báo: Không tìm thấy câu hỏi cho id: {', '.join(missing_ids)}")

    if selected_ids is None:
        print(f"🚀 Bắt đầu chạy Batch cho Agentic RAG ({len(selected_questions)} câu hỏi)...")
    else:
        selected_labels = ", ".join(str(item.get("id")) for item in selected_questions)
        print(f"🚀 Bắt đầu chạy lại Agentic RAG ({len(selected_questions)}/{len(questions)} câu hỏi, id: {selected_labels})...")

    chat_history = []
    if not no_history:
        chat_history = load_history_from_db(SESSION_ID, limit=25)
        if chat_history:
            print(f"📦 Đã khôi phục {len(chat_history)//2} phiên hỏi-đáp từ Database!")

    for idx, item in enumerate(selected_questions):
        question_text = item.get("cau_hoi", "")
        print(f"\n[{idx+1}/{len(selected_questions)}] Đang xử lý ID {item.get('id')}: {question_text}")

        # 1. Gọi Đồ thị Agentic RAG
        inputs = {"question": question_text, "chat_history": chat_history}
        result = app.invoke(inputs)

        # 2. Trích xuất câu trả lời và tài liệu từ GraphState
        answer = result.get("generation", "")
        # Nếu Router rẽ nhánh sang direct_chat (chào hỏi), documents sẽ không tồn tại, ta để mảng rỗng
        context_docs = result.get("documents", []) 

        # 3. Đóng gói kết quả ra file JSON
        results.append({
            "id": item.get("id"),
            "cau_hoi": question_text,
            "tra_loi": answer,
            "context": serialize_context(context_docs),
        })

        # ==========================================
        # 4. LƯU XUỐNG DATABASE (Tùy chọn)
        # ==========================================
        sources_list = []
        if context_docs:
            for doc in context_docs:
                sources_list.append({
                    "content_snippet": doc.page_content[:200] + "...", 
                    "metadata": doc.metadata
                })
        
        # Lưu xuống Postgres với Session ID riêng của Agentic
        save_chat_turn_to_db(SESSION_ID, question_text, answer, sources_list)
        # ==========================================

        if not no_history:
            chat_history.append(HumanMessage(content=question_text))
            chat_history.append(AIMessage(content=answer))
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]

    if selected_ids is not None:
        existing_results = load_existing_results(output_path)
        results = merge_results(existing_results, results, questions)

    # Xuất file kết quả riêng cho Agentic
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Đã hoàn thành! Kết quả lưu tại: {output_path.name}")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Chay batch Agentic RAG, co the chi dinh id cau hoi.")
    parser.add_argument("--ids", type=parse_id_selection, help="Danh sach id can chay, vi du: 1 hoac 1,3,5-7")
    parser.add_argument("--input", type=Path, default=base_dir / "questions.json", help="File cau hoi dau vao.")
    parser.add_argument("--output", type=Path, default=base_dir / "answers_agentic.json", help="File ket qua dau ra.")
    parser.add_argument("--no-history", action="store_true", help="Khong dung lich su chat DB, moi cau hoi chay doc lap.")
    args = parser.parse_args()

    run_batch(args.input, args.output, selected_ids=args.ids, no_history=args.no_history)
