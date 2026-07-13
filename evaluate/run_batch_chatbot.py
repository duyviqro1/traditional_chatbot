import argparse
import json
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, AIMessage

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# 1. BẮT BUỘC IMPORT THÊM HÀM LƯU DATABASE TỪ DATABASE.PY
# (Thêm hàm init_db để đảm bảo database sẵn sàng trước khi chạy)
from database import load_history_from_db, save_chat_turn_to_db, init_db

# Thay vì import SESSION_ID bị lỗi, ta import đối tượng qdrant
from chatbot import chat_with_medical_bot, qdrant 

# Tự định nghĩa Session ID riêng cho luồng chạy test hàng loạt
SESSION_ID = "22d2690d-2d1b-4ef0-ba90-419764bac0f9"


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


def load_questions(file_path: Path) -> list:
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_answers(file_path: Path) -> dict:
    if not file_path.exists():
        return {}
    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {item.get("id"): item for item in data if "id" in item}
    except Exception:
        return {}


def serialize_context(docs: list) -> list:
    context_items = []
    for doc in docs:
        context_items.append({
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        })
    return context_items


def run_batch(input_path: Path, output_path: Path, target_ids: set | None = None) -> None:
    # Khởi tạo DB (phòng trường hợp DB chưa được tạo)
    init_db()
    
    questions = load_questions(input_path)
    existing_answers = load_existing_answers(output_path)

    chat_history = load_history_from_db(SESSION_ID, limit=25)

    for item in questions:
        q_id = item.get("id")
        if target_ids and str(q_id) not in target_ids:
            continue
        question_text = item.get("cau_hoi", "")
        
        # 3. ĐỒNG BỘ: Truyền thêm biến 'qdrant' vào hàm
        answer, context_docs = chat_with_medical_bot(question_text, chat_history, qdrant)

        existing_answers[q_id] = {
            "id": item.get("id"),
            "cau_hoi": question_text,
            "tra_loi": answer,
            "context": serialize_context(context_docs),
        }

        # ==========================================
        # BỔ SUNG LOGIC LƯU DATABASE
        # ==========================================
        sources_list = []
        for doc in context_docs:
            sources_list.append({
                "content_snippet": doc.page_content[:200] + "...", 
                "metadata": doc.metadata
            })
            
        # Lưu xuống Postgres
        save_chat_turn_to_db(SESSION_ID, question_text, answer, sources_list)
        # ==========================================

        chat_history.append(HumanMessage(content=question_text))
        chat_history.append(AIMessage(content=answer))
        
        # Sửa lại thành 20 tin nhắn để đồng bộ với bộ nhớ của chatbot gốc
        if len(chat_history) > 20: 
            chat_history = chat_history[-20:]

    merged_results = []
    for item in questions:
        q_id = item.get("id")
        if q_id in existing_answers:
            merged_results.append(existing_answers[q_id])

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(merged_results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    cli_parser = argparse.ArgumentParser(
        description="Chạy batch chatbot thường, có thể chỉ định id câu hỏi."
    )
    cli_parser.add_argument(
        "legacy_ids",
        nargs="*",
        help="Cách cũ: truyền trực tiếp id, ví dụ: 1 hoặc 1 3 5",
    )
    cli_parser.add_argument(
        "--ids",
        type=parse_id_selection,
        help="Danh sách id cần chạy, ví dụ: 1 hoặc 1,3,5-7",
    )
    cli_parser.add_argument(
        "--input",
        default=str(base_dir / "questions.json"),
        help="File câu hỏi đầu vào.",
    )
    cli_parser.add_argument(
        "--output",
        default=str(base_dir / "answers.json"),
        help="File lưu câu trả lời.",
    )
    args = cli_parser.parse_args()

    input_file = Path(args.input)
    output_file = Path(args.output)
    target_ids = args.ids
    if target_ids is None and args.legacy_ids:
        target_ids = parse_id_selection(",".join(args.legacy_ids))

    if target_ids:
        print(f"🚀 Bắt đầu chạy lại các ID: {sorted(target_ids)}")
    else:
        print(f"🚀 Bắt đầu chạy test hàng loạt từ: {input_file}")
    run_batch(input_file, output_file, target_ids=target_ids)
    print(f"✅ Đã hoàn tất! Kết quả được lưu tại: {output_file}")
