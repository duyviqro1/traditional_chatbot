import html
import json
import psycopg2
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from chatbot import chat_with_medical_bot, qdrant
from database import (
    init_db,
    verify_user,
    create_user,
    create_session,
    list_sessions,
    delete_session,
    load_messages_from_history,
    load_history_from_db,
    save_chat_turn_to_db,
    update_session_title,
    get_session_title,
)

st.set_page_config(page_title="Trợ lý Lakehouse", page_icon="💬", layout="wide")

NEW_CHAT_TITLE = "Cuộc trò chuyện mới"


@st.cache_resource
def ensure_database_initialized():
    init_db()
    return True


def ensure_state():
    defaults = {
        "logged_in": False,
        "user_id": None,
        "username": None,
        "current_session_id": "NEW",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def format_source_name(raw_source: str) -> str:
    if not raw_source or raw_source == "Không rõ nguồn":
        return "Không rõ nguồn"
    
    # 1. Tách chuỗi theo dấu "/" và lấy phần cuối cùng (tên file)
    filename = raw_source.split("/")[-1]  # VD: "cay_thuoc_thong_dung.md"
    
    # 2. Xóa đuôi file (.md, .pdf...) bằng cách tách theo dấu "." từ bên phải
    name_without_ext = filename.rsplit(".", 1)[0]  # VD: "cay_thuoc_thong_dung"
    
    # 3. Thay dấu gạch dưới "_" thành khoảng trắng và viết hoa chữ cái đầu
    clean_name = name_without_ext.replace("_", " ").capitalize()  # VD: "Cay thuoc thong dung"
    
    # (Tùy chọn) Nếu bạn muốn có DẤU tiếng Việt hoàn chỉnh, bạn có thể tạo một từ điển ở đây:
    dict_ten_sach = {
        "cay thuoc thong dung": "Cây thuốc thông dụng",
        "duoclieutrungcap": "Dược liệu trung cấp",
        "so tay thuoc nam": "Sổ tay thuốc Nam"
    }
    
    # Nếu tên đã xử lý có trong từ điển thì lấy tên có dấu, không thì giữ nguyên
    return dict_ten_sach.get(clean_name.lower(), clean_name)


def short_title(text: str, max_len: int = 48) -> str:
    cleaned = " ".join(text.strip().split())
    if len(cleaned) <= max_len:
        return cleaned
    words = cleaned.split(" ")
    title = ""
    for word in words:
        candidate = f"{title} {word}".strip()
        if len(candidate) > max_len:
            break
        title = candidate
    return title or cleaned[:max_len].rstrip()

def apply_styles():
    css = """
    :root {
        --app-bg: #f6f7fb;
        --app-text: #101418;
        --border: #d0d7de;
    }
    """
    st.markdown(
        f"""
        <style>
        {css}
        .stApp {{
            background: var(--app-bg);
            color: var(--app-text);
        }}
        section[data-testid="stSidebar"] {{
            background: var(--app-bg);
            border-right: 1px solid var(--border);
        }}
        div[data-testid="column"] {{
            align-self: center;
        }}
        
        /* Căn giữa mũi tên của nút popover */
        div[data-testid="stPopover"] button {{
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            padding-left: 0 !important;
            padding-right: 6px !important;
        }}
        div[data-testid="stPopover"] button svg {{
            margin: 0 !important;
        }}

        /* =========================================
           XÓA HOÀN TOÀN LOGO CỦA BOT (ASSISTANT)
           ========================================= */
        /* Ép ẩn phần tử chứa Avatar (phần tử đầu tiên trong khối chat) */
        div[data-testid="stChatMessage"] > div:first-child {{
            display: none !important;
            width: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
        }}
        
        /* Căn lề kéo nội dung sát sang trái, bỏ gap */
        div[data-testid="stChatMessage"] {{
            gap: 0 !important;
            padding: 0 !important;
            background-color: transparent !important;
        }}
        
        /* Đóng khung nội dung của Bot thành bong bóng chat màu trắng */
        div[data-testid="stChatMessageContent"] {{
            background-color: #ffffff;
            color: #101418;
            padding: 12px 16px !important;
            border-radius: 4px 16px 16px 16px; /* Bo góc bong bóng bên trái */
            border: 1px solid var(--border);
            margin: 0 !important;
            margin-bottom: 1rem !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def generate_title_with_llm(user_text: str) -> str:
    title_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Tạo tiêu đề ngắn gọn bằng tiếng Việt có dấu, tối đa 6 từ, không dấu câu, không giải thích.",
        ),
        ("human", "{input}"),
    ])
    chain = title_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0) | StrOutputParser()
    try:
        title = chain.invoke({"input": user_text}).strip()
        return title or short_title(user_text)
    except Exception:
        return short_title(user_text)


def login_screen():
    st.title("Đăng nhập")

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Tên đăng nhập")
        password = st.text_input("Mật khẩu", type="password")
        submitted = st.form_submit_button("Đăng nhập")
        if submitted:
            user_id = verify_user(username, password)
            if user_id:
                st.session_state.logged_in = True
                st.session_state.user_id = user_id
                st.session_state.username = username
                st.session_state.current_session_id = "NEW"
                st.success("Đăng nhập thành công.")
                st.rerun()
            else:
                st.error("Sai thông tin đăng nhập.")

    with st.expander("Tạo tài khoản mới"):
        with st.form("register_form", clear_on_submit=True):
            new_username = st.text_input("Tên đăng nhập mới")
            new_password = st.text_input("Mật khẩu mới", type="password")
            register = st.form_submit_button("Đăng ký")
            if register:
                try:
                    create_user(new_username, new_password)
                    st.success("Tạo tài khoản thành công. Hãy đăng nhập.")
                except psycopg2.IntegrityError:
                    st.error("Tên đăng nhập đã tồn tại.")
                except Exception:
                    st.error("Không thể tạo tài khoản. Vui lòng thử lại.")


def create_new_chat():
    session_id = create_session(
        st.session_state.user_id,
        st.session_state.username,
        NEW_CHAT_TITLE,
    )
    st.session_state.current_session_id = session_id


def sidebar_sessions():
    with st.sidebar:
        st.header("Lịch sử hội thoại")
        
        # CHỈ GÁN TRẠNG THÁI "NEW", KHÔNG TẠO VÀO DB
        if st.button("Hội thoại mới", use_container_width=True):
            st.session_state.current_session_id = "NEW"
            st.rerun()

        st.write("---")

        sessions = list_sessions(st.session_state.user_id)
        
        for session in sessions:
            col_title, col_menu = st.columns([0.85, 0.15])
            
            with col_title:
                is_current = session["id"] == st.session_state.current_session_id
                label = f"💬 {session['title']}" if is_current else session["title"]
                
                if st.button(label, key=f"btn_{session['id']}", use_container_width=True):
                    st.session_state.current_session_id = session["id"]
                    st.rerun()
            
            with col_menu:
                with st.popover("", use_container_width=True):
                    new_title = st.text_input(
                        "Đổi tên cuộc hội thoại",
                        value=session["title"],
                        key=f"rename_{session['id']}",
                    )
                    if st.button(
                        "Lưu tên mới",
                        key=f"save_{session['id']}",
                        use_container_width=True,
                    ):
                        trimmed = new_title.strip()
                        if trimmed:
                            update_session_title(session["id"], trimmed)
                            st.success("Đã cập nhật tên cuộc hội thoại.")
                            st.rerun()
                        else:
                            st.warning("Tên cuộc hội thoại không được để trống.")

                    confirm_delete = st.checkbox("Xác nhận xóa", key=f"conf_{session['id']}")
                    if st.button("Xóa", key=f"del_{session['id']}", disabled=not confirm_delete, type="primary", use_container_width=True):
                        delete_session(st.session_state.user_id, session["id"])

                        if st.session_state.current_session_id == session["id"]:
                            st.session_state.current_session_id = "NEW"

                        st.success("Đã xóa!")
                        st.rerun()

        st.write("---")
        if st.button("Đăng xuất", use_container_width=True):
            st.session_state.clear()  
            st.rerun()

def render_messages(session_id: str):
    messages = load_messages_from_history(str(session_id))
    for message in messages:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            escaped_content = html.escape(content).replace("\n", "<br>")
            st.markdown(
                f"""
                <div style="display: flex; justify-content: flex-end; margin-bottom: 1rem;">
                    <div style="background-color: #2f6fed; color: white; padding: 12px 16px; border-radius: 16px 16px 4px 16px; max-width: 75%; font-size: 16px; line-height: 1.5; box-shadow: 0 1px 2px rgba(0,0,0,0.1);">
                        {escaped_content}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            with st.chat_message("assistant"):
                st.markdown(content)
                
                
def main_screen():
    st.title("NamY")
    st.caption("Trợ lý ảo hỗ trợ tư vấn thuốc Nam.")
    sidebar_sessions()

    if st.session_state.current_session_id is None:
        st.session_state.current_session_id = "NEW"

    current_session_id = st.session_state.current_session_id
    
    # NẾU LÀ PHÒNG MỚI CHƯA CÓ TRONG DB THÌ KHÔNG LOAD MESSAGES
    if current_session_id != "NEW":
        render_messages(current_session_id)
    else:
        st.markdown("<h3 style='text-align: center; color: #555; margin-top: 2rem;'>Hôm nay tôi có thể giúp gì cho bạn?</h3>", unsafe_allow_html=True)

    prompt = st.chat_input("Nhập câu hỏi của bạn...")
    if prompt:
        if current_session_id == "NEW":
            new_id = create_session(
                st.session_state.user_id,
                st.session_state.username,
                NEW_CHAT_TITLE,
            )
            st.session_state.current_session_id = new_id
            current_session_id = new_id
            
        history = load_history_from_db(str(current_session_id), limit=5)
        
        escaped_content = html.escape(prompt).replace("\n", "<br>")
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-end; margin-bottom: 1rem;">
                <div style="background-color: #2f6fed; color: white; padding: 12px 16px; border-radius: 16px 16px 4px 16px; max-width: 75%; font-size: 16px; line-height: 1.5; box-shadow: 0 1px 2px rgba(0,0,0,0.1);">
                    {escaped_content}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # -------------------------------------------------------
        
        with st.spinner("Đang trả lời..."):
            answer, context_docs = chat_with_medical_bot(prompt, history, qdrant)

        answer = answer or "Xin lỗi, hệ thống không trả lời được."

        sources_list = []
        for doc in context_docs or []:
            sources_list.append({
                "content_snippet": doc.page_content[:200] + "...",
                "metadata": doc.metadata,
            })

        save_chat_turn_to_db(str(current_session_id), prompt, answer, sources_list)

        current_title = get_session_title(current_session_id)
        if current_title == NEW_CHAT_TITLE:
            update_session_title(current_session_id, generate_title_with_llm(prompt))

        st.rerun()

def main():
    ensure_database_initialized()
    ensure_state()
    apply_styles()

    if not st.session_state.logged_in:
        login_screen()
        return

    main_screen()


if __name__ == "__main__":
    main()
