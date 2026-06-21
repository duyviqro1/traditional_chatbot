\c rag_lakehouse

-- ==========================================================
-- NHÓM 1: QUẢN LÝ QUY TRÌNH NẠP DỮ LIỆU (METADATA TRACKING)
-- ==========================================================
CREATE TABLE IF NOT EXISTS processed_files (
    id SERIAL PRIMARY KEY,
    file_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    etag TEXT UNIQUE,          -- Dùng để nhận diện file mới/file bị sửa đổi
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'SUCCESS'
);

-- ==========================================================
-- NHÓM 2: TỐI ƯU TỐC ĐỘ TRẢ LỜI (QUERY CACHING)
-- Giúp trả lời ngay lập tức các câu hỏi đã từng hỏi mà không cần chạy LLM
-- ==========================================================
CREATE TABLE IF NOT EXISTS query_cache (
    id SERIAL PRIMARY KEY,
    query_hash TEXT UNIQUE,    -- Mã băm của câu hỏi để tìm kiếm cực nhanh
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_hit_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP -- Theo dõi lần cuối dùng đến câu trả lời này
);

-- ==========================================================
-- NHÓM 3: LỊCH SỬ CUỘC HỘI THOẠI (CHAT HISTORY)
-- Giúp AI nhớ được ngữ cảnh (Context) của các câu hỏi trước đó
-- ==========================================================
CREATE TABLE IF NOT EXISTS chat_history (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,  -- Định danh phiên chat (ví dụ ID của người dùng)
    user_query TEXT NOT NULL,
    ai_response TEXT NOT NULL,
    source_nodes JSONB,        -- Lưu các đoạn văn bản (chunks) mà AI đã dùng để trả lời (Citations)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ==========================================================
-- NHÓM 4: ĐÁNH GIÁ VÀ CẢI THIỆN (FEEDBACK & EVALUATION)
-- Rất quan trọng cho báo cáo KLTN để chứng minh hệ thống tốt lên theo thời gian
-- ==========================================================
CREATE TABLE IF NOT EXISTS user_feedback (
    id SERIAL PRIMARY KEY,
    message_id INTEGER REFERENCES chat_history(id),
    rating INTEGER CHECK (rating IN (-1, 1)), -- 1: Like, -1: Dislike
    feedback_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Phân quyền cho user admin
ALTER TABLE processed_files OWNER TO admin;
ALTER TABLE query_cache OWNER TO admin;
ALTER TABLE chat_history OWNER TO admin;
ALTER TABLE user_feedback OWNER TO admin;