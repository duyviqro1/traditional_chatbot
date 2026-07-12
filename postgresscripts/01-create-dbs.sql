-- 1. Database cho Airflow
CREATE USER airflow WITH PASSWORD 'airflow';
CREATE DATABASE airflow;
GRANT ALL PRIVILEGES ON DATABASE airflow TO airflow;
-- Lệnh quan trọng để sửa lỗi của bạn:
\c airflow
GRANT ALL ON SCHEMA public TO airflow;
ALTER SCHEMA public OWNER TO airflow;

-- 2. Database cho Hive Metastore
CREATE USER hive WITH PASSWORD 'hive';
CREATE DATABASE metastore;
GRANT ALL PRIVILEGES ON DATABASE metastore TO hive;
\c metastore
GRANT ALL ON SCHEMA public TO hive;
ALTER SCHEMA public OWNER TO hive;

-- 3. Database cho Chatbot AI
CREATE USER admin WITH PASSWORD 'admin';
CREATE DATABASE rag_lakehouse;
GRANT ALL PRIVILEGES ON DATABASE rag_lakehouse TO admin;
\c rag_lakehouse
GRANT ALL ON SCHEMA public TO admin;
ALTER SCHEMA public OWNER TO admin;