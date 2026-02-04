import pymysql
import os
from dotenv import load_dotenv

load_dotenv()

db_url = os.getenv("DATABASE_URL", "mysql+pymysql://root:@localhost:3306/truthlens_db")

try:
    # Parsing URL logic
    url_str = db_url.replace("mysql+pymysql://", "")
    user_pass, rest = url_str.split("@")
    if ":" in user_pass: user, password = user_pass.split(":")
    else: user, password = user_pass, ""
    host_port, dbname = rest.split("/")
    if ":" in host_port: host, port = host_port.split(":")
    else: host, port = host_port, "3306"
    port = int(port)

    print(f"Connecting to database: {dbname}...")
    connection = pymysql.connect(host=host, user=user, password=password, port=port, database=dbname)

    try:
        with connection.cursor() as cursor:
            tables = [
                """
                CREATE TABLE IF NOT EXISTS fake_news_scans (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT,
                    content_text TEXT,
                    label VARCHAR(50),
                    confidence_score FLOAT,
                    emotional_tone VARCHAR(255),
                    source_credibility VARCHAR(255),
                    semantic_consistency VARCHAR(255),
                    analysis_text TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                );
                """,
                """
                CREATE TABLE IF NOT EXISTS image_scans (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT,
                    image_url VARCHAR(500),
                    label VARCHAR(50),
                    confidence_score FLOAT,
                    visual_artifacts VARCHAR(255),
                    pixel_consistency VARCHAR(255),
                    metadata_analysis VARCHAR(255),
                    analysis_text TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                );
                """,
                 """
                CREATE TABLE IF NOT EXISTS video_scans (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT,
                    video_url VARCHAR(500),
                    label VARCHAR(50),
                    confidence_score FLOAT,
                    frame_consistency VARCHAR(255),
                    audio_visual_sync VARCHAR(255),
                    blinking_patterns VARCHAR(255),
                    analysis_text TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                );
                """,
                 """
                CREATE TABLE IF NOT EXISTS audio_scans (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT,
                    audio_url VARCHAR(500),
                    label VARCHAR(50),
                    confidence_score FLOAT,
                    spectral_analysis VARCHAR(255),
                    voice_cloning_signature VARCHAR(255),
                    background_noise VARCHAR(255),
                    analysis_text TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                );
                """,
                 """
                CREATE TABLE IF NOT EXISTS ai_text_scans (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT,
                    content_text TEXT,
                    label VARCHAR(50),
                    confidence_score FLOAT,
                    perplexity VARCHAR(255),
                    burstiness VARCHAR(255),
                    repetitive_patterns VARCHAR(255),
                    analysis_text TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                );
                """,
                 """
                CREATE TABLE IF NOT EXISTS malware_scans (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT,
                    target VARCHAR(500),
                    scan_type VARCHAR(50),
                    label VARCHAR(50),
                    threat_score INT,
                    threat_level VARCHAR(50),
                    signature_match VARCHAR(255),
                    heuristic_score VARCHAR(50),
                    analysis_text TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                );
                """
            ]

            print(f"Syncing {len(tables)} new tables...")
            for sql in tables:
                cursor.execute(sql)
            
            print("✅ tables synced.")

        connection.commit()
        print("\n🎉 Full database schema synchronized successfully!")

    finally:
        connection.close()

except Exception as e:
    print(f"❌ Critical error: {e}")
