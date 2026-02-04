try:
    import pymysql
except ImportError:
    import pip
    pip.main(['install', 'pymysql'])
    import pymysql
import os
try:
    from dotenv import load_dotenv
except ImportError:
    import pip
    pip.main(['install', 'python-dotenv'])
    from dotenv import load_dotenv

load_dotenv()

# Parse the DATABASE_URL
db_url = os.getenv("DATABASE_URL", "mysql+pymysql://root:@localhost:3306/truthlens_db")

try:
    # Remove prefix
    url_str = db_url.replace("mysql+pymysql://", "")
    
    # Split user:pass and rest
    user_pass, rest = url_str.split("@")
    if ":" in user_pass:
        user, password = user_pass.split(":")
    else:
        user = user_pass
        password = ""
        
    # Split host:port and dbname
    host_port, dbname = rest.split("/")
    if ":" in host_port:
        host, port = host_port.split(":")
        port = int(port)
    else:
        host = host_port
        port = 3306

    print(f"Connecting to database: {dbname}...")
    connection = pymysql.connect(
        host=host,
        user=user,
        password=password,
        port=port,
        database=dbname
    )

    try:
        with connection.cursor() as cursor:
            print("Updating 'users' table schema...")
            
            # Add full_name column
            try:
                cursor.execute("ALTER TABLE users ADD COLUMN full_name VARCHAR(255) NULL;")
                print("✅ Added 'full_name' column.")
            except pymysql.err.OperationalError as e:
                if e.args[0] == 1060: print("ℹ️ 'full_name' column already exists.")
                else: print(f"❌ Error adding 'full_name': {e}")

            # Add phone_number column
            try:
                cursor.execute("ALTER TABLE users ADD COLUMN phone_number VARCHAR(50) NULL;")
                print("✅ Added 'phone_number' column.")
            except pymysql.err.OperationalError as e:
                if e.args[0] == 1060: print("ℹ️ 'phone_number' column already exists.")
                else: print(f"❌ Error adding 'phone_number': {e}")

            # Add avatar column
            try:
                cursor.execute("ALTER TABLE users ADD COLUMN avatar VARCHAR(500) NULL;")
                print("✅ Added 'avatar' column.")
            except pymysql.err.OperationalError as e:
                if e.args[0] == 1060: print("ℹ️ 'avatar' column already exists.")
                else: print(f"❌ Error adding 'avatar': {e}")

            # Add video_hash column to video_scans
            try:
                cursor.execute("ALTER TABLE video_scans ADD COLUMN video_hash VARCHAR(64) NULL;")
                cursor.execute("CREATE INDEX idx_video_hash ON video_scans(video_hash);")
                print("✅ Added 'video_hash' column and index.")
            except pymysql.err.OperationalError as e:
                if e.args[0] == 1060: print("ℹ️ 'video_hash' column already exists.")
                else: print(f"❌ Error adding 'video_hash': {e}")

        connection.commit()
        print("🎉 Database schema updated successfully!")

    finally:
        connection.close()

except Exception as e:
    print(f"❌ Critical error: {e}")
