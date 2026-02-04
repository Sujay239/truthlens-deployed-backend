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
            print("Updating 'analysis_logs' table schema...")
            
            # 1. Add analysis_summary
            try:
                cursor.execute("ALTER TABLE analysis_logs ADD COLUMN analysis_summary JSON NULL;")
                print("✅ Added 'analysis_summary' column.")
            except pymysql.err.OperationalError as e:
                # 1060 = Duplicate column name
                if e.args[0] == 1060: print("ℹ️ 'analysis_summary' column already exists.")
                else: print(f"❌ Error adding 'analysis_summary': {e}")

            # 2. Add file_size
            try:
                cursor.execute("ALTER TABLE analysis_logs ADD COLUMN file_size VARCHAR(50) NULL;")
                print("✅ Added 'file_size' column.")
            except pymysql.err.OperationalError as e:
                if e.args[0] == 1060: print("ℹ️ 'file_size' column already exists.")
                else: print(f"❌ Error adding 'file_size': {e}")

            # 3. Add media_url
            try:
                cursor.execute("ALTER TABLE analysis_logs ADD COLUMN media_url VARCHAR(500) NULL;")
                print("✅ Added 'media_url' column.")
            except pymysql.err.OperationalError as e:
                if e.args[0] == 1060: print("ℹ️ 'media_url' column already exists.")
                else: print(f"❌ Error adding 'media_url': {e}")
            
        connection.commit()
        print("🎉 Database schema updated successfully!")

    finally:
        connection.close()

except Exception as e:
    print(f"❌ Critical error: {e}")
