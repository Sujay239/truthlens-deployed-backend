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
            
            # Add reset_token
            try:
                cursor.execute("ALTER TABLE users ADD COLUMN reset_token VARCHAR(500) NULL;")
                print("✅ Added 'reset_token' column.")
            except pymysql.err.OperationalError as e:
                if e.args[0] == 1060: print("ℹ️ 'reset_token' column already exists.")
                else: print(f"❌ Error adding 'reset_token': {e}")

            # Add reset_token_expiry
            try:
                cursor.execute("ALTER TABLE users ADD COLUMN reset_token_expiry DATETIME NULL;")
                print("✅ Added 'reset_token_expiry' column.")
            except pymysql.err.OperationalError as e:
                if e.args[0] == 1060: print("ℹ️ 'reset_token_expiry' column already exists.")
                else: print(f"❌ Error adding 'reset_token_expiry': {e}")

        connection.commit()
        print("🎉 Database schema updated successfully!")

    finally:
        connection.close()

except Exception as e:
    print(f"❌ Critical error: {e}")
