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
            
            # 1. Add first_name
            try:
                cursor.execute("ALTER TABLE users ADD COLUMN first_name VARCHAR(100) NULL;")
                print("✅ Added 'first_name' column.")
            except pymysql.err.OperationalError as e:
                if e.args[0] == 1060: print("ℹ️ 'first_name' column already exists.")
                else: print(f"❌ Error adding 'first_name': {e}")

            # 2. Add last_name
            try:
                cursor.execute("ALTER TABLE users ADD COLUMN last_name VARCHAR(100) NULL;")
                print("✅ Added 'last_name' column.")
            except pymysql.err.OperationalError as e:
                if e.args[0] == 1060: print("ℹ️ 'last_name' column already exists.")
                else: print(f"❌ Error adding 'last_name': {e}")

            # 3. Migrate data from full_name to first_name and last_name
            print("Migrating existing data...")
            cursor.execute("SELECT id, full_name FROM users WHERE full_name IS NOT NULL")
            users = cursor.fetchall()
            
            for user in users:
                user_id = user[0]
                full_name = user[1]
                
                first_name = ""
                last_name = ""
                
                parts = full_name.split(" ", 1)
                first_name = parts[0]
                if len(parts) > 1:
                    last_name = parts[1]
                
                cursor.execute(
                    "UPDATE users SET first_name = %s, last_name = %s WHERE id = %s",
                    (first_name, last_name, user_id)
                )
            
            print(f"✅ Migrated data for {len(users)} users.")

        connection.commit()
        print("🎉 Database schema updated successfully!")

    finally:
        connection.close()

except Exception as e:
    print(f"❌ Critical error: {e}")
