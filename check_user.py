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
    user_pass, rest = url_str.split("@")
    if ":" in user_pass:
        user, password = user_pass.split(":")
    else:
        user = user_pass
        password = ""
    host_port, dbname = rest.split("/")
    if ":" in host_port:
        host, port = host_port.split(":")
        port = int(port)
    else:
        host = host_port
        port = 3306

    connection = pymysql.connect(host=host, user=user, password=password, port=port, database=dbname)

    try:
        with connection.cursor() as cursor:
            print("Checking last 5 users...")
            cursor.execute("SELECT id, username, email, first_name, last_name, full_name FROM users ORDER BY id DESC LIMIT 5")
            users = cursor.fetchall()
            for u in users:
                print(f"ID: {u[0]} | User: {u[1]} | First: {u[3]} | Last: {u[4]} | Full: {u[5]}")

    finally:
        connection.close()

except Exception as e:
    print(f"❌ Error: {e}")
