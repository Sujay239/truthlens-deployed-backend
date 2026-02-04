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

db_url = os.getenv("DATABASE_URL", "mysql+pymysql://root:@localhost:3306/truthlens_db")

try:
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
            # We know the specific username from the user's request
            target_username = "sujay kumarkotal292"
            
            print(f"Checking user: {target_username}")
            cursor.execute("SELECT id, first_name, last_name FROM users WHERE username = %s", (target_username,))
            result = cursor.fetchone()
            
            if result:
                user_id = result[0]
                first_name = result[1]
                last_name = result[2]
                print(f"Found User ID: {user_id}. Current First: '{first_name}', Last: '{last_name}'")
                
                if not first_name or not last_name:
                    print("⚠️ Names are missing! Manually fixing them now...")
                    
                    # Heuristic fix based on username/email/known data
                    # Username "sujay kumarkotal292" -> First: Sujay, Last: Kumarkotal
                    new_first = "Sujay"
                    new_last = "Kumarkotal"
                    
                    cursor.execute(
                        "UPDATE users SET first_name = %s, last_name = %s, full_name = %s WHERE id = %s",
                        (new_first, new_last, f"{new_first} {new_last}", user_id)
                    )
                    connection.commit()
                    print(f"✅ User fixed! Updated to First: {new_first}, Last: {new_last}")
                else:
                    print("✅ User already has names.")
            else:
                print("❌ User not found in DB!")

    finally:
        connection.close()

except Exception as e:
    print(f"❌ Error: {e}")
