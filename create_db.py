import pymysql
import os
from dotenv import load_dotenv
from urllib.parse import urlparse

load_dotenv()

# Parse the DATABASE_URL to get connection details
# Format: mysql+pymysql://user:pass@host:port/dbname
db_url = os.getenv("DATABASE_URL", "mysql+pymysql://root:@localhost:3306/truthlens_db")

# Simple parsing assuming standard format
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
        
    connection = pymysql.connect(
        host=host,
        user=user,
        password=password,
        port=port
    )
    
    try:
        with connection.cursor() as cursor:
            print(f"Attempting to create database: {dbname}")
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {dbname}")
            print("Database created successfully or already exists.")
    finally:
        connection.close()

except Exception as e:
    print(f"Error creating database: {e}")
