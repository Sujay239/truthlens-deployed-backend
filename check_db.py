from app.database import engine
from sqlalchemy import inspect
from sqlalchemy.orm import sessionmaker

def check_connection():
    print("Testing database connection...")
    try:
        # Try to connect
        connection = engine.connect()
        print("✅ Successfully connected to MySQL!")
        
        # Check for tables
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        print(f"📁 Found tables: {tables}")
        
        if "users" in tables:
            print("✅ 'users' table exists.")
        else:
            print("⚠️ 'users' table NOT found (Run main.py to create it).")
            
        connection.close()
    except Exception as e:
        print("❌ Connection failed!")
        print(f"Error: {e}")

if __name__ == "__main__":
    check_connection()
