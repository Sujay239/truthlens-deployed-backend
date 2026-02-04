from app.database import engine
from app import models
from sqlalchemy import text

def update_schema():
    with engine.connect() as conn:
        # Check if reset_token column exists
        try:
            conn.execute(text("SELECT reset_token FROM users LIMIT 1"))
            print("Column 'reset_token' already exists.")
        except:
            print("Adding 'reset_token' column...")
            conn.execute(text("ALTER TABLE users ADD COLUMN reset_token VARCHAR(500) NULL"))
            print("Added 'reset_token'.")

        # Check if reset_token_expiry column exists
        try:
            conn.execute(text("SELECT reset_token_expiry FROM users LIMIT 1"))
            print("Column 'reset_token_expiry' already exists.")
        except:
            print("Adding 'reset_token_expiry' column...")
            conn.execute(text("ALTER TABLE users ADD COLUMN reset_token_expiry DATETIME NULL"))
            print("Added 'reset_token_expiry'.")
        
        conn.commit()

if __name__ == "__main__":
    print("Updating database schema...")
    update_schema()
    print("Schema update complete.")
