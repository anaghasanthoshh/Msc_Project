import os
from dotenv import load_dotenv
import psycopg2
#need to install binary version pf psycopg2

def get_db_connection():
    # Load .env file
    load_dotenv()
    # Retrieve database credentials from environment
    DB_NAME = os.getenv("DATABASE_NAME")
    DB_USER = os.getenv("DATABASE_USER")
    DB_PASSWORD = os.getenv("DATABASE_PASSWORD")
    DB_HOST = os.getenv("DATABASE_HOST")
    DB_PORT = os.getenv("DATABASE_PORT")
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT  )
        return conn
    except Exception as e:
        print("failed to connect to db")

