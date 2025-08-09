# deployment/storage.py
import sqlite3
import os

class Storage:
    def __init__(self, db_path: str = 'deployment/predictions.db'):
        """
        Initializes SQLite connection and ensures the log table and file exist.
        """
        db_abs_path = os.path.abspath(db_path)
        db_dir = os.path.dirname(db_abs_path)

        # Ensure the target directory exists
        os.makedirs(db_dir, exist_ok=True)

        # Check for write permissions BEFORE connecting
        if not os.access(db_dir, os.W_OK):
            raise PermissionError(
                f"No write permissions for the directory: '{db_dir}'. "
                f"Please check the folder's permissions."
            )

        print("Permissions check passed.")
        self.conn = sqlite3.connect(db_abs_path, check_same_thread=False)
        self._create_table()

    def _create_table(self):
        """
        Creates the logs table
        """
        self.conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS logs(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                text TEXT,
                label TEXT,
                confidence REAL,
                features TEXT
            );
            ''')
        self.conn.commit()

    def log(self, timestamp: str, text: str, label: str, confidence: float, features: str):
        """
        Inserts a new prediction record into the logs table.
        """
        self.conn.execute(
            '''
            INSERT INTO logs(timestamp, text, label, confidence, features)
            VALUES (?,?,?,?,?);
            ''',
            (timestamp, text, label, confidence, features)
        )
        self.conn.commit()