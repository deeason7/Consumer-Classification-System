# deployment/storage.py
import sqlite3
import os
import datetime
from typing import List, Dict, Optional

class Storage:
    def __init__(self, db_path: str = 'deployment/predictions.db'):
        """
        Manages the SQLite database for storing and tracking consumer complaints
        """
        db_abs_path = os.path.abspath(db_path)
        db_dir = os.path.dirname(db_abs_path)

        # Ensure the target directory exists
        os.makedirs(db_dir, exist_ok=True)

        self.conn = sqlite3.connect(db_abs_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_table()

    def _create_table(self):
        """
        Creates the 'complaints' table.
        """
        self.conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS complaints(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                submission_timestamp TEXT NOT NULL,
                last_updated_timestamp TEXT,
                complaint_text TEXT NOT NULL,
                product TEXT,
                company TEXT,
                predicted_sentiment TEXT,
                confidence_score REAL,
                priority_score REAL,
                status TEXT DEFAULT 'Submitted',
                agent_notes TEXT
            );
            ''')
        self.conn.commit()

    def submit_complaint(self, text: str, product: str, company: str, label: str, confidence: float) -> int:
        """Logs a new complaint to the database and returns its unique ID.
        Calculates a priority score based on a sentiment.
        """
        timestamp = datetime.datetime.now().isoformat()

        # Calculate priority score (higher is more urgent)
        if label == "extreme_negative":
            priority_score = 2.0 + confidence

        elif label == "negative":
            priority_score = 1.0 + confidence

        else:
            priority_score = 0.0 + confidence

        cursor = self.conn.cursor()
        cursor.execute(
            '''
            INSERT INTO complaints (submission_timestamp, complaint_text, product, company, 
                                    predicted_sentiment, confidence_score, priority_score, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            ''',
            (timestamp, text, product, company, label, confidence, priority_score, 'Submitted')
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_complaint_by_id(self, complaint_id: int) -> Optional[Dict]:
        """Fetches a single complaint by its ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM complaints WHERE id = ?", (complaint_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_all_complaints_by_priority(self) -> List[Dict]:
        """Fetches all complaints, ordered by the highest priority first."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM complaints ORDER BY priority_score DESC, submission_timestamp ASC"
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def update_complaint_status(self, complaint_id: int, new_status: str, agent_notes: str):
        """Updates the status and notes for a specific complaint."""
        timestamp = datetime.datetime.now().isoformat()
        self.conn.execute(
            '''
            UPDATE complaints
            SET status = ?, agent_notes = ?, last_updated_timestamp = ?
            WHERE id = ?
            ''',
            (new_status, agent_notes, timestamp, complaint_id)
        )
        self.conn.commit()