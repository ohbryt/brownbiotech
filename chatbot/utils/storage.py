"""
Chat storage using SQLite for persistence
"""
import sqlite3
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List


class ChatStorage:
    """
    Simple SQLite-based storage for chat history.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_dir = Path.home() / ".brown-biotech-chatbot"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = db_dir / "chat_history.db"
        
        self.db_path = str(db_path)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                intent TEXT,
                sources TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                cost REAL DEFAULT 0.0,
                model TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                ended_at DATETIME,
                message_count INTEGER DEFAULT 0
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_message(self, query: str, response: str, intent: str, 
                    sources: str = "", cost: float = 0.0, model: str = ""):
        """Save a query-response pair."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO messages (query, response, intent, sources, cost, model)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (query, response, intent, sources, cost, model))
        
        conn.commit()
        conn.close()
    
    def get_history(self, limit: int = 10, offset: int = 0) -> List[dict]:
        """Get recent chat history."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, query, response, intent, sources, timestamp, cost
            FROM messages
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        """, (limit, offset))
        
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            results.append({
                "id": row[0],
                "query": row[1],
                "response": row[2][:200] + "..." if len(row[2]) > 200 else row[2],
                "intent": row[3],
                "sources": row[4],
                "timestamp": row[5],
                "cost": row[6]
            })
        
        return results
    
    def get_stats(self) -> dict:
        """Get usage statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total messages
        cursor.execute("SELECT COUNT(*) FROM messages")
        total = cursor.fetchone()[0]
        
        # Today's messages
        cursor.execute("""
            SELECT COUNT(*) FROM messages 
            WHERE DATE(timestamp) = DATE('now')
        """)
        today = cursor.fetchone()[0]
        
        # This week
        cursor.execute("""
            SELECT COUNT(*) FROM messages 
            WHERE timestamp >= datetime('now', '-7 days')
        """)
        week = cursor.fetchone()[0]
        
        # Average cost
        cursor.execute("SELECT AVG(cost) FROM messages WHERE cost > 0")
        avg_cost = cursor.fetchone()[0] or 0.0
        
        # Top intents
        cursor.execute("""
            SELECT intent, COUNT(*) as count 
            FROM messages 
            GROUP BY intent 
            ORDER BY count DESC 
            LIMIT 5
        """)
        top_intents = cursor.fetchall()
        
        conn.close()
        
        return {
            "total": total,
            "today": today,
            "week": week,
            "avg_cost": round(avg_cost, 4),
            "top_intents": [{"intent": i[0], "count": i[1]} for i in top_intents]
        }
    
    def search_history(self, query: str, limit: int = 10) -> List[dict]:
        """Search history by query text."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, query, response, intent, sources, timestamp
            FROM messages
            WHERE query LIKE ? OR response LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (f"%{query}%", f"%{query}%", limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                "id": row[0],
                "query": row[1],
                "response": row[2][:200] + "..." if len(row[2]) > 200 else row[2],
                "intent": row[3],
                "sources": row[4],
                "timestamp": row[5]
            }
            for row in rows
        ]
    
    def clear_history(self, before_date: Optional[datetime] = None):
        """Clear history before a certain date."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if before_date:
            cursor.execute("DELETE FROM messages WHERE timestamp < ?", (before_date.isoformat(),))
        else:
            cursor.execute("DELETE FROM messages")
        
        conn.commit()
        deleted = cursor.rowcount
        conn.close()
        
        return deleted
