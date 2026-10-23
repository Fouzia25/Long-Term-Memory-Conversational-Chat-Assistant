# memory_store.py: long-term memory that survives restarts, stored in SQLite.
#
# Each memory is one row in the "memories" table: id, text, created_at, and its
# embedding (1,536 numbers stored as raw bytes). Semantic search loads the
# embeddings and compares them with numpy (cosine similarity, as in Step 5).

import sqlite3
import uuid
from datetime import datetime

import numpy as np
from langchain_openai import OpenAIEmbeddings


def _to_blob(vector):
    """Pack a list of floats into bytes for the BLOB column (4 bytes per number)."""
    return np.asarray(vector, dtype=np.float32).tobytes()


def _from_blob(blob):
    """Unpack bytes from the BLOB column back into a numpy array."""
    return np.frombuffer(blob, dtype=np.float32)


class MemoryStore:
    def __init__(self, path="memories.db"):
        self.path = path
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        # Creates the database file and table the first time; does nothing afterwards.
        self._run(
            "CREATE TABLE IF NOT EXISTS memories ("
            "  id TEXT PRIMARY KEY,"
            "  text TEXT NOT NULL,"
            "  created_at TEXT NOT NULL,"
            "  embedding BLOB NOT NULL"
            ")"
        )

    def _run(self, sql, params=()):
        """Run one SQL statement, save (commit) any changes, and return the result rows.

        A fresh connection each time means the terminal and web versions (and
        Streamlit's threads) can all use the database safely.
        """
        connection = sqlite3.connect(self.path)
        try:
            rows = connection.execute(sql, params).fetchall()
            connection.commit()
            return rows
        finally:
            connection.close()

    def add(self, text):
        """Embed and store a new memory. Returns its ID."""
        memory_id = uuid.uuid4().hex[:8]  # A short random ID like 'a3f9c2e1'.
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M")
        vector = self.embeddings.embed_query(text)
        self.add_existing(memory_id, text, created_at, vector)
        return memory_id

    def add_existing(self, memory_id, text, created_at, vector):
        """Store a memory that already has an ID, date and vector (used for migration)."""
        # The ? placeholders let sqlite3 insert values safely (never build SQL with f-strings).
        self._run(
            "INSERT OR REPLACE INTO memories (id, text, created_at, embedding) VALUES (?, ?, ?, ?)",
            (memory_id, text, created_at, _to_blob(vector)),
        )

    def add_if_new(self, text, duplicate_threshold=0.85):
        """Add a memory unless a near-identical one already exists.

        Returns the new ID, or None if it was a duplicate.
        """
        closest = self.search(text, k=1)
        if closest and closest[0][1] >= duplicate_threshold:
            return None
        return self.add(text)

    def replace(self, old_id, new_text):
        """Replace an outdated memory with an updated one. Returns the new ID.

        No duplicate check here: the new text is meant to be similar to the old one
        (e.g. 'lives in Karachi' replacing 'lives in Lahore').
        """
        self.delete(old_id)
        return self.add(new_text)

    def search_with_ids(self, query, k=4):
        """Return up to k relevant memories as dicts with id, text and score."""
        rows = self._run("SELECT id, text, embedding FROM memories")
        if not rows:  # Nothing saved yet: skip the embedding API call.
            return []

        query_vector = np.asarray(self.embeddings.embed_query(query), dtype=np.float32)
        vectors = np.array([_from_blob(embedding) for _, _, embedding in rows])

        # Cosine similarity of the query with every memory at once.
        scores = vectors @ query_vector / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vector))
        best = np.argsort(scores)[::-1][:k]  # Indexes of the k highest scores.
        return [{"id": rows[i][0], "text": rows[i][1], "score": float(scores[i])} for i in best]

    def search(self, query, k=4):
        """Return up to k memories most relevant to the query, as (text, score) pairs."""
        return [(m["text"], m["score"]) for m in self.search_with_ids(query, k=k)]

    def list_all(self):
        """Return every memory as a dict with id, text and created_at, oldest first."""
        rows = self._run("SELECT id, text, created_at FROM memories ORDER BY created_at, rowid")
        return [{"id": memory_id, "text": text, "created_at": created_at} for memory_id, text, created_at in rows]

    def delete(self, memory_id):
        """Delete one memory by ID. Returns True if it existed."""
        if not self._run("SELECT 1 FROM memories WHERE id = ?", (memory_id,)):
            return False
        self._run("DELETE FROM memories WHERE id = ?", (memory_id,))
        return True

    def clear(self):
        """Delete every memory."""
        self._run("DELETE FROM memories")
