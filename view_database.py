# view_database.py: look inside memories.db with plain SQL (read-only, no API calls).
#
# Run:  python view_database.py              (the real database)
#       python view_database.py test_memories.db

import sqlite3
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "memories.db"
connection = sqlite3.connect(path)

# sqlite_master is SQLite's own table that describes every table in the file.
(table_sql,) = connection.execute("SELECT sql FROM sqlite_master WHERE name = 'memories'").fetchone()
print(f"Table structure in {path}:\n{table_sql}\n")

rows = connection.execute(
    "SELECT id, text, created_at, length(embedding) FROM memories ORDER BY created_at, rowid"
).fetchall()
print(f"{len(rows)} memories:")
for memory_id, text, created_at, embedding_bytes in rows:
    print(f"  [{memory_id}] {text}  ({created_at}, embedding: {embedding_bytes} bytes)")

connection.close()
