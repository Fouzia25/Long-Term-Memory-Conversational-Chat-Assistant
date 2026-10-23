# migrate_json_to_sqlite.py: copy memories from the old memories.json into memories.db.
#
# Run once:  python migrate_json_to_sqlite.py
# The saved vectors are reused, so no embedding API calls are made.
# memories.json is left untouched as a backup. Running this again is safe.

import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()  # OpenAIEmbeddings in MemoryStore needs the key, even though we don't call it here.

from memory_store import MemoryStore

OLD_FILE = "memories.json"
NEW_FILE = "memories.db"

if not os.path.exists(OLD_FILE):
    sys.exit(f"No {OLD_FILE} found, so there is nothing to migrate.")

with open(OLD_FILE, encoding="utf-8") as f:
    old_memories = json.load(f)  # {id: {"id", "text", "vector", "metadata": {"created_at"}}}

store = MemoryStore(path=NEW_FILE)
for memory_id, item in old_memories.items():
    store.add_existing(memory_id, item["text"], item["metadata"]["created_at"], item["vector"])
    print(f"Migrated [{memory_id}] {item['text']}")

print(f"\nDone: {len(old_memories)} memories copied. {NEW_FILE} now holds {len(store.list_all())} memories.")
print(f"{OLD_FILE} was not changed; keep it as a backup or delete it once you're happy.")
