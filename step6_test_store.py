# Step 6: Test that MemoryStore keeps memories across restarts.
#
# Run in two separate phases (two separate Python processes):
#   python step6_test_store.py add     -> adds memories, then the program ends
#   python step6_test_store.py check   -> a brand-new process reads them back

import sys
from dotenv import load_dotenv
from memory_store import MemoryStore

load_dotenv()
memory = MemoryStore(path="test_memories.db")  # A separate database just for testing.

phase = sys.argv[1] if len(sys.argv) > 1 else "check"

if phase == "add":
    memory.clear()  # Start from a clean slate each time we run the test.
    for fact in [
        "The user's name is Raeesah.",
        "The user is vegetarian.",
        "The user has a cat called Luna.",
    ]:
        print(f"Added [{memory.add(fact)}] {fact}")
    print("\nSaved to test_memories.db. Program ending now...")

elif phase == "check":
    print("Fresh start. Memories loaded from disk:")
    for m in memory.list_all():
        print(f"   [{m['id']}] {m['text']}  (saved {m['created_at']})")

    print("\nSearch 'any pets?':", memory.search("any pets?", k=1))

    first_id = memory.list_all()[0]["id"]
    memory.delete(first_id)
    print(f"\nDeleted [{first_id}]. Remaining: {[m['text'] for m in memory.list_all()]}")
