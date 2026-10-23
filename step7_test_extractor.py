# Step 7: Test automatic memory extraction and duplicate handling.

from dotenv import load_dotenv

load_dotenv()  # Must run before importing modules that create OpenAI clients.

from memory_extractor import extract_memories
from memory_store import MemoryStore

# (previous AI message, user message) pairs: some worth saving, some not.
test_turns = [
    ("", "hi there!"),
    ("", "btw I just moved to Lahore, so tired lol"),
    ("", "What's the capital of France?"),
    ("What's your cat called?", "Luna"),
    ("", "I'm a vegetarian and I'm allergic to peanuts"),
    ("", "Like I said, I don't eat meat."),  # A repeat: should be caught as a duplicate.
]

memory = MemoryStore(path="test_memories.db")
memory.clear()

for previous_ai, user_message in test_turns:
    print(f"\nUser: {user_message}" + (f"   (AI had asked: '{previous_ai}')" if previous_ai else ""))
    # Embeddings find related memories; the LLM decides whether the new fact is truly new.
    related = memory.search_with_ids(user_message, k=5)
    facts = [item.text for item in extract_memories(user_message, previous_ai, known_memories=related)]
    if not facts:
        print("   -> nothing to remember")
    for fact in facts:
        memory_id = memory.add_if_new(fact)
        status = f"saved [{memory_id}]" if memory_id else "skipped (duplicate)"
        print(f"   -> {fact}  {status}")

print("\nFinal memories:", [m["text"] for m in memory.list_all()])
