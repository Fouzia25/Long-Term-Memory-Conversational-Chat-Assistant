# Step 5: Embeddings and vector stores (semantic search).

import math
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

load_dotenv()

# An embedding model turns text into a vector (a list of numbers) representing its meaning.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# --- Part A: look at embeddings by hand ---
vector = embeddings.embed_query("I'm vegetarian")
print(f"An embedding has {len(vector)} numbers. First five: {[round(x, 4) for x in vector[:5]]}\n")


def cosine_similarity(a, b):
    """How close two vectors point in the same direction: ~1 = same meaning."""
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


base = "I'm vegetarian"
for other in ["I don't eat meat", "I love vegetables", "My laptop is a Dell"]:
    score = cosine_similarity(embeddings.embed_query(base), embeddings.embed_query(other))
    print(f"similarity('{base}', '{other}') = {score:.3f}")

# --- Part B: a vector store does the searching for us ---
facts = [
    "The user is vegetarian.",
    "The user's name is Raeesah.",
    "The user is learning Generative AI with Python.",
    "The user has a cat called Luna.",
    "The user's favourite colour is green.",
    "The user is allergic to peanuts.",
]

# Embeds every fact and stores (text, vector) pairs in memory.
store = InMemoryVectorStore.from_texts(facts, embedding=embeddings)

for question in ["What should I cook tonight?", "Do I have any pets?", "What am I studying?"]:
    print(f"\nQuestion: {question}")
    # Embeds the question, then returns the k closest facts with their scores.
    for doc, score in store.similarity_search_with_score(question, k=2):
        print(f"   {score:.3f}  {doc.page_content}")
