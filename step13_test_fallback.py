# Step 13 test: concise search answers, and graceful fallback when web search fails.
#
# Asks questions only, so no new memories should be saved.

import chatbot
from langchain_core.runnables import RunnableLambda


def show(question, history):
    reply, activity = chatbot.chat_turn(question, history, use_web_search=True)
    print(f"\nQ: {question}")
    print(f"   searched={activity['searched']}  sources={len(activity['sources'])}  "
          f"memories used={len(activity['retrieved'])}  search_error={activity['search_error']}")
    print(f"   A ({len(reply)} characters): {reply}")


print("===== 1. Normal: web search available =====")
show("What's the weather like where I live today?", [])


# Replace the search chain with one that always fails, like a network outage would.
def broken_search(_inputs):
    raise ConnectionError("Simulated network failure")


chatbot.search_chain = RunnableLambda(broken_search)

print("\n===== 2. Web search broken (simulated) =====")
show("What is the latest news about artificial intelligence today?", [])
show("Where do I live?", [])  # Memory should still work without web search.
