# Step 3: A chat loop with short-term memory (a sliding window of recent messages).

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

USE_MEMORY = True        # Set to False to see the chatbot "forget" everything.
MAX_RECENT_MESSAGES = 6  # How many recent messages to keep (3 user + 3 AI turns).

system_message = SystemMessage(content="You are a friendly assistant. Keep answers short.")
history = []  # Short-term memory: lives only in RAM, lost when the program stops.

print("Chatbot ready! Type 'quit' to exit.\n")

while True:
    user_text = input("You: ").strip()
    if user_text.lower() in ("quit", "exit"):
        break
    if not user_text:
        continue

    history.append(HumanMessage(content=user_text))

    if USE_MEMORY:
        # Keep only the last N messages so the prompt doesn't grow forever.
        recent = history[-MAX_RECENT_MESSAGES:]
    else:
        # No memory: the model only ever sees the current message.
        recent = [history[-1]]

    # The system message always goes first, then the recent conversation.
    response = llm.invoke([system_message] + recent)

    # Save the AI's reply too, so the next turn includes it.
    history.append(AIMessage(content=response.content))

    print(f"AI: {response.content}")
    print(f"   [sent {len(recent) + 1} messages, {response.usage_metadata['input_tokens']} input tokens]\n")
