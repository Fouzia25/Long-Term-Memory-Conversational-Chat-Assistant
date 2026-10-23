# Step 4: Prompt templates and chains.

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

MAX_RECENT_MESSAGES = 6

# The prompt layout, defined once. {memories} and {input} are blanks to fill in;
# MessagesPlaceholder is a slot where a whole list of messages gets inserted.
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a friendly assistant. Keep answers short.\n"
     "Things you know about the user from past conversations:\n{memories}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# A chain: fill the prompt -> send to the model -> extract the reply text.
chain = prompt | llm | StrOutputParser()

# Placeholder memories for now. In Step 8 these will come from real storage.
fake_memories = "- The user's favourite colour is green.\n- The user is learning GenAI."

# --- Look inside: what does the template produce? ---
filled = prompt.invoke({
    "memories": fake_memories,
    "chat_history": [HumanMessage(content="Hello"), AIMessage(content="Hi there!")],
    "input": "What colour do I like?",
})
print("=== What the model will receive ===")
for message in filled.to_messages():
    print(f"[{message.type}] {message.content}")
print("===================================\n")

# --- The chat loop, now using the chain ---
history = []
print("Chatbot ready! Type 'quit' to exit.\n")

while True:
    user_text = input("You: ").strip()
    if user_text.lower() in ("quit", "exit"):
        break
    if not user_text:
        continue

    reply = chain.invoke({
        "memories": fake_memories,
        "chat_history": history[-MAX_RECENT_MESSAGES:],
        "input": user_text,
    })

    # The new message isn't in the history yet (the template adds it as {input}),
    # so we add both sides of the exchange after the reply.
    history.append(HumanMessage(content=user_text))
    history.append(AIMessage(content=reply))

    print(f"AI: {reply}\n")
