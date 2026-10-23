# Step 2: Send one message to OpenAI through LangChain.

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Read OPENAI_API_KEY from the .env file into the environment.
# ChatOpenAI finds the key there automatically.
load_dotenv()

# Create the chat model.
# temperature controls randomness: 0 = predictable, 1 = more creative.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# A conversation is a list of messages, each with a role.
messages = [
    SystemMessage(content="You are a friendly assistant. Keep answers short."),
    HumanMessage(content="Hi! My name is Raeesah. What is LangChain in one sentence?"),
]

# Send the messages to the model and get back an AIMessage.
response = llm.invoke(messages)

print("Reply text:\n", response.content)
print("\nResponse type:", type(response).__name__)
print("Token usage:", response.usage_metadata)
