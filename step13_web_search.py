# Step 13: Real-time web search with OpenAI's built-in web_search tool.
#
# Asks a mix of questions: some need current information (should search),
# some don't (should answer directly without paying for a search).

from datetime import date
from dotenv import load_dotenv

load_dotenv()  # Must run before importing modules that create OpenAI clients.

from langchain_core.messages import SystemMessage, HumanMessage
from web_search import search_llm, parse_response

# Today's date comes from the computer's clock, so it's always current.
system = SystemMessage(content=(
    f"Today's date is {date.today():%d %B %Y}. "
    "Use web search for current or time-sensitive questions. "
    "Answer general knowledge questions directly. Keep answers to 2 sentences."
))

questions = [
    "What is 12 multiplied by 8?",                       # No search needed
    "What is photosynthesis?",                           # General knowledge
    "Who won the 2026 FIFA World Cup?",                  # After the model's training cutoff
    "What is the latest stable version of Python?",      # Changes over time
]

for question in questions:
    print(f"\nQ: {question}")
    message = search_llm.invoke([system, HumanMessage(content=question)])
    result = parse_response(message)

    print(f"   Searched the web: {'YES' if result['searched'] else 'no'}")
    print(f"   A: {result['text']}")
    for source in result["sources"]:
        print(f"   Source: {source['title']}\n           {source['url']}")
