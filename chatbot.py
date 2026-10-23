# chatbot.py: a conversational assistant with short-term and long-term memory.
#
# Each turn:
#   1. Retrieve  - search long-term memory for facts related to the message
#   2. Prompt    - fill the template with those memories + recent chat history
#   3. Reply     - send to the LLM
#   4. Extract   - ask the LLM what is worth remembering, and save it

from datetime import date

from dotenv import load_dotenv

load_dotenv()  # Must run before importing modules that create OpenAI clients.

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from memory_store import MemoryStore
from memory_extractor import extract_memories
from web_search import search_llm, parse_response

# --- Settings ---
MAX_RECENT_MESSAGES = 10  # Short-term memory window (5 user + 5 AI messages).
MEMORIES_TO_RETRIEVE = 4  # How many long-term memories to put in the prompt.
SHOW_DEBUG = True         # Print what was retrieved and saved each turn.
WEB_SEARCH_DEFAULT = False  # Start with web search off (each search costs about 1 cent).

# --- Components ---
memory = MemoryStore(path="memories.db")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a warm, helpful personal assistant. Keep answers concise.\n"
     "Today's date is {today}.\n"
     "\n"
     "Things you remember about the user from past conversations:\n"
     "{memories}\n"
     "\n"
     "Use these memories naturally when they are relevant, but don't list them "
     "unprompted. Some may be unrelated to the current message, so ignore those. "
     "Memories are facts about the user only; never search the web for facts about the user. "
     "If asked about the user or the conversation and the answer isn't in your memories "
     "or the conversation, say you don't know rather than guessing.\n"
     "\n"
     "{web_instructions}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

SEARCH_INSTRUCTIONS = (
    "You can search the web. Search when the question needs current, recent or "
    "time-sensitive information (news, events, prices, weather, sports results, latest "
    "versions, anything that may have changed since your training). Answer general "
    "knowledge, maths, and questions about the user directly without searching.\n"
    "For current events and facts about the world, trust web search results over your "
    "training knowledge and over the user's memories. Cite your sources for facts taken "
    "from the web. If you can't find or verify current information, say so clearly. "
    "Never invent sources, links or dates.\n"
    "Summarise search results in a few sentences. Don't paste long lists, hourly "
    "forecasts, headings or extra links unless the user asks for details."
)

NO_SEARCH_INSTRUCTIONS = (
    "Web search is unavailable right now. If the question needs current or recent "
    "information, say that you can't verify it at the moment and that your knowledge "
    "may be out of date. Never invent sources, links or dates."
)

WEB_OFF_INSTRUCTIONS = (
    "Web search is turned off. If the question needs current or recent information, "
    "say that your knowledge may be out of date and suggest turning on web search. "
    "Never invent sources, links or dates."
)

# Main chain: the model may search the web. It streams AIMessageChunks; once the
# stream ends, parse_response reads the text and sources from the complete message.
search_chain = prompt | search_llm

# Fallback chain (the original one): used if web search fails. Streams plain text.
chain = prompt | llm | StrOutputParser()


def format_memories(memories):
    """Turn a list of memory strings into bullet points for the prompt."""
    if not memories:
        return "(nothing yet)"
    return "\n".join(f"- {m}" for m in memories)


def stream_chat_turn(user_text, history, activity, use_web_search=WEB_SEARCH_DEFAULT):
    """Run one full turn (retrieve, reply, extract), yielding the reply piece by piece.

    use_web_search: if False, the original chain answers (no internet, no search cost).
    `activity` is an empty dict from the caller. When the generator finishes it holds
    what happened: "retrieved", "saved" and "updated" (memory), "searched" and
    "sources" (web), and "search_error" (None, or a short message if web search failed).
    """
    # 1. Retrieve relevant long-term memories (with IDs, so outdated ones can be replaced).
    related = memory.search_with_ids(user_text, k=MEMORIES_TO_RETRIEVE)
    relevant = [m["text"] for m in related]

    # 2 + 3. Fill the prompt and stream the reply. The model searches the web if it needs to.
    inputs = {
        "today": f"{date.today():%A, %d %B %Y}",  # From the computer's clock, never hard-coded.
        "memories": format_memories(relevant),
        "chat_history": history[-MAX_RECENT_MESSAGES:],
        "input": user_text,
    }
    pieces = []           # The reply text, as shown so far.
    full_message = None   # All chunks added together, to read sources from at the end.
    search_error = None
    if use_web_search:
        try:
            for chunk in search_chain.stream({**inputs, "web_instructions": SEARCH_INSTRUCTIONS}):
                full_message = chunk if full_message is None else full_message + chunk
                if chunk.text:
                    pieces.append(chunk.text)
                    yield chunk.text
        except Exception as error:
            # Web search failed (network, API or model problem).
            search_error = f"{type(error).__name__}: {str(error)[:200]}"
    else:
        # Web search switched off: the original chain, no internet.
        for piece in chain.stream({**inputs, "web_instructions": WEB_OFF_INSTRUCTIONS}):
            pieces.append(piece)
            yield piece

    used_fallback = search_error is not None and not pieces
    if used_fallback:
        # Nothing was shown yet, so answer without web search instead.
        for piece in chain.stream({**inputs, "web_instructions": NO_SEARCH_INSTRUCTIONS}):
            pieces.append(piece)
            yield piece
    elif search_error:
        # Part of the answer was already shown: keep it, and say it was cut off.
        notice = "\n\n*(This answer was cut off by an error.)*"
        pieces.append(notice)
        yield notice

    reply = "".join(pieces)
    if used_fallback or full_message is None:
        searched, sources = False, []
    else:
        web = parse_response(full_message)
        searched, sources = web["searched"], web["sources"]

    # 4. Extract new facts, using the AI's previous message as context
    #    and the retrieved memories so known facts aren't saved twice.
    previous_ai = history[-1].content if history else ""
    old_texts = {m["id"]: m["text"] for m in related}
    saved, updated = [], []
    for item in extract_memories(user_text, previous_ai, known_memories=related):
        if item.replaces_id:
            memory.replace(item.replaces_id, item.text)
            updated.append(f"{old_texts[item.replaces_id]} -> {item.text}")
        elif memory.add_if_new(item.text):
            saved.append(item.text)

    # Update short-term memory with both sides of the exchange.
    history.append(HumanMessage(content=user_text))
    history.append(AIMessage(content=reply))
    activity.update({
        "retrieved": relevant, "saved": saved, "updated": updated,
        "searched": searched, "sources": sources, "search_error": search_error,
    })


def chat_turn(user_text, history, use_web_search=WEB_SEARCH_DEFAULT):
    """Run one full turn without streaming. Returns (reply, activity)."""
    activity = {}
    reply = "".join(stream_chat_turn(user_text, history, activity, use_web_search))
    return reply, activity


HELP_TEXT = """Commands:
  /memories       List everything I remember about you
  /forget <id>    Delete one memory (IDs are shown by /memories)
  /forget all     Delete every memory
  /web on|off     Turn web search on or off (about 1 cent per search)
  /help           Show this message
  quit            Exit"""


def handle_command(command, settings):
    """Handle a /command without calling the LLM.

    settings: this session's options, e.g. {"web_search": False}.
    """
    parts = command.split(maxsplit=1)
    name = parts[0].lower()
    argument = parts[1].strip() if len(parts) > 1 else ""

    if name == "/memories":
        memories = memory.list_all()
        if not memories:
            print("I don't have any memories saved yet.\n")
            return
        print(f"I remember {len(memories)} thing(s):")
        for m in memories:
            print(f"  [{m['id']}] {m['text']}  ({m['created_at']})")
        print()

    elif name == "/forget":
        if not argument:
            print("Usage: /forget <id>  or  /forget all\n")
        elif argument.lower() == "all":
            confirm = input("Delete ALL memories? This cannot be undone. (yes/no): ")
            if confirm.strip().lower() == "yes":
                memory.clear()
                print("All memories deleted.\n")
            else:
                print("Cancelled.\n")
        elif memory.delete(argument):
            print(f"Forgot memory [{argument}].\n")
        else:
            print(f"No memory with ID [{argument}]. Use /memories to see IDs.\n")

    elif name == "/web":
        if argument.lower() in ("on", "off"):
            settings["web_search"] = argument.lower() == "on"
        state = "ON" if settings["web_search"] else "OFF"
        print(f"Web search is {state}. Use /web on or /web off to change it.\n")

    elif name == "/help":
        print(HELP_TEXT + "\n")

    else:
        print(f"Unknown command '{name}'. Type /help for commands.\n")


def main():
    history = []  # Short-term memory: this session only.
    settings = {"web_search": WEB_SEARCH_DEFAULT}
    web_state = "on" if settings["web_search"] else "off"
    print(f"Chatbot ready! ({len(memory.list_all())} memories loaded, web search {web_state}). "
          "Type /help for commands.\n")

    while True:
        # Strip whitespace and the invisible BOM character Windows can add to piped input.
        user_text = input("You: ").strip().lstrip("\ufeff")
        if user_text.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if not user_text:
            continue
        if user_text.startswith("/"):
            handle_command(user_text, settings)
            continue  # Commands never reach the LLM.

        # Print the reply piece by piece as it streams in.
        activity = {}
        print("AI: ", end="", flush=True)
        for piece in stream_chat_turn(user_text, history, activity, settings["web_search"]):
            print(piece, end="", flush=True)
        print()

        if SHOW_DEBUG:
            print(f"   [retrieved: {activity['retrieved'] or 'none'}]")
            if activity["saved"]:
                print(f"   [saved: {activity['saved']}]")
            if activity["updated"]:
                print(f"   [updated: {activity['updated']}]")
            if activity["searched"]:
                print("   [searched the web]")
        if activity["search_error"]:
            print(f"   [web search failed, answered without it: {activity['search_error']}]")
        for source in activity["sources"]:
            print(f"   Source: {source['title']} - {source['url']}")
        print()


if __name__ == "__main__":
    main()
