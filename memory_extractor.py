# memory_extractor.py: uses the LLM to decide what is worth remembering.

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# The exact shape we want the LLM to return. The Field descriptions are sent
# to the model too, so they act as extra instructions.
class MemoryItem(BaseModel):
    text: str = Field(description="The fact, as one short sentence starting with 'The user'.")
    replaces_id: str | None = Field(
        description="ID of a known memory this fact updates or contradicts, otherwise null."
    )


class ExtractedMemories(BaseModel):
    memories: list[MemoryItem] = Field(
        description="Facts worth remembering about the user. Empty list if there are none."
    )


EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You extract long-term memories about the user from their latest message.\n"
     "\n"
     "SAVE lasting personal facts: name, location, job, studies, family, pets, "
     "preferences, dislikes, allergies, diet, goals, important dates and plans.\n"
     "DO NOT SAVE: greetings, small talk, temporary feelings ('I'm tired'), "
     "questions the user asks, or facts about the world that aren't about the user.\n"
     "\n"
     "Rules:\n"
     "- Write each memory as one short, standalone sentence starting with 'The user'.\n"
     "- One fact per memory.\n"
     "- Use the assistant's previous message only as context to understand short replies.\n"
     "- Do not repeat facts that are already known, even if worded differently.\n"
     "- If a new fact updates or contradicts a known memory (e.g. the user moved, "
     "changed jobs, or no longer has something), set replaces_id to that memory's ID. "
     "Otherwise set replaces_id to null.\n"
     "- If there is nothing worth saving, return an empty list."),
    ("human",
     "Already known about the user:\n{known_memories}\n\n"
     "Assistant's previous message: {previous_ai_message}\n"
     "User's latest message: {user_message}"),
])

# temperature=0: we want consistent, careful decisions, not creativity.
_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# with_structured_output makes the chain return an ExtractedMemories object, not text.
_extraction_chain = EXTRACTION_PROMPT | _llm.with_structured_output(ExtractedMemories)


def extract_memories(user_message, previous_ai_message="", known_memories=None):
    """Return a list of MemoryItem objects (text + replaces_id) found in the user's message.

    known_memories: related saved memories as dicts with "id" and "text", so the LLM
    can skip repeats and point at outdated memories by ID.
    """
    known_memories = known_memories or []
    known = "\n".join(f"- [{m['id']}] {m['text']}" for m in known_memories) or "(nothing yet)"
    result = _extraction_chain.invoke({
        "user_message": user_message,
        "previous_ai_message": previous_ai_message or "(none)",
        "known_memories": known,
    })

    # Never trust IDs from the LLM blindly: it could invent one that doesn't exist.
    valid_ids = {m["id"] for m in known_memories}
    for item in result.memories:
        if item.replaces_id not in valid_ids:
            item.replaces_id = None
    return result.memories
