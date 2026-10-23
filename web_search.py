# web_search.py: real-time web search using OpenAI's built-in web_search tool.
#
# The tool only works with OpenAI's Responses API, so this model is created with
# use_responses_api=True. The model decides by itself whether a question needs a
# search; OpenAI runs the search and returns the answer with citations attached.

from langchain_openai import ChatOpenAI

# Change this one line if OpenAI stops supporting web search on this model
# (gpt-4.1-mini is listed as supported in OpenAI's web search guide).
SEARCH_MODEL = "gpt-4o-mini"

_llm = ChatOpenAI(
    model=SEARCH_MODEL,
    temperature=0.7,
    use_responses_api=True,          # web_search is a Responses API tool
    output_version="responses/v1",   # put search calls and citations into message.content
)

# tool_choice defaults to "auto": the model searches only when it decides it needs to.
search_llm = _llm.bind_tools([{"type": "web_search"}])


def parse_response(message):
    """Split a Responses API AIMessage into its answer, sources, and whether a search ran.

    Sources come only from OpenAI's url_citation annotations, never from the
    answer text, so the app can't display a link the search didn't return.
    """
    searched = False
    sources = []
    for block in message.content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "web_search_call":
            searched = True
        for annotation in block.get("annotations") or []:
            if annotation.get("type") != "url_citation":
                continue
            if any(s["url"] == annotation["url"] for s in sources):
                continue  # The same page can be cited several times; list it once.
            sources.append({"title": annotation.get("title") or annotation["url"], "url": annotation["url"]})
    return {"text": message.text, "searched": searched, "sources": sources}
