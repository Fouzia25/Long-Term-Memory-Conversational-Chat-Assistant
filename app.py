# app.py: a web interface for the memory chatbot, built with Streamlit.
# Run with:  streamlit run app.py
#
# Streamlit reruns this whole script from top to bottom every time the user
# clicks a button or sends a message. st.session_state keeps values between reruns.

import streamlit as st

# Importing chatbot.py loads the API key, the memory store and the AI chain.
# Python imports a module only once, so these are created once, not on every rerun.
# (The terminal chat loop doesn't start, thanks to `if __name__ == "__main__":`.)
from chatbot import WEB_SEARCH_DEFAULT, memory, stream_chat_turn

st.set_page_config(page_title="Memory chat assistant", page_icon=":material/neurology:")

# --- Short-term memory for this browser tab (survives reruns, not restarts) ---
if "history" not in st.session_state:
    st.session_state.history = []         # HumanMessage/AIMessage objects sent to the AI
    st.session_state.activity = []        # What the memory system did on each turn
    st.session_state.show_activity = True # Value of the "Show memory activity" toggle
    st.session_state.web_search = WEB_SEARCH_DEFAULT  # Value of the "Search the web" toggle


def start_new_conversation():
    """Clear short-term memory. Long-term memories are kept."""
    st.session_state.history = []
    st.session_state.activity = []


def show_web_results(activity):
    """Show web search status and clickable sources under the AI's reply.

    Always shown (not hidden by the activity toggle): citations must stay visible.
    """
    if activity.get("search_error"):
        st.warning(
            "I answered without current information, so anything time-sensitive "
            "may be out of date.",
            title="Web search unavailable",
            icon=":material/wifi_off:",
        )
    if not activity.get("searched"):
        return
    st.badge("Searched the web", icon=":material/travel_explore:", color="blue")
    sources = activity.get("sources", [])
    if not sources:
        st.caption(":material/info: No source links were returned for this search. Please verify this information.")
        return
    st.caption("Sources")
    for number, source in enumerate(sources, start=1):
        title = source["title"].replace("[", "(").replace("]", ")")  # Brackets would break the link.
        st.markdown(f"{number}. [{title}]({source['url']})")


def show_activity(activity, turn_number):
    """Show what the memory system did for one turn, under the AI's reply."""
    for fact in activity["saved"]:
        st.caption(f":material/bookmark_add: Remembered: {fact}")
    for change in activity["updated"]:
        st.caption(f":material/sync: Updated: {change}")
    if activity["retrieved"]:
        label = f"Saved memories checked ({len(activity['retrieved'])})"
        with st.expander(label, icon=":material/psychology:", key=f"used_{turn_number}"):
            for fact in activity["retrieved"]:
                st.markdown(f"- {fact}")


# --- Sidebar: view and delete long-term memories ---
with st.sidebar:
    st.header("What I remember")
    memories = memory.list_all()

    if not memories:
        st.caption("Nothing saved yet. Tell me about yourself!")

    for m in memories:
        with st.container(border=True, horizontal=True, vertical_alignment="center"):
            st.markdown(f"{m['text']}  \n:gray[{m['created_at']}]")
            st.button(
                ":material/delete:",
                key=f"delete_{m['id']}",
                help="Forget this memory",
                type="tertiary",
                on_click=memory.delete,
                args=(m["id"],),
            )

    if memories:
        with st.popover("Forget everything", icon=":material/delete_forever:", width="stretch"):
            st.warning("This deletes every memory and cannot be undone.")
            st.button("Yes, delete all", type="primary", on_click=memory.clear)

    st.divider()
    st.button(
        "New conversation",
        icon=":material/refresh:",
        width="stretch",
        help="Clears the recent chat. Long-term memories are kept.",
        on_click=start_new_conversation,
    )
    st.toggle(
        "Search the web",
        key="web_search",
        help="Look up current information online when needed. Each search costs about 1 cent.",
    )
    st.toggle("Show memory activity", key="show_activity")

# --- Main area: the chat ---
st.title("Memory chat assistant")
if st.session_state.web_search:
    st.caption("I remember important things about you, and I search the web for current information.")
else:
    st.caption("I remember important things about you. Turn on **Search the web** in the sidebar for current information.")

if not st.session_state.history:
    with st.chat_message("assistant"):
        st.markdown(f"Hi! I currently remember **{len(memories)}** thing(s) about you. What's on your mind?")

# Redraw the conversation so far. History alternates: user, AI, user, AI...
for i, message in enumerate(st.session_state.history):
    role = "user" if message.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)
        if role == "assistant":
            show_web_results(st.session_state.activity[i // 2])
            if st.session_state.show_activity:
                show_activity(st.session_state.activity[i // 2], turn_number=i // 2)

# Handle a new message.
if prompt := st.chat_input("Message your assistant", submit_mode="disable"):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        thinking = st.empty()
        thinking.caption(":material/hourglass_top: Thinking...")

        activity = {}  # Filled in by stream_chat_turn once the reply is complete.

        def reply_pieces():
            """Stream the reply, removing the 'Thinking...' note when the first piece arrives."""
            pieces = stream_chat_turn(prompt, st.session_state.history, activity, st.session_state.web_search)
            for number, piece in enumerate(pieces):
                if number == 0:
                    thinking.empty()
                yield piece
            thinking.empty()  # In case the reply was empty.

        st.write_stream(reply_pieces())  # Shows each piece as it arrives.
    st.session_state.activity.append(activity)
    # Rerun so the sidebar and the conversation are redrawn with the new memories.
    st.rerun()
