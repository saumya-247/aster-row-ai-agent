import uuid
import streamlit as st
from src.agent import AgentCore, ConversationSession

# Configure page layout and title
st.set_page_config(
    page_title="Aster & Row Support Assistant",
    page_icon="🎒",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_session():
    """Initializes agent core and conversation state."""
    if "agent" not in st.session_state:
        st.session_state.agent = AgentCore()

    if "session" not in st.session_state or "messages" not in st.session_state:
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        st.session_state.session = ConversationSession(session_id=session_id)
        st.session_state.messages = []


def reset_conversation():
    """Resets the active conversation session."""
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    st.session_state.session = ConversationSession(session_id=session_id)
    st.session_state.messages = []


initialize_session()

# Sidebar: Controls, Model Status & Example Questions
with st.sidebar:
    st.title("🎒 Aster & Row")
    st.markdown("### Customer Support AI")

    # Engine Status Indicator
    llm_available = st.session_state.agent.llm_provider.is_available()
    if llm_available:
        model_name = st.session_state.agent.llm_provider.model_name
        st.success(f"🟢 **Live LLM Connected**\nModel: `{model_name}`")
    else:
        st.info("🔵 **Deterministic Engine Active**\nOperating in grounded zero-drift mode.")

    st.divider()

    if st.button("🔄 New Conversation", use_container_width=True, type="primary"):
        reset_conversation()
        st.rerun()

    st.divider()

    st.markdown("#### 💡 Example Questions")
    example_prompts = [
        "How long do I have to return an unused backpack?",
        "My TrailPlus membership was active when I ordered. What is my return window?",
        "Where is ORD-1007 and when will it arrive?",
        "Can I put the entire Breeze Tumbler in the dishwasher?",
        "Do you ship to Canada, and how long does it take?",
        "A final-sale bag arrived with a broken zipper. Can I get help?",
        "What is the warranty period for bags vs drinkware?",
    ]

    for ex in example_prompts:
        if st.button(ex, key=f"ex_{ex[:20]}", use_container_width=True):
            st.session_state["pending_prompt"] = ex
            st.rerun()

    st.divider()
    st.caption("🕒 Support Hours: Monday – Friday, 9:00 AM – 6:00 PM EST")
    st.caption("🛡️ Privacy Protected: Zero PII or internal company metrics exposed.")


# Main Chat Interface
st.title("🎒 Aster & Row Support Assistant")
st.markdown(
    "Welcome! I can assist you with **order tracking**, **returns & exchanges**, **shipping rates**, "
    "**product care**, **warranty coverage**, and **membership benefits**."
)

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Display citations if present
        sources = msg.get("sources", [])
        if sources:
            with st.expander("📚 Verified Sources & Citations", expanded=False):
                for src in sources:
                    st.markdown(f"- `{src}`")

        # Display Human Handoff notice if applicable
        if msg.get("requires_handoff"):
            st.warning("⚠️ **Human Support Escalation:** This case has been flagged for human review or requires specialist confirmation.")


# Handle User Input (from chat_input or example button click)
user_input = st.chat_input("Ask a question about your order, return, or company policy...")

# Check if an example prompt was clicked
if "pending_prompt" in st.session_state and st.session_state["pending_prompt"]:
    user_input = st.session_state.pop("pending_prompt")

if user_input:
    # Render user query
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Consulting company knowledge base & order systems..."):
            try:
                response = st.session_state.agent.handle_message(
                    user_input, session=st.session_state.session
                )
                answer = response.answer
                sources = response.sources
                handoff = response.requires_handoff
                tool_calls = response.tool_calls
            except Exception as e:
                # Friendly error handling without leaking stack trace
                answer = (
                    "I apologize, but I encountered an issue processing your request. "
                    "Please try again or contact human customer support for assistance."
                )
                sources = []
                handoff = True
                tool_calls = []

            # Display answer
            st.markdown(answer)

            # Display sources
            if sources:
                with st.expander("📚 Verified Sources & Citations", expanded=False):
                    for src in sources:
                        st.markdown(f"- `{src}`")

            # Display handoff indicator if needed
            if handoff:
                st.warning("⚠️ **Human Support Escalation:** This case has been flagged for human review or requires specialist confirmation.")

            # Record in session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "requires_handoff": handoff,
                "tool_calls": tool_calls,
            })
