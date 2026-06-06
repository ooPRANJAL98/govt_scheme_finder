# app.py
import streamlit as st
from rag_chain import get_rag_chain
from datetime import datetime

st.set_page_config(
    page_title="Govt Scheme Finder",
    page_icon="🏛️",
    layout="wide"
)

@st.cache_resource
def load_chain():
    return get_rag_chain()

chain, retriever = load_chain()

# ── Session state init ──────────────────────────────────────────────
if "chats" not in st.session_state:
    st.session_state.chats = {}

if "active_chat" not in st.session_state:
    st.session_state.active_chat = None

# ── Helper functions ────────────────────────────────────────────────
def create_new_chat():
    chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.chats[chat_id] = {
        "title": "New Chat",
        "messages": [],
        "created": datetime.now().strftime("%d %b, %I:%M %p")
    }
    st.session_state.active_chat = chat_id
    st.rerun()

def delete_chat(chat_id):
    del st.session_state.chats[chat_id]
    if st.session_state.active_chat == chat_id:
        st.session_state.active_chat = (
            list(st.session_state.chats.keys())[-1]
            if st.session_state.chats else None
        )
    st.rerun()

def get_active_messages():
    if st.session_state.active_chat:
        return st.session_state.chats[st.session_state.active_chat]["messages"]
    return []

def add_message(role, content):
    chat = st.session_state.chats[st.session_state.active_chat]
    chat["messages"].append({"role": role, "content": content})
    # Auto-title chat from first user message
    if role == "user" and chat["title"] == "New Chat":
        chat["title"] = content[:35] + "..." if len(content) > 35 else content

# ── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏛️ Scheme Finder")
    st.divider()

    # New chat button
    if st.button("➕  New Chat", use_container_width=True, type="primary"):
        create_new_chat()

    st.divider()

    # List all chats
    if st.session_state.chats:
        st.markdown("**💬 Your Chats**")
        for chat_id in reversed(list(st.session_state.chats.keys())):
            chat = st.session_state.chats[chat_id]
            is_active = chat_id == st.session_state.active_chat

            col1, col2 = st.columns([5, 1])
            with col1:
                label = ("▶ " if is_active else "") + chat["title"]
                if st.button(
                    label,
                    key=f"chat_{chat_id}",
                    use_container_width=True,
                    type="secondary"
                ):
                    st.session_state.active_chat = chat_id
                    st.rerun()
            with col2:
                if st.button("🗑", key=f"del_{chat_id}", help="Delete chat"):
                    delete_chat(chat_id)
    else:
        st.caption("No chats yet. Click ➕ New Chat to start!")

    st.divider()
    st.caption("📦 Schemes loaded: 10")
    st.caption("🤖 Model: Gemini 2.5 Flash")
    st.caption("🔒 Embeddings: Local (MiniLM)")

# ── Main area ───────────────────────────────────────────────────────

# No chat open yet
if not st.session_state.active_chat:
    st.markdown("""
    <div style='text-align:center; padding: 80px 0 20px 0;'>
        <h1>🏛️ Government Scheme Finder</h1>
        <p style='font-size:18px; color:gray;'>
            Find schemes you qualify for — just describe your situation
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("➕  Start a New Chat", use_container_width=True, type="primary"):
            create_new_chat()

# Active chat open
else:
    messages = get_active_messages()
    chat_title = st.session_state.chats[st.session_state.active_chat]["title"]

    st.markdown(f"### {chat_title}")
    st.divider()

    # Show example buttons only when chat is empty
    if not messages:
        st.markdown("#### 👋 Hi! Describe your situation and I'll find matching schemes.")
        st.markdown("**Or try one of these:**")

        examples = [
            "🌾 Small farmer in Himachal Pradesh with 2 acres of land",
            "📚 Girl student, SC category, Class 11, income below 2 lakh",
            "💼 Unemployed rural youth aged 22, looking for skill training",
            "🏠 Living in kutcha house in rural HP, want to build pucca house",
            "💊 Kidney disease patient in Himachal Pradesh, need financial help",
            "💰 Want to start a small business, need loan without collateral",
            "👧 Want to open savings account for my daughter aged 5",
            "🎓 Want free coaching for JEE, family income below 2.5 lakh in HP",
        ]

        col1, col2 = st.columns(2)
        for i, example in enumerate(examples):
            col = col1 if i % 2 == 0 else col2
            if col.button(example, use_container_width=True, key=f"ex_{i}"):
                add_message("user", example)
                with st.spinner("Finding matching schemes..."):
                    answer = chain.invoke(example)
                add_message("assistant", answer)
                st.rerun()

    # Display chat messages
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Describe your situation or ask a follow-up question..."):
        add_message("user", prompt)

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching schemes..."):
                answer = chain.invoke(prompt)
            st.markdown(answer)

        add_message("assistant", answer)
        st.rerun()