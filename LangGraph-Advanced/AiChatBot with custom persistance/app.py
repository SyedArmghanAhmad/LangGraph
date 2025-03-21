import streamlit as st
from chatbot import ChatState, graph, memory
import uuid

# Ultra-modern CSS with glass morphism and animations
st.markdown("""
<style>
/* ===== CSS Variables ===== */
:root {
    --primary: #6366f1;
    --primary-hover: #4f46e5;
    --bg: #0f172a;
    --card-bg: rgba(30, 41, 59, 0.7); /* Glass morphism effect */
    --text: #f8fafc;
    --accent: #10b981;
    --gradient: linear-gradient(135deg, #6366f1 0%, #10b981 100%);
    --glass-border: rgba(255, 255, 255, 0.1);
}

/* ===== Base Styles ===== */
body {
    background-color: var(--bg);
    color: var(--text);
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    line-height: 1.6;
    margin: 0;
    padding: 0;
    overflow: hidden;
}

/* ===== Animated Background ===== */
body::before {
    content: '';
    position: fixed;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.1) 10%, transparent 10.01%);
    background-size: 20px 20px;
    animation: moveBackground 20s linear infinite;
    z-index: -1;
    opacity: 0.5;
}

@keyframes moveBackground {
    from {
        transform: translate(0, 0);
    }
    to {
        transform: translate(20px, 20px);
    }
}

/* ===== Chat Container ===== */
.stApp {
    max-width: 800px;
    margin: 0 auto;
    padding: 2rem 1rem;
    backdrop-filter: blur(20px);
    background: rgba(15, 23, 42, 0.8); /* Semi-transparent background */
    border-radius: 24px;
    border: 1px solid var(--glass-border);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
}

/* ===== Header ===== */
header {
    text-align: center;
    margin-bottom: 2.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 2px solid var(--glass-border);
}

.stTitle {
    font-size: 2.5rem;
    font-weight: 700;
    background: var(--gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
    animation: float 3s ease-in-out infinite;
}

@keyframes float {
    0%, 100% {
        transform: translateY(0);
    }
    50% {
        transform: translateY(-10px);
    }
}

.stCaption {
    color: #94a3b8;
    font-size: 1.1rem;
}

/* ===== Chat Messages ===== */
.chat-message {
    max-width: 75%;
    margin: 1rem 0;
    padding: 1.25rem;
    border-radius: 1.25rem;
    animation: messageIn 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    transform-origin: bottom;
    position: relative;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    backdrop-filter: blur(10px);
    border: 1px solid var(--glass-border);
}

.chat-message:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px -2px rgba(0, 0, 0, 0.15);
}

.user-message {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.8), rgba(79, 70, 229, 0.8));
    margin-left: auto;
    border-bottom-right-radius: 4px;
    color: white;
}

.bot-message {
    background: var(--card-bg);
    margin-right: auto;
    border-bottom-left-radius: 4px;
}

/* ===== Input Field ===== */
.stTextInput input {
    background: var(--card-bg) !important;
    color: var(--text) !important;
    border: 2px solid var(--glass-border) !important;
    border-radius: 12px !important;
    padding: 1rem 1.25rem !important;
    font-size: 1rem;
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
}

.stTextInput input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
}

/* ===== Spinner ===== */
.stSpinner > div {
    border-color: var(--primary) transparent transparent transparent !important;
    width: 28px !important;
    height: 28px !important;
}

/* ===== Scrollbar ===== */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: var(--bg);
}

::-webkit-scrollbar-thumb {
    background: var(--card-bg);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--primary);
}

/* ===== Sidebar ===== */
.stSidebar {
    background: var(--bg) !important;
    border-right: 1px solid var(--glass-border);
}

.stButton>button {
    background: var(--card-bg) !important;
    color: var(--text) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 8px !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.3s ease !important;
    backdrop-filter: blur(10px);
}

.stButton>button:hover {
    background: var(--primary) !important;
    border-color: var(--primary) !important;
    transform: translateY(-1px);
}

/* ===== Timestamp ===== */
.message-timestamp {
    font-size: 0.75rem;
    color: rgba(255, 255, 255, 0.4);
    margin-top: 0.5rem;
    display: block;
}

/* ===== Gradient Border ===== */
.gradient-border {
    position: relative;
    background: var(--bg);
}

.gradient-border::before {
    content: '';
    position: absolute;
    top: -2px;
    left: -2px;
    right: -2px;
    bottom: -2px;
    background: var(--gradient);
    border-radius: inherit;
    z-index: -1;
}
</style>
""", unsafe_allow_html=True)

# Reload memory at the start
memory.vectorstore = memory._load_memory()

# Generate a consistent session ID
if "session_id" not in st.session_state:
    st.session_state.session_id = "default_session"

# Session state initialization
if "chat_state" not in st.session_state:
    st.session_state.chat_state = ChatState(session_id=st.session_state.session_id)
if "processing" not in st.session_state:
    st.session_state.processing = False

# App layout
st.title("🤖 Futuristic AI Assistant")
st.caption("Experience next-gen conversational AI")

# Chat container
chat_container = st.container()

def display_messages():
    with chat_container:
        for idx, (user_msg, bot_msg) in enumerate(st.session_state.chat_state.history):
            cols = st.columns([1, 14])
            with cols[1]:
                # User message
                st.markdown(
                    f'<div class="chat-message user-message">{user_msg}</div>', 
                    unsafe_allow_html=True
                )
                
                # Bot message
                st.markdown(
                    f'<div class="chat-message bot-message">{bot_msg}</div>',
                    unsafe_allow_html=True
                )

# Handle user input
user_input = st.chat_input("Type your message...")

if user_input and not st.session_state.processing:
    try:
        st.session_state.processing = True
        
        # Create temp state
        temp_state = ChatState(
            session_id=st.session_state.session_id,
            history=st.session_state.chat_state.history,
            user_input=user_input
        )
        
        # Process with spinner
        with st.spinner(""):
            updated_state = graph.invoke(temp_state)
        
        # Update chat state
        st.session_state.chat_state = ChatState(
            session_id=st.session_state.session_id,
            history=updated_state.get("history", []),
            user_input=""
        )
        
    except Exception as e:
        st.error(f"Error processing message: {str(e)}")
    finally:
        st.session_state.processing = False
        memory.vectorstore.save_local(memory.db_path)

# Display messages
display_messages()