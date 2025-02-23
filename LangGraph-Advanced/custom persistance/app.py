import streamlit as st
from chatbot import ChatState, graph, memory

# Custom CSS with enhanced animations
st.markdown("""
<style>
:root {
    --primary: #1E90FF;
    --bg: #0E1117;
    --card-bg: #1A1D24;
}

body {
    background-color: var(--bg);
    color: white;
}

.stTextInput input {
    background: var(--card-bg) !important;
    color: white !important;
    border: 1px solid #2E2E2E !important;
    border-radius: 12px !important;
    padding: 12px !important;
}

.chat-message {
    max-width: 80%;
    margin: 12px 0;
    padding: 16px;
    border-radius: 20px;
    animation: messageIn 0.3s cubic-bezier(0.18, 0.89, 0.32, 1.28);
    transform-origin: bottom;
}

.user-message {
    background: var(--primary);
    margin-left: auto;
    border-bottom-right-radius: 4px;
}

.bot-message {
    background: var(--card-bg);
    margin-right: auto;
    border-bottom-left-radius: 4px;
}

@keyframes messageIn {
    from {
        opacity: 0;
        transform: translateY(20px) scale(0.95);
    }
    to {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
}

.stSpinner > div {
    border-color: var(--primary) transparent transparent transparent !important;
}
</style>
""", unsafe_allow_html=True)

# Session state initialization
if "chat_state" not in st.session_state:
    st.session_state.chat_state = ChatState(session_id="default_session")
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
            session_id=st.session_state.chat_state.session_id,
            history=st.session_state.chat_state.history,
            user_input=user_input
        )
        
        # Process with spinner
        with st.spinner(""):
            updated_state = graph.invoke(temp_state)
        
        # Update chat state
        st.session_state.chat_state = ChatState(
            session_id=st.session_state.chat_state.session_id,
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