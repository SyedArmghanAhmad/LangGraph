from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from custom_memory import ChatMemory
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

# Define ChatState as Pydantic model
class ChatState(BaseModel):
    session_id: str
    history: list = []
    user_input: str = ""

# Initialize LLM and memory
llm = ChatGroq(model_name="llama3-70b-8192", temperature=0.7)  # Increased temperature for more creative responses
memory = ChatMemory()

def process_message(state: ChatState) -> dict:
    """Handles user input and bot response."""
    if not state.user_input:
        return {}

    # Retrieve relevant context from memory
    context = memory.get_memory(state.user_input, session_id=state.session_id, k=10)
    
    # Include conversation history in the prompt
    conversation_history = "\n".join([f"User: {msg[0]}\nBot: {msg[1]}" for msg in state.history])
    
    # Generate response (with advanced prompt engineering)
    prompt = f"""You are a helpful and friendly AI assistant. Your goal is to provide natural, conversational responses based on the user's input and the context provided. Use the following context and conversation history to answer the user's question:

    ### Context:
    {context}

    ### Conversation History:
    {conversation_history}

    ### User Input:
    {state.user_input}

    ### Instructions:
    1. If the user asks about themselves (e.g., "What do you know about me?" or "What are my hobbies?"), retrieve all relevant memories and summarize them in a natural way.
    2. If the user provides new information (e.g., "I love football"), save it to memory and acknowledge it.
    3. If the user asks a general question (e.g., "How are you?"), respond naturally based on the conversation history.
    4. Always maintain a friendly and conversational tone.

    ### Assistant Response:
    """
    
    response = llm.invoke(prompt)
    
    # Save to memory
    memory.save_memory(state.session_id, state.user_input, response.content)
    
    # Return state updates
    return {
        "history": state.history + [(state.user_input, response.content)],
        "user_input": ""  # Clear user input after processing
    }

# Build LangGraph
graph = StateGraph(ChatState)
graph.add_node("chat", process_message)
graph.set_entry_point("chat")
graph.add_edge("chat", END)
graph = graph.compile()

# Run chat session
if __name__ == "__main__":
    state = ChatState(session_id="session_1")
    
    print("Chatbot initialized. Type 'exit' to quit.")
    while True:
        user_input = input("👤 You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        # Create temporary state with user input
        temp_state = ChatState(
            session_id=state.session_id,
            history=state.history,
            user_input=user_input
        )
        
        # Process through graph
        updated_state = graph.invoke(temp_state)
        
        # Merge updates into main state
        state = ChatState(
            session_id=state.session_id,
            history=updated_state.get("history", state.history),
            user_input=updated_state.get("user_input", "")
        )
        
        if state.history:
            print(f"🤖 Bot: {state.history[-1][1]}")

    # Save memory before exiting
    memory.vectorstore.save_local(memory.db_path)