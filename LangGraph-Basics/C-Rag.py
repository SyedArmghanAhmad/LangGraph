import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from typing import List, Dict, Any
from typing_extensions import TypedDict

# LangChain imports
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Constants
MODEL_NAME = "llama3-70b-8192"  # Groq model to use
MAX_RETRIES = 2  # Maximum number of query rewrites
CHUNK_SIZE = 1500  # Increased chunk size for better context
CHUNK_OVERLAP = 300  # Increased overlap for better continuity
groq_api_key = os.getenv("GROQ_API_KEY")  # Fetch the API key

# Type definitions
class AgentState(TypedDict):
    question: str
    documents: List[str]
    grades: List[str]
    llm_output: str
    on_topic: bool
    retry_count: int  # Track retries for query rewriting

# Core models
class GradeQuestion(BaseModel):
    score: str = Field(description="'Yes' if question is answerable using documents, 'No' otherwise")

class GradeDocuments(BaseModel):
    score: str = Field(description="'Yes' if document is relevant, 'No' otherwise")

# Utility functions
@st.cache_resource
def get_embeddings():
    """Initialize and cache HuggingFace embeddings"""
    return HuggingFaceEmbeddings()

def validate_environment():
    """Check for required environment variables"""
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")

def initialize_chromadb():
    """Ensure ChromaDB is properly initialized"""
    try:
        # Test ChromaDB connection
        test_embeddings = get_embeddings()
        Chroma.from_texts(
            texts=["test"],
            embedding=test_embeddings,
            persist_directory=tempfile.mkdtemp()
        )
        return True
    except Exception as e:
        st.error(f"ChromaDB initialization failed: {str(e)}")
        return False

# Document processing
def process_pdf(uploaded_file) -> Chroma:
    """Process PDF file and create vector store"""
    try:
        # Create a temporary directory for ChromaDB
        persist_directory = tempfile.mkdtemp()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            loader = PyPDFLoader(tmp.name)
            pages = loader.load_and_split()
            st.write(f"Loaded {len(pages)} pages from the PDF")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        splits = text_splitter.split_documents(pages)
        st.write(f"Split into {len(splits)} document chunks")
        
        # Use persistent ChromaDB
        vector_store = Chroma.from_documents(
            documents=splits,
            embedding=get_embeddings(),
            persist_directory=persist_directory
        )
        st.write("Vector store created successfully")
        return vector_store
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        raise

# Workflow components
def initialize_llm() -> ChatGroq:
    """Initialize Groq LLM with validation"""
    validate_environment()
    return ChatGroq(model_name=MODEL_NAME, temperature=0, groq_api_key=groq_api_key)  # Pass API key here

def create_classifier_chain():
    """Create question classification chain"""
    system = """You are a relevance classifier. Determine if a question can be answered using documents.
    Respond "Yes" if the question is answerable with text content, "No" for personal/opinion/temporal questions."""
    
    return (
        ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "User question: {question}")
        ])
        | initialize_llm().with_structured_output(GradeQuestion)
    )

def create_document_grader_chain():
    """Create document grading chain"""
    system = """Evaluate document relevance. Respond "Yes" if the document contains:
    - Keywords related to the question
    - Semantic meaning related to the question"""
    
    return (
        ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "Document: {document}\nQuestion: {question}")
        ])
        | initialize_llm().with_structured_output(GradeDocuments)
    )

def create_rewriter_chain():
    """Create query rewriter chain"""
    system = "Improve this question for document retrieval while preserving its intent"
    
    return (
        ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "Original question: {question}")
        ])
        | initialize_llm()
        | StrOutputParser()
    )

def create_answer_chain():
    """Create answer generation chain"""
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    return (
        ChatPromptTemplate.from_template(template)
        | initialize_llm()
        | StrOutputParser()
    )

# Workflow nodes
def question_classifier(state: AgentState) -> AgentState:
    """Classify if question is answerable"""
    chain = create_classifier_chain()
    result = chain.invoke({"question": state["question"]})
    return {**state, "on_topic": result.score}

def document_grader(state: AgentState) -> AgentState:
    """Grade document relevance"""
    chain = create_document_grader_chain()
    scores = [
        chain.invoke({"document": doc, "question": state["question"]}).score
        for doc in state["documents"]
    ]
    return {**state, "grades": scores}

def rewriter(state: AgentState) -> AgentState:
    """Rewrite query for better retrieval"""
    chain = create_rewriter_chain()
    new_question = chain.invoke({"question": state["question"]})
    return {**state, "question": new_question}

def retrieve_docs(state: AgentState, retriever) -> AgentState:
    """Retrieve relevant documents"""
    try:
        docs = retriever.get_relevant_documents(state["question"], k=5)  # Retrieve top 5 documents
        st.write(f"Retrieved {len(docs)} documents for query: {state['question']}")
        return {**state, "documents": [doc.page_content for doc in docs]}
    except Exception as e:
        st.error(f"Error retrieving documents: {str(e)}")
        return {**state, "documents": []}

def generate_answer(state: AgentState) -> AgentState:
    """Generate the final answer"""
    try:
        chain = create_answer_chain()
        context = "\n\n".join(state["documents"])
        st.write(f"Using context: {context[:500]}...")  # Show first 500 chars of context
        result = chain.invoke({"question": state["question"], "context": context})
        st.write("Generated answer:", result)
        return {**state, "llm_output": result}
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        return {**state, "llm_output": "Failed to generate answer."}

def off_topic_response(state: AgentState) -> AgentState:
    """Handle unanswerable questions"""
    state["llm_output"] = "I couldn't find relevant information in the document to answer this question."
    return state

# Workflow setup
def create_workflow(retriever) -> StateGraph:
    """Configure and return LangGraph workflow"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    nodes = {
        "topic_decision": question_classifier,
        "retrieve_docs": lambda state: retrieve_docs(state, retriever),
        "document_grader": document_grader,
        "rewrite_query": rewriter,
        "generate_answer": generate_answer,
        "off_topic_response": off_topic_response
    }
    for name, func in nodes.items():
        workflow.add_node(name, func)
    
    # Add edges
    workflow.add_edge("off_topic_response", END)
    workflow.add_edge("retrieve_docs", "document_grader")
    workflow.add_edge("generate_answer", END)
    
    # Conditional edges
    def should_retry(state):
        if "retry_count" not in state:
            state["retry_count"] = 0
        if any(g == "Yes" for g in state["grades"]):
            return "generate"
        if state["retry_count"] < MAX_RETRIES:
            state["retry_count"] += 1
            return "rewrite"
        return "give_up"

    workflow.add_conditional_edges(
        "document_grader",
        should_retry,
        {"generate": "generate_answer", "rewrite": "rewrite_query", "give_up": "off_topic_response"}
    )
    
    workflow.add_edge("rewrite_query", "retrieve_docs")
    workflow.set_entry_point("topic_decision")
    
    return workflow.compile()

# UI Components
def init_chat_interface():
    """Initialize chat session state"""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Upload a PDF and ask me anything about it!"}
        ]

def display_chat_history():
    """Render chat messages"""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

def handle_user_input(app):
    """Process user input and update UI"""
    if prompt := st.chat_input("Ask about the document"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("Analyzing..."):
            result = app.invoke({
                "question": prompt,
                "documents": [],
                "grades": [],
                "llm_output": "",
                "on_topic": False,
                "retry_count": 0
            })
        
        display_response(result)

def display_response(result: Dict[str, Any]):
    """Show response and metadata"""
    with st.chat_message("assistant"):
        if result["llm_output"]:
            st.markdown(result["llm_output"])
        else:
            st.warning("No response generated by the LLM.")
        
        with st.expander("View analysis details"):
            st.subheader("Topic Relevance")
            st.write(f"On-topic: {result['on_topic']}")
            
            st.subheader("Document Relevance Scores")
            if result["documents"]:
                st.table({"Document": result["documents"], "Score": result["grades"]})
            else:
                st.warning("No documents retrieved.")

# Main application
def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Document QA Assistant",
        page_icon="📄",
        layout="centered"
    )
    st.title("📄 Intelligent Document Assistant")
    
    # Check ChromaDB initialization
    if not initialize_chromadb():
        st.error("Failed to initialize ChromaDB. Please check your setup.")
        return
    
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file:
        try:
            vector_store = process_pdf(uploaded_file)
            app = create_workflow(vector_store.as_retriever())
            
            init_chat_interface()
            display_chat_history()
            handle_user_input(app)
            
        except Exception as e:
            st.error(f"Error processing request: {str(e)}")

if __name__ == "__main__":
    main()