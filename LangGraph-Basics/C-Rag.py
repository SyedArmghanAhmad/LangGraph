import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Initialize embeddings
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings()

# PDF Processing
def process_pdf(uploaded_file):
    """Process uploaded PDF file and create vector store"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        loader = PyPDFLoader(tmp.name)
        pages = loader.load_and_split()
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(pages)
    
    # Use in-memory ChromaDB
    return Chroma.from_documents(splits, get_embeddings(), persist_directory=None)

# State definition
class AgentState(TypedDict):
    question: str
    documents: list[str]
    grades: list[str]
    llm_output: str
    on_topic: bool

# Core workflow components

class GradeQuestion(BaseModel):
    """Boolean value to check whether a question is related to the restaurant Bella Vista"""
    score: str = Field(description="Question is about restaurant? If yes -> 'Yes' if not -> 'No'")

def question_classifier(state: AgentState):
    """Classify if the question is on-topic"""
    question = state["question"]
    system = """You are a grader assessing the relevance of a retrieved document to a user question. \n
        Only answer if the question is about one of the following topics:
        1. Information about the owner of Bella Vista (Antonio Rossi).
        2. Prices of dishes at Bella Vista.
        3. Opening hours of Bella Vista.
        4. Available menus at Bella Vista.

        If the question IS about these topics response with "Yes", otherwise respond with "No".
        """
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: {question}"),
        ]
    )
    llm = ChatGroq(model_name="llama3-70b-8192", temperature=0)
    structured_llm = llm.with_structured_output(GradeQuestion)
    grader_llm = grade_prompt | structured_llm
    result = grader_llm.invoke({"question": question})
    state["on_topic"] = result.score
    return state

def on_topic_router(state: AgentState):
    """Route based on whether the question is on-topic"""
    on_topic = state["on_topic"]
    if on_topic == "Yes":
        return "on_topic"
    return "off_topic"

def off_topic_response(state: AgentState):
    """Handle off-topic questions"""
    state["llm_output"] = "I can't respond to that!"
    return state

class GradeDocuments(BaseModel):
    """Boolean values to check for relevance on retrieved documents."""
    score: str = Field(description="Documents are relevant to the question, 'Yes' or 'No'")

def document_grader(state: AgentState):
    """Grade the relevance of retrieved documents"""
    document = state["documents"]
    question = state["question"]
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n
        If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
        Give a binary score 'Yes' or 'No' score to indicate whether the document is relevant to the question."""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )
    llm = ChatGroq(model_name="llama3-70b-8192", temperature=0)
    structured_llm = llm.with_structured_output(GradeDocuments)
    grader_llm = grade_prompt | structured_llm
    scores = []
    for doc in document:
        result = grader_llm.invoke({"document": doc, "question": question})
        scores.append(result.score)
    state["grades"] = scores
    return state

def gen_router(state: AgentState):
    """Route based on document grading results"""
    grades = state["grades"]
    if any(grade.lower() == "yes" for grade in grades):
        return "generate"
    else:
        return "rewrite_query"

def rewriter(state: AgentState):
    """Rewrite the question for better retrieval"""
    question = state["question"]
    system = """You a question re-writer that converts an input question to a better version that is optimized \n
        for retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
        ]
    )
    llm = ChatGroq(model_name="llama3-70b-8192", temperature=0)
    question_rewriter = re_write_prompt | llm | StrOutputParser()
    output = question_rewriter.invoke({"question": question})
    state["question"] = output
    return state

def generate_answer(state: AgentState):
    """Generate the final answer based on retrieved documents"""
    llm = ChatGroq(model_name="llama3-70b-8192", temperature=0)
    question = state["question"]
    context = state["documents"]
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template=template)
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question, "context": context})
    state["llm_output"] = result
    return state
def retrieve_docs(state: AgentState, retriever):  # Add retriever as parameter
    """Retrieve relevant documents based on the question"""
    question = state['question']
    documents = retriever.get_relevant_documents(query=question)
    state['documents'] = [doc.page_content for doc in documents]
    return state

# Initialize workflow
def initialize_workflow(retriever):  # Accept retriever as argument
    """Initialize LangGraph workflow"""
    workflow = StateGraph(AgentState)
    
    # Pass retriever to retrieve_docs using lambda
    workflow.add_node("retrieve_docs", lambda state: retrieve_docs(state, retriever))
    
    # Rest of the workflow setup remains the same
    workflow.add_node("topic_decision", question_classifier)
    workflow.add_node("off_topic_response", off_topic_response)
    workflow.add_node("rewrite_query", rewriter)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("document_grader", document_grader)

    workflow.add_edge("off_topic_response", END)
    workflow.add_edge("retrieve_docs", "document_grader")
    workflow.add_conditional_edges(
        "topic_decision",
        on_topic_router,
        {"on_topic": "retrieve_docs", "off_topic": "off_topic_response"},
    )
    workflow.add_conditional_edges(
        "document_grader",
        gen_router,
        {"generate": "generate_answer", "rewrite_query": "rewrite_query"},
    )
    workflow.add_edge("rewrite_query", "retrieve_docs")
    workflow.add_edge("generate_answer", END)
    workflow.set_entry_point("topic_decision")
    return workflow.compile()

# Streamlit UI
def main():
    st.title("📄 Document QA Assistant")
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    
    if uploaded_file:
        # Process PDF and create retriever
        vector_store = process_pdf(uploaded_file)
        retriever = vector_store.as_retriever()
        
        # Initialize workflow WITH the retriever
        app = initialize_workflow(retriever)
        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if prompt := st.chat_input("Ask about the document"):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Run workflow
            result = app.invoke({
                "question": prompt,
                "documents": [],
                "grades": [],
                "llm_output": "",
                "on_topic": False
            })
            
            # Display results
            with st.chat_message("assistant"):
                st.markdown(result["llm_output"])
                
                with st.expander("Retrieved Context"):
                    for doc in result["documents"]:
                        st.write(doc)
                        
                with st.expander("Grading Results"):
                    st.write(f"On Topic: {result['on_topic']}")
                    st.write("Document Scores:", result["grades"])
            
            # Add assistant response
            st.session_state.messages.append({
                "role": "assistant", 
                "content": result["llm_output"]
            })

if __name__ == "__main__":
    main()