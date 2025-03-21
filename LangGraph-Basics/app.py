import streamlit as st
from PyPDF2 import PdfReader
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from langchain_core.pydantic_v1 import BaseModel, Field
import tempfile
import os
from dotenv import load_dotenv
from chromadb.config import Settings as ChromaSettings  # Add this import

# Load environment variables
load_dotenv()

# Initialize embedding function
embedding_function = HuggingFaceEmbeddings()

# Define the AgentState
class AgentState(TypedDict):
    question: str
    grades: list[str]
    llm_output: str
    documents: list[str]
    on_topic: bool

# Define the GradeQuestion model
class GradeQuestion(BaseModel):
    score: str = Field(description="Question is relevant to the document? If yes -> 'Yes' if not -> 'No'")

# Define the GradeDocuments model
class GradeDocuments(BaseModel):
    score: str = Field(description="Documents are relevant to the question, 'Yes' or 'No'")

# Define the workflow functions
def retrieve_docs(state: AgentState):
    question = state['question']
    documents = retriever.get_relevant_documents(query=question)
    state['documents'] = [doc.page_content for doc in documents]
    return state

def question_classifier(state: AgentState):
    question = state["question"]
    system = """You are a grader assessing whether a user's question is relevant to the content of a document. \n
        Your task is to determine if the question is related to the topics covered in the document. \n
        If the question IS relevant to the document's content, respond with "Yes". Otherwise, respond with "No".
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
    on_topic = state["on_topic"]
    if on_topic == "Yes":
        return "on_topic"
    return "off_topic"

def off_topic_response(state: AgentState):
    state["llm_output"] = "I can't respond to that!"
    return state

def document_grader(state: AgentState):
    document = state["documents"]
    question = state["question"]
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n
        If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
        Give a binary score 'Yes' or 'No' score to indicate whether the document is relevant to the question."""

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Retrieved document: \n\n {document} \n\n User question: {question}",
            ),
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
    grades = state["grades"]
    if any(grade.lower() == "yes" for grade in grades):
        return "generate"
    else:
        return "rewrite_query"

def rewriter(state: AgentState):
    question = state["question"]
    system = """You a question re-writer that converts an input question to a better version that is optimized \n
        for retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ]
    )
    llm = ChatGroq(model_name="llama3-70b-8192", temperature=0)
    question_rewriter = re_write_prompt | llm | StrOutputParser()
    output = question_rewriter.invoke({"question": question})
    state["question"] = output
    return state

def generate_answer(state: AgentState):
    llm = ChatGroq(model_name="llama3-70b-8192", temperature=0)
    question = state["question"]
    context = state["documents"]

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(
        template=template,
    )

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question, "context": context})
    state["llm_output"] = result
    return state

# Build the workflow
workflow = StateGraph(AgentState)

workflow.add_node("topic_decision", question_classifier)
workflow.add_node("off_topic_response", off_topic_response)
workflow.add_node("retrieve_docs", retrieve_docs)
workflow.add_node("rewrite_query", rewriter)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("document_grader", document_grader)

workflow.add_edge("off_topic_response", END)
workflow.add_edge("retrieve_docs", "document_grader")
workflow.add_conditional_edges(
    "topic_decision",
    on_topic_router,
    {
        "on_topic": "retrieve_docs",
        "off_topic": "off_topic_response",
    },
)
workflow.add_conditional_edges(
    "document_grader",
    gen_router,
    {
        "generate": "generate_answer",
        "rewrite_query": "rewrite_query",
    },
)
workflow.add_edge("rewrite_query", "retrieve_docs")
workflow.add_edge("generate_answer", END)

workflow.set_entry_point("topic_decision")

app_workflow = workflow.compile()

# Initialize retriever
retriever = None

# Streamlit UI
st.title("PDF Question Answering System")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file is not None:
    # Extract text from PDF
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    # Create documents
    docs = [Document(page_content=text, metadata={"source": uploaded_file.name})]

    # Initialize Chroma with explicit client settings for local persistence
    client_settings = ChromaSettings(
        chroma_db_impl="duckdb+parquet",  # Use DuckDB with Parquet for local storage
        persist_directory=".chroma",  # Directory to store the Chroma database
        anonymized_telemetry=False  # Disable telemetry
    )

    # Initialize Chroma vector store
    db = Chroma.from_documents(
        documents=docs,
        embedding=embedding_function,
        client_settings=client_settings,
        collection_name="pdf_collection",  # Name of the collection
        persist_directory=".chroma"  # Persist data to disk
    )
    retriever = db.as_retriever()

    st.success("PDF uploaded and processed successfully!")

# Ask a question
if retriever is not None:
    question = st.text_input("Ask a question about the PDF:")
    if question:
        state = {"question": question}
        result = app_workflow.invoke(state)
        st.write("Answer:")
        st.write(result["llm_output"])
else:
    st.warning("Please upload a PDF first.")