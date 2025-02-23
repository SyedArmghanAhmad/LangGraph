import os
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel

class ChatMemory:
    def __init__(self, db_path: str = "faiss_memory"):
        self.db_path = db_path
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = self._load_memory()

    def _load_memory(self):
        """Load or create FAISS index"""
        if os.path.exists(self.db_path):
            return FAISS.load_local(
                self.db_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        return FAISS.from_texts(["Start of conversation"], self.embeddings)

    def save_memory(self, session_id: str, message: str, response: str):
        """Save interaction to memory"""
        text = f"User: {message}\nBot: {response}"
        self.vectorstore.add_texts([text])
        self.vectorstore.save_local(self.db_path)
        print(f"💾 Saved memory for session {session_id}")

    def get_memory(self, query: str, k: int = 3) -> List[str]:
        """Retrieve relevant memories"""
        return [doc.page_content 
                for doc in self.vectorstore.similarity_search(query, k=k)]

if __name__ == "__main__":
    # Test with clear separation
    test_memory = ChatMemory(db_path="test_memory")
    test_memory.save_memory("test", "Hello", "Hi! How can I help?")
    print("Test retrieval:", test_memory.get_memory("greeting"))