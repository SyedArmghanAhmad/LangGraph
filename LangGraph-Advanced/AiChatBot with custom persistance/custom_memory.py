import os
from typing import List, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class ChatMemory:
    def __init__(self, db_path: str = "faiss_memory"):
        self.db_path = db_path
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = self._load_memory()

    def _load_memory(self) -> FAISS:
        """Load or create FAISS index"""
        try:
            if os.path.exists(self.db_path):
                print("📂 Loading existing memory...")
                return FAISS.load_local(
                    self.db_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            print("🆕 Creating new memory...")
            return FAISS.from_texts(["Start of conversation"], self.embeddings)
        except Exception as e:
            print(f"❌ Error loading memory: {e}")
            return FAISS.from_texts(["Start of conversation"], self.embeddings)

    def save_memory(self, session_id: str, message: str, response: str):
        """Save interaction to memory"""
        try:
            text = f"Session: {session_id}\nUser: {message}\nBot: {response}"
            metadata = {"session_id": session_id, "type": "interaction"}
            self.vectorstore.add_texts([text], metadatas=[metadata])
            self.vectorstore.save_local(self.db_path)
            print(f"💾 Saved memory for session {session_id}")
        except Exception as e:
            print(f"❌ Error saving memory: {e}")

    def get_memory(self, query: str, session_id: Optional[str] = None, k: int = 10) -> List[str]:
        """Retrieve relevant memories"""
        try:
            # Filter memories by session ID if provided
            if session_id:
                query = f"Session: {session_id}\n{query}"
            docs = self.vectorstore.similarity_search(query, k=k)
            return [doc.page_content for doc in docs]
        except Exception as e:
            print(f"❌ Error retrieving memory: {e}")
            return []

if __name__ == "__main__":
    # Test memory functionality
    test_memory = ChatMemory(db_path="test_memory")
    test_memory.save_memory("test_session", "Hello", "Hi! How can I help?")
    print("Test retrieval:", test_memory.get_memory("greeting", session_id="test_session"))