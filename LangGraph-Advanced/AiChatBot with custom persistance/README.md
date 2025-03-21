# 🤖 Futuristic AI Assistant with Advanced LangGraph & Custom Memory Persistence

Welcome to the **Futuristic AI Assistant**, a state-of-the-art conversational AI powered by **LangGraph**, **LangChain**, **Groq**, **HuggingFace**, and **Streamlit**. This project combines cutting-edge language models with a sleek, modern interface and advanced memory persistence to deliver an unparalleled user experience. Dive into the future of conversational AI with a system that remembers, learns, and adapts to your needs.

---

## ✨ **Features**

- **Advanced LangGraph Integration**: Leverage the power of LangGraph for stateful, context-aware conversations.
- **Custom Memory Persistence**: Built with **FAISS** and **HuggingFace Embeddings**, the AI remembers past interactions for seamless continuity.
- **Groq-Powered LLM**: Utilizes **Groq's Llama3-70b** model for lightning-fast, intelligent responses.
- **Glass Morphism UI**: A stunning, futuristic user interface with glass morphism effects and dynamic animations.
- **Dynamic Context Retrieval**: The AI retrieves relevant context from memory to provide personalized responses.
- **Streamlit-Powered Interface**: A beautifully designed, interactive web app for effortless user interaction.
- **Advanced Prompt Engineering**: Sophisticated prompt design ensures natural, human-like conversations.

---

## 🚀 **Getting Started**

### **Prerequisites**

- Python 3.8+
- Streamlit
- LangChain
- LangGraph
- FAISS
- HuggingFace Transformers
- Groq API Key

### **Installation**

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/futuristic-ai-assistant.git
   cd futuristic-ai-assistant
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables:

   ```bash
   cp .env.example .env
   ```

   Edit the `.env` file to include your **Groq API Key** and **HuggingFace API Key**.

### **Running the Application**

1. Start the Streamlit app:

   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to `http://localhost:8501` to interact with the AI assistant.

---

## 🛠️ **Configuration**

### **Environment Variables**

- `GROQ_API_KEY`: Your Groq API key for accessing the **Llama3-70b** model.
- `HUGGINGFACE_API_KEY`: Your HuggingFace API key for embeddings.

### **Customization**

- **Memory Storage**: Adjust the `db_path` in `custom_memory.py` to change the storage location.
- **UI Styling**: Modify the CSS in `app.py` to customize the appearance.
- **Prompt Engineering**: Tweak the prompt template in `chatbot.py` for specialized use cases.

---

## 📂 **Project Structure**

```bash
futuristic-ai-assistant/
│
├── app.py                  # Main Streamlit application with UI and interactions
├── chatbot.py              # Chatbot logic, LangGraph integration, and state management
├── custom_memory.py        # Custom memory persistence using FAISS and HuggingFace
├── requirements.txt        # List of dependencies
├── .env.example            # Example environment variables
├── README.md               # Project documentation
└── image.png               # Demo screenshot
```

---

## 🤖 **How It Works**

1. **User Input**: The user types a message in the chat interface.
2. **Context Retrieval**: The system retrieves relevant context from **FAISS-based memory** using **HuggingFace embeddings**.
3. **Response Generation**: The **Groq Llama3-70b** model generates a response based on the input and context.
4. **Memory Update**: The interaction is saved to memory for future reference.
5. **UI Update**: The chat interface is updated with the new messages, complete with smooth animations and glass morphism effects.

---

## 🌟 **Highlights**

### **Advanced LangGraph Integration**

- **Stateful Conversations**: LangGraph ensures the AI maintains context across interactions.
- **Custom Workflows**: Define complex conversational workflows with ease.

### **Custom Memory Persistence**

- **FAISS Vector Store**: Efficient, scalable memory storage for quick retrieval.
- **HuggingFace Embeddings**: High-quality embeddings for accurate context matching.

### **Groq-Powered LLM**

- **Llama3-70b**: One of the most advanced language models for natural, human-like responses.
- **Lightning-Fast**: Groq's hardware acceleration ensures near-instant responses.

### **Aesthetic UI**

- **Glass Morphism**: A modern, frosted glass effect for a futuristic look.
- **Dynamic Animations**: Smooth transitions and hover effects for an immersive experience.
- **Interactive Elements**: Buttons, input fields, and chat bubbles designed for usability and beauty.

---

## 📄 **License**

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

- **LangChain & LangGraph**: For providing the framework for building stateful, context-aware conversational AI.
- **Groq**: For their cutting-edge language models and hardware acceleration.
- **HuggingFace**: For embeddings and transformer models.
- **Streamlit**: For the interactive and beautiful web interface.
- **FAISS**: For efficient vector storage and retrieval.

---

## 🌌 **Experience the Future of Conversational AI**

The **Futuristic AI Assistant** is more than just a chatbot—it's a glimpse into the future of human-AI interaction. With advanced memory persistence, stateful conversations, and a stunning interface, this project redefines what's possible with conversational AI.

**Ready to dive in?** Clone the repo, fire up the app, and start chatting with the future! 🚀

---

For any questions, feedback, or contributions, feel free to open an issue or submit a pull request. Let's build the future together! 🌟
