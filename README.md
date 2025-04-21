# 🔥 LangChain x Groq AI Projects & Tools

This repository contains a collection of mini-projects, experiments, and tool integrations that demonstrate the use of **LangChain** with **Groq LLMs** (`llama3-8b-8192`), **FAISS for vector search**, **semantic document understanding**, and external data ingestion via web or file. Each project is modular, well-commented, and ready to be extended or integrated.

---

## 📁 Contents of the Repository

### 🔑 Environment Configuration
- Securely loads **Groq API**, **Google API**, and **ElevenLabs API** keys via environment variables.
- Ensures best practices for key management using `os.getenv()` and `process.env`.

---

### 🧠 LangChain + Groq LLM Integration
- Initializes and runs `ChatGroq` using LLaMA3 model.
- Simple prompt examples using `PromptTemplate`, `LLMChain`, and `SequentialChain`.
- Example: Generate a paragraph about a topic and summarize it.

---

### 🛠️ Tools and Agent Initialization
- Loads LangChain tools like **Wikipedia**.
- Sets up intelligent agents using `initialize_agent` with `ZERO_SHOT_REACT_DESCRIPTION`.
- Sample: Agent answers real-world questions like Elon Musk's birth date.

---

### 📄 URL and File Loading + Chunking
- Handles content loading from:
  - Local `.txt` files
  - Live news article URLs
- Splits large content into chunks using `RecursiveCharacterTextSplitter`.

---

### 🧬 Embedding & Vector Search (FAISS)
- Uses **SentenceTransformers** to create vector embeddings.
- Builds a **FAISS index** for fast similarity search.
- Example: Performs word analogies with vector math.

---

### 💬 Mini Project 01 – Q&A Bot from Web Articles
- Loads articles from URLs and uses FAISS + LangChain to answer user questions.
- Includes a strict fallback if the question isn’t related to the document.
- Combines smart chunking, embedding, search, and LLM response generation.

---

### 💻 Mini Project 02 – MongoDB Atlas Query Assistant (WIP)
- Sets the foundation for an AI assistant that understands natural language and generates MongoDB queries using LangChain and prompt engineering.

---

### 📬 Mini Project 03 – Smart Mail Agent
- An **AI-driven email assistant** that:
  - Connects to Gmail (via Google API).
  - Retrieves unread or filtered emails.
  - Uses Groq-powered LLMs to summarize, prioritize, or take intelligent actions based on email content.
- Designed for productivity and intelligent inbox automation.
- Supports customization for specific use cases like:
  - Prioritizing job-related emails.
  - Summarizing newsletters.
  - Detecting urgent messages based on semantic cues.

---

### 🔊 Voice-based LLM Integration (JavaScript)
- Example code to run voice-based LLM calls.
- Loads API keys (`GROQ`, `ElevenLabs`) securely via `process.env`.
- VoiceID setup included for ElevenLabs.

---

## 📦 Requirements

- Python 3.8+
- Install dependencies:
  ```bash
  pip install -r requirements.txt


  
> ⚠️ **Note: This repository is currently a Work In Progress (WIP).**
> 
> While the base functionality works, things might be rough around the edges!  
> I'm constantly building, updating, cleaning, and improving.
> 
> **Contributions, suggestions, bug fixes, and improvements are always welcome!**
> If you have an idea or improvement in mind, feel free to jump in.  
> This is a learning journey — let’s grow it together.
>
> Hope it helps. Stay Hard! 💪

---
