# 🧠 StudyMate AI — Agentic RAG Quiz & Chat Assistant

**StudyMate AI** is an intelligent study assistant built on an **Agentic RAG (Retrieval-Augmented Generation)** framework using **LangChain**, and **LLMs**. It allows you to upload documents (PDF, PPTX, DOCX), interact with them through chat, and test your understanding via automatically generated MCQs and short answer quizzes — all in one app.

---

## ✨ Features

- 📂 **Multi-format Support**: Upload multiple files — PDF, DOCX, and PPTX
- ⚙️ **Agentic RAG Workflow**:
  - Smart tool selection based on your query:
    - `default_response` → handles greetings, generic questions
    - `summary_generate` → generates a summary from the document
    - `rag_generate` → document-specific answers using vector search
- 🧪 **Interactive Quiz Modes**:
  - 📝 **MCQ Mode**: Auto-generated multiple choice questions (A–D)
  - ✏️ **Short Answer Mode**: Conceptual questions with LLM-validated answers
- 💬 **Chat Mode**: Ask freeform questions about the uploaded content
- 🧠 **Memory Buffer**: Keeps track of recent conversation context

---

## 📘 Agentic Approach to RAG

Traditional RAG systems are powerful, but they often struggle with tasks like:

- Generating document summaries
- Responding to basic conversations (e.g., "hi", "who are you?")
- Managing multi-step interactions

**StudyMate AI** solves this using an *Agentic* architecture:

🧭 A tool-routing agent chooses the correct path:

| Query Type               | Action Taken                     |
|--------------------------|----------------------------------|
| General Chat             | Uses LLM for default response    |
| "Summarize this PDF"     | Activates `summary_generate`     |
| "What is concept X?"     | Uses RAG via `rag_generate`      |

This keeps the chatbot fluent, task-aware, and truly helpful in educational use cases.

---

## 🧪 Quiz Modes (LLM-evaluated)

You can test your learning with two advanced modes:

- **MCQ Mode**:
  - Generates multiple choice questions based on the document
  - Provides explanations with answers

- **Short Answer Mode**:
  - Generates conceptual, definition-style questions
  - Your answers are evaluated using an LLM and scored

Both modes dynamically generate questions as you proceed, with "Next" and "Exit" options for control.

---

## 📁 Supported File Types

- PDF (`.pdf`)
- Word Documents (`.docx`)
- PowerPoint Presentations (`.pptx`)
- Multi-file uploads supported

---

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/your-username/studymate-ai.git
cd studymate-ai
pip install -r requirements.txt
