import os.path
import traceback

import streamlit as st
from data.loader import loader
from pathlib import Path
from main_rag import process_query
import time
from quiz.generate_mcq import generate_mcq_questions, parse_mcqs
from quiz.generate_short import generate_short_questions, parse_short_answers, evaluate_short_answer
from rag_initializer import get_llm, get_chunk
import tempfile


def init_quiz_state():
    defaults = {
        "messages": [],
        "quiz_mode": None,
        "quiz_questions": [],
        "current_q_index": 0,
        "quiz_score": 0,
        "total_answered": 0,
        "quiz_active": False,
        "chunk_index": 0
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def main():
    st.set_page_config(page_title="Chatbot", layout="centered")
    st.title("🧠 RAG Chatbot")

    try:
        init_quiz_state()
        # --- Upload section ---
        st.sidebar.header("📄 Upload PDFs")
        uploaded_files = st.sidebar.file_uploader(
            "Upload one or more files",
            type=["pdf", "docx", "pptx"],
            accept_multiple_files=True
        )
        if "uploaded_pdf_paths" not in st.session_state:
            with st.expander("📘 How to use this app", expanded=True):
                st.markdown("""
                - 📤 **Upload** one or more PDF, DOCX, or PPTX files using the sidebar.
                - 🧠 Use **MCQ** or **Short Answer** mode to test your knowledge.
                - 🗣️ Ask **follow-up questions or generate summary** about the content in chat mode.
                - 🔁 You can **exit the quiz** anytime and switch to chatting.
                - 🚫 Uploaded files are **not stored** and are deleted after session ends.
                - 🔄 For best results, we recommend uploading individual chapters rather than the entire book.
                - 📤 If you do choose to upload the full book, consider removing the introductory pages to ensure higher-quality quiz questions.
                """)

        if uploaded_files and "uploaded_pdf_paths" not in st.session_state:
            st.cache_resource.clear()
            for key in ["documents", "vector_store", "chunked_docs"]:
                st.session_state.pop(key, None)

            temp_paths = []
            for uploaded_file in uploaded_files:
                suffix = os.path.splitext(uploaded_file.name)[-1]
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                temp_file.write(uploaded_file.read())
                temp_file.flush()
                temp_paths.append(temp_file.name)

            st.session_state.uploaded_pdf_paths = temp_paths

            # Load only once after upload

            with st.spinner("Reading the document content..."):
                docs = loader()
                st.session_state.documents = docs

        # --- Guard: If no documents yet, don't show input ---
        if "documents" not in st.session_state:
            st.info("👋 Please upload PDF(s) to begin chatting.")
            st.stop()

        # --- Chat CSS & History ---
        st.markdown("""<style>
        .chat-bubble {
            padding: 12px 16px;
            border-radius: 16px;
            margin: 10px 0;
            max-width: 60%;
            line-height: 1.5;
            font-size: 16px;
            word-wrap: break-word;
            display: inline-block;
            clear: both;
        }
        .bot {
            background-color: #f0f0f0;
            color: #000;
            text-align: left;
            float: left;
            margin-left: 10px;
        }
        .user {
            background-color: #cce5df;
            color: #000;
            text-align: right;
            float: right;
            margin-right: 10px;
        }
        </style>""", unsafe_allow_html=True)

        # --- Start Quiz Buttons ---
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📝 MCQ Mode"):
                st.session_state.quiz_mode = "mcq"
                st.session_state.quiz_score = 0
                st.session_state.total_answered = 0
                st.session_state.current_q_index = 0
                st.session_state.chunk_index = 0
                st.session_state.quiz_active = True
                with st.spinner("Generating questions..."):
                    raw_qas, next_chunk_index = generate_mcq_questions(
                        get_chunk(), get_llm(), total_mcq_limit=2, start_idx=st.session_state.chunk_index
                    )
                    st.session_state.chunk_index = next_chunk_index
                    st.session_state.quiz_questions = parse_mcqs(raw_qas)
                st.rerun()

        with col2:
            if st.button("✏️ Short Answer Mode"):
                st.session_state.quiz_mode = "short"
                st.session_state.quiz_score = 0
                st.session_state.total_answered = 0
                st.session_state.current_q_index = 0
                st.session_state.chunk_index = 0
                st.session_state.quiz_active = True
                with st.spinner("Generating questions..."):
                    raw_qas, next_chunk_index = generate_short_questions(
                        get_chunk(), get_llm(), total_question_limit=2, start_idx=st.session_state.chunk_index
                    )
                    st.session_state.chunk_index = next_chunk_index
                    st.session_state.quiz_questions = parse_short_answers(raw_qas)
                st.rerun()

        # --- Exit Quiz ---
        if st.session_state.quiz_active:
            if st.button("❌ Exit Quiz"):
                score = st.session_state.quiz_score
                total = st.session_state.total_answered or 1
                st.success(f"Quiz Ended. Final Score: {score}/{total}")
                for key in ["quiz_mode", "quiz_questions", "current_q_index", "quiz_score", "quiz_active", "total_answered", "chunk_index"]:
                    st.session_state[key] = None if key == "quiz_mode" else 0 if key in ["quiz_score", "current_q_index", "total_answered", "chunk_index"] else []
                st.rerun()

        # --- Dynamic Question Generator ---
        def ensure_next_questions():
            remaining = len(st.session_state.quiz_questions) - st.session_state.current_q_index
            if remaining < 2:
                with st.spinner("Preparing more questions..."):
                    if st.session_state.quiz_mode == "mcq":
                        raw_qas, next_chunk_index = generate_mcq_questions(
                            get_chunk(), get_llm(), total_mcq_limit=2, start_idx=st.session_state.chunk_index
                        )
                        st.session_state.chunk_index = next_chunk_index
                        new_qs = parse_mcqs(raw_qas)
                    else:
                        raw_qas, next_chunk_index = generate_short_questions(
                            get_chunk(), get_llm(), total_question_limit=2, start_idx=st.session_state.chunk_index
                        )
                        st.session_state.chunk_index = next_chunk_index
                        new_qs = parse_short_answers(raw_qas)
                    st.session_state.quiz_questions.extend(new_qs)

        if st.session_state.quiz_active:
            q_idx = st.session_state.current_q_index
            quiz = st.session_state.quiz_questions

            if q_idx < len(quiz):
                q = quiz[q_idx]
                st.subheader(f"Q{q_idx + 1}: {q['question']}")
                answered_key = f"answered_{q_idx}"

                if st.session_state.quiz_mode == "mcq":
                    options_dict = q['options']
                    option_keys = list(options_dict.keys())

                    selected = st.radio(
                        "Options",
                        options=option_keys,
                        format_func=lambda key: f"{key}. {options_dict[key]}",
                        key=f"mcq_radio_{q_idx}"
                    )

                    if answered_key not in st.session_state and st.button("Submit Answer"):
                        is_correct = selected == q['answer']
                        if is_correct:
                            st.success("✅ Correct!")
                            st.session_state.quiz_score += 1
                        else:
                            st.error(f"❌ Incorrect. Correct answer: {q['answer']}")
                        st.info(f"💡 Explanation: {q['explanation']}")
                        st.session_state.total_answered += 1
                        st.session_state[answered_key] = True
                        st.rerun()

                    if answered_key in st.session_state:
                        is_correct = selected == q['answer']
                        if is_correct:
                            st.success("✅ Correct!")
                        else:
                            st.error(f"❌ Incorrect. Correct answer: {q['answer']}")
                        st.info(f"💡 Explanation: {q['explanation']}")

                elif st.session_state.quiz_mode == "short":
                    user_ans = st.text_input("Your Answer", key=f"short_ans_{q_idx}")

                    if answered_key not in st.session_state and st.button("Submit Answer"):
                        correct_answer = q["answer"]
                        eval_result = evaluate_short_answer(q['question'], user_ans, correct_answer, get_llm())
                        st.session_state[answered_key] = {
                            "eval_result": eval_result,
                            "user_ans": user_ans
                        }
                        if eval_result["verdict"] == "Correct":
                            st.session_state.quiz_score += 1
                        st.session_state.total_answered += 1
                        st.rerun()

                    if answered_key in st.session_state:
                        result = st.session_state[answered_key]
                        eval_result = result["eval_result"]

                        if eval_result["verdict"] == "Correct":
                            st.success("✅ Correct (LLM validated)!")
                        else:
                            st.error(f"❌ Incorrect. Ideal answer: {q['answer']}")
                        st.info(f"💡 Explanation: {eval_result['explanation']}")

                # Always show Next Question (for skipping or after answer)
                if st.button("Next Question"):
                    st.session_state.current_q_index += 1
                    if answered_key in st.session_state:
                        del st.session_state[answered_key]
                    ensure_next_questions()
                    st.rerun()

            else:
                st.success(f"🎉 Quiz Complete! Final Score: {st.session_state.quiz_score}/{st.session_state.total_answered}")
                if st.button("Restart"):
                    for key in ["quiz_mode", "quiz_questions", "current_q_index", "quiz_score", "quiz_active", "total_answered", "chunk_index"]:
                        st.session_state[key] = None if key == "quiz_mode" else 0 if isinstance(st.session_state[key], int) else []
                    st.rerun()

        # --- Chat Mode ---
        if not st.session_state.quiz_active:

            for msg in st.session_state.messages:
                role_class = "user" if msg["role"] == "user" else "bot"
                st.markdown(
                    f'<div class="chat-bubble {role_class}">{msg["content"]}</div>',
                    unsafe_allow_html=True
                )

            if prompt := st.chat_input("Type your question here..."):
                st.session_state.messages.append({"role": "user", "content": prompt})

                with st.spinner("Thinking..."):
                    response = process_query(prompt)

                st.session_state.messages.append({"role": "bot", "content": response})
                st.rerun()

    except Exception:
        st.error("🚨 An unexpected error occurred.")
        st.text(traceback.format_exc())


if __name__ == "__main__":
    main()