import re
import time


def generate_mcq_questions(chunks: list, llm, total_mcq_limit: int = 10, start_idx = 0):
    mcqs = []
    total_questions_generated = 0

    for i in range(start_idx, len(chunks)):
        chunk = chunks[i]
        remaining = total_mcq_limit - total_questions_generated
        if remaining <= 0:
            break  # We've generated enough

        questions_this_chunk = min(2, remaining)
        print(f"\n📄 Chunk {i+1} — generating {questions_this_chunk} questions")
        prompt = f"""
        You are an expert education assistant helping students prepare for exams.

        From the passage below, generate {questions_this_chunk} high-quality multiple-choice questions (MCQs). 
        Each question should be either:
        - a **definition-based question** (asking what a term means), or
        - a **conceptual question** (testing understanding of ideas or logic).

        Instructions:
        - Each question must have **4 options (A–D)**.
        - Clearly mark the correct answer.
        - Provide a **brief explanation** of why the correct option is right.
        - Avoid trivial or vague questions. Focus on meaningful content.

        Passage:
        {chunk.page_content}

        Format:
        Q1: <question>
        A. Option A
        B. Option B
        C. Option C
        D. Option D
        Answer: <correct letter>
        Explanation: <why this is the correct answer>
        """

        try:
            response = llm.invoke(prompt)
            result = response.content if hasattr(response, "content") else response
            print("🧠 MCQs generated:\n", result)
            mcqs.append(result)

            count = result.count("Q")
            total_questions_generated += count

        except Exception as e:
            print(f"⚠️ Error generating MCQs for chunk {i}: {e}")

        time.sleep(1.5)  # Token rate-safe pause

    return "\n\n".join(mcqs), i + 1


def parse_mcqs(text):
    questions = re.split(r'Q\d+:', text)[1:]
    parsed = []
    for q in questions:
        lines = q.strip().split("\n")
        question = lines[0].strip()
        options = {l[0]: l[3:].strip() for l in lines[1:5]}
        answer = [l for l in lines if l.startswith("Answer:")][0].split(":")[1].strip()
        explanation = [l for l in lines if l.startswith("Explanation:")][0].split(":", 1)[1].strip()
        parsed.append({"question": question, "options": options, "answer": answer, "explanation": explanation})
    return parsed


