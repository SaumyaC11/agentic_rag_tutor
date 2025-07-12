import re
import time


def generate_short_questions(chunks: list, llm, total_question_limit: int = 10, start_idx: int = 0):
    short_qas = []
    total_questions_generated = 0

    for i in range(start_idx, len(chunks)):
        chunk = chunks[i]
        remaining = total_question_limit - total_questions_generated
        if remaining <= 0:
            break

        questions_this_chunk = min(2, remaining)
        print(f"\n📄 Chunk {i+1} — generating {questions_this_chunk} short questions")
        prompt = f"""
        You are an expert teacher helping students prepare for exams.

        From the following academic passage, generate {questions_this_chunk} conceptual or definition-based short-answer questions that test important ideas, terms, or reasoning.
        
        Each question should focus on:
        - Key definitions
        - Core concepts
        - Important ideas explained in the passage
        
        For each question, provide:
        - The question (in clear academic style)
        - The ideal answer (1–2 sentences)
        - A brief explanation of why the answer is correct
        
        Passage:
        {chunk.page_content}
        
        Format:
        Q1: <question>
        Answer: <ideal answer>
        Explanation: <why this is correct>

        """

        try:
            response = llm.invoke(prompt)
            result = response.content if hasattr(response, "content") else response
            print("✍️ Short QAs generated:\n", result)
            short_qas.append(result)

            count = result.count("Q")
            total_questions_generated += count

        except Exception as e:
            print(f"⚠️ Error generating short-answer questions for chunk {i}: {e}")

        time.sleep(1.5)  # Respect Groq TPM

    return "\n\n".join(short_qas), i + 1


def parse_short_answers(text):
    questions = re.split(r'Q\d+:', text)[1:]
    parsed = []
    for q in questions:
        lines = q.strip().split("\n")
        question = lines[0].strip()
        answer = [l for l in lines if l.startswith("Answer:")][0].split(":", 1)[1].strip()
        explanation = [l for l in lines if l.startswith("Explanation:")][0].split(":", 1)[1].strip()
        parsed.append({"question": question, "answer": answer, "explanation": explanation})
    return parsed


def evaluate_short_answer(question, user_ans, ideal_ans, llm):
    eval_prompt = f"""
    Question: {question}
    Ideal Answer: {ideal_ans}
    Student Answer: {user_ans}
    
    Evaluate if the student's answer is correct or not. Respond with 'Correct' or 'Incorrect' and a short explanation.
    """
    response = llm.invoke(eval_prompt).content
    if response.lower().startswith("correct"):
        return {"verdict": "Correct", "explanation": response}
    elif response.lower().startswith("incorrect"):
        return {"verdict": "Incorrect", "explanation": response}
    else:
        return {"verdict": "Unclear", "explanation": response}

