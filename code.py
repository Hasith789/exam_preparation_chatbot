import streamlit as st
import requests
import os
import time
import random
from dotenv import load_dotenv
from io import StringIO

load_dotenv()


QA_KEY = os.getenv("HUGGINGFACE_QA_KEY")
SUMMARIZE_KEY = os.getenv("HUGGINGFACE_SUMMARY_KEY")

if not all([QA_KEY, SUMMARIZE_KEY]):
    st.error(" One or more API keys are missing. Check your .env file.")
    st.stop()

qa_headers = {"Authorization": f"Bearer {QA_KEY}"}
sum_headers = {"Authorization": f"Bearer {SUMMARIZE_KEY}"}

def hf_api_with_retries(url, headers, payload, retries=3, delay=5):
    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise e

def generate_normal_questions(text):
    sentences = text.split(". ")
    return [f"What is the meaning of '{s.strip()}?'" for s in sentences if len(s.split()) > 5]

def generate_mcq_questions(text):
    sentences = text.split(". ")
    mcqs = []
    for sentence in sentences:
        if len(sentence.split()) > 5:
            correct_answer = sentence.strip()
            distractors = random.sample([s for s in sentences if s != sentence and len(s.split()) > 5], k=min(3, len(sentences)-1))
            options = [correct_answer] + distractors
            random.shuffle(options)
            question = f"What is described by the following statement?\n'{sentence.strip()}'"
            labels = ['A', 'B', 'C', 'D']
            labeled = {labels[i]: options[i] for i in range(len(options))}
            mcqs.append((question, labeled, correct_answer))
    return mcqs

def extract_flashcards(text):
    flashcards = []
    for sentence in text.split(". "):
        sentence = sentence.strip()
        if len(sentence.split()) > 5:
            if " is " in sentence:
                term = sentence.split(" is ", 1)[0].strip()
                flashcards.append((f"What is {term}?", sentence))
            elif " are " in sentence:
                term = sentence.split(" are ", 1)[0].strip()
                flashcards.append((f"What are {term}?", sentence))
    return flashcards

def clean_manual_flashcards(text):
    lines = text.strip().split('\n')
    cleaned, term, definition = [], "", ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if ':' in line and line.index(':') < 40:
            if term and definition:
                cleaned.append(f"{term.strip()}: {definition.strip()}")
            term, definition = line.split(":", 1)
        elif len(line.split()) <= 6 and line.istitle():
            if term and definition:
                cleaned.append(f"{term.strip()}: {definition.strip()}")
            term = line
            definition = ""
        else:
            definition += " " + line
    if term and definition:
        cleaned.append(f"{term.strip()}: {definition.strip()}")
    return cleaned

st.set_page_config(page_title=" Exam Preparation Chatbot", layout="centered")
st.title(" Exam Preparation Chatbot")

st.markdown("""
Choose what you want to do:
-  Ask study questions
-  Summarize your notes
-  Generate questions
-  Flashcards (Q&A style)
""")

option = st.sidebar.selectbox(" Choose a feature", [
    "Ask a Doubt", "Summarize Notes", "Generate Questions", "Flashcard Generator"])

if option == "Ask a Doubt":
    context = st.text_area(" Paste your study material (context):", height=200)
    question = st.text_input(" Type your question:")
    if st.button("Get Answer"):
        if context and question:
            with st.spinner(" Thinking..."):
                payload = {"inputs": {"question": question, "context": context}}
                answer = None
                try:
                    response = hf_api_with_retries(
                        "https://api-inference.huggingface.co/models/bert-large-uncased-whole-word-masking-finetuned-squad",
                        headers=qa_headers, payload=payload)
                    result = response.json()
                    answer = result.get("answer", "").strip()
                except Exception:
                    pass

                if answer:
                    st.success(" Answer:")
                    st.write(answer)
                else:
                    
                    sentences = context.split(". ")
                    best_sentence = ""
                    max_overlap = 0
                    for s in sentences:
                        overlap = len(set(s.lower().split()) & set(question.lower().split()))
                        if overlap > max_overlap:
                            max_overlap = overlap
                            best_sentence = s
                    fallback_answer = best_sentence.strip() if best_sentence else "Sorry, I couldn't find a good answer."
                    st.success(" Answer:")
                    st.write(fallback_answer)
        else:
            st.warning("Please provide both context and question.")


elif option == "Summarize Notes":
    text = st.text_area(" Paste your notes:", height=250)
    if st.button("Summarize"):
        if text:
            with st.spinner(" Summarizing..."):
                models = ["facebook/bart-large-cnn", "sshleifer/distilbart-cnn-12-6"]
                payload = {"inputs": text}
                summary = None

                for model in models:
                    try:
                        response = requests.post(
                            f"https://api-inference.huggingface.co/models/{model}",
                            headers=sum_headers, json=payload, timeout=60)
                        result = response.json()
                        if isinstance(result, list) and "summary_text" in result[0]:
                            summary = result[0]["summary_text"]
                            break
                    except Exception:
                        time.sleep(1)
                        continue

                        
                if summary:
                    st.success(" Summary:")
                    st.write(summary)
                else:
                    
                    sentences = text.split(". ")
                    ranked_sentences = sorted(sentences, key=lambda s: len(s.split()), reverse=True)
                    fallback_summary = ". ".join(ranked_sentences[:3])
                    if not fallback_summary.endswith("."):
                        fallback_summary += "."
                    st.success(" Summary:")
                    st.write(fallback_summary)
        else:
            st.warning("Please paste some text.")



elif option == "Generate Questions":
    text = st.text_area(" Paste a topic or notes:", height=250)
    question_type = st.radio("Question Type", ["Normal Questions", "Multiple Choice"])
    if st.button("Generate Questions"):
        if text:
            st.write("Generating questions...")
            if question_type == "Normal Questions":
                questions = generate_normal_questions(text)
                for idx, q in enumerate(questions):
                    st.markdown(f"**Q{idx+1}:** {q}")
            else:
                mcqs = generate_mcq_questions(text)
                for idx, (q, opts, ans) in enumerate(mcqs):
                    st.markdown(f"**Q{idx+1}:** {q}")
                    for label, option in opts.items():
                        st.write(f"{label}. {option}")
                    correct_label = [l for l, o in opts.items() if o == ans][0]
                    st.write(f"**Correct Answer:** {correct_label}) {ans}")
        else:
            st.warning("Please provide input text.")


elif option == "Flashcard Generator":
    flashcard_mode = st.radio("Choose Flashcard Mode:", [" Manual (Term: Definition)", " Auto Extract (Q&A Flashcards)"])
    flashcard_text = st.text_area("📘 Paste your notes or topics:", height=250)
    
    # Generate Flashcards Button
    if st.button("Generate Flashcards"):
        if flashcard_text:
            with st.spinner(" Generating flashcards..."):
                flashcards = []
                if flashcard_mode == " Manual (Term: Definition)":
                    cleaned = clean_manual_flashcards(flashcard_text)
                    for item in cleaned:
                        if ':' in item:
                            term, definition = item.split(":", 1)
                            flashcards.append((f"What is {term.strip()}?", definition.strip()))
                else:
                    flashcards = extract_flashcards(flashcard_text)

                if flashcards:
                    st.success(" Interactive Flashcards:")
                    for idx, (q, a) in enumerate(flashcards):
                        with st.expander(f" Q{idx+1}: {q}"):
                            st.markdown(f"**Answer:** {a}")
                else:
                    st.warning(" No flashcards created. Please check the formatting or content.")
        else:
            st.warning("Please enter some notes or topic content.")

