import streamlit as st
st.set_page_config(page_title="Breast Cancer Detection Bot", page_icon="🩺")

import pandas as pd
import requests
import json
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import os
from datetime import datetime
from fpdf import FPDF

# === Title ===
st.title("🩺 Breast Cancer Detection Support Chatbot")

# === API Key Input ===
api_key = st.sidebar.text_input("Enter your OpenRouter API Key", type="password")

# === Mode Selection ===
mode = st.sidebar.radio("Choose mode", ["Detection Support", "General Chat"])

# === LLaMA Function ===
YOUR_SITE_URL = "https://your-site.com"
YOUR_SITE_NAME = "BreastCancerChatbot"

def query_llama3(messages):
    if not api_key:
        st.error("Please enter your OpenRouter API key in the sidebar.")
        st.stop()

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": YOUR_SITE_URL,
        "X-Title": YOUR_SITE_NAME,
    }
    payload = {
        "model": "meta-llama/llama-3.3-8b-instruct:free",
        "messages": messages,
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code != 200:
        raise Exception(f"Error {response.status_code}: {response.text}")
    return response.json()["choices"][0]["message"]["content"]

# === Load Memory ===
@st.cache_resource
def load_case_memory():
    data = [
        {"age": 52, "symptoms": "Lump in left breast, nipple discharge", "family_history": "Yes", "breast_density": "High", "final_diagnosis": "Breast Cancer"},
        {"age": 45, "symptoms": "Pain in breast, irregular lump", "family_history": "No", "breast_density": "Medium", "final_diagnosis": "No Cancer"},
        {"age": 60, "symptoms": "Skin dimpling, swelling", "family_history": "Yes", "breast_density": "High", "final_diagnosis": "Breast Cancer"},
        {"age": 38, "symptoms": "No symptoms, detected on mammogram", "family_history": "No", "breast_density": "High", "final_diagnosis": "Breast Cancer"},
        {"age": 29, "symptoms": "Breast tenderness during periods", "family_history": "No", "breast_density": "Low", "final_diagnosis": "No Cancer"},
        {"age": 67, "symptoms": "Retraction of nipple, thickened skin", "family_history": "Yes", "breast_density": "High", "final_diagnosis": "Breast Cancer"},
        {"age": 50, "symptoms": "Breast pain, no lump, no discharge", "family_history": "No", "breast_density": "Medium", "final_diagnosis": "No Cancer"},
        {"age": 41, "symptoms": "Irregular shape found on ultrasound", "family_history": "Yes", "breast_density": "High", "final_diagnosis": "Breast Cancer"},
        {"age": 55, "symptoms": "New small lump, past DCIS", "family_history": "Yes", "breast_density": "High", "final_diagnosis": "Breast Cancer"},
        {"age": 62, "symptoms": "No symptoms, routine screening abnormal", "family_history": "No", "breast_density": "Medium", "final_diagnosis": "No Cancer"},
    ]
    cases = [
        Document(
            page_content=f"Age: {r['age']}, Symptoms: {r['symptoms']}, Family History: {r['family_history']}, Density: {r['breast_density']} → {r['final_diagnosis']}"
        ) for r in data
    ]
    return DocArrayInMemorySearch.from_documents(cases, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

retriever = load_case_memory().as_retriever(search_kwargs={"k": 5})

# === General Chat Mode ===
if mode == "General Chat":
    st.subheader("💬 Ask any question about breast cancer")
    st.markdown("""
        <style>
            .stTextInput > div > input {
                width: 100% !important;
                padding: 0.75rem;
                border-radius: 8px;
                font-size: 16px;
            }
        </style>
    """, unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "system", "content": "You are a helpful breast cancer assistant."}]

    # Show chat messages
    for msg in st.session_state.chat_history[1:]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Full-width form input
    with st.form("general_chat_form", clear_on_submit=True):
        user_question = st.text_input("Type your question here:")
        send = st.form_submit_button("Send")
        if send and user_question:
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            reply = query_llama3(st.session_state.chat_history)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()

# === Detection Mode ===
elif mode == "Detection Support":
    questions = [
        "How old are you? (e.g., 45, 52, 60)",
        "Please describe any symptoms you're experiencing in your breasts.",
        "Do you have any family history of breast cancer?",
        "Have you had a biopsy or mammogram? What were the results?",
        "Do you know your breast density? (Low, Medium, High)",
        "What is your hormone receptor status? (ER, PR, HER2)",
        "Any other relevant information you'd like to share?"
    ]

    if "qa_pairs" not in st.session_state:
        st.session_state.qa_pairs = []
    if "q_index" not in st.session_state:
        st.session_state.q_index = 0

    st.markdown("""
        <style>
            .stTextInput > div > input {
                width: 100% !important;
                padding: 0.75rem;
                border-radius: 8px;
                font-size: 16px;
            }
        </style>
    """, unsafe_allow_html=True)

    if st.session_state.q_index < len(questions):
        question = questions[st.session_state.q_index]
        with st.form("detection_form", clear_on_submit=True):
            answer = st.text_input(question)
            next_q = st.form_submit_button("Next")
            if next_q and answer:
                st.session_state.qa_pairs.append((question, answer))
                st.session_state.q_index += 1
                st.rerun()
    else:
        st.success("Analyzing your responses...")
        chat_history = "\n".join([f"{q} {a}" for q, a in st.session_state.qa_pairs])
        retrieved = retriever.get_relevant_documents(chat_history)
        examples = "\n".join([doc.page_content for doc in retrieved])

        prompt = f"""
You are a medical assistant helping detect breast cancer.
Use the patient's answers and compare them with known examples to assess risk.

Patient responses:
{chat_history}

Similar past cases:
{examples}

Now, provide a summary of the patient's risk and recommend what they should do next.
"""

        result = query_llama3([
            {"role": "system", "content": "You are a helpful breast cancer assistant."},
            {"role": "user", "content": prompt}
        ])
        st.markdown("### 🧠 Detection Result")
        st.markdown(result)

        new_case = {f"q{i+1}": a for i, (_, a) in enumerate(st.session_state.qa_pairs)}
        new_case["llm_diagnosis"] = result
        pd.DataFrame([new_case]).to_csv("chat_logs.csv", mode="a", header=not os.path.exists("chat_logs.csv"), index=False)

        class PDFReport(FPDF):
            def header(self):
                self.set_font("Arial", "B", 16)
                self.set_text_color(40, 40, 40)
                self.cell(0, 10, "RF Vision Diagnostic Center", ln=True, align="C")
                self.set_font("Arial", "", 11)
                self.cell(0, 10, "Breast Cancer Screening & Diagnosis Report", ln=True, align="C")
                self.ln(10)

            def footer(self):
                self.set_y(-15)
                self.set_font("Arial", "I", 8)
                self.set_text_color(128)
                self.cell(0, 10, f"Page {self.page_no()}", align="C")

        pdf = PDFReport()
        pdf.add_page()
        pdf.set_font("Arial", size=11)
        pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.ln(5)

        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Patient Responses:", ln=True)
        pdf.set_font("Arial", '', 11)
        for q, a in st.session_state.qa_pairs:
            pdf.multi_cell(0, 8, f"{q}\n-> {a}")

            pdf.ln(2)

        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "AI Diagnosis Summary:", ln=True)
        pdf.set_font("Arial", '', 11)
        pdf.multi_cell(0, 10, result)

        pdf_output = pdf.output(dest='S').encode('latin1')
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_output,
            file_name="breast_cancer_diagnosis_report.pdf",
            mime="application/pdf"
        )

        if st.button("Start Over"):
            st.session_state.qa_pairs = []
            st.session_state.q_index = 0
            st.rerun()

# === Sidebar Info ===
st.sidebar.title("About")
st.sidebar.info("""
This chatbot helps assess breast cancer risk from your responses and answers general questions using LLaMA 3.3 with RAG.
""")
