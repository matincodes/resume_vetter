import os
import re
import spacy
import requests
import dateparser
from datetime import datetime
from docx import Document
from PyPDF2 import PdfReader
import streamlit as st

# ----------------------------
# Configuration
# ----------------------------
HF_API_TOKEN = "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"  # Replace with your token
MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"

# ----------------------------
# Core Functions
# ----------------------------
def extract_text(file_path):
    """Extract text from PDF/DOCX files"""
    text = ""
    if file_path.name.endswith('.pdf'):
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text()
    elif file_path.name.endswith('.docx'):
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    return text

def get_ai_analysis(resume_text, job_desc=None):
    """Get AI analysis using Hugging Face API"""
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    
    prompt = f"""Analyze this resume and provide structured feedback:
    
    [Resume Content]
    {resume_text[:3000]}
    
    """
    
    if job_desc:
        prompt += f"[Job Description]\n{job_desc[:1000]}\n\n"
        
    prompt += """Format your response with these sections:
    - Key Strengths: (3 bullet points)
    - Potential Weaknesses: (3 bullet points)
    - Skill Gaps: (list missing relevant skills)
    - Improvement Suggestions: (3 actionable items)
    - Match Score: (1-100 rating)
    
    Keep responses concise and professional."""
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 600,
            "temperature": 0.5,
            "return_full_text": False
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()[0]['generated_text']
        return f"API Error: {response.text}"
    except Exception as e:
        return f"Analysis failed: {str(e)}"

# ----------------------------
# Resume Analysis Functions
# ----------------------------
def parse_experience(text):
    """Calculate total work experience"""
    dates = re.findall(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d{4})\b', text)
    total_exp = 0
    current_date = datetime.now()

    for i in range(0, len(dates), 2):
        start_str = dates[i]
        end_str = dates[i+1] if i+1 < len(dates) else 'Present'

        start_date = dateparser.parse(start_str)
        end_date = dateparser.parse(end_str) if end_str.lower() != 'present' else current_date

        if start_date and end_date:
            delta = (end_date - start_date).days / 365.25
            total_exp += max(delta, 0)  # Avoid negative durations

    return round(total_exp, 1)

def extract_skills(text):
    """Extract technical skills using spaCy"""
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text.lower())
    
    skills = {
        'Languages': {'python', 'java', 'javascript', 'c++', 'sql'},
        'Frameworks': {'react', 'node.js', 'django', 'flask', 'spring'},
        'Tools': {'docker', 'git', 'aws', 'jenkins', 'kubernetes'}
    }
    
    found_skills = {category: set() for category in skills}
    
    for token in doc:
        for category, keywords in skills.items():
            if token.text in keywords:
                found_skills[category].add(token.text.capitalize())
                
    return {k: list(v) for k, v in found_skills.items() if v}

# ----------------------------
# Streamlit Interface
# ----------------------------
def main():
    st.title("AI Resume Analyzer (Free Version)")
    
    # User Inputs
    hf_token = st.text_input("Hugging Face Token:", type="password")
    uploaded_file = st.file_uploader("Upload Resume", type=['pdf', 'docx'])
    job_desc = st.text_area("Job Description (optional):", height=150)
    
    if uploaded_file and hf_token:
        with st.spinner("Analyzing..."):
            resume_text = extract_text(uploaded_file)
            
            # Basic Analysis
            experience = parse_experience(resume_text)
            skills = extract_skills(resume_text)
            
            # AI Analysis
            ai_analysis = get_ai_analysis(resume_text, job_desc)
            
            # Display Results
            st.subheader("Basic Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Experience", f"{experience} years")
                st.write("**Languages:**", ", ".join(skills.get('Languages', [])))
                
            with col2:
                st.metric("Key Skills", len(skills))
                st.write("**Frameworks:**", ", ".join(skills.get('Frameworks', [])))
            
            st.subheader("AI Evaluation")
            st.markdown(f"```\n{ai_analysis}\n```")  # Preserve formatting
            
            # Extract and display match score
            if "Match Score:" in ai_analysis:
                score = re.search(r"Match Score: (\d+)", ai_analysis)
                if score:
                    st.progress(int(score.group(1))/100)

if __name__ == "__main__":
    main()