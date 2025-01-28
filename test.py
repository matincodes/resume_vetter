import os
import re
import pandas as pd
import spacy
import dateparser
from datetime import datetime
from docx import Document
from PyPDF2 import PdfReader
from difflib import SequenceMatcher
import streamlit as st
from dateutil.relativedelta import relativedelta
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import undetected_chromedriver as uc

# ----------------------------
# Core Functions (Updated)
# ----------------------------

def extract_text(file_path):
    """Extract text from PDF/DOCX files."""
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

def extract_name(text):
    """Extract name from the resume header."""
    lines = text.split('\n')
    for line in lines[:10]:  # Search in the first 10 lines for the name
        if line.strip().isupper() and len(line.split()) <= 3:  # Check for uppercase and short length
            return line.strip()
    return "Name not found"

def parse_email(text):
    """Extract email using stricter regex."""
    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    return email_match.group(0) if email_match else "N/A"

def parse_experience(text):
    """Accurate experience calculation."""
    date_ranges = re.findall(r'(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d{1,2}/\d{4}|\d{4})\b|Present)', text, re.IGNORECASE)
    total_experience = 0
    current_date = datetime.now()

    for i in range(0, len(date_ranges), 2):
        start_str = date_ranges[i]
        end_str = date_ranges[i + 1] if i + 1 < len(date_ranges) else 'Present'

        start_date = dateparser.parse(start_str, settings={'RELATIVE_BASE': datetime.now()})
        end_date = dateparser.parse(end_str, settings={'RELATIVE_BASE': datetime.now()}) if end_str.lower() != 'present' else current_date

        if start_date and end_date and start_date <= end_date:
            delta = relativedelta(end_date, start_date)
            total_experience += delta.years + delta.months / 12

    return round(total_experience, 1)

def parse_skills(text):
    """Deduplicated skills using sets."""
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)

    skills_db = {
        'Languages': ['python', 'java', 'javascript', 'typescript', 'c++', 'sql'],
        'Frameworks': ['react', 'node.js', 'django', 'flask', 'next.js', 'graphql'],
        'Tools': ['docker', 'kubernetes', 'aws', 'jenkins', 'git', 'rabbitmq']
    }

    skills = {category: set() for category in skills_db}
    for token in doc:
        token_lower = token.text.lower()
        for category, keywords in skills_db.items():
            if token_lower in keywords:
                skills[category].add(token_lower)

    return {category: list(skills[category]) for category in skills}

def score_projects(projects):
    """Expanded complexity keywords."""
    complexity_keywords = [
        'scalable', 'microservices', 'ci/cd', 'optimized', 'containerized',
        'authentication', 'api', 'cloud', 'database', 'latency'
    ]
    score = 0
    for project in projects:
        for keyword in complexity_keywords:
            if keyword in project.lower():
                score += 1
    return score

def check_plagiarism(projects):
    """Plagiarism check against known projects."""
    known_projects = [
        "Built a REST API with Django",
        "Developed a chatbot using NLP",
        "Architected scalable microservices"
    ]
    plagiarism_score = 0
    for project in projects:
        max_similarity = max([
            SequenceMatcher(None, project, kp).ratio() 
            for kp in known_projects
        ])
        plagiarism_score += max_similarity
    return (plagiarism_score / len(projects)) * 100 if projects else 0

def categorize_candidate(experience, project_score):
    """Separate Level and Proficiency Tier."""
    if experience <= 1 and project_score <= 2:
        return 1, 'Beginner'
    elif 1 < experience <= 3 and 2 < project_score <= 4:
        return 3, 'Intermediate'
    elif 3 < experience <= 5 and project_score > 4:
        return 4, 'Professional'
    elif experience > 5 and project_score > 5:
        return 6, 'Expert'
    else:
        return 2, 'Intermediate'

def scrape_linkedin(linkedin_url):
    """Scrape LinkedIn profile for skills (headless browser)."""
    chrome_options = uc.ChromeOptions()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

    driver = uc.Chrome(options=chrome_options)
    skills = []

    try:
        driver.get(linkedin_url)
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".pv-skill-categories-section"))
        )
        skills_section = driver.find_element(By.CSS_SELECTOR, ".pv-skill-categories-section")
        skills = [skill.text.split('\n')[0] for skill in skills_section.find_elements(By.TAG_NAME, "li")]
    except Exception as e:
        print(f"Error scraping LinkedIn: {e}")
    finally:
        driver.quit()

    return skills

# ----------------------------
# Streamlit App (Updated)
# ----------------------------

def main():
    st.title("AI-Powered CV Vetter")
    uploaded_file = st.file_uploader("Upload a resume (PDF or DOCX)", type=['pdf', 'docx'])

    if uploaded_file:
        text = extract_text(uploaded_file)

        # Extract data
        name = extract_name(text)
        email = parse_email(text)
        experience = parse_experience(text)
        skills = parse_skills(text)
        projects = re.findall(r'- (.*?)(?=\n- |$)', text, re.DOTALL)
        project_score = score_projects(projects)
        plagiarism_score = check_plagiarism(projects)
        level, tier = categorize_candidate(experience, project_score)

        # Scrape LinkedIn
        linkedin_url_match = re.search(r'linkedin\.com/in/[\w-]+', text)
        linkedin_skills = []
        if linkedin_url_match:
            linkedin_url = f"https://www.{linkedin_url_match.group(0)}"
            linkedin_skills = scrape_linkedin(linkedin_url)
            skills['LinkedIn Skills'] = linkedin_skills

        # Determine interview readiness
        interview_ready = "Yes" if level >= 3 and project_score >= 3 else "No"

        # Display results
        st.subheader("Candidate Summary")
        st.write(f"**Name:** {name}")
        st.write(f"**Email:** {email}")
        st.write(f"**Experience:** {experience} years")
        st.write(f"**Skills:** {skills}")
        st.write(f"**Project Complexity Score:** {project_score}/10")
        st.write(f"**Plagiarism Risk:** {plagiarism_score:.1f}%")
        st.write(f"**Level:** {level}")
        st.write(f"**Proficiency Tier:** {tier}")
        st.write(f"**Interview Ready?** {interview_ready}")

if __name__ == "__main__":
    main()
