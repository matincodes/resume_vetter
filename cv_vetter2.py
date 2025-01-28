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
nlp = spacy.load("en_core_web_sm")
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"

# ----------------------------
# Core Functions
# ----------------------------
def extract_text(file_path):
    """Improved text extraction with error handling"""
    try:
        text = ""
        if file_path.name.endswith('.pdf'):
            reader = PdfReader(file_path)
            for page in reader.pages:
                text += page.extract_text() or ""  # Handle None returns
        elif file_path.name.endswith('.docx'):
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
        return text.strip()
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return ""

def extract_personal_details(text):
    """Extract name, email, and LinkedIn with validation"""
    details = {'name': 'Not Found', 'email': 'Not Found', 'linkedin': 'Not Found'}
    
    # Email extraction
    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    if email_match:
        details['email'] = email_match.group(0)
    
    # LinkedIn extraction
    linkedin_match = re.search(
        r'(https?://)?(www\.)?linkedin\.com/(in|company)/[a-zA-Z0-9-]+/?', 
        text
    )
    if linkedin_match:
        details['linkedin'] = linkedin_match.group(0)
    
    # Name extraction with multiple fallbacks
    name_patterns = [
        r"^([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)$",  # Title case names
        r"^[A-Z\s]{5,}$",  # All caps names
        r"(?i)\b(name|about)\b[\s:]*(.+)$"  # Label-based names
    ]
    
    for line in text.split('\n')[:20]:  # Check first 20 lines
        line = line.strip()
        for pattern in name_patterns:
            name_match = re.match(pattern, line)
            if name_match and len(name_match.group().split()) <= 4:
                details['name'] = name_match.group().title()
                return details
    return details

def analyze_linkedin_profile(profile_url):
    """Validate LinkedIn profile accessibility"""
    if profile_url == 'Not Found':
        return {'valid': False, 'issues': ['No LinkedIn found'], 'recommendations': []}
    
    issues = []
    try:
        response = requests.head(profile_url, timeout=5)
        if response.status_code != 200:
            issues.append(f"Profile unavailable (HTTP {response.status_code})")
    except Exception as e:
        issues.append(f"Connection error: {str(e)}")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'recommendations': [
            "Ensure profile is public",
            "Add profile picture",
            "Include detailed experience"
        ] if len(issues) == 0 else []
    }

def calculate_experience(dates):
    """Calculate total experience using dateparser"""
    total_exp = 0
    current_date = datetime.now()
    
    for i in range(0, len(dates)-1, 2):
        start_str = dates[i]
        end_str = dates[i+1] if i+1 < len(dates) else 'Present'
        
        start_date = dateparser.parse(start_str)
        end_date = dateparser.parse(end_str) if end_str.lower() != 'present' else current_date
        
        if start_date and end_date:
            if start_date > current_date:
                continue  # Ignore future dates
            delta = (end_date - start_date).days / 365.25
            total_exp += max(delta, 0)  # Avoid negative durations
            
    return round(total_exp, 1)

def extract_skills(text):
    """Extract technical skills using spaCy"""
    skills_db = {
        'Languages': {'python', 'java', 'javascript', 'c++', 'sql'},
        'Frameworks': {'react', 'node.js', 'django', 'flask', 'spring'},
        'Tools': {'docker', 'git', 'aws', 'jenkins', 'kubernetes'}
    }
    
    doc = nlp(text.lower())
    found_skills = {category: set() for category in skills_db}
    
    for token in doc:
        for category, keywords in skills_db.items():
            if token.text in keywords:
                found_skills[category].add(token.text.capitalize())
                
    return {k: list(v) for k, v in found_skills.items() if v}

def score_projects(text):
    """Score project complexity through keyword analysis"""
    keywords = {
        'basic': 1, 
        'intermediate': 2,
        'advanced': 3,
        'scalable': 3,
        'microservices': 3,
        'ci/cd': 2,
        'optimized': 2,
        'containerized': 3
    }
    return sum(score for word, score in keywords.items() if word in text.lower())

def determine_proficiency(experience, project_score):
    """Determine proficiency tier and level"""
    if experience <= 1:
        return ("Entry Level", 1) if project_score < 2 else ("Junior", 2)
    elif 1 < experience <= 3:
        return ("Mid-Level", 3) if project_score < 4 else ("Professional", 4)
    elif 3 < experience <= 5:
        return ("Senior", 5) if project_score < 6 else ("Expert", 6)
    else:
        return ("Principal", 7) if project_score < 8 else ("Architect", 8)

def get_hf_analysis(api_key, text, job_desc=None):
    """Get AI insights from Hugging Face API"""
    headers = {"Authorization": f"Bearer {api_key}"}
    
    prompt = f"""Analyze this resume and provide structured feedback:
    
    [Resume Content]
    {text[:2500]}
    
    [Job Description]
    {job_desc[:500] if job_desc else 'N/A'}
    
    Provide analysis in this format:
    - Key Strengths: 3 bullet points
    - Weakness Alerts: 3 potential concerns
    - Skill Gaps: Missing but required skills
    - Optimization Tips: 3 actionable suggestions
    - Culture Fit: Compatibility assessment
    
    Keep responses concise and focused on technical hiring."""
    
    try:
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": prompt, "parameters": {"max_new_tokens": 600}}
        )
        
        if response.status_code == 200:
            return response.json()[0]['generated_text']
        return f"API Error: {response.text}"
    except Exception as e:
        return f"AI Analysis Failed: {str(e)}"

# ----------------------------
# Analysis Pipeline
# ----------------------------
def full_analysis_pipeline(resume_text, hf_api_key=None, job_desc=None):
    """Complete analysis workflow"""
    results = {}
    
    # Personal Details
    results['personal'] = extract_personal_details(resume_text)
    
    # Experience Calculation
    dates = re.findall(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d{4})\b', resume_text)
    results['experience'] = calculate_experience(dates)
    
    # Skill Analysis
    results['skills'] = extract_skills(resume_text)
    
    # Project Scoring
    results['project_score'] = score_projects(resume_text)
    
    # Proficiency Tier
    results['proficiency'] = determine_proficiency(
        results['experience'], 
        results['project_score']
    )
    
    # LinkedIn Analysis
    results['linkedin'] = analyze_linkedin_profile(
        results['personal']['linkedin']
    )
    
    # Interview Readiness
    results['readiness'] = {
        'ready': results['proficiency'][1] >= 3 and results['project_score'] >= 4,
        'score': results['proficiency'][1] + (1 if results['project_score'] >=4 else 0),
        'factors': [
            f"Proficiency Level: {results['proficiency'][1]}",
            f"Project Score: {results['project_score']}",
            f"LinkedIn Valid: {results['linkedin']['valid']}"
        ]
    }
    
    # AI Analysis (if API key provided)
    if hf_api_key:
        with st.spinner("Generating AI Insights..."):
            results['ai_analysis'] = get_hf_analysis(
                hf_api_key,
                resume_text,
                job_desc
            )
    
    return results

# ----------------------------
# Streamlit Interface
# ----------------------------
def main():
    st.set_page_config(page_title="Resume Vetter Pro", layout="wide")
    
    st.title("📄 Resume Vetter Pro")
    st.caption("Hybrid Local + AI Resume Analysis System")
    
    with st.sidebar:
        hf_api_key = st.text_input("Hugging Face API Key (optional):", 
                                 type="password",
                                 help="Get your key from huggingface.co/settings/tokens")
        job_desc = st.text_area("Job Description (optional):", height=200)
    
    uploaded_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=['pdf', 'docx'])
    
    if uploaded_file:
        resume_text = extract_text(uploaded_file)
        
        if not resume_text:
            st.error("Failed to extract text from file")
            return
            
        results = full_analysis_pipeline(resume_text, hf_api_key, job_desc)
        
        # Main Dashboard
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Personal Details Card
            st.subheader("👤 Candidate Profile")
            st.markdown(f"""
                - **Name**: {results['personal']['name']}
                - **Email**: {results['personal']['email']}
                - **LinkedIn**: {results['personal']['linkedin']}
                """)
            
            # Proficiency Card
            st.subheader("📊 Proficiency Analysis")
            st.markdown(f"""
                - **Years Experience**: {results['experience']}
                - **Project Score**: {results['project_score']}/10
                - **Proficiency Tier**: {results['proficiency'][0]}
                - **Tier Level**: {results['proficiency'][1]}
                """)
            
            # Readiness Card
            st.subheader("✅ Interview Readiness")
            readiness_icon = "🟢" if results['readiness']['ready'] else "🔴"
            st.markdown(f"""
                {readiness_icon} **Status**: {"Ready" if results['readiness']['ready'] else "Needs Improvement"}
                - **Score**: {results['readiness']['score']}/4
                - **Factors**:
                    {", ".join(results['readiness']['factors'])}
                """)
        
        with col2:
            # Skills & LinkedIn Analysis
            tab1, tab2, tab3 = st.tabs(["Technical Skills", "LinkedIn Review", "AI Insights"])
            
            with tab1:
                st.subheader("🛠 Technical Skills Breakdown")
                for category, skills in results['skills'].items():
                    st.markdown(f"**{category}**")
                    st.write(", ".join(skills) or "No skills detected")
                    st.progress(min(len(skills)/10, 1.0))
            
            with tab2:
                st.subheader("🔗 LinkedIn Profile Analysis")
                if results['linkedin']['valid']:
                    st.success("✅ Profile is accessible and valid")
                    st.markdown("**Recommendations:**")
                    for rec in results['linkedin']['recommendations']:
                        st.write(f"- {rec}")
                else:
                    st.error("❌ Profile issues detected")
                    st.markdown("**Issues:**")
                    for issue in results['linkedin']['issues']:
                        st.write(f"- {issue}")
            
            with tab3:
                if hf_api_key:
                    st.subheader("🤖 AI-Powered Insights")
                    if "Error" in results.get('ai_analysis', ''):
                        st.error(results['ai_analysis'])
                    else:
                        st.markdown(f"""```plaintext
{results.get('ai_analysis', 'AI analysis not available')}
```""")
                else:
                    st.info("🔒 Enable AI insights by adding Hugging Face API key in sidebar")

if __name__ == "__main__":
    main()