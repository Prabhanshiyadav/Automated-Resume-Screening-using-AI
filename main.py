import streamlit as st
import pdfplumber
import docx2txt
import re
import nltk
import pandas as pd
from io import BytesIO
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK stopwords
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

# Function to extract text from files
def extract_text(file):
    if file.name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            return "\n".join([page.extract_text() or "" for page in pdf.pages]).strip()
    elif file.name.endswith(".docx"):
        return docx2txt.process(file).strip()
    return ""

# Function to extract email and phone number
def extract_contact_info(text):
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    phone_pattern = r"\b\d{10}\b|\(\d{3}\) \d{3}-\d{4}|\d{3}-\d{3}-\d{4}"
    emails = re.findall(email_pattern, text)
    phones = re.findall(phone_pattern, text)
    return emails[0] if emails else "Not Found", phones[0] if phones else "Not Found"

# Function to extract relevant skills
def extract_skills(text, skills):
    return ", ".join(set(skills) & set(text.lower().split())) or "No Matching Skills"

# Function to summarize text
def summarize_text(text, num_sentences=3):
    return " ".join(sent_tokenize(text)[:num_sentences])

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    return " ".join([word for word in word_tokenize(text) if word not in stop_words])

# Function to rank resumes
def rank_resumes(resume_texts, job_desc):
    if not job_desc.strip(): return [0] * len(resume_texts)
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    tfidf_matrix = vectorizer.fit_transform([job_desc] + resume_texts)
    return (cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0] * 100).tolist()

# Streamlit UI
st.title("🚀 AI Resume Screening Dashboard")

# Upload Resumes
uploaded_files = st.file_uploader("📤 Upload Resumes (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True)
job_desc = st.text_area("📝 Paste Job Description Here", height=150)
skills_input = st.text_input("🎯 Required Skills (comma-separated)", "Python, Machine Learning, Data Analysis")

# Process Resumes
if uploaded_files and job_desc.strip():
    required_skills = {skill.strip().lower() for skill in skills_input.split(',')}
    resumes_data = []

    for file in uploaded_files:
        text = extract_text(file)
        if text:
            email, phone = extract_contact_info(text)
            matched_skills = extract_skills(text, required_skills)
            summary = summarize_text(text)
            resumes_data.append({
                "Name": file.name,
                "Contact": f"📧 {email} | 📞 {phone}",
                "Skills Matched": matched_skills,
                "Summary": summary,
                "Text": text
            })

    if resumes_data:
        st.success(f"✅ {len(resumes_data)} Resume(s) Processed!")
        resume_texts = [preprocess_text(res["Text"]) for res in resumes_data]
        scores = rank_resumes(resume_texts, preprocess_text(job_desc))

        # Display ranked resumes
        st.subheader("📌 Ranked Resumes:")
        results = []

        for res, score in sorted(zip(resumes_data, scores), key=lambda x: x[1], reverse=True):
            st.write(f"🔹 **{res['Name']}** - Match Score: **{round(score, 2)}%**")
            st.write(f"📬 **Contact Info:** {res['Contact']}")
            st.write(f"🔍 **Skills Matched:** {res['Skills Matched']}")
            st.write(f"📄 **Summary:** {res['Summary']}")
            st.write("---")
            results.append({"Resume": res['Name'], "Match Score": round(score, 2), "Contact": res['Contact'], "Skills": res['Skills Matched'], "Summary": res['Summary']})

        # Provide CSV download
        df_results = pd.DataFrame(results)
        output = BytesIO()
        df_results.to_csv(output, index=False)
        output.seek(0)
        st.download_button("📥 Download Results (CSV)", data=output, file_name="ranked_resumes.csv", mime="text/csv")
    else:
        st.warning("⚠️ No valid resumes found!")
else:
    if not job_desc.strip():
        st.warning("⚠️ Please enter a job description!")
