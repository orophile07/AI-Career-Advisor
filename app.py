# app.py

import streamlit as st
from main_mod import analyze_resume_vs_jd
import json

# Streamlit page setup
st.set_page_config(page_title="AI Career Advisor", layout="centered")
st.title("🤖 AI Career Advisor")
st.markdown("Upload your **Resume (PDF)** and paste a **Job Description (JD)** to get career insights.")

# File upload and job description input
uploaded_resume = st.file_uploader("📄 Upload Resume (PDF)", type="pdf")
job_desc = st.text_area("📝 Paste Job Description Here")

# Run analysis if both inputs are provided
if uploaded_resume and job_desc:
    with open("temp_resume.pdf", "wb") as f:
        f.write(uploaded_resume.read())
    
    with st.spinner("Analyzing your resume... ⏳"):
        result = analyze_resume_vs_jd("temp_resume.pdf", job_desc)

    # Display results if analysis succeeds
    if isinstance(result, dict) and "error" not in result:
        st.subheader("📊 Career Report")
        
        st.markdown(f"**🎯 Match Score:** `{result['match_score']}`")
        
        st.markdown("### 🔍 Missing Skills")
        for i, skill in enumerate(result["missing_skills"], start=1):
            st.markdown(f"{i}. {skill}")
            
        st.markdown("### 💡 Recommendations")
        for i, rec in enumerate(result["recommendations"], start=1):
            st.markdown(f"{i}. {rec}")
            
        st.markdown("### 🗣️ Feedback")
        st.markdown(f"> {result['feedback']}")
