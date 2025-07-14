# main_mod.py

# Importing the necessary libraries
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from langchain.document_loaders import PyMuPDFLoader
import os
from dotenv import load_dotenv
import json, re, ast
import tempfile

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("groq_api_key")

# Initialize Groq LLM
llm = ChatGroq(temperature=0, model_name="llama3-8b-8192", api_key=groq_api_key)

# Prompt template for resume and job description analysis
template = """
You are a career advisor AI assistant. Analyze how well the candidate's resume fits the job description.

RESUME:
{resume_text}

JOB DESCRIPTION:
{jd_text}

Perform:
1. Extract name, skills, experience, and education from the resume.
2. Extract required skills and responsibilities from the job description.
3. Compare resume vs JD and do a skill gap analysis — only include skills relevant to the JD.
4. Suggest 2-3 resume improvements that would help match this JD better.
5. Provide short, personalized career advice based on the resume and JD.

Only return this JSON:
{{
  "match_score": <int from 0-100>,
  "missing_skills": ["skill1", "skill2", ...],
  "recommendations": ["Tip 1", "Tip 2", ...],
  "feedback": "Short personalized advice."
}}
"""

# Create a prompt template
prompt = PromptTemplate(
    input_variables=["resume_text", "jd_text"],
    template=template
)

# Create LLM chain with the prompt
chain = LLMChain(llm=llm, prompt=prompt)

# Extract text content from PDF resume
def extract_resume_text(uploaded_file): # Renamed from pdf_path → uploaded_file
    # Change: Save uploaded file to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Pass temp file path to the loader
    loader = PyMuPDFLoader(tmp_path)
    pages = loader.load()
    return "\n".join([page.page_content for page in pages])

# Analyze resume against job description using the LLM chain
def analyze_resume_vs_jd(resume_path, jd_text):
    resume_text = extract_resume_text(resume_path)
    response = chain.run(resume_text=resume_text, jd_text=jd_text)

    # Try extracting JSON from LLM response (inside triple backticks)
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Fallback to Python dict parsing if JSON fails
    try:
        parsed = ast.literal_eval(response.strip())
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Return raw output on failure
    return {
        "error": "Could not parse response as valid JSON or Python dict.",
        "raw_output": response
    }
