# main_mod.py

# Importing the necessary libraries
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from langchain.document_loaders import PyMuPDFLoader
import os
from dotenv import load_dotenv
import json, re, ast

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("groq_api_key")

# Initialize Groq LLM
llm = ChatGroq(temperature=0, model_name="llama3-8b-8192", api_key=groq_api_key)

# Prompt template for resume and job description analysis
template = """
You are a career advisor AI assistant. Analyze the following:

RESUME:
{resume_text}

JOB DESCRIPTION:
{jd_text}

Perform:
1. Extract name, skills, experience, education from the resume.
2. Extract required skills and responsibilities from the job description.
3. Do a skill gap analysis.
4. Suggest resume improvements.
5. Provide personalized career advice.

Output JSON like:
{{
  "match_score": <int from 0-100>,
  "missing_skills": [...],
  "recommendations": [...],
  "feedback": "<string>"
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
def extract_resume_text(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
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
