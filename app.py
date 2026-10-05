import os
import json
import streamlit as st
import pdfplumber
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Page Setup
st.set_page_config(
    page_title="LEGALSCAN | AI Legal Document Analyzer",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️️ LEGALSCAN")
st.caption("Automated Legal Risk Analysis, Clause Extraction & Summary Tool")

# Initialize Gemini API Client
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("⚠️ GEMINI_API_KEY not found in .env file. Please check your configuration.")
    st.stop()


@st.cache_resource
def get_gemini_client(key: str):
    return genai.Client(api_key=key)


client = get_gemini_client(api_key)


# Pydantic Schema for Strict Structured Analysis
class KeyClause(BaseModel):
    clause_title: str = Field(description="Title of the clause (e.g., Termination, Liability Limit, Indemnity)")
    summary: str = Field(description="Simplified non-legalese explanation of what this clause entails.")
    risk_level: str = Field(description="Risk rating: 'Low', 'Medium', or 'High'")


class LegalAnalysisReport(BaseModel):
    document_type: str = Field(
        description="Detected type of document (e.g., Non-Disclosure Agreement, Employment Contract, Service Agreement)")
    executive_summary: str = Field(description="Concise 2-3 sentence overview of the legal agreement.")
    overall_risk_score: str = Field(
        description="Overall risk level: 'Low Risk', 'Moderate Risk', or 'High Risk / Severe'")
    key_clauses: list[KeyClause] = Field(description="Extracted key clauses with individual risk assessments.")
    red_flags_and_pitfalls: list[str] = Field(
        description="One-sided clauses, ambiguous terms, or unfavorable legal obligations.")
    recommended_negotiation_points: list[str] = Field(
        description="Actionable points to negotiate or revise prior to signing.")


SYSTEM_INSTRUCTION = """
You are LEGALSCAN, an expert AI legal document review analyst.
Analyze the user-provided legal text/document and output a strict JSON breakdown according to the requested schema.
Break down legal terminology into plain, easy-to-understand language while accurately highlighting potential risks and liabilities.
"""


def extract_text_from_pdf(pdf_file):
    """Extracts raw text from an uploaded PDF file."""
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text


def render_analysis(report: dict):
    """Renders the JSON legal analysis into a clear dashboard layout."""

    # 1. Executive Summary & Document Type Header
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.subheader(f"📄 Document Type: {report.get('document_type', 'Legal Agreement')}")
        st.write(report.get("executive_summary", ""))

    with col_b:
        risk = report.get("overall_risk_score", "Moderate Risk").title()
        if "High" in risk or "Severe" in risk:
            st.error(f"🚨 **Overall Risk:**\n\n### {risk}")
        elif "Moderate" in risk or "Medium" in risk:
            st.warning(f"⚠️ **Overall Risk:**\n\n### {risk}")
        else:
            st.success(f"🟢 **Overall Risk:**\n\n### {risk}")

    st.divider()

    # 2. Key Clause Analysis
    st.subheader("📑 Clause Breakdown & Risk Ratings")
    clauses = report.get("key_clauses", [])
    if clauses:
        clause_cols = st.columns(min(len(clauses), 3) if len(clauses) > 0 else 1)
        for idx, clause in enumerate(clauses):
            col_target = clause_cols[idx % len(clause_cols)]
            with col_target:
                r_level = clause.get("risk_level", "Low").title()
                badge = "🟢" if "Low" in r_level else ("⚠️" if "Medium" in r_level else "🚨")

                with st.expander(f"{badge} **{clause.get('clause_title', 'Clause')}** ({r_level})", expanded=True):
                    st.write(clause.get("summary", ""))

    st.divider()

    # 3. Red Flags & Negotiation Recommendations
    col_red, col_neg = st.columns(2)

    with col_red:
        st.subheader("🚩 Red Flags & Hidden Pitfalls")
        for flag in report.get("red_flags_and_pitfalls", []):
            st.markdown(f"- {flag}")

    with col_neg:
        st.subheader("💡 Recommended Negotiation Points")
        for point in report.get("recommended_negotiation_points", []):
            st.markdown(f"- {point}")


# Sidebar Controls
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload PDF or TXT contract", type=["pdf", "txt"])

    st.info("Upload a contract or copy text into the box to start analysis.")

# Main Input Interface
raw_text = ""

if uploaded_file is not None:
    if uploaded_file.name.endswith(".pdf"):
        raw_text = extract_text_from_pdf(uploaded_file)
    else:
        raw_text = uploaded_file.read().decode("utf-8")

    st.text_area("Extracted Document Text Preview", raw_text[:1500] + ("..." if len(raw_text) > 1500 else ""),
                 height=150)

else:
    raw_text = st.text_area("Or Paste Legal Text / Contract Here:", height=250,
                            placeholder="Paste contract text here...")

if st.button("🔍 Analyze Document", type="primary", use_container_width=True):
    if not raw_text.strip():
        st.warning("Please upload a document or paste text first.")
    else:
        with st.spinner("Analyzing legal clauses"):
            try:
                response = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=[raw_text],
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=0.1,
                        response_mime_type="application/json",
                        response_schema=LegalAnalysisReport,
                    ),
                )

                analysis_data = json.loads(response.text)
                render_analysis(analysis_data)

            except Exception as e:
                st.error(f"Error during legal document analysis: {e}")