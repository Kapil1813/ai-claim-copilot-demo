# app_gpt_claim_copilot_v11.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import random
import os
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------
# Load OpenAI API Key
# ---------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------
# Page Setup
# ---------------------------
st.set_page_config(page_title="AI Claim Copilot", layout="wide")
st.title("🏥 AI Claim Copilot")
st.subheader("Medical Coding & Claim Denial Risk Analyzer")

# ---------------------------
# CPT Procedure Reference
# ---------------------------
CPT_REFERENCE = {
    "72148": "MRI Lumbar Spine",
    "83036": "HbA1c Diabetes Test",
    "85025": "Complete Blood Count",
    "80061": "Lipid Panel",
    "80048": "Basic Metabolic Panel",
    "99213": "Office Visit, Established Patient"
}

# ---------------------------
# Session State
# ---------------------------
if "claims_df" not in st.session_state:
    st.session_state.claims_df = pd.DataFrame(columns=[
        "claim_id", "diagnosis_code", "procedure_code", "denial_risk"
    ])

if "ai_icd" not in st.session_state:
    st.session_state.ai_icd = ""

if "ai_cpt" not in st.session_state:
    st.session_state.ai_cpt = []

if "confidence" not in st.session_state:
    st.session_state.confidence = 0

if "explanation" not in st.session_state:
    st.session_state.explanation = ""

# ---------------------------
# AI Code Extraction
# ---------------------------
def extract_codes(note):
    """
    Uses OpenAI GPT API to extract ICD-10 and CPT codes with explanation.
    Falls back to demo random codes if API fails.
    """
    if client is None:
        # Demo fallback
        icd = random.choice(["E11.9", "E11.65", "I10", "E78.5"])
        cpt = random.sample(list(CPT_REFERENCE.keys()), 2)
        confidence = random.randint(80, 95)
        explanation = "Based on the clinical note, these codes match documented conditions and tests."
        return icd, cpt, confidence, explanation

    # Advanced GPT Prompt for better CPT extraction
    prompt = f"""
You are a medical coding assistant. Extract **ICD-10** and **CPT** codes from the clinical note below. 
Provide the JSON response only in this format:

{{
  "icd": "",
  "cpt": [],
  "confidence": 0,
  "explanation": ""
}}

Clinical Note:
{note}

Rules:
1. Include all relevant CPT codes.
2. Provide confidence as 0-100 percentage.
3. Include short explanation of why each code was selected.
4. Do not include extra text outside JSON.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        result = json.loads(response.choices[0].message.content)
        icd = result.get("icd", "E11.9")
        cpt = result.get("cpt", ["83036"])
        confidence = result.get("confidence", 85)
        explanation = result.get("explanation", "")
        return icd, cpt, confidence, explanation

    except Exception as e:
        # fallback demo
        return "E11.9", ["83036"], 80, "Fallback explanation due to API error"

# ---------------------------
# Denial Risk Logic
# ---------------------------
CPT_DENIAL_RISK = {
    "72148": 65,
    "83036": 25,
    "85025": 30,
    "80061": 20,
    "80048": 20,
    "99213": 15
}

def calculate_denial_risk(cpt):
    return CPT_DENIAL_RISK.get(cpt, 20)

# ---------------------------
# Clinical Note Input
# ---------------------------
st.header("📝 Enter Clinical Note")
clinical_note = st.text_area("Clinical Note", height=120)

generate = st.button("Generate CPT & ICD-10")

# ---------------------------
# Generate Codes
# ---------------------------
if generate:
    icd, cpt, confidence, explanation = extract_codes(clinical_note)
    st.session_state.ai_icd = icd
    st.session_state.ai_cpt = cpt
    st.session_state.confidence = confidence
    st.session_state.explanation = explanation

# ---------------------------
# Show AI Suggestion
# ---------------------------
if st.session_state.ai_icd:
    st.success(
        f"AI Suggested ICD-10: {st.session_state.ai_icd} | CPT: {', '.join(st.session_state.ai_cpt)} | Confidence: {st.session_state.confidence}%"
    )
    st.subheader("AI Explanation")
    st.info(st.session_state.explanation)

# ---------------------------
# Editable Codes
# ---------------------------
if st.session_state.ai_icd:
    st.subheader("Edit Codes (Optional)")
    edited_icd = st.text_input("ICD-10", value=st.session_state.ai_icd)
    edited_cpt = st.text_input(
        "CPT Codes (comma separated)",
        value=",".join(st.session_state.ai_cpt)
    )
    submit = st.button("Submit Final Codes")

    if submit:
        cpt_list = [c.strip() for c in edited_cpt.split(",")]
        for c in cpt_list:
            risk = calculate_denial_risk(c)
            new_claim = pd.DataFrame([{
                "claim_id": len(st.session_state.claims_df) + 1,
                "diagnosis_code": edited_icd,
                "procedure_code": c,
                "denial_risk": risk
            }])
            st.session_state.claims_df = pd.concat(
                [st.session_state.claims_df, new_claim],
                ignore_index=True
            )
        st.success("✅ Claim added to table")

# ---------------------------
# Claim Predictions Table
# ---------------------------
df = st.session_state.claims_df
if len(df) > 0:
    st.subheader("📋 Claim Predictions Table")
    st.dataframe(df, use_container_width=True)

# ---------------------------
# Average Denial Risk by CPT
# ---------------------------
if len(df) > 0:
    st.subheader("📊 Average Denial Risk by CPT Code")
    summary = df.groupby("procedure_code")["denial_risk"].mean().reset_index()
    summary.columns = ["CPT Code", "Avg Denial Risk"]
    summary["Procedure"] = summary["CPT Code"].map(CPT_REFERENCE)
    summary = summary[["CPT Code", "Procedure", "Avg Denial Risk"]]
    summary["Avg Denial Risk"] = summary["Avg Denial Risk"].astype(int).astype(str) + "%"
    st.table(summary)

# ---------------------------
# AI Confidence Gauge
# ---------------------------
if st.session_state.ai_icd:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=st.session_state.confidence,
        title={'text': "AI Confidence %"},
        gauge={'axis': {'range': [0, 100]}}
    ))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Privacy Notice
# ---------------------------
st.markdown("""
⚠️ **Demo Privacy Notice**  
This tool is for demonstration only.  
Do **not** enter real patient data.
""")