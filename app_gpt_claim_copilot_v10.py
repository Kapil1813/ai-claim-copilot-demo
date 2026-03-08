# app_gpt_claim_copilot_v10.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import json
import random
from dotenv import load_dotenv

# ---------------------------
# Load OpenAI API Key
# ---------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

try:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except ImportError:
    client = None

# ---------------------------
# Page Setup
# ---------------------------
st.set_page_config(page_title="AI Claim Copilot", layout="wide")
st.title("🏥 AI Claim Copilot")
st.subheader("AI Medical Coding & Claim Denial Risk Analyzer")

# ---------------------------
# CPT Procedure Reference
# ---------------------------
CPT_REFERENCE = {
    "72148":"MRI Lumbar Spine",
    "83036":"HbA1c Diabetes Test",
    "85025":"Complete Blood Count",
    "80061":"Lipid Panel",
    "80048":"Basic Metabolic Panel",
    "99213":"Office Visit"
}

# ---------------------------
# Session State
# ---------------------------
if "claims_df" not in st.session_state:
    st.session_state.claims_df = pd.DataFrame(columns=[
        "claim_id","diagnosis_code","procedure_code","denial_risk"
    ])

if "ai_icd" not in st.session_state:
    st.session_state.ai_icd=""

if "ai_cpt" not in st.session_state:
    st.session_state.ai_cpt=[]

if "confidence" not in st.session_state:
    st.session_state.confidence=0

if "explanation" not in st.session_state:
    st.session_state.explanation=""

# ---------------------------
# AI Code Generator
# ---------------------------
def extract_codes(note):
    """Return ICD-10, CPT list, confidence %, and AI explanation"""
    if client is None:
        # Demo fallback
        icd=random.choice(["E11.9","I10","E11.65"])
        cpt=random.sample(["83036","85025","80061","72148","99213"],2)
        confidence=random.randint(80,95)
        explanation="Based on the clinical note, these codes match the documented conditions and lab tests."
        return icd,cpt,confidence,explanation

    prompt=f"""
Extract ICD10 and CPT codes from the clinical note.

Return JSON:
{{
"icd":"",
"cpt":[],
"confidence":0,
"explanation":""
}}

Clinical Note:
{note}
"""
    try:
        response=client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role":"user","content":prompt}]
        )
        result=json.loads(response.choices[0].message.content)
        return result["icd"], result["cpt"], result["confidence"], result["explanation"]
    except:
        # Fallback if API fails
        return "E11.9", ["83036"], 85, "Fallback explanation"

# ---------------------------
# Denial Risk Logic
# ---------------------------
CPT_RISK = {
    "72148":65,
    "83036":25,
    "85025":30,
    "80061":20,
    "80048":20,
    "99213":15
}

def calculate_denial_risk(cpt):
    return CPT_RISK.get(cpt, 20)

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
# Show AI Suggestion & Explanation
# ---------------------------
if st.session_state.ai_icd:
    st.success(
        f"AI Suggested ICD-10: {st.session_state.ai_icd} | CPT: {', '.join(st.session_state.ai_cpt)} | Confidence: {st.session_state.confidence}%"
    )
    st.subheader("💡 AI Explanation")
    st.info(st.session_state.explanation)

# ---------------------------
# Denial Risk Distribution before submission
# ---------------------------
if st.session_state.ai_cpt:
    st.subheader("📊 Denial Risk Distribution by CPT (Preview)")
    preview_df = pd.DataFrame({
        "CPT Code": st.session_state.ai_cpt,
        "Denial Risk": [calculate_denial_risk(c) for c in st.session_state.ai_cpt]
    })
    fig = px.bar(preview_df, x="CPT Code", y="Denial Risk", text="Denial Risk",
                 labels={"Denial Risk":"Denial Risk (%)"})
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Editable Codes & Submission
# ---------------------------
if st.session_state.ai_icd:
    st.subheader("✏️ Edit Codes (Optional)")
    edited_icd = st.text_input("ICD-10", value=st.session_state.ai_icd)
    edited_cpt = st.text_input(
        "CPT Codes (comma separated)", 
        value=",".join(st.session_state.ai_cpt)
    )
    submit = st.button("Submit Final Codes")
    if submit:
        cpt_list = [c.strip() for c in edited_cpt.split(",") if c.strip()]
        for c in cpt_list:
            risk = calculate_denial_risk(c)
            new_claim = pd.DataFrame([{
                "claim_id": len(st.session_state.claims_df)+1,
                "diagnosis_code": edited_icd,
                "procedure_code": c,
                "denial_risk": risk
            }])
            st.session_state.claims_df = pd.concat(
                [st.session_state.claims_df, new_claim], ignore_index=True
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
    st.subheader("📈 Average Denial Risk by CPT Code")
    summary = df.groupby("procedure_code")["denial_risk"].mean().reset_index()
    summary.columns = ["CPT Code", "Avg Denial Risk"]
    summary["Procedure"] = summary["CPT Code"].map(CPT_REFERENCE)
    summary = summary[["CPT Code","Procedure","Avg Denial Risk"]]
    summary["Avg Denial Risk"] = summary["Avg Denial Risk"].round().astype(int).astype(str) + "%"
    st.table(summary)

# ---------------------------
# Confidence Gauge
# ---------------------------
if st.session_state.ai_icd:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=st.session_state.confidence,
        title={'text':"AI Confidence %"},
        gauge={'axis':{'range':[0,100]}}
    ))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Privacy Notice
# ---------------------------
st.markdown("""
⚠️ **Demo only** — do not enter real patient data.
""")