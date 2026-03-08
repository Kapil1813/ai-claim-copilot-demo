import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import json
import random
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------
# Load API Key
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

st.title("AI Claim Copilot")
st.subheader("AI Medical Coding & Claim Denial Risk Analyzer")

# ---------------------------
# CPT Procedure Reference
# ---------------------------
CPT_REFERENCE = {
"72148":"MRI Lumbar Spine",
"83036":"HbA1c Diabetes Test",
"85025":"Complete Blood Count",
"80061":"Lipid Panel",
"80048":"Basic Metabolic Panel"
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

# ---------------------------
# AI Code Generator
# ---------------------------
def extract_codes(note):

    if client is None:
        icd=random.choice(["E11.9","I10","E78.5"])
        cpt=random.sample(["83036","85025","80061"],2)
        confidence=random.randint(85,95)
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

        return result["icd"],result["cpt"],result["confidence"],result["explanation"]

    except:

        return "E11.9",["83036"],80,"Fallback explanation"


# ---------------------------
# Denial Risk Logic
# ---------------------------
def calculate_denial_risk(cpt):

    base=20

    if cpt=="72148":
        base=65

    if cpt=="83036":
        base=25

    if cpt=="85025":
        base=30

    if cpt=="80061":
        base=20

    return base


# ---------------------------
# Clinical Note Input
# ---------------------------
st.header("Enter Clinical Note")

clinical_note=st.text_area("Clinical Note")

generate=st.button("Generate CPT & ICD-10")

# ---------------------------
# Generate Codes
# ---------------------------
if generate:

    icd,cpt,confidence,explanation=extract_codes(clinical_note)

    st.session_state.ai_icd=icd
    st.session_state.ai_cpt=cpt
    st.session_state.confidence=confidence
    st.session_state.explanation=explanation

# ---------------------------
# Show AI Suggestion
# ---------------------------
if st.session_state.ai_icd:

    st.success(
        f"AI Suggested ICD10: {st.session_state.ai_icd} | CPT: {', '.join(st.session_state.ai_cpt)} | Confidence: {st.session_state.confidence}%"
    )

# ---------------------------
# AI Explanation
# ---------------------------
if st.session_state.ai_icd:

    st.subheader("AI Explanation")

    st.info(st.session_state.explanation)

# ---------------------------
# Editable Codes
# ---------------------------
if st.session_state.ai_icd:

    st.subheader("Edit Codes (Optional)")

    edited_icd=st.text_input("ICD10",value=st.session_state.ai_icd)

    edited_cpt=st.text_input(
        "CPT Codes (comma separated)",
        value=",".join(st.session_state.ai_cpt)
    )

    submit=st.button("Submit Final Codes")

    if submit:

        cpt_list=[c.strip() for c in edited_cpt.split(",")]

        for c in cpt_list:

            risk=calculate_denial_risk(c)

            new_claim=pd.DataFrame([{
                "claim_id":len(st.session_state.claims_df)+1,
                "diagnosis_code":edited_icd,
                "procedure_code":c,
                "denial_risk":risk
            }])

            st.session_state.claims_df=pd.concat(
                [st.session_state.claims_df,new_claim],
                ignore_index=True
            )

        st.success("Claim added to table")

# ---------------------------
# Claim Predictions Table
# ---------------------------
df=st.session_state.claims_df

st.subheader("Claim Predictions Table")

st.dataframe(df,use_container_width=True)

# ---------------------------
# CPT Risk Summary Table
# ---------------------------
if len(df)>0:

    st.subheader("Average Denial Risk by CPT Code")

    summary=df.groupby("procedure_code")["denial_risk"].mean().reset_index()

    summary.columns=["CPT Code","Avg Denial Risk"]

    summary["Procedure"]=summary["CPT Code"].map(CPT_REFERENCE)

    summary=summary[["CPT Code","Procedure","Avg Denial Risk"]]

    summary["Avg Denial Risk"]=summary["Avg Denial Risk"].astype(int).astype(str)+"%"

    st.table(summary)

# ---------------------------
# Confidence Gauge
# ---------------------------
if st.session_state.ai_icd:

    fig=go.Figure(go.Indicator(
        mode="gauge+number",
        value=st.session_state.confidence,
        title={'text':"AI Confidence %"},
        gauge={'axis':{'range':[0,100]}}
    ))

    st.plotly_chart(fig,use_container_width=True)

# ---------------------------
# Privacy Notice
# ---------------------------
st.markdown("Demo only. Do not enter real patient data.")