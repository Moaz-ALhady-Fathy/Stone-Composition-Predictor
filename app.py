import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Stone Composition Predictor (Two-Stage)", page_icon="🧪", layout="centered")

# Feature list and model paths
FEATURES = ["Sex", "Stone_Burden", "age", "BMI", "Urine_Crystals", "Diabetes", "Urine_PH", "HU", "Urine_culture", "Organism"]
COMBINED_LABEL = 10.0
PATH_STAGE1 = "stone_stage1_model.joblib"
PATH_STAGE2 = "stone_stage2_model.joblib"

# Class labels
CLASS_MAP = {
    1.0: "calcium oxalate",
    2.0: "calcium phosphate",
    3.0: "Uric acid",
    4.0: "infection stones",
    5.0: "cystine",
    10.0: "merged (2/4/5) — add Stage-2 to refine"
}

# Encoding maps
BIN_MAP = {
    "Sex": {"Male": 1, "Female": 2},
    "Diabetes": {"Yes": 2, "No": 0},
    "Urine_culture": {"Sterile": 1, "Infected": 2},
    "Urine_Crystals": {
        "No crystals": 0,
        "Calcium Oxalate": 1,
        "Uric acid": 2,
        "Cystine": 3,
        "Calcium Phosphate": 4
    },
    "Stone_Burden": {
        "Single": 1,
        "Multiple": 2,
        "Staghorn": 3
    },
    "Organism": {
        "Sterile": 1,
        "E Coli": 2,
        "Klebsiella": 3,
        "Pseudomonas": 4,
        "Staph Epidermedis": 5,
        "Staph Aurious": 6,
        "Enterobacter": 7,
        "Candida": 8,
        "Proteus mirabilis": 9,
        "Enterococcus faecalis": 10,
        "Acinetobacter baumannii": 11,
        "Streptococcus spp": 12,
        "Yeast": 13
    }
}

# Dropdown order
BIN_ORDER = {
    "Sex": list(BIN_MAP["Sex"].keys()),
    "Diabetes": list(BIN_MAP["Diabetes"].keys()),
    "Urine_culture": list(BIN_MAP["Urine_culture"].keys()),
    "Urine_Crystals": list(BIN_MAP["Urine_Crystals"].keys()),
    "Stone_Burden": list(BIN_MAP["Stone_Burden"].keys()),
    "Organism": list(BIN_MAP["Organism"].keys())
}

# UI
st.title("Prediction Of Urinary Stone Composition")
st.caption("Two-stage inference • Developed by **Moaz Alhadi**")

# Load models
m1 = joblib.load(PATH_STAGE1) if os.path.exists(PATH_STAGE1) else None
m2 = joblib.load(PATH_STAGE2) if os.path.exists(PATH_STAGE2) else None
m1_status = m1 is not None
m2_status = m2 is not None

cols = st.columns(2)
with cols[0]:
    st.markdown(f"**Stage-1 model:** {'✅ Loaded' if m1_status else '❌ Missing'}")
with cols[1]:
    st.markdown(f"**Stage-2 model:** {'✅ Loaded' if m2_status else '⚠️ Optional (used when Stage-1 predicts 10)'}")

if not m1_status:
    st.stop()

st.divider()
st.subheader("Enter variables")

with st.form("single_pred"):
    c1, c2 = st.columns(2)

    with c1:
        sex = st.selectbox("Sex", BIN_ORDER["Sex"], index=0)
        sb = st.selectbox("Stone Burden", BIN_ORDER["Stone_Burden"], index=0)
        age = st.number_input("Age", value=40.0, step=1.0, format="%.3f", min_value=0.0, max_value=120.0)
        bmi = st.number_input("BMI", value=25.0, step=0.1, format="%.3f", min_value=0.0, max_value=100.0)
        ucr = st.selectbox("Urine Crystals", BIN_ORDER["Urine_Crystals"], index=0)

    with c2:
        diab = st.selectbox("Diabetes", BIN_ORDER["Diabetes"], index=1)
        uph = st.number_input("Urine PH", value=6.0, step=0.1, format="%.3f")
        hu = st.number_input("HU", value=0.0, step=1.0, format="%.3f")
        ucx = st.selectbox("Urine Culture", BIN_ORDER["Urine_culture"], index=1)
        org = st.selectbox("Organism", BIN_ORDER["Organism"], index=0)

    go = st.form_submit_button("🚀 Predict")

if go:
    try:
        row = {
            "Sex": BIN_MAP["Sex"][sex],
            "Stone_Burden": BIN_MAP["Stone_Burden"][sb],
            "age": age,
            "BMI": bmi,
            "Urine_Crystals": BIN_MAP["Urine_Crystals"][ucr],
            "Diabetes": BIN_MAP["Diabetes"][diab],
            "Urine_PH": uph,
            "HU": hu,
            "Urine_culture": BIN_MAP["Urine_culture"][ucx],
            "Organism": BIN_MAP["Organism"][org],
        }
        X = pd.DataFrame([row])[FEATURES]

        y1 = m1.predict(X)
        final = y1
        used_stage2 = False

        if (y1[0] == COMBINED_LABEL) and m2_status:
            y2 = m2.predict(X)
            final = y2
            used_stage2 = True

        pred_num = float(final[0])
        pred_name = CLASS_MAP.get(pred_num, str(pred_num))

        if used_stage2:
            st.success(f"Predicted class (refined): **{pred_name}**")
            st.caption("Stage-1 predicted 10 → refined by Stage-2.")
        else:
            st.success(f"Predicted class: **{pred_name}**")
            if y1[0] == COMBINED_LABEL and not m2_status:
                st.warning("Stage-1 predicted the merged class (10: merged 2/4/5). Add stage-2 model to refine into: calcium phosphate / infection stones / cystine.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("Dropdowns map to numeric values used during training.")
st.markdown("© Developed by **Moaz Alhadi**")