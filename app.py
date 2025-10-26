import streamlit as st
import pandas as pd
import joblib
import lightgbm as lgb
from collections import Counter
import numpy as np
import os

# --- จำนวนเหตุการณ์ในแต่ละ session ---
EVENT_COUNT = 5

# --- ฟังก์ชันโหลดโมเดลแบบ universal ---
def load_model(path):
    """
    โหลดโมเดล LightGBM อัตโนมัติ
    - ถ้าเป็น sklearn wrapper (LGBMRanker) ใช้ joblib.load()
    - ถ้าเป็น native Booster ใช้ lgb.Booster(model_file)
    """
    try:
        model = joblib.load(path)
        if hasattr(model, "predict"):
            print(f"✅ Loaded sklearn model: {path}")
            return model
        else:
            raise TypeError("Not sklearn model")
    except Exception:
        print(f"⚙️ Loading as Booster model: {path}")
        return lgb.Booster(model_file=path)

# --- โหลดโมเดลทั้ง 3 ตัว ---
models = {
    "clicks": load_model("models/lgbm_ranker_clicks.pkl"),
    "carts": load_model("models/lgbm_ranker_carts.pkl"),
    "orders": load_model("models/lgbm_ranker_orders.pkl")
}

# --- โหลดข้อมูล global popularity ---
global_popularity_counter = joblib.load("global_popularity_counter.joblib")

# --- ฟังก์ชันหลักสำหรับรันโมเดล ---
def run_model(session):
    results = {}
    events = session["events"]

    for event_type in ["clicks", "carts", "orders"]:
        model = models[event_type]

        session_length = len(events)
        history_aids_counter = Counter([e["aid"] for e in events])

        X_data = []

        # --- สร้าง feature สำหรับแต่ละ event ---
        for event in events:
            aid_str = event["aid"]
            aid_int = int(aid_str)

            features = [
                global_popularity_counter.get(aid_int, 0),  # global popularity
                session_length,                              # session length
                history_aids_counter.get(aid_str, 0)         # frequency ของ aid ใน session
            ]
            X_data.append(features)

        # --- แปลงเป็น DataFrame ---
        X_df = pd.DataFrame(X_data, columns=[
            "global_popularity",
            "session_length",
            "aid_frequency"
        ])

        st.subheader(f"🧩 Input features for {event_type}")
        st.write(X_df.head())

        # --- ทำนาย ---
        try:
            if isinstance(model, lgb.Booster):
                # สำหรับ native Booster ต้องส่ง numpy array
                preds = model.predict(X_df.values, num_iteration=model.best_iteration)
            else:
                # สำหรับ sklearn wrapper ใช้ DataFrame ได้ตรงๆ
                preds = model.predict(X_df)

            results[event_type] = preds.tolist()
        except Exception as e:
            st.error(f"Error predicting {event_type}: {e}")
            results[event_type] = []

    return results


# --- ส่วน UI ของ Streamlit ---
if __name__ == "__main__":
    samples = pd.read_json(path_or_buf="test_trimmed.jsonl", lines=True)

    st.title("🧠 OTTO: Multi-Objective Recommender System")

    with st.container():
        sample = st.selectbox("Choose Sample No.", samples.index)
        selected_sample = samples.iloc[sample]

    with st.container(border=True, height=320):
        st.write(f"**Session ID:** `{selected_sample['session']}`")
        st.write("**Events:**")
        st.write(selected_sample["events"])

    with st.container():
        if st.button("🚀 Run Model", type="primary", use_container_width=True):
            st.divider()
            result = run_model(selected_sample)
            st.subheader("🔮 Prediction Results")
            st.write(result)
