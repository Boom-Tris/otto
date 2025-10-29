import streamlit as st
import pandas as pd
import joblib
import lightgbm as lgb
import gdown
import os

from collections import Counter

# --- ค่าคงที่ (ควรตั้งให้ตรงกับ Cell 2 ใน Notebook) ---
# (นี่คือค่าจากโค้ดที่คุณส่งมา)
N_ITEMS_FROM_HISTORY = 5
N_CO_VISITS_PER_ITEM = 40
N_CANDIDATES_PER_SESSION = 200

# --- ฟังก์ชันโหลดโมเดล (เหมือนเดิม) ---
def load_model(path):
    """
    โหลดโมเดล LightGBM อัตโนมัติ
    - ถ้าเป็น sklearn wrapper (LGBMRanker) ใช้ joblib.load()
    - ถ้าเป็น native Booster ใช้ lgb.Booster(model_file)
    """
    try:
        # พยายามโหลดเป็น sklearn wrapper ก่อน
        model = joblib.load(path)
        if hasattr(model, "predict"):
            print(f"✅ Loaded sklearn model: {path}")
            return model
        else:
            raise TypeError("Not sklearn model")
    except Exception:
        # ถ้าไม่สำเร็จ, โหลดเป็น native Booster
        print(f"⚙️ Loading as Booster model: {path}")
        return lgb.Booster(model_file=path)

# --- 1. โหลดทุกอย่าง (Models + Maps) ---
@st.cache_resource
def load_all_assets():
    # โหลด co_visitation_map จาก google drive
    print("--- 0. Downloading co_visitation_map from Google Drive")
    
    if not os.path.exists("co_visitation_map.joblib"):
        gdown.download("https://drive.google.com/uc?id=1PMs1-swsSPwyH-_nL2IqX5Bz5p34mb9N", "co_visitation_map.joblib", quiet=False)
    
    if not os.path.exists("top_20_fallback.joblib"):
        gdown.download("https://drive.google.com/uc?id=14FwUZeiXM9ZRfyS45EMzOuAJtQkPRjSt", "top_20_fallback.joblib", quiet=False)

    if not os.path.exists("global_popularity_counter.joblib"):
        gdown.download("https://drive.google.com/uc?id=1Q5OdFMmGh34fw-ZYfcC3GwQ4d60932YQ", "global_popularity_counter.joblib", quiet=False)
    
    if not os.path.exists("lgbm_ranker_clicks.pkl"):
        gdown.download("https://drive.google.com/uc?id=1oniZl-45sxldTYNeB8ruY_4VY2_TD02g", "lgbm_ranker_clicks.pkl", quiet=False)
    
    if not os.path.exists("lgbm_ranker_carts.pkl"):
        gdown.download("https://drive.google.com/uc?id=1ZFD7d9ehJ-T5C4aErJ08UCiyP5mKbcri", "lgbm_ranker_carts.pkl", quiet=False)
    
    if not os.path.exists("lgbm_ranker_orders.pkl"):
        gdown.download("https://drive.google.com/uc?id=1oWaLZi6Jzrc2YmSKqF3SnHorFGACcVQQ", "lgbm_ranker_orders.pkl", quiet=False)

    # โหลด Model และ Maps ทั้งหมด (ใช้ cache เพื่อความเร็ว)
    print("--- 1. Loading all assets... ---")
    try:
        models = {
            "clicks": load_model("lgbm_ranker_clicks.pkl"),
            "carts": load_model("lgbm_ranker_carts.pkl"),
            "orders": load_model("lgbm_ranker_orders.pkl")
        }
        
        # (เราใช้ .joblib ตามที่ Notebook บันทึก)
        global_popularity_counter = joblib.load("global_popularity_counter.joblib")
        co_visitation_map = joblib.load("co_visitation_map.joblib")
        top_20_fallback = joblib.load("top_20_fallback.joblib")

        # ดึงชื่อ Feature 4 ตัวจากโมเดล
        # (เราสมมติว่า model 'clicks' เป็น sklearn wrapper ที่มี .feature_name_)
        # ถ้า Error ให้ลองเปลี่ยนเป็น ['co_visitation_score', 'global_popularity', 'session_length', 'history_clicks_on_candidate']
        if hasattr(models['clicks'], 'feature_name_'):
             feature_names = models['clicks'].feature_name_
        else:
             # Fallback สำหรับ native booster
             feature_names = models['clicks'].feature_name()

        print(f"Models loaded. Expecting features: {feature_names}")
        
        return models, global_popularity_counter, co_visitation_map, top_20_fallback, feature_names

    except FileNotFoundError as e:
        st.error(f"❌ ERROR: ไม่พบไฟล์โมเดลหรือ Map! - {e}")
        st.error("กรุณารัน NewV_2.ipynb (Cell 3, 5, 6) [cite: NewV_2.ipynb, Cell 3, 5, 6] และบันทึกไฟล์ .joblib และ .pkl มาไว้ที่นี่ก่อน")
        st.stop()
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดไฟล์: {e}")
        st.stop()

# --- โหลดข้อมูล ---
models, global_popularity_counter, co_visitation_map, top_20_fallback, FEATURE_NAMES = load_all_assets()


# --- 2. ฟังก์ชันช่วยจัดอันดับ (เหมือน Cell 8) [cite: NewV_2.ipynb, Cell 8] ---
def get_top_20_recs(scores, candidates, fallback):
    """
    เรียงลำดับคะแนน, เลือก Top 20, และเติมด้วย fallback ถ้าไม่พอ
    """
    ranked_candidates = [aid for _, aid in sorted(zip(scores, candidates), reverse=True)]
    top_20 = ranked_candidates[:20]
    if len(top_20) < 20:
        top_20.extend([aid for aid in fallback if aid not in top_20])
        top_20 = top_20[:20]
    return top_20


# --- 3. ฟังก์ชันหลักสำหรับรันโมเดล (เขียนใหม่ตาม Cell 8) [cite: NewV_2.ipynb, Cell 8] ---
def run_model_pipeline(session_data):
    """
    รัน 2-Stage Pipeline (Candidate Generation -> Ranking)
    """
    events = session_data["events"]
    
    if not events:
        # ถ้า session ว่าง, คืนค่า fallback
        fallback_str = " ".join(map(str, top_20_fallback))
        return {
            "clicks": fallback_str,
            "carts": fallback_str,
            "orders": fallback_str
        }, pd.DataFrame(columns=FEATURE_NAMES) # คืน DF ว่าง

    # --- 3.1. Stage 1: Candidate Generation (เหมือน Cell 8) [cite: NewV_2.ipynb, Cell 8] ---
    session_candidate_pool = Counter()
    history_aids = [event['aid'] for event in events]
    history_aids_set = set(history_aids)
    history_aids_recent = history_aids[-N_ITEMS_FROM_HISTORY:]

    for aid in set(history_aids_recent):
        top_co_visited = co_visitation_map.get(aid, Counter()).most_common(N_CO_VISITS_PER_ITEM)
        for co_aid, score in top_co_visited:
            if co_aid not in history_aids_set: # ไม่แนะนำของที่อยู่ในประวัติแล้ว
                session_candidate_pool[co_aid] += score
                
    dynamic_candidates = [aid for aid, score in session_candidate_pool.most_common(N_CANDIDATES_PER_SESSION)]
    # รวม "ตัวเลือก" จาก co-visit และ "ตัวเลือก" ยอดฮิต (fallback)
    final_candidate_list = list(dict.fromkeys(dynamic_candidates + top_20_fallback))
    final_candidate_list = final_candidate_list[:N_CANDIDATES_PER_SESSION]

    if not final_candidate_list:
        # (กรณีที่ 2: ถ้าหา candidate ไม่ได้เลย)
        fallback_str = " ".join(map(str, top_20_fallback))
        return {
            "clicks": fallback_str,
            "carts": fallback_str,
            "orders": fallback_str
        }, pd.DataFrame(columns=FEATURE_NAMES)

    # --- 3.2. Stage 2: Ranking (สร้าง 4 Features เหมือน Cell 8) [cite: NewV_2.ipynb, Cell 8] ---
    session_length = len(events)
    history_aids_counter = Counter(history_aids)
    X_test_session_list = []
    
    for candidate_aid in final_candidate_list:
        
        # Feature 1: Co-visitation Score [cite: NewV_2.ipynb, Cell 4, 8]
        co_visit_score = 0
        for history_aid in history_aids_set:
            co_visit_score += co_visitation_map.get(history_aid, Counter()).get(candidate_aid, 0)

        # Feature 2: Global Popularity [cite: NewV_2.ipynb, Cell 4, 8]
        global_pop = global_popularity_counter.get(candidate_aid, 0)
        
        # Feature 3: Session Length [cite: NewV_2.ipynb, Cell 4, 8]
        session_len = session_length
        
        # Feature 4: History Clicks (ความถี่ในอดีต) [cite: NewV_2.ipynb, Cell 4, 8]
        history_clicks = history_aids_counter.get(candidate_aid, 0)

        features = [
            co_visit_score,
            global_pop,
            session_len,
            history_clicks
        ]
        X_test_session_list.append(features)

    # สร้าง DataFrame Input
    X_df = pd.DataFrame(X_test_session_list, columns=FEATURE_NAMES, index=final_candidate_list)
    
    # --- 3.3. ทำนายและจัดอันดับ ---
    try:
        # ตรวจสอบประเภทโมเดลก่อน predict
        if isinstance(models['clicks'], lgb.Booster):
            # Native booster
            scores_clicks = models['clicks'].predict(X_df.values, num_iteration=models['clicks'].best_iteration)
            scores_carts = models['carts'].predict(X_df.values, num_iteration=models['carts'].best_iteration)
            scores_orders = models['orders'].predict(X_df.values, num_iteration=models['orders'].best_iteration)
        else:
            # Sklearn wrapper
            scores_clicks = models['clicks'].predict(X_df)
            scores_carts = models['carts'].predict(X_df)
            scores_orders = models['orders'].predict(X_df)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return {}, pd.DataFrame()


    # จัดอันดับ
    top_20_clicks = get_top_20_recs(scores_clicks, final_candidate_list, top_20_fallback)
    top_20_carts = get_top_20_recs(scores_carts, final_candidate_list, top_20_fallback)
    top_20_orders = get_top_20_recs(scores_orders, final_candidate_list, top_20_fallback)

    results = {
        "clicks": " ".join(map(str, top_20_clicks)),
        "carts": " ".join(map(str, top_20_carts)),
        "orders": " ".join(map(str, top_20_orders))
    }
    
    # เพิ่มคะแนนกลับเข้าไปใน DF เพื่อโชว์ (สำหรับ Debug)
    X_df['score_clicks'] = scores_clicks
    X_df['score_carts'] = scores_carts
    X_df['score_orders'] = scores_orders
    
    # คืนค่า 2 อย่าง: ผลลัพธ์ (dict) และ DF (สำหรับโชว์)
    return results, X_df.sort_values('score_orders', ascending=False) # เรียงตาม Order score


# --- 4. ส่วน UI ของ Streamlit (แก้ไขใหม่) ---
if __name__ == "__main__":
    try:
        # (โหลดแค่ 100 แถวมาเป็นตัวอย่าง)
        samples = pd.read_json(path_or_buf="test_trimmed.jsonl", lines=True, nrows=100)
    except FileNotFoundError:
        st.error("❌ ไม่พบไฟล์ test_trimmed.jsonl")
        st.info("กรุณาดาวน์โหลดไฟล์ test_trimmed.jsonl จาก Kaggle มาไว้ที่เดียวกับ app.py")
        st.stop()
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการอ่าน test_trimmed.jsonl: {e}")
        st.stop()


    st.title("🧠 OTTO: Recommender System (v2 - 4 Features)")

    with st.container():
        sample_index = st.selectbox("Choose Sample Session No.", samples.index)
        selected_sample = samples.iloc[sample_index]

    with st.container(border=True, height=320):
        st.write(f"**Session ID:** `{selected_sample['session']}`")
        st.write("**Events (History):**")
        st.json(selected_sample["events"]) # ใช้ st.json ให้อ่านง่ายขึ้น

    with st.container():
        if st.button("🚀 Run Prediction Pipeline", type="primary", use_container_width=True):
            st.divider()
            
            # --- เรียกฟังก์ชันใหม่ ---
            with st.spinner("Finding Candidates and Ranking..."):
                results, features_df = run_model_pipeline(selected_sample)
            
            st.subheader("🔮 Prediction Results (Top 20)")
            st.info("นี่คือ Top 20 'aid' ที่โมเดลทำนายว่าจะเกิดขึ้นในอนาคต")
            st.json(results) # ใช้ st.json ให้อ่านง่ายขึ้น
            
            st.divider()
            st.subheader(f"📊 Features & Scores for {len(features_df)} Candidates")
            st.info("นี่คือ Input 4-Features (และ Scores) ที่โมเดลใช้ (เรียงตาม Order Score)")
            st.dataframe(features_df.head(20))
