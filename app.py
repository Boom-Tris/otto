import streamlit as st
import pandas as pd
import joblib
import lightgbm as lgb
import gdown
import os
from collections import Counter

# --- ค่าคงที่ ---
N_ITEMS_FROM_HISTORY = 5
N_CO_VISITS_PER_ITEM = 40
N_CANDIDATES_PER_SESSION = 200

# --- ฟังก์ชันโหลดโมเดล ---
def load_model(path):
    try:
        model = joblib.load(path)
        if hasattr(model, "predict"):
            return model
        else:
            raise TypeError("Not sklearn model")
    except Exception:
        return lgb.Booster(model_file=path)

# --- ฟังก์ชันโหลด assets + ดาวน์โหลด Google Drive ---
@st.cache_resource(show_spinner=False)
def load_assets():
    os.makedirs("assets", exist_ok=True)
    os.makedirs("assets/models", exist_ok=True)

    # --- โหลด co_visitation_map จาก Google Drive ---
    map_path_raw = "assets/co_visitation_map_raw.joblib"
    map_path_compressed = "assets/co_visitation_map_compressed.joblib"

    if not os.path.exists(map_path_raw):
        gdown.download(
            "https://drive.google.com/uc?id=1YFHmmMXYzm0AtjakazsAziwNpk03hT58",
            map_path_raw,
            quiet=False
        )
    
    # --- ลดขนาดไฟล์และ compress ---
    if not os.path.exists(map_path_compressed):
        st.info("Compressing co_visitation_map to reduce size...")
        co_map_raw = joblib.load(map_path_raw)
        # ตัด subset: top 40 co-visits ต่อ aid
        co_map_subset = {aid: Counter(dict(c.most_common(40))) for aid, c in co_map_raw.items()}
        joblib.dump(co_map_subset, map_path_compressed, compress=3)
        del co_map_raw

    # โหลด compressed map
    co_visitation_map = joblib.load(map_path_compressed)

    # --- โหลดโมเดล ---
    models = {
        "clicks": load_model("assets/models/lgbm_ranker_clicks.pkl"),
        "carts": load_model("assets/models/lgbm_ranker_carts.pkl"),
        "orders": load_model("assets/models/lgbm_ranker_orders.pkl")
    }

    # --- โหลด fallback + global popularity ---
    global_popularity_counter = joblib.load("assets/global_popularity_counter.joblib")
    top_20_fallback = joblib.load("assets/top_20_fallback.joblib")

    # --- feature names ---
    if hasattr(models['clicks'], 'feature_name_'):
        feature_names = models['clicks'].feature_name_
    else:
        feature_names = ['co_visitation_score', 'global_popularity', 'session_length', 'history_clicks_on_candidate']

    return models, global_popularity_counter, co_visitation_map, top_20_fallback, feature_names

# --- ฟังก์ชัน Top 20 ---
def get_top_20_recs(scores, candidates, fallback):
    ranked_candidates = [aid for _, aid in sorted(zip(scores, candidates), reverse=True)]
    top_20 = ranked_candidates[:20]
    if len(top_20) < 20:
        top_20.extend([aid for aid in fallback if aid not in top_20])
        top_20 = top_20[:20]
    return top_20

# --- Pipeline ---
def run_model_pipeline(session_data, models, global_popularity_counter, co_visitation_map, top_20_fallback, FEATURE_NAMES):
    events = session_data["events"]
    if not events:
        fallback_str = " ".join(map(str, top_20_fallback))
        return {"clicks": fallback_str, "carts": fallback_str, "orders": fallback_str}, pd.DataFrame(columns=FEATURE_NAMES)

    # Stage 1: Candidate Generation
    session_candidate_pool = Counter()
    history_aids = [event['aid'] for event in events]
    history_aids_set = set(history_aids)
    history_aids_recent = history_aids[-N_ITEMS_FROM_HISTORY:]

    for aid in set(history_aids_recent):
        top_co_visited = co_visitation_map.get(aid, Counter()).most_common(N_CO_VISITS_PER_ITEM)
        for co_aid, score in top_co_visited:
            if co_aid not in history_aids_set:
                session_candidate_pool[co_aid] += score

    dynamic_candidates = [aid for aid, _ in session_candidate_pool.most_common(N_CANDIDATES_PER_SESSION)]
    final_candidate_list = list(dict.fromkeys(dynamic_candidates + top_20_fallback))[:N_CANDIDATES_PER_SESSION]

    if not final_candidate_list:
        fallback_str = " ".join(map(str, top_20_fallback))
        return {"clicks": fallback_str, "carts": fallback_str, "orders": fallback_str}, pd.DataFrame(columns=FEATURE_NAMES)

    # Stage 2: Ranking
    session_length = len(events)
    history_aids_counter = Counter(history_aids)
    X_test_session_list = []

    for candidate_aid in final_candidate_list:
        co_visit_score = sum(co_visitation_map.get(history_aid, Counter()).get(candidate_aid, 0) for history_aid in history_aids_set)
        global_pop = global_popularity_counter.get(candidate_aid, 0)
        history_clicks = history_aids_counter.get(candidate_aid, 0)
        X_test_session_list.append([co_visit_score, global_pop, session_length, history_clicks])

    X_df = pd.DataFrame(X_test_session_list, columns=FEATURE_NAMES, index=final_candidate_list)

    # Predict
    try:
        if isinstance(models['clicks'], lgb.Booster):
            scores_clicks = models['clicks'].predict(X_df.values, num_iteration=models['clicks'].best_iteration)
            scores_carts = models['carts'].predict(X_df.values, num_iteration=models['carts'].best_iteration)
            scores_orders = models['orders'].predict(X_df.values, num_iteration=models['orders'].best_iteration)
        else:
            scores_clicks = models['clicks'].predict(X_df)
            scores_carts = models['carts'].predict(X_df)
            scores_orders = models['orders'].predict(X_df)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return {}, pd.DataFrame()

    top_20_clicks = get_top_20_recs(scores_clicks, final_candidate_list, top_20_fallback)
    top_20_carts = get_top_20_recs(scores_carts, final_candidate_list, top_20_fallback)
    top_20_orders = get_top_20_recs(scores_orders, final_candidate_list, top_20_fallback)

    results = {
        "clicks": " ".join(map(str, top_20_clicks)),
        "carts": " ".join(map(str, top_20_carts)),
        "orders": " ".join(map(str, top_20_orders))
    }

    X_df['score_clicks'] = scores_clicks
    X_df['score_carts'] = scores_carts
    X_df['score_orders'] = scores_orders

    return results, X_df.sort_values('score_orders', ascending=False)

# --- Streamlit UI ---
st.title("🧠 OTTO: Recommender System (v2 - Auto Download + Compress)")

# โหลด sample session JSON เล็ก
try:
    samples = pd.read_json("test_trimmed.jsonl", lines=True, nrows=100)
except Exception as e:
    st.error(f"ไม่พบหรืออ่านไฟล์ test_trimmed.jsonl ไม่ได้: {e}")
    st.stop()

sample_index = st.selectbox("Choose Sample Session No.", samples.index)
selected_sample = samples.iloc[sample_index]

st.write(f"**Session ID:** `{selected_sample['session']}`")
st.json(selected_sample["events"])

if st.button("🚀 Run Prediction Pipeline"):
    with st.spinner("Downloading & Loading Models + Maps (compressed)..."):
        models, global_popularity_counter, co_visitation_map, top_20_fallback, FEATURE_NAMES = load_assets()

    with st.spinner("Finding Candidates and Ranking..."):
        results, features_df = run_model_pipeline(selected_sample, models, global_popularity_counter, co_visitation_map, top_20_fallback, FEATURE_NAMES)

    st.subheader("🔮 Prediction Results (Top 20)")
    st.json(results)

    st.subheader(f"📊 Features & Scores for {len(features_df)} Candidates")
    st.dataframe(features_df.head(20))
