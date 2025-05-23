# cricket_dashboard_scorecard.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
from io import BytesIO
import os

# ---------------------------
# Page Setup
# ---------------------------
st.set_page_config(page_title="Live Cricket Player Stats Dashboard", layout="wide")
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>🏏 CricAPI Player Stats Dashboard</h1>", unsafe_allow_html=True)

# ---------------------------
# Gemini Setup
# ---------------------------
try:
    GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel("models/gemini-2.0-flash")
    gemini_ready = True
except Exception as e:
    st.error(f"❌ Gemini setup failed: {e}")
    gemini_ready = False

# ---------------------------
# Dataset Configuration
# ---------------------------
data_dir = "archive (2)"
formats = ["ODI", "TEST", "T20"]

batting_df, bowling_df, all_round_df = [], [], []

try:
    all_players_df = pd.read_csv(os.path.join(data_dir, "all_players.csv"))
    all_players_df.columns = all_players_df.columns.str.strip().str.lower()
    all_players_df.rename(columns={"name": "player"}, inplace=True)
except Exception as e:
    st.error(f"Failed to load all_players.csv: {e}")
    all_players_df = pd.DataFrame()

for fmt in formats:
    try:
        bat = pd.read_csv(os.path.join(data_dir, f"{fmt}_batting.csv"))
        bat.columns = bat.columns.str.strip().str.lower()
        bat = bat.merge(all_players_df, on="id", how="left")
        bat["format"] = fmt
        batting_df.append(bat)
    except Exception as e:
        st.warning(f"⚠️ Could not load {fmt}_batting.csv: {e}")

    try:
        bowl = pd.read_csv(os.path.join(data_dir, f"{fmt}_bowling.csv"))
        bowl.columns = bowl.columns.str.strip().str.lower()
        bowl = bowl.merge(all_players_df, on="id", how="left")
        bowl["format"] = fmt
        bowling_df.append(bowl)
    except Exception as e:
        st.warning(f"⚠️ Could not load {fmt}_bowling.csv: {e}")

    try:
        allr = pd.read_csv(os.path.join(data_dir, f"{fmt}_all_round.csv"))
        allr.columns = allr.columns.str.strip().str.lower()
        allr = allr.merge(all_players_df, on="id", how="left")
        allr["format"] = fmt
        all_round_df.append(allr)
    except Exception as e:
        st.warning(f"⚠️ Could not load {fmt}_all_round.csv: {e}")

batting_df = pd.concat(batting_df, ignore_index=True) if batting_df else pd.DataFrame()
bowling_df = pd.concat(bowling_df, ignore_index=True) if bowling_df else pd.DataFrame()
all_round_df = pd.concat(all_round_df, ignore_index=True) if all_round_df else pd.DataFrame()

# ---------------------------
# Player Names List
# ---------------------------
try:
    all_players = set(batting_df["player"]) | set(bowling_df["player"]) | set(all_round_df["player"])
except KeyError:
    all_players = set()

player_names = sorted(all_players)

# ---------------------------
# Helper Function
# ---------------------------
def get_stats(df, player):
    if df.empty or player not in df["player"].values:
        return pd.DataFrame()
    data = df[df["player"] == player].drop(columns=["player"])
    return data.groupby("format").sum(numeric_only=True).reset_index()

# ---------------------------
# UI and Logic
# ---------------------------
if not player_names:
    st.error("No player names could be loaded. Please check your CSV files.")

selected = st.multiselect("Select Player(s) for Comparison", player_names, max_selections=3)

show_tabs = st.button("📊 Fetch Player Stats")

if show_tabs and selected:
    tabs = st.tabs(["Batting", "Bowling", "All-Round"])
    dfs = {
        "Batting": batting_df,
        "Bowling": bowling_df,
        "All-Round": all_round_df
    }

    for label, tab in zip(["Batting", "Bowling", "All-Round"], tabs):
        with tab:
            compare_df = dfs[label][dfs[label]["player"].isin(selected)]
            if compare_df.empty:
                st.warning(f"No {label.lower()} records found.")
                continue

            st.markdown(f"### 🏌️ Player {label} Performance", unsafe_allow_html=True)
            st.dataframe(compare_df, use_container_width=True)

            st.markdown("---")
            st.markdown(f"### 📊 {label} Comparison Chart", unsafe_allow_html=True)
            fig = go.Figure()
            metrics = ["runs", "average_score", "strike_rate"] if label == "Batting" else ["wk"]
            for metric in metrics:
                for player in selected:
                    subset = compare_df[compare_df["player"] == player].groupby("format").sum(numeric_only=True).reset_index()
                    fig.add_trace(go.Bar(x=subset["format"], y=subset[metric], name=f"{player} - {metric}"))
            fig.update_layout(barmode="group", title=f"{label} Metrics Comparison", xaxis_title="Format", yaxis_title="Value")
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Gemini Summary
# ---------------------------
if selected and gemini_ready:
    st.subheader("🤖 Gemini AI Summary")
    if st.button("🧠 Generate Summary with Gemini"):
        with st.spinner("Calling Gemini..."):
            try:
                prompt = "Summarize the cricket performance of the following players:\n"
                for player in selected:
                    prompt += f"\nPlayer: {player}\n"
                    for _, row in get_stats(batting_df, player).iterrows():
                        prompt += f"Batting - {row['format']}: {row.get('runs', '-')} runs, Avg: {row.get('average_score', '-')}, SR: {row.get('strike_rate', '-')}\n"
                    for _, row in get_stats(bowling_df, player).iterrows():
                        prompt += f"Bowling - {row['format']}: {row.get('wk', '-')} wickets\n"

                response = model.generate_content(prompt)
                summary_text = response.text

                # Save to summary.txt
                with open("summary.txt", "w", encoding="utf-8") as f:
                    f.write(summary_text)

                st.success("✅ Gemini Summary:")
                st.markdown(summary_text)
                st.info("📝 Summary has been saved to 'summary.txt'")
            except Exception as e:
                st.error(f"❌ Gemini generation failed: {e}")