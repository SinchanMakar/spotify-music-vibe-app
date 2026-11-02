from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# This application displays the results of the existing Spotify K-Means run.
# It intentionally does NOT import, fit, or retrain a machine-learning model.

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "spotify_outputs"

st.set_page_config(
    page_title="Spotify Music Taste Clustering",
    page_icon="🎧",
    layout="wide",
)

st.markdown(
    """
    <style>
      .stApp { background: #0e1712; color: #f5f7f5; }
      [data-testid="stSidebar"] { background: #14231a; }
      h1, h2, h3 { color: #1ed760 !important; }
      div[data-testid="stMetric"] {
        background: #17291e; padding: 14px; border-radius: 12px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎧 Spotify Music Taste Clustering")
st.caption("An interactive viewer for the results of your existing K-Means model.")
st.info(
    "This website uses the saved clustering outputs from your completed ML project. "
    "It does not train or retrain a model, and it does not predict labels for new tracks."
)

@st.cache_data
def load_data():
    tracks_path = OUTPUTS / "engineered_playlists.csv"
    means_path = OUTPUTS / "cluster_feature_means.csv"
    quality_path = OUTPUTS / "data_quality_summary.csv"
    summary_path = OUTPUTS / "cluster_summary.txt"

    for path in (tracks_path, means_path, quality_path, summary_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing file: {path.relative_to(ROOT)}. "
                "Make sure the spotify_outputs folder is included in the deployment."
            )

    tracks = pd.read_csv(tracks_path)
    means = pd.read_csv(means_path)
    quality = pd.read_csv(quality_path)
    summary = summary_path.read_text(encoding="utf-8")
    return tracks, means, quality, summary

try:
    tracks, means, quality, summary = load_data()
except Exception as exc:
    st.error(f"Could not load the existing project outputs: {exc}")
    st.stop()

expected_track_columns = {"Track_Name", "Artist", "Vibe_Cluster_Label"}
expected_mean_columns = {
    "Cluster_ID", "danceability", "energy", "acousticness", "tempo",
    "Vibe_Cluster_Label",
}
if not expected_track_columns.issubset(tracks.columns):
    st.error("engineered_playlists.csv does not have the expected columns.")
    st.stop()
if not expected_mean_columns.issubset(means.columns):
    st.error("cluster_feature_means.csv does not have the expected columns.")
    st.stop()

tracks = tracks.dropna(subset=["Vibe_Cluster_Label"]).copy()
tracks["Vibe_Cluster_Label"] = tracks["Vibe_Cluster_Label"].astype(str)
vibe_counts = (
    tracks["Vibe_Cluster_Label"]
    .value_counts()
    .rename_axis("Vibe")
    .reset_index(name="Tracks")
)

# Read the existing data-quality metrics.
quality_values = dict(zip(quality["Metric"].astype(str), quality["Value"]))
input_rows = int(float(quality_values.get("Input rows", len(tracks))))
valid_rows = int(float(quality_values.get("Valid rows clustered", len(tracks))))
excluded_rows = int(float(quality_values.get("Rows excluded", max(0, input_rows-valid_rows))))

m1, m2, m3, m4 = st.columns(4)
m1.metric("Tracks clustered", f"{valid_rows:,}")
m2.metric("Vibe clusters", f"{tracks['Vibe_Cluster_Label'].nunique()}")
m3.metric("Rows excluded", f"{excluded_rows:,}")
m4.metric("Selected K", str(len(means)))

tab_overview, tab_playlists, tab_model, tab_quality = st.tabs(
    ["Overview", "Explore playlists", "Model results", "Data quality"]
)

with tab_overview:
    st.subheader("Your three discovered music vibes")
    cols = st.columns(len(means))
    for col, (_, row) in zip(cols, means.iterrows()):
        with col:
            st.markdown(f"### {row['Vibe_Cluster_Label']}")
            count = int(
                (tracks["Vibe_Cluster_Label"] == row["Vibe_Cluster_Label"]).sum()
            )
            st.metric("Tracks", f"{count:,}")
            st.caption(
                f"Energy {row['energy']:.2f} · "
                f"Danceability {row['danceability']:.2f}"
            )

    st.subheader("Tracks per vibe")
    count_fig = px.bar(
        vibe_counts,
        x="Vibe",
        y="Tracks",
        color="Vibe",
        text="Tracks",
        title="Distribution of tracks across the existing clusters",
    )
    count_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1712",
        plot_bgcolor="#14231a",
        showlegend=False,
    )
    st.plotly_chart(count_fig, use_container_width=True)

    st.subheader("Existing cluster feature averages")
    st.dataframe(means, use_container_width=True, hide_index=True)

with tab_playlists:
    st.subheader("Explore your generated playlists")
    vibes = sorted(tracks["Vibe_Cluster_Label"].unique())
    selected_vibe = st.selectbox("Choose a music vibe", vibes)
    search_text = st.text_input("Search by track name or artist")
    playlist = tracks.loc[
        tracks["Vibe_Cluster_Label"] == selected_vibe,
        ["Track_Name", "Artist", "Vibe_Cluster_Label"],
    ].copy()

    if search_text.strip():
        query = search_text.strip()
        mask = (
            playlist["Track_Name"].astype(str).str.contains(query, case=False, na=False)
            | playlist["Artist"].astype(str).str.contains(query, case=False, na=False)
        )
        playlist = playlist.loc[mask]

    st.write(f"**{len(playlist):,} tracks**")
    st.dataframe(playlist, use_container_width=True, hide_index=True, height=480)
    st.download_button(
        "⬇️ Download selected vibe playlist",
        data=playlist.to_csv(index=False).encode("utf-8"),
        file_name="engineered_playlists.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.download_button(
        "Download all existing clustered tracks",
        data=tracks[
            ["Track_Name", "Artist", "Vibe_Cluster_Label"]
        ].to_csv(index=False).encode("utf-8"),
        file_name="all_engineered_playlists.csv",
        mime="text/csv",
        use_container_width=True,
    )

with tab_model:
    st.subheader("Original model evaluation plots")
    st.markdown("These are the plots saved by your original ML run.")
    plot_files = [
        ("elbow_method.png", "Elbow method — inertia for K=1 to K=10"),
        ("silhouette_scores.png", "Silhouette comparison — K=3 to K=6"),
        ("spotify_vibe_clusters.png", "Danceability vs energy — existing cluster assignments"),
    ]
    for filename, caption in plot_files:
        path = OUTPUTS / filename
        if path.exists():
            st.image(str(path), caption=caption, use_container_width=True)
        else:
            st.warning(f"Plot file not found: {filename}")

    st.subheader("Saved model summary")
    st.code(summary, language="text")

with tab_quality:
    st.subheader("Dataset quality summary from the original run")
    st.dataframe(quality, use_container_width=True, hide_index=True)
    st.caption(
        "The numbers above are read from the saved output files; "
        "this website does not reprocess the source dataset."
    )

st.divider()
st.caption(
    "Spotify Music Taste Clustering · Existing ML results viewer · "
    "No model training or retraining occurs in this app."
)
