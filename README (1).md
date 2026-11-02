# Spotify Music Taste Clustering — Streamlit Deployment

This repository contains a Streamlit website for **viewing the outputs of the existing Spotify K-Means project**.

## Important: no model retraining

The app does not call `KMeans.fit`, `fit_predict`, `StandardScaler.fit`, or any training routine.
It reads the existing files under `spotify_outputs/` and displays:
- Existing cluster profiles and vibe labels
- Existing cluster counts
- Existing elbow, silhouette, and cluster scatter plots
- Existing generated playlists, with search and CSV download

It does not assign a cluster to previously unseen tracks. The original project output package contains cluster labels and summary files, but not a serialized fitted K-Means model and scaler.

## Files

```text
.
├── app.py
├── requirements.txt
└── spotify_outputs/
    ├── engineered_playlists.csv
    ├── cluster_feature_means.csv
    ├── cluster_summary.txt
    ├── data_quality_summary.csv
    ├── elbow_method.png
    ├── silhouette_scores.png
    └── spotify_vibe_clusters.png
```

## Run locally (optional)

1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the app:

   ```bash
   streamlit run app.py
   ```

5. Open the local address Streamlit prints (usually http://localhost:8501).

## Deploy on Streamlit Community Cloud

1. Sign in to GitHub or create an account at https://github.com/.
2. Create a **new repository** (for example, `spotify-music-vibe-app`).
3. Upload `app.py`, `requirements.txt`, and the entire `spotify_outputs` folder.
   - Keep the folder name exactly `spotify_outputs`.
   - Do not upload only the CSV files; the app also uses the saved plots and summary.
4. Confirm the repository file tree matches the structure shown above.
5. Open https://share.streamlit.io/ and sign in with GitHub.
6. Choose **Create app** / **Deploy an app**.
7. Select your repository, branch (usually `main`), and main file path `app.py`.
8. Deploy. Streamlit Community Cloud will install packages from `requirements.txt`.
9. When deployment completes, open the app URL and test the Overview, Explore playlists, Model results, and Data quality tabs.

## Updating the site

If you replace any existing output file in `spotify_outputs/` and push the change to GitHub, Streamlit Cloud will redeploy the app.

## Troubleshooting

- **Missing file error:** ensure every file listed in `spotify_outputs/` is committed to GitHub.
- **App cannot find outputs:** verify the folder is named exactly `spotify_outputs` and is at the same level as `app.py`.
- **Dependency error:** check that `requirements.txt` is in the repository root.
- **Large repository warning:** `engineered_playlists.csv` is several MB; this should be fine for a small demo repository, but avoid committing the original 19 MB source dataset because this viewer does not need it.
