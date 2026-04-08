"""
Ship of Theseus — Style-Drift Dashboard
Streamlit app for exploring how LLM paraphrasers erode linguistic identity.

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_PROCESSED = ROOT / "data" / "processed"
EXP_BASELINE = ROOT / "experiments" / "baseline_similarity"
EXP_STYLOMETRY = ROOT / "experiments" / "stylometry"
EXP_NER = ROOT / "experiments" / "ner"
EXP_FORENSICS = ROOT / "experiments" / "identity_forensics"
FIGURES = ROOT / "figures"

DATASETS = ["sci_gen", "wp", "xsum", "eli5", "cmv", "tldr", "yelp"]
STAGES = ["T0", "T1", "T2", "T3"]

st.set_page_config(
    page_title="Ship of Theseus — Style-Drift Dashboard",
    page_icon="⛵",
    layout="wide",
)


# ── Data loading (cached) ───────────────────────────────────────────────────
@st.cache_data
def load_scored_chains():
    """Load all scored chain CSVs with similarity metrics."""
    frames = []
    for ds in DATASETS:
        path = EXP_BASELINE / f"{ds}_chains_scored.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["dataset"] = ds
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@st.cache_data
def load_experiment_csv(path):
    if Path(path).exists():
        return pd.read_csv(path)
    return pd.DataFrame()


# ── Load data ────────────────────────────────────────────────────────────────
chains = load_scored_chains()
stylometry_summary = load_experiment_csv(EXP_STYLOMETRY / "feature_summary_by_stage.csv")
ner_summary = load_experiment_csv(EXP_NER / "ner_retention_summary.csv")
ner_family = load_experiment_csv(EXP_NER / "ner_retention_by_family.csv")
ner_domain = load_experiment_csv(EXP_NER / "ner_retention_by_domain.csv")
attribution = load_experiment_csv(EXP_FORENSICS / "attribution_results_balanced.csv")

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("⛵ Ship of Theseus")
st.sidebar.markdown("**Iterative Decay of Human Style through LLM Paraphrasing**")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "Overview",
    "Document Explorer",
    "Similarity Decay",
    "Stylometric Drift",
    "Entity Retention (NER)",
    "Authorship Attribution",
    "Multi-Modal Audit",
])

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("⛵ The Ship of Theseus — Style-Drift Dashboard")
    st.markdown("""
    > *If every plank of a ship is replaced one by one, is it still the same ship?*

    This dashboard explores what happens when LLM paraphrasers iteratively rewrite
    human text. We track **four layers of linguistic identity** across T0 → T3:
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Lexical (BLEU-2)", "0.28 at T3", "-72%")
    col2.metric("Content (NER Jaccard)", "0.41 at T3", "-59%")
    col3.metric("Semantic (BERTScore)", "0.81 at T3", "-19%")
    col4.metric("Structure (POS Cos)", "0.95 at T3", "-5%")

    st.markdown("""
    ### Decay Hierarchy
    **Lexical skin** is replaced first, **concrete entities** erode significantly,
    **semantic meaning** mostly survives, and the **syntactic skeleton** barely changes.

    ---
    **Corpus:** 7 datasets, 19,342 source documents, 420,012 total rows

    **Paraphrasers:** ChatGPT, PaLM2, Pegasus (slight/full), Dipper (low/high)

    **Iterations:** T0 (original) → T1 → T2 → T3
    """)

    if not chains.empty:
        st.markdown("### Corpus at a Glance")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Documents per dataset**")
            st.bar_chart(chains.groupby("dataset").size(), height=300)
        with col2:
            st.markdown("**Paraphraser family distribution**")
            st.bar_chart(chains.groupby("family").size(), height=300)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Document Explorer — T0 vs T3 side-by-side
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Document Explorer":
    st.title("📄 Document Explorer — T0 vs T3")
    st.markdown("Compare original text with its 3rd-generation paraphrase side-by-side.")

    if chains.empty:
        st.error("No scored chain data found.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            sel_dataset = st.selectbox("Dataset", DATASETS)
        with col2:
            available_families = chains[chains["dataset"] == sel_dataset]["family"].unique()
            sel_family = st.selectbox("Paraphraser Family", sorted(available_families))

        filtered = chains[(chains["dataset"] == sel_dataset) & (chains["family"] == sel_family)]

        if len(filtered) == 0:
            st.warning("No data for this combination.")
        else:
            doc_idx = st.slider("Document index", 0, len(filtered) - 1, 0)
            row = filtered.iloc[doc_idx]

            # Similarity scores
            st.markdown("### Similarity Scores (vs. T0)")
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("BLEU", f"{row.get('bleu_T3', 'N/A'):.3f}" if pd.notna(row.get('bleu_T3')) else "N/A")
            mcol2.metric("ROUGE-1", f"{row.get('rouge1_T3', 'N/A'):.3f}" if pd.notna(row.get('rouge1_T3')) else "N/A")
            mcol3.metric("ROUGE-L", f"{row.get('rougeL_T3', 'N/A'):.3f}" if pd.notna(row.get('rougeL_T3')) else "N/A")
            mcol4.metric("BERTScore", f"{row.get('bertscore_T3', 'N/A'):.3f}" if pd.notna(row.get('bertscore_T3')) else "N/A")

            st.markdown("---")

            # Side-by-side text
            left, right = st.columns(2)
            with left:
                st.markdown("#### T0 — Original")
                st.markdown(f"**Source:** `{row['source']}` | **Key:** `{row['key']}`")
                st.text_area("T0 Text", row["T0"], height=300, disabled=True, key="t0_text")
            with right:
                st.markdown(f"#### T3 — After 3 Iterations ({sel_family})")
                st.text_area("T3 Text", row["T3"], height=300, disabled=True, key="t3_text")

            # Show intermediate stages in expander
            with st.expander("Show T1 and T2 (intermediate stages)"):
                st.markdown("**T1:**")
                st.write(row["T1"])
                st.markdown("**T2:**")
                st.write(row["T2"])


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Similarity Decay
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Similarity Decay":
    st.title("📉 Similarity Decay Across Iterations")

    if chains.empty:
        st.error("No scored chain data found.")
    else:
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            sel_datasets = st.multiselect("Datasets", DATASETS, default=DATASETS)
        with col2:
            sel_metric = st.selectbox("Metric", ["bleu", "rouge1", "rougeL", "bertscore"])

        filtered = chains[chains["dataset"].isin(sel_datasets)]

        # Compute means by family and stage
        fig, ax = plt.subplots(figsize=(10, 5))
        family_colors = {
            'chatgpt': '#E53935', 'dipper': '#1E88E5',
            'pegasus': '#43A047', 'palm': '#FDD835',
        }

        for family in sorted(filtered["family"].unique()):
            fam_df = filtered[filtered["family"] == family]
            means = [1.0]
            for stage in ["T1", "T2", "T3"]:
                col = f"{sel_metric}_{stage}"
                if col in fam_df.columns:
                    means.append(fam_df[col].mean())
            ax.plot(STAGES, means, "o-", color=family_colors.get(family, "gray"),
                    label=family.capitalize(), linewidth=2, markersize=7)

        ax.set_ylabel(f"{sel_metric} (vs. T0)")
        ax.set_xlabel("Paraphrase Stage")
        ax.set_title(f"{sel_metric.upper()} Decay by Paraphraser Family")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)
        st.pyplot(fig)

        # Per-dataset breakdown
        st.markdown("### Per-Dataset Breakdown")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        for ds in sel_datasets:
            ds_df = filtered[filtered["dataset"] == ds]
            means = [1.0]
            for stage in ["T1", "T2", "T3"]:
                col = f"{sel_metric}_{stage}"
                if col in ds_df.columns:
                    means.append(ds_df[col].mean())
            ax2.plot(STAGES, means, "o-", label=ds, linewidth=2, markersize=6)

        ax2.set_ylabel(f"{sel_metric} (vs. T0)")
        ax2.set_xlabel("Paraphrase Stage")
        ax2.legend()
        ax2.grid(alpha=0.3)
        ax2.set_ylim(0, 1.05)
        st.pyplot(fig2)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Stylometric Drift
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Stylometric Drift":
    st.title("🔬 Stylometric Feature Drift")

    if stylometry_summary.empty:
        st.error("No stylometry data found.")
    else:
        st.markdown("Mean stylometric features across T0 → T3 (wp, xsum, yelp).")
        st.dataframe(stylometry_summary, use_container_width=True)

        # Show saved figures
        fig_path = FIGURES / "stylometry"
        for fig_file in sorted(fig_path.glob("*.png")):
            st.image(str(fig_file), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Entity Retention (NER)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Entity Retention (NER)":
    st.title("🏷️ Named Entity Retention — Content Erasure")

    if ner_summary.empty:
        st.error("No NER data found. Run notebook 05 first.")
    else:
        st.markdown("### Overall Entity Retention")
        st.dataframe(ner_summary, use_container_width=True)

        tab1, tab2, tab3 = st.tabs(["By Family", "By Domain", "Figures"])

        with tab1:
            if not ner_family.empty:
                st.dataframe(ner_family, use_container_width=True)

        with tab2:
            if not ner_domain.empty:
                st.dataframe(ner_domain, use_container_width=True)

        with tab3:
            fig_path = FIGURES / "ner"
            for fig_file in sorted(fig_path.glob("*.png")):
                st.image(str(fig_file), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Authorship Attribution
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Authorship Attribution":
    st.title("🔍 Authorship Attribution — Point of No Return")

    if chains.empty:
        st.error("No data found.")
    else:
        st.markdown("""
        A classifier trained on T0 (Human vs. AI) is evaluated on T1/T2/T3.
        How quickly does authorship identity collapse?
        """)

        # --- Attribution accuracy table ---
        if not attribution.empty:
            st.dataframe(attribution, use_container_width=True)

        # --- Per-family attribution data (from paper) ---
        family_attr = pd.DataFrame({
            "Family": ["ChatGPT", "Dipper", "PaLM", "Pegasus"],
            "T0": [0.980, 0.975, 0.959, 0.961],
            "T1": [0.743, 0.757, 0.878, 0.958],
            "T2": [0.831, 0.726, 0.857, 0.926],
            "T3": [0.696, 0.734, 0.841, 0.916],
        })

        st.markdown("---")
        st.markdown("### Interactive Attribution Collapse")

        # Filters
        col1, col2 = st.columns(2)
        with col1:
            sel_stages = st.multiselect(
                "Select stages to display",
                STAGES, default=STAGES, key="attr_stages"
            )
        with col2:
            sel_families = st.multiselect(
                "Select paraphraser families",
                family_attr["Family"].tolist(),
                default=family_attr["Family"].tolist(),
                key="attr_families"
            )

        # Interactive line chart: attribution collapse by family
        filtered_attr = family_attr[family_attr["Family"].isin(sel_families)]
        family_colors_map = {
            "ChatGPT": "#E53935", "Dipper": "#1E88E5",
            "PaLM": "#FDD835", "Pegasus": "#43A047",
        }

        fig_attr = go.Figure()
        for _, row in filtered_attr.iterrows():
            vals = [row[s] for s in STAGES if s in sel_stages]
            stages_filtered = [s for s in STAGES if s in sel_stages]
            fig_attr.add_trace(go.Scatter(
                x=stages_filtered, y=vals, mode="lines+markers",
                name=row["Family"],
                line=dict(color=family_colors_map.get(row["Family"], "gray"), width=3),
                marker=dict(size=10),
            ))

        fig_attr.update_layout(
            title="Attribution Accuracy by Paraphraser Family",
            xaxis_title="Paraphrase Stage",
            yaxis_title="Accuracy",
            yaxis=dict(range=[0.5, 1.02]),
            template="plotly_white",
            height=450,
            legend=dict(font=dict(size=14)),
        )
        fig_attr.add_hline(y=0.5, line_dash="dash", line_color="gray",
                           annotation_text="Chance (50%)")
        st.plotly_chart(fig_attr, use_container_width=True)

        # --- Interactive t-SNE ---
        st.markdown("---")
        st.markdown("### Identity Trajectory (t-SNE)")
        st.markdown("Visualize how text identity drifts across iterations in embedding space.")

        @st.cache_data(show_spinner="Computing t-SNE embeddings (first load only)...")
        def compute_tsne(chains_df, sample_per_dataset=100):
            """TF-IDF → PCA(50) → t-SNE(2) for all stages."""
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import PCA
            from sklearn.manifold import TSNE

            # Sample for speed
            sampled = chains_df.groupby("dataset").apply(
                lambda x: x.sample(min(sample_per_dataset, len(x)), random_state=42),
                include_groups=False,
            ).reset_index(drop=True)
            # Re-add dataset column lost by include_groups=False
            sampled_full = chains_df.groupby("dataset").apply(
                lambda x: x.sample(min(sample_per_dataset, len(x)), random_state=42),
            ).reset_index(drop=True)

            texts, labels_stage, labels_family, labels_dataset = [], [], [], []
            for _, row in sampled_full.iterrows():
                for stage in STAGES:
                    if pd.notna(row.get(stage)):
                        texts.append(str(row[stage]))
                        labels_stage.append(stage)
                        labels_family.append(row.get("family", "unknown"))
                        labels_dataset.append(row.get("dataset", "unknown"))

            tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), sublinear_tf=True)
            X = tfidf.fit_transform(texts)
            X_pca = PCA(n_components=50, random_state=42).fit_transform(X.toarray())
            X_tsne = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_pca)

            return pd.DataFrame({
                "t-SNE 1": X_tsne[:, 0],
                "t-SNE 2": X_tsne[:, 1],
                "Stage": labels_stage,
                "Family": labels_family,
                "Dataset": labels_dataset,
            })

        tsne_df = compute_tsne(chains)

        # t-SNE filters
        col1, col2 = st.columns(2)
        with col1:
            tsne_color = st.radio("Color by", ["Stage", "Family"], horizontal=True)
        with col2:
            if tsne_color == "Stage":
                tsne_filter = st.multiselect(
                    "Filter stages", STAGES, default=STAGES, key="tsne_stages"
                )
            else:
                all_families = tsne_df["Family"].unique().tolist()
                tsne_filter = st.multiselect(
                    "Filter families", all_families, default=all_families, key="tsne_families"
                )

        # Filter
        if tsne_color == "Stage":
            tsne_filtered = tsne_df[tsne_df["Stage"].isin(tsne_filter)]
            color_col = "Stage"
            color_map = {"T0": "#1565C0", "T1": "#7B1FA2", "T2": "#E65100", "T3": "#C62828"}
            cat_order = {"Stage": [s for s in STAGES if s in tsne_filter]}
        else:
            tsne_filtered = tsne_df[tsne_df["Family"].isin(tsne_filter)]
            color_col = "Family"
            color_map = {"chatgpt": "#E53935", "dipper": "#1E88E5",
                         "pegasus": "#43A047", "palm": "#FDD835"}
            cat_order = {"Family": [f for f in tsne_filtered["Family"].unique()]}

        fig_tsne = px.scatter(
            tsne_filtered, x="t-SNE 1", y="t-SNE 2",
            color=color_col, color_discrete_map=color_map,
            category_orders=cat_order,
            opacity=0.5, hover_data=["Dataset", "Stage", "Family"],
            title=f"Identity Trajectory — colored by {color_col}",
            height=550,
        )
        fig_tsne.update_traces(marker=dict(size=5))
        fig_tsne.update_layout(template="plotly_white",
                               legend=dict(font=dict(size=14)))
        st.plotly_chart(fig_tsne, use_container_width=True)

        # --- Per-domain error analysis ---
        error_domain = load_experiment_csv(EXP_FORENSICS / "error_analysis_by_domain.csv")
        if not error_domain.empty:
            st.markdown("---")
            st.markdown("### Per-Domain Attribution Accuracy at T3")
            fig_domain = px.bar(
                error_domain.sort_values("accuracy", ascending=True),
                x="accuracy", y="dataset", orientation="h",
                color="accuracy", color_continuous_scale="RdYlGn",
                range_color=[0.5, 1.0],
                title="Which domains are most vulnerable to identity loss?",
                height=350,
            )
            fig_domain.update_layout(template="plotly_white", yaxis_title="",
                                     xaxis_title="Accuracy at T3")
            st.plotly_chart(fig_domain, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Multi-Modal Audit
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Multi-Modal Audit":
    st.title("🎯 Multi-Modal Audit — Skeleton vs. Skin vs. Cargo")

    st.markdown("""
    The central finding: **different linguistic layers decay at dramatically different rates.**
    """)

    # Multi-modal comparison data
    modal_data = {
        "Layer": ["Structure (POS Cosine)", "Semantics (BERTScore F1)",
                   "Semantics (SBERT Cosine)", "Content (NER Jaccard)",
                   "Lexical (BLEU-2)"],
        "T0": [1.0, 1.0, 1.0, 1.0, 1.0],
        "T1": [0.965, 0.91, 0.79, 0.509, 0.54],
        "T2": [0.955, 0.86, 0.75, 0.446, 0.38],
        "T3": [0.946, 0.81, 0.73, 0.410, 0.28],
    }
    modal_df = pd.DataFrame(modal_data)

    st.dataframe(modal_df, use_container_width=True)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#1565C0", "#43A047", "#7CB342", "#E53935", "#9E9E9E"]
    markers = ["o", "^", "v", "s", "D"]
    linestyles = ["-", "--", "--", "-", ":"]

    for i, row in modal_df.iterrows():
        vals = [row["T0"], row["T1"], row["T2"], row["T3"]]
        ax.plot(STAGES, vals, marker=markers[i], color=colors[i],
                linestyle=linestyles[i], label=row["Layer"],
                linewidth=2.5, markersize=8)

    ax.set_xlabel("Paraphrase Stage", fontsize=12)
    ax.set_ylabel("Similarity to T0", fontsize=12)
    ax.set_title("Multi-Modal Decay: Which Layers Survive?", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    st.markdown("""
    ### The Paradox of Authorial Identity

    > If every linguistic marker is replaced by an AI, but meaning remains — **who is the author?**

    Our empirical data shows:
    - **The syntactic skeleton survives** (POS structure barely changes)
    - **The semantic hull holds** (BERTScore stays above 0.80)
    - **But the concrete cargo is eroding** (half of named entities lost by T3)
    - **The lexical skin is almost entirely replaced** (BLEU drops to 0.28)

    The "ship" retains its **shape** and **purpose**, but the **planks** (words),
    **cargo** (entities), and **paint** (style) have been systematically replaced.
    The authorship signal persists at 79.5% accuracy even at T3 — attenuated but not erased.
    The point of no return has not been reached within three iterations.
    """)


# ── Footer ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown(
    "*CS 6120 — Northeastern University, Seattle*\n\n"
    "Jae Hun Cho & Catherine Nguyen"
)
