# Ship of Theseus NLP Project

**Iterative Decay of Human Style through LLM Paraphrasing**

*CS 6120 Final Project — Northeastern University, Seattle, 2025*

*Jae Hun Cho & Catherine Nguyen*

---

If you replace every plank of a ship one by one, is it still the same ship? We apply this question to text: if an LLM paraphrases a document repeatedly (T0 → T1 → T2 → T3), at what point is the original author no longer detectable?

## Research Questions

**RQ1 — Style vs. Content Decay:** Which linguistic layers erode first? We compare decay rates across lexical (BLEU), semantic (BERTScore, SBERT), structural (POS cosine), and content (NER) metrics.

**RQ2 — Point of No Return:** At what iteration does authorship attribution collapse? A classifier trained on T0 (Human vs. AI) is tested on T1-T3 to track identity erosion.

**RQ3 — Paraphraser Fingerprints:** Do different paraphrasers leave distinct stylistic signatures? Can we identify which model produced the text?

## Key Findings

| Layer | Metric | T3 Score | Decay |
|-------|--------|----------|-------|
| Structure | POS Cosine | 0.870 | -13% |
| Semantics | BERTScore F1 | 0.810 | -19% |
| Semantics | SBERT Cosine | 0.730 | -27% |
| Content | NER Jaccard | 0.410 | -59% |
| Lexical | BLEU-2 | 0.280 | -72% |

- **The point of no return is not reached by T3** — attribution accuracy remains at 79.5%, well above the 50% chance baseline.
- **All paraphrasers leave identifiable fingerprints** (F1 = 0.55 vs. 0.25 chance). ChatGPT is most identifiable (F1 = 0.70).
- **Decay hierarchy:** Lexical > Content > Semantics > Structure — words are replaced long before meaning is lost.

## Dataset

The [Ship of Theseus Corpus](https://aclanthology.org/2024.acl-long.271/) (Tripto et al., ACL 2024) contains iterative paraphrase chains across 7 source datasets:

| Dataset | Domain | Style |
|---------|--------|-------|
| sci_gen | Scientific | Formal, technical |
| wp | Creative fiction | Narrative, diverse |
| xsum | News summaries | Journalistic |
| eli5 | Explanatory QA | Informal, varied |
| cmv | Debate/persuasion | Argumentative |
| tldr | Summarization | Compressed |
| yelp | Reviews | Colloquial |

**19,342 source documents, 420,012 total rows.**

## Paraphrasers

Four paraphraser families with distinct architectures:

- **ChatGPT** — Instruction-tuned LLM (GPT-3.5). Aggressively rewrites style and vocabulary, causing the most identity erosion (-28.4pp by T3).
- **PaLM2** — Google's instruction-tuned LLM. Moderate rewriting, falls between ChatGPT and Pegasus.
- **Pegasus** (`slight` / `full`) — Seq2seq summarization model. Preserves original style most faithfully (-4.5pp by T3) because it shortens rather than restyles.
- **Dipper** (`low` / `high`) — Discourse-level paraphraser purpose-built for rewriting. `high` variant causes the steepest entity erosion.

Variants are grouped under their shared family for cross-system comparison.

## Iterations and Version Names

Each document is tracked across four stages:
- **T0**: Original human-authored or AI-generated text
- **T1**: First paraphrase
- **T2**: Second paraphrase (of T1)
- **T3**: Third paraphrase (of T2)

The `version_name` column records the rewrite chain. For example, `chatgpt_chatgpt_chatgpt` = T3 produced by applying ChatGPT three times in succession.

## Run the Dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard provides:
- Document explorer (T0 vs T3 side-by-side)
- Similarity decay curves by metric and paraphraser family
- Stylometric drift analysis with POS heatmaps
- NER entity retention tracking
- Authorship attribution analysis
- Multi-modal audit comparing all linguistic layers

## Project Structure

```
ShipOfTheseus-NLP-Project/
├── app.py                          # Streamlit dashboard
├── requirements.txt                # Python dependencies
├── src/
│   ├── models/
│   │   ├── attribution.py         # Human vs. AI classification (RQ2)
│   │   └── fingerprint.py         # Paraphraser identification (RQ3)
│   ├── features/
│   │   ├── stylometry.py          # Linguistic feature extraction + POS cosine
│   │   └── ner.py                 # Named entity retention analysis
│   ├── similarity/
│   │   ├── sbert.py               # SBERT sentence embeddings
│   │   ├── bertscore.py           # BERTScore semantic matching
│   │   └── bleu_rouge.py          # Lexical overlap metrics
│   ├── data/
│   │   ├── load_data.py           # CSV loading & column derivation
│   │   ├── split_data.py          # Chain building (T0 → T3)
│   │   └── preprocess.py          # Text cleaning
│   └── utils/
│       └── config.py              # Paths, dataset/paraphraser definitions
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_similarity.ipynb
│   ├── 03_stylometry_analysis.ipynb
│   ├── 04_identity_forensics.ipynb
│   └── 05_ner_analysis.ipynb
├── scripts/
│   └── compute_sbert_and_error_analysis.py
├── experiments/                    # Pre-computed results (loaded by dashboard)
│   ├── baseline_similarity/       # Scored chains, SBERT results
│   ├── stylometry/                # Feature summaries, POS cosine
│   ├── identity_forensics/        # Attribution, fingerprinting, error analysis
│   └── ner/                       # Entity retention metrics
└── paper/
    └── update2.tex                # LaTeX paper
```

## References

- Tripto et al., "A Ship of Theseus: Curious Cases of Paraphrasing in LLM-Generated Texts," *ACL*, 2024.
