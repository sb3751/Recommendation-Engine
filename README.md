🛒 Intelligent Hybrid Recommendation Engine

Content-Based + Collaborative Filtering (SVD)

📌 Project Overview

This project implements an end-to-end Intelligent Recommendation System using real user–item interaction data.
It incrementally builds and evaluates multiple recommendation approaches and finally integrates them into a hybrid recommendation engine, similar to systems used in modern e-commerce and content platforms.

The system is designed with:

Reproducibility

Explainability

Academic rigor

Industry relevance

in mind.

🎯 Objectives

Design a recommendation engine using user behavior data

Compare memory-based and model-based collaborative filtering

Address cold-start problems

Evaluate systems using RMSE and Precision@K

Build an industry-grade hybrid recommender

Discuss bias, fairness, and ethical implications

📚 Literature Survey (Conceptual Foundation)

This project is inspired by and aligned with established research in recommender systems:

Core Approaches

Content-Based Filtering
Recommends items similar to those a user previously liked based on item features.

Collaborative Filtering (CF)
Learns from interactions across multiple users.

Matrix Factorization (SVD)
Projects users and items into a shared latent space to reduce sparsity.

Hybrid Recommendation Systems
Combine multiple models to offset individual weaknesses.

Key Observations from Literature

RMSE alone does not correlate well with recommendation usefulness.

Ranking metrics (Precision@K, Recall@K) are more indicative of user satisfaction.

Pure CF suffers from cold-start and popularity bias.

Hybrid systems are the de facto standard in industry and research.

This project explicitly demonstrates these findings through experimentation.

🧱 System Architecture & Design
High-Level Architecture
Raw Data (MovieLens)
        ↓
Preprocessing & Time-Aware Split
        ↓
┌─────────────────────────────┐
│ Content-Based Recommender   │
│  - Genre TF-IDF             │
│  - Cosine Similarity        │
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│ Collaborative Filtering     │
│  - User-Based CF            │
│  - SVD Matrix Factorization │
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│ Hybrid Recommendation Engine│
│  - Weighted Fusion          │
│  - Cold-Start Robust        │
└─────────────────────────────┘
        ↓
Evaluation & Ethical Analysis

📂 Repository Structure
recommendation-engine/
│
├── data/
│   ├── raw/                  # Original MovieLens dataset
│   └── processed/            # Train/Test CSVs
│
├── src/
│   ├── content_model.py      # Content-based recommender
│   ├── collaborative_model.py# User-based CF
│   ├── svd_model.py          # SVD-based CF
│   ├── hybrid_model.py       # Hybrid engine
│   ├── metrics.py            # RMSE & Precision@K
│
├── run_phase1.py             # Data preprocessing
├── run_phase4_eval.py        # CF evaluation
├── run_phase5_svd.py         # SVD execution
├── run_phase6_eval_svd.py    # SVD evaluation
├── run_phase7_hybrid.py      # Hybrid recommender
│
├── visuals/                  # (Optional) plots & diagrams
├── report/                   # (Optional) report drafts
└── README.md

⚙️ Requirements
Software

Python 3.10+ (tested on Python 3.13)

pip

Python Libraries
pandas
numpy
scikit-learn


Install all dependencies using:

pip install pandas numpy scikit-learn

▶️ How to Run the Project (From Scratch)
Step 1: Clone or Copy Repository
git clone <repository-url>
cd recommendation-engine

Step 2: Add Dataset

Download MovieLens 100K and place it as:

data/raw/ml-100k/


Required files:

u.data

u.item

u.user

Step 3: Preprocess Data
python run_phase1.py


Creates:

data/processed/train.csv
data/processed/test.csv

Step 4: Run Individual Models
Content-Based Recommendation
python src/content_model.py

User-Based Collaborative Filtering
python src/collaborative_model.py

SVD Collaborative Filtering
python run_phase5_svd.py

Step 5: Evaluation
Evaluate User-Based CF
python run_phase4_eval.py

Evaluate SVD
python run_phase6_eval_svd.py


Metrics:

RMSE

Precision@5

Step 6: Hybrid Recommendation Engine
python run_phase7_hybrid.py


Outputs final hybrid recommendations.

🧪 What You Can Experiment With

By running and modifying the project, you can:

Compare CF vs SVD vs Hybrid

Observe how Precision@K improves with SVD

Tune hybrid weight α

Study cold-start behavior

Analyze sparsity effects

Modify relevance thresholds

Add diversity constraints

This repository is intentionally designed for experimentation.

❄️ Cold-Start Handling
Scenario	Strategy
New User	Content-based + popularity prior
New Item	Content similarity only
Sparse User	Adaptive hybrid weighting

These strategies are documented and implemented in the hybrid design.

⚖️ Bias, Fairness & Ethics

The project explicitly acknowledges:

Popularity Bias

Filter Bubbles

Exposure Inequality

Mitigation strategies discussed:

Diversity-aware re-ranking

Controlled exploration

Fair exposure constraints

This aligns with modern Responsible AI expectations.

🚀 Expansion Possibilities

This project can be extended into:

Technical Enhancements

Bias-aware SVD

Implicit feedback modeling

Neural collaborative filtering

Approximate nearest neighbors (ANN)

Online learning

System-Level Enhancements

REST API (FastAPI)

Real-time inference

Power BI / dashboard visualization

User profiling & segmentation

Research Extensions

Fairness metrics

Explainable recommendations

Cross-domain recommendation

Long-tail optimization

### Baseline 1: Popularity-Based Recommendation

As a non-personalized baseline, 
we implemented a popularity-based recommender that ranks items by global interaction frequency. 
This model ignores user preferences and also,
serves as a control to evaluate the value of personalization.

popular_items = (
    train.groupby("item_id")
    .size()
    .sort_values(ascending=False)
    .head(5)
)

### Baseline 2: Content-Based Recommendation

The content-based recommender uses item metadata (genres) 
to recommend items similar to those previously liked by a user. 
This baseline performs well under cold-start conditions but lacks collaborative personalization.

### Comparative Analysis of Recommendation Models

| Model | Personalization | Precision@5 | Cold-Start Handling |
|-----|----------------|-------------|---------------------|
| Popularity-Based | ❌ | Low | ✅ |
| Content-Based | ⚠️ | Medium | ✅ |
| User-Based CF | ✅ | 0.00 | ❌ |
| SVD (Matrix Factorization) | ✅ | ~0.20 | ❌ |
| Hybrid (Content + SVD) | ✅ | Best | ✅ |

The results show that while collaborative filtering models improve personalization, 
they struggle under cold-start conditions. 
The hybrid model achieves the best overall performance 
by combining collaborative signals with content-based robustness.

### Cold-Start Analysis

**New Users**
For users with no interaction history, 
collaborative filtering assumptions break due to the absence of similarity signals. 
The system falls back to content-based recommendations and 
popularity priors until sufficient interactions are collected.

**New Items**
New items lack interaction data and cannot be placed reliably in the collaborative latent space. 
Content-based similarity enables immediate recommendation based on item metadata.

**Sparse Interaction Profiles**
Users with limited interactions yield unstable similarity estimates. 
The hybrid model mitigates this by dynamically relying more on content-based signals in early stages.

### Limitations

- Evaluation is performed offline and does not capture real-time user feedback.
- Rating prediction accuracy (RMSE) does not fully reflect user satisfaction.
- Fairness metrics are discussed conceptually but not enforced algorithmically.
- The dataset represents explicit feedback and may not generalize to implicit interaction settings.

### Future Work

- Bias-aware matrix factorization
- Diversity-constrained recommendation
- Implicit feedback modeling
- Neural collaborative filtering
- Online learning and A/B testing
