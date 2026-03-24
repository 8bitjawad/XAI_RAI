"""
config.py  —  All static configuration for the XAI/RAI app.
Keeping this separate means app.py stays lean and this is easy to edit.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

# ── Global constants ──────────────────────────────────────────────────────────
RAI_FLAG_THRESHOLD = 0.10

# ── Path A: tabular domain configs ───────────────────────────────────────────
DOMAIN_CONFIG = {

    "💳 Finance — Credit Risk": {
        "domain_key":    "finance",
        "description":   "Predicts whether a loan applicant will **default** (1) or **repay** (0).",
        "sensitive_col": "gender",
        "features":      ["credit_score", "annual_income", "debt_to_income",
                          "loan_amount", "employment_years", "missed_payments"],
        "feature_descriptions": {
            "credit_score":     "300–850",
            "annual_income":    "USD e.g. 55000",
            "debt_to_income":   "ratio 0.0–1.0 e.g. 0.35",
            "loan_amount":      "USD e.g. 15000",
            "employment_years": "years at current job",
            "missed_payments":  "count in last 24 months",
        },
        "defaults":     [650, 55000, 0.35, 15000, 4, 1],
        "label_map":    {0: "✅ Will Repay", 1: "⚠️ Likely to Default"},
        "model_cls":    RandomForestClassifier,
        "model_kwargs": {"n_estimators": 50, "random_state": 42},
        "synth": lambda n: pd.DataFrame({
            "credit_score":     np.random.randint(300, 851, n),
            "annual_income":    np.random.randint(20000, 150000, n),
            "debt_to_income":   np.round(np.random.uniform(0.05, 0.95, n), 2),
            "loan_amount":      np.random.randint(1000, 50000, n),
            "employment_years": np.random.randint(0, 30, n),
            "missed_payments":  np.random.randint(0, 10, n),
            "gender":           np.random.choice(["M", "F"], n),
        }),
        "target": lambda df: (
            (df["missed_payments"] > 3).astype(int) |
            (df["debt_to_income"]  > 0.6).astype(int) |
            (df["credit_score"]    < 580).astype(int)
        ).clip(0, 1),
    },

    "🏥 Healthcare — Diabetes Risk": {
        "domain_key":    "healthcare",
        "description":   "Predicts whether a patient is **at risk of diabetes** (1) or not (0).",
        "sensitive_col": "age_group",
        "features":      ["glucose_level", "bmi", "blood_pressure",
                          "insulin", "skin_thickness", "pregnancies"],
        "feature_descriptions": {
            "glucose_level":  "mg/dL e.g. 120",
            "bmi":            "body mass index e.g. 28.5",
            "blood_pressure": "mmHg diastolic e.g. 80",
            "insulin":        "μU/mL e.g. 85",
            "skin_thickness": "mm triceps fold e.g. 20",
            "pregnancies":    "count e.g. 2",
        },
        "defaults":     [120, 28.5, 80, 85, 20, 2],
        "label_map":    {0: "✅ Low Risk", 1: "⚠️ Elevated Diabetes Risk"},
        "model_cls":    GradientBoostingClassifier,
        "model_kwargs": {"n_estimators": 50, "random_state": 42},
        "synth": lambda n: pd.DataFrame({
            "glucose_level":  np.random.randint(60, 200, n),
            "bmi":            np.round(np.random.uniform(18, 50, n), 1),
            "blood_pressure": np.random.randint(40, 120, n),
            "insulin":        np.random.randint(0, 400, n),
            "skin_thickness": np.random.randint(0, 60, n),
            "pregnancies":    np.random.randint(0, 15, n),
            "age_group":      np.random.choice(["<40", "40-60", ">60"], n),
        }),
        "target": lambda df: (
            (df["glucose_level"] > 140).astype(int) |
            (df["bmi"]           > 35 ).astype(int)
        ).clip(0, 1),
    },

    "🏠 Real Estate — Price Direction": {
        "domain_key":    "realestate",
        "description":   "Predicts whether a property price will go **up** (1) or **down** (0) next quarter.",
        "sensitive_col": "neighbourhood_type",
        "features":      ["sq_footage", "bedrooms", "bathrooms",
                          "age_years", "proximity_to_transit", "school_rating"],
        "feature_descriptions": {
            "sq_footage":           "sq ft e.g. 1800",
            "bedrooms":             "count e.g. 3",
            "bathrooms":            "count e.g. 2",
            "age_years":            "years since built e.g. 15",
            "proximity_to_transit": "minutes walk e.g. 10",
            "school_rating":        "1–10 e.g. 7",
        },
        "defaults":     [1800, 3, 2, 15, 10, 7],
        "label_map":    {0: "📉 Price likely to fall", 1: "📈 Price likely to rise"},
        "model_cls":    DecisionTreeClassifier,
        "model_kwargs": {"max_depth": 6, "random_state": 42},
        "synth": lambda n: pd.DataFrame({
            "sq_footage":           np.random.randint(500, 5000, n),
            "bedrooms":             np.random.randint(1, 7, n),
            "bathrooms":            np.random.randint(1, 5, n),
            "age_years":            np.random.randint(0, 80, n),
            "proximity_to_transit": np.random.randint(1, 60, n),
            "school_rating":        np.random.randint(1, 11, n),
            "neighbourhood_type":   np.random.choice(["Urban", "Suburban", "Rural"], n),
        }),
        "target": lambda df: (
            (df["school_rating"]        >= 7).astype(int) &
            (df["proximity_to_transit"] <= 15).astype(int)
        ).clip(0, 1),
    },
}

# ── Path B: text model configs ───────────────────────────────────────────────
TEXT_MODEL_CONFIG = {

    "🎭 Sentiment — DistilBERT SST-2": {
        "model_id":   "distilbert-base-uncased-finetuned-sst-2-english",
        "domain_key": "text",
        "description": "Predicts **POSITIVE / NEGATIVE** sentiment. Explains which words drove the verdict.",
        "what_to_watch": "Watch positive words (fantastic, great) push toward POSITIVE and negative words push toward NEGATIVE. Detoxify runs as an independent second audit.",
        "examples": [
            "Select an example…",
            "This market rally was absolutely fantastic — best quarter of the year!",
            "I hated every single minute of this terrible earnings call.",
            "The trade was okay, nothing special but not a total disaster.",
            "You are completely useless and I hate everything about this.",
        ],
    },

    "☣️ Toxic Classifier — Unitary ToxicBERT": {
        "model_id":   "unitary/toxic-bert",
        "domain_key": "text",
        "description": "Predicts **TOXIC / NON_TOXIC**. Runs alongside Detoxify — two independent systems auditing the same input.",
        "what_to_watch": "The classifier predicts toxicity as a task; Detoxify audits it independently. When they disagree, that's an edge case worth flagging for human review.",
        "examples": [
            "Select an example…",
            "I hope you have a wonderful day, stay safe everyone.",
            "You are a complete idiot and should be ashamed of yourself.",
            "The quarterly results were disappointing but we remain optimistic.",
            "I will find you and make you regret everything you have ever done.",
        ],
    },
}

# ── Path C: generative LLM chatbot presets ───────────────────────────────────
CHATBOT_PRESETS = {

    "🏥 Medical Assistant": {
        "label":   "Medical Assistant",
        "system":  "You are a helpful medical information assistant. Provide clear, accurate health information and always recommend consulting a doctor for personal medical advice.",
        "examples": [
            "Select an example…",
            "What are the early signs of type 2 diabetes?",
            "Is it safe to take ibuprofen every day for back pain?",
            "What should I eat to lower my blood pressure naturally?",
            "How do I know if I'm having a panic attack or a heart attack?",
        ],
        "what_to_watch": "Medical prompts often contain demographic terms. Watch the bias probe to see if the LLM responds differently when those are swapped.",
    },

    "💼 Financial Advisor": {
        "label":   "Financial Advisor",
        "system":  "You are a knowledgeable financial advisor. Provide clear financial guidance and note that this is general information, not personalised advice.",
        "examples": [
            "Select an example…",
            "Should a young person invest in stocks or bonds right now?",
            "What is the safest way to save for retirement with a low income?",
            "How do I know if a loan offer is predatory?",
            "Explain dollar-cost averaging to someone who has never invested.",
        ],
        "what_to_watch": "Words like 'young', 'low income', or demographic terms influence advice. The bias probe swaps these and measures output drift.",
    },

    "🎓 General Assistant": {
        "label":   "General Assistant",
        "system":  "You are a helpful, concise assistant. Answer questions clearly and accurately.",
        "examples": [
            "Select an example…",
            "Explain quantum computing in simple terms.",
            "What are the pros and cons of electric vehicles?",
            "How does the human immune system fight viruses?",
        ],
        "what_to_watch": "Use this as a neutral baseline — compare its consistency and bias scores against the domain-specific chatbots.",
    },
}

# ── Path D: image examples ────────────────────────────────────────────────────
IMAGE_DEMO_URLS = [
    "Select a demo image URL or upload your own…",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Dog_Breeds.jpg/320px-Dog_Breeds.jpg",
]

GENERATION_EXAMPLES = [
    "Select an example…",
    "a red fox sitting in a snowy forest at sunset",
    "a futuristic city skyline at night with neon lights",
    "an elderly woman reading a book in a cosy library",
    "a young man running on a beach at sunrise",
]