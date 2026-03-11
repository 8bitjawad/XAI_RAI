"""
llm_engine.py  —  Domain-aware LLM explanation engine
======================================================
Builds a rich, domain-specific prompt from the full XAI result dict
so the LLM gives genuinely useful explanations rather than generic ones.

Usage:
    from llm_engine import generate_explanation

    explanation = generate_explanation(result, domain="healthcare")
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL              = os.getenv("OPENROUTER_MODEL")

# ── Domain context blocks ─────────────────────────────────────────────────────
# Each domain gets a system prompt that gives the LLM genuine expertise
# in that field, so its explanation uses correct terminology and framing.

DOMAIN_SYSTEM_PROMPTS = {

    "healthcare": """You are a clinical decision-support AI assistant with expertise 
in internal medicine, endocrinology, and preventive care. You explain machine learning 
risk assessments to patients and their families in plain English. You always:
- Explain what each risk factor means medically and why it matters
- Put numbers in clinical context (e.g. what a glucose level of 160 means vs normal)
- Suggest concrete, actionable next steps a patient could take
- Use a calm, empathetic tone — never alarmist
- Flag when the model's confidence is low and a doctor should be consulted
- Never diagnose. Always frame as a risk assessment tool, not a diagnosis.""",

    "finance": """You are a senior credit risk analyst and financial advisor. You explain 
loan and credit decisions to applicants in plain English. You always:
- Explain what each financial factor means and how lenders interpret it
- Put numbers in real-world context (e.g. what a debt-to-income ratio of 0.6 means)
- Give specific, actionable steps the applicant can take to improve their profile
- Explain both why a decision was made AND what would need to change to reverse it
- Use a professional but approachable tone
- Be transparent about the model's confidence level""",

    "realestate": """You are a real estate market analyst and property investment advisor. 
You explain property price predictions to buyers, sellers, and investors. You always:
- Explain which property and market factors are driving the prediction
- Put metrics in real-world context (e.g. what a school rating of 8 means for demand)
- Discuss both the upside and downside risks
- Suggest what the buyer or seller could do given this prediction
- Use a confident but measured tone — markets are uncertain and you acknowledge that""",

    "trading": """You are a quantitative finance analyst explaining algorithmic trading 
model predictions to portfolio managers. You always:
- Explain what each price/volume signal means technically
- Put the prediction in market context
- Flag confidence levels and what they mean for position sizing
- Discuss potential risks that could invalidate the prediction
- Use precise financial language but remain accessible""",

    "text": """You are an AI safety and content analyst explaining sentiment and toxicity 
analysis results. You always:
- Explain which words or phrases drove the sentiment prediction and why
- Explain what each toxicity score means in plain English
- Flag any concerning patterns in the content
- Suggest how the text could be revised if needed
- Be objective and non-judgmental in tone""",

    "general": """You are an expert AI system explainer. You explain machine learning 
predictions clearly to non-technical users. You always:
- Explain what each feature means in plain English
- Explain why each factor pushed the prediction in a particular direction
- Give concrete, actionable next steps based on the prediction
- Acknowledge uncertainty when model confidence is low"""
}

# ── Feature-level clinical/domain context ────────────────────────────────────
# These give the LLM the knowledge to interpret raw numbers meaningfully.

FEATURE_CONTEXT = {

    # Healthcare
    "glucose_level": {
        "unit": "mg/dL",
        "ranges": "Normal: 70–99. Pre-diabetic: 100–125. Diabetic: 126+.",
        "implication": "High glucose is the primary indicator of diabetes risk."
    },
    "bmi": {
        "unit": "kg/m²",
        "ranges": "Underweight: <18.5. Normal: 18.5–24.9. Overweight: 25–29.9. Obese: 30+.",
        "implication": "BMI above 30 significantly raises diabetes and cardiovascular risk."
    },
    "blood_pressure": {
        "unit": "mmHg (diastolic)",
        "ranges": "Normal: <80. Elevated: 80–89. High: 90+.",
        "implication": "Persistently high blood pressure damages blood vessels and raises diabetes risk."
    },
    "insulin": {
        "unit": "μU/mL",
        "ranges": "Fasting normal: 2–25. Elevated: 25+.",
        "implication": "High fasting insulin suggests insulin resistance, a precursor to Type 2 diabetes."
    },

    # Finance
    "credit_score": {
        "unit": "points (300–850)",
        "ranges": "Poor: 300–579. Fair: 580–669. Good: 670–739. Very Good: 740–799. Excellent: 800+.",
        "implication": "The single strongest predictor of repayment likelihood."
    },
    "debt_to_income": {
        "unit": "ratio",
        "ranges": "Excellent: <0.20. Good: 0.20–0.35. Concerning: 0.36–0.50. High risk: >0.50.",
        "implication": "Shows how much of monthly income is already committed to debt payments."
    },
    "missed_payments": {
        "unit": "count (last 24 months)",
        "ranges": "0 is ideal. 1–2 is a yellow flag. 3+ is a strong default predictor.",
        "implication": "Past payment behaviour is the strongest behavioural signal lenders use."
    },

    # Real estate
    "school_rating": {
        "unit": "score (1–10)",
        "ranges": "Low: 1–4. Average: 5–7. High: 8–10.",
        "implication": "School ratings are one of the top 3 drivers of residential property demand."
    },
    "proximity_to_transit": {
        "unit": "minutes walk",
        "ranges": "Excellent: <5. Good: 5–15. Average: 15–30. Poor: 30+.",
        "implication": "Transit access directly affects desirability and resale value."
    },
}


def _build_feature_context_block(shap_values: dict) -> str:
    """
    For each feature in the SHAP output, attach its clinical/domain
    context if we have it. This is what makes explanations non-generic.
    """
    lines = []
    for feature, importance in shap_values.items():
        direction = "increased" if importance > 0 else "reduced"
        line = f"- {feature}: SHAP importance {round(importance, 4)} ({direction} risk)"
        if feature in FEATURE_CONTEXT:
            ctx = FEATURE_CONTEXT[feature]
            line += (
                f"\n    Unit: {ctx['unit']}"
                f"\n    Reference ranges: {ctx['ranges']}"
                f"\n    Clinical meaning: {ctx['implication']}"
            )
        lines.append(line)
    return "\n".join(lines)


def _build_prompt(result: dict, domain: str, sample_values: dict | None = None) -> str:
    """
    Builds a rich, structured prompt from the full XAI result dict.
    Much more information than the original — gives the LLM everything
    it needs to produce a genuinely useful domain-specific explanation.
    """
    prediction  = result["prediction"]
    confidence  = round(result["confidence"] * 100, 2)
    shap_values = result.get("shap_values", {})
    lime_rules  = result.get("lime_rules", [])
    top_factors = result.get("top_factors", [])

    # Top factors in plain text
    top_factors_text = "\n".join(
        [f"  {i+1}. {f} (importance: {round(v, 4)})"
         for i, (f, v) in enumerate(top_factors)]
    )

    # LIME rules in plain text
    lime_text = "\n".join(
        [f"  - {rule} (weight: {round(w, 4)})" for rule, w in lime_rules]
    ) if lime_rules else "  Not available"

    # Rich feature context block
    feature_context = _build_feature_context_block(shap_values)

    # Actual input values if we have them
    sample_text = ""
    if sample_values:
        sample_text = "\nActual patient/applicant values provided:\n" + "\n".join(
            [f"  {k}: {v}" for k, v in sample_values.items()]
        )

    # Confidence framing
    if confidence >= 90:
        confidence_note = "The model is highly confident in this prediction."
    elif confidence >= 70:
        confidence_note = "The model has moderate-to-high confidence. A professional review is still advisable."
    else:
        confidence_note = (
            "The model confidence is relatively low. "
            "This prediction is uncertain and professional consultation is strongly recommended."
        )

    prompt = f"""
You are explaining a machine learning prediction to a non-technical person.

=== PREDICTION RESULT ===
Outcome: {prediction}
Confidence: {confidence}%
Confidence note: {confidence_note}
{sample_text}

=== TOP FACTORS (SHAP — what drove the prediction) ===
{top_factors_text}

=== DECISION RULES (LIME — how the model reasoned) ===
{lime_text}

=== DETAILED FEATURE ANALYSIS ===
{feature_context}

=== YOUR TASK ===
Write a clear, in-depth explanation for this {domain} prediction. Structure your response as:

1. SUMMARY (2–3 sentences — what the model predicted and how confident it is)
2. KEY DRIVERS (explain each top factor in plain language with its real-world meaning)
3. WHAT THE NUMBERS MEAN (put the actual values in clinical/domain context using the ranges above)
4. WHAT THIS MEANS FOR YOU (practical, actionable advice specific to this domain)
5. IMPORTANT CAVEATS (when to seek professional help, limitations of the model)

Be specific. Use the actual numbers. Do not be generic.
"""
    return prompt


def generate_explanation(
    result: dict,
    domain: str = "general",
    sample_values: dict | None = None,
) -> str:
    """
    Main entry point. Generates a domain-aware, in-depth LLM explanation.

    Parameters
    ----------
    result        : the dict returned by adapter.predict()
    domain        : one of 'healthcare', 'finance', 'realestate', 'trading',
                    'text', 'general'
    sample_values : optional dict of the raw input values (e.g. {'glucose': 160})
                    — if provided, the LLM can reference actual numbers
    """
    
    if os.getenv("USE_LLM", "true").lower() == "false":
        return "⚠️ LLM disabled in dev mode. Set USE_LLM=true to enable."

    system_prompt = DOMAIN_SYSTEM_PROMPTS.get(domain, DOMAIN_SYSTEM_PROMPTS["general"])
    user_prompt   = _build_prompt(result, domain, sample_values)

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type":  "application/json",
            },
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                "temperature": 0.3,   # lower = more consistent, factual output
            },
            timeout=30,
        )
        data = response.json()
        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return f"Explanation unavailable: {str(e)}"