"""
fake_data.py
------------
Centralized hardcoded data for the TechVenture SFM frontend prototype.

Nothing in this project should hardcode a value inside a UI component.
Every number, label, or string shown on screen is imported from here.

When the real ML pipeline (SFTNN / XGBoost / TabNet / Integrated Gradients)
is wired up, this file is the ONLY place that should be replaced by live
model output. The component layer should not need to change shape.
"""

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

SIDEBAR = {
    "brand_name": "TechVenture",
    "brand_subtitle": "SFM",
    "brand_icon": "assets/images/logo.png",
    "nav_items": [
        {"key": "home", "label": "Dashboard", "icon": "layout-dashboard"},
        {"key": "predictor", "label": "Predictor", "icon": "target"},
        {"key": "research_poster", "label": "Research Poster", "icon": "file-bar-chart-2"},
        {"key": "documentation", "label": "Documentation", "icon": "book-open"},
        {"key": "settings", "label": "Settings", "icon": "settings"},
    ],
    "assistant": {
        "name": "Insight",
        "tagline": "Your AI Assistant",
    },
}

TOPBAR = {
    "page_titles": {
        "home": ("Dashboard", "Overview of TechVenture SFM."),
        "predictor": ("Predictor Dashboard", "Predict startup success using region-aware machine learning."),
        "research_poster": ("Research Poster", "Empirical evaluation and explainable AI insights."),
        "documentation": ("Documentation", "How the SFTNN pipeline and dashboard fit together."),
        "settings": ("Settings", "Workspace and account preferences."),
    },
    "notifications_count": 3,
    "user": {
        "name": "J. Dela Cruz",
        "role": "Researcher",
        "avatar": "assets/images/avatar.png",
    },
}

# ---------------------------------------------------------------------------
# PRESETS (Predictor page quick-start cards)
# ---------------------------------------------------------------------------

PRESETS = [
    {
        "key": "sea_saas",
        "title": "Southeast Asia SaaS",
        "subtitle": "High growth potential",
        "icon": "sparkles",
        "accent": "primary-green",
        "values": {
            "region": "Southeast Asia",
            "industry": "SaaS",
            "investor_count": 12,
            "total_funding": 2_500_000,
            "funding_rounds": 3,
            "months_between_rounds": 8,
        },
    },
    {
        "key": "africa_fintech",
        "title": "Africa FinTech",
        "subtitle": "Emerging opportunity",
        "icon": "trending-up",
        "accent": "accent-purple",
        "values": {
            "region": "Africa",
            "industry": "FinTech",
            "investor_count": 6,
            "total_funding": 900_000,
            "funding_rounds": 2,
            "months_between_rounds": 11,
        },
    },
    {
        "key": "na_established",
        "title": "North America AI",
        "subtitle": "Established market",
        "icon": "cpu",
        "accent": "primary-green",
        "values": {
            "region": "North America",
            "industry": "Artificial Intelligence",
            "investor_count": 21,
            "total_funding": 8_000_000,
            "funding_rounds": 4,
            "months_between_rounds": 6,
        },
    },
]

# ---------------------------------------------------------------------------
# PREDICTOR FORM
# ---------------------------------------------------------------------------

PREDICTOR_FORM = {
    "regions": [
        "North America",
        "Europe",
        "Southeast Asia",
        "Latin America",
        "Africa",
        "East Asia",
        "Middle East",
    ],
    "industries": [
        "SaaS",
        "FinTech",
        "Artificial Intelligence",
        "E-commerce",
        "HealthTech",
        "EdTech",
        "Logistics",
    ],
    "defaults": {
        "region": "Southeast Asia",
        "industry": "SaaS",
        "investor_count": 12,
        "total_funding": 2_500_000,
        "funding_rounds": 3,
        "months_between_rounds": 8,
    },
    "funding_rounds_range": {"min": 1, "max": 15},
    "input_modes": ["Single Startup", "Bulk CSV Upload"],
}

# ---------------------------------------------------------------------------
# PREDICTION RESULTS
# ---------------------------------------------------------------------------

PREDICTION_RESULTS = {
    "mode_badge": "Single Prediction",
    "sftnn": {
        "label": "SFTNN Contextual Probability",
        "value": 84.2,
        "delta_label": "+13.2% vs baseline",
        "tag": "Champion Model",
        "accent": "primary-green",
    },
    "xgboost": {
        "label": "XGBoost Baseline Probability",
        "value": 71.0,
        "delta_label": "Baseline Model",
        "tag": None,
        "accent": "accent-purple",
    },
    "trend_series": [61, 64, 63, 67, 70, 74, 78, 81, 84.2],
}

SPATIAL_MODULATION = {
    "regional_scaling": {
        "label": "Regional Scaling",
        "sublabel": "Funding Velocity Multiplier",
        "value_label": "+42%",
        "range": (0, 200),
        "position_pct": 42,
        "icon": "trending-up",
    },
    "regional_shift": {
        "label": "Regional Shift",
        "sublabel": "Base Probability Shift",
        "value_label": "+14.0%",
        "range": (-20, 20),
        "position_pct": 85,
        "icon": "move-vertical",
    },
}

STATUS_CARDS = [
    {
        "key": "dataset_integrity",
        "title": "Dataset Integrity",
        "description": "Emerging markets face high registry data scarcity.",
        "tone": "warning",
        "icon": "alert-triangle",
    },
    {
        "key": "inference_latency",
        "title": "Inference Latency",
        "description": "4.8ms average per prediction.",
        "tone": "info",
        "icon": "gauge",
    },
    {
        "key": "model_status",
        "title": "Model Status",
        "description": "All systems operational. Model v1.2.3.",
        "tone": "success",
        "icon": "check-circle",
    },
    {
        "key": "academic_disclaimer",
        "title": "Academic Disclaimer",
        "description": "For research purposes only. Not financial advice.",
        "tone": "neutral",
        "icon": "info",
    },
]

# ---------------------------------------------------------------------------
# RESEARCH POSTER
# ---------------------------------------------------------------------------

RESEARCH_OBJECTIVE = (
    "TechVenture-SFM replaces standard, flat categorical representations with "
    "dynamic regional embeddings to overcome spatial non-stationarity and "
    "target right-censoring in startup success prediction."
)

SFT_EQUATION = {
    "title": "SFT Parameter Transformation",
    "latex": r"y = \gamma_r \odot h + \beta_r",
    "caption": "Where \u03b3\u1d63 is regional scaling and \u03b2\u1d63 is regional shift.",
}

MODEL_METRICS = {
    "title": "Benchmark Performance (Global Test Set)",
    "columns": ["Model", "F1-Score", "ROC-AUC", "PR-AUC"],
    "rows": [
        {"model": "SFTNN (Proposed)", "f1": 0.842, "roc_auc": 0.913, "pr_auc": 0.887, "highlight": True},
        {"model": "MLP", "f1": 0.732, "roc_auc": 0.842, "pr_auc": 0.798, "highlight": False},
        {"model": "XGBoost", "f1": 0.710, "roc_auc": 0.821, "pr_auc": 0.776, "highlight": False},
        {"model": "TabNet", "f1": 0.748, "roc_auc": 0.864, "pr_auc": 0.812, "highlight": False},
    ],
}

BAR_CHART_DATA = {
    "title": "Regional F1-Score Comparison",
    "series": ["SFTNN", "XGBoost"],
    "regions": ["North America", "Europe", "Southeast Asia", "Latin America", "Africa"],
    "values": {
        "SFTNN": [0.88, 0.86, 0.79, 0.77, 0.74],
        "XGBoost": [0.83, 0.80, 0.65, 0.55, 0.58],
    },
}

REGIONAL_PARAMETERS = {
    "scaling": {
        "title": "Regional Feature Scaling (Multiplicative \u03b3)",
        "columns": ["Region", "Funding Velocity", "Total Funding", "Team Experience", "Market Size"],
        "rows": [
            {"region": "North America", "funding_velocity": 1.00, "total_funding": 1.00, "team_experience": 1.00, "market_size": 1.00},
            {"region": "Europe", "funding_velocity": 0.96, "total_funding": 0.98, "team_experience": 1.03, "market_size": 1.02},
            {"region": "Southeast Asia", "funding_velocity": 1.42, "total_funding": 0.78, "team_experience": 1.18, "market_size": 1.25},
            {"region": "Latin America", "funding_velocity": 1.36, "total_funding": 0.82, "team_experience": 1.14, "market_size": 1.20},
            {"region": "Africa", "funding_velocity": 1.38, "total_funding": 0.75, "team_experience": 1.16, "market_size": 1.22},
        ],
    },
    "offsets": {
        "title": "Regional Offsets (Additive \u03b2)",
        "columns": ["Region", "Offset"],
        "rows": [
            {"region": "North America", "offset": 0.00},
            {"region": "Europe", "offset": 0.02},
            {"region": "Southeast Asia", "offset": 0.14},
            {"region": "Latin America", "offset": 0.12},
            {"region": "Africa", "offset": 0.11},
        ],
    },
}

# ---------------------------------------------------------------------------
# DOCUMENTATION (placeholder content)
# ---------------------------------------------------------------------------

DOCUMENTATION_SECTIONS = [
    {
        "title": "Overview",
        "body": (
            "TechVenture SFM is a research prototype dashboard for a region-aware "
            "startup success prediction system built around the SFTNN architecture. "
            "This frontend is intentionally decoupled from the ML pipeline so it can "
            "be wired to live inference with minimal refactoring."
        ),
    },
    {
        "title": "Architecture",
        "body": (
            "The dashboard is organized into layout, card, chart, and form components "
            "under components/, with all sample values centralized in fake_data.py. "
            "Pages are thin composition layers that assemble components and pass in data."
        ),
    },
    {
        "title": "Roadmap",
        "body": (
            "Planned next steps include wiring the Predictor form to live SFTNN, XGBoost, "
            "and TabNet inference, replacing static chart data with real evaluation output, "
            "and adding a bulk CSV inference mode."
        ),
    },
]

# ---------------------------------------------------------------------------
# SETTINGS (placeholder content)
# ---------------------------------------------------------------------------

SETTINGS_SECTIONS = [
    {
        "title": "Workspace",
        "fields": [
            {"label": "Project name", "value": "TechVenture SFM"},
            {"label": "Model version", "value": "v1.2.3"},
            {"label": "Environment", "value": "Prototype / No backend"},
        ],
    },
    {
        "title": "Notifications",
        "fields": [
            {"label": "Prediction alerts", "value": "Enabled"},
            {"label": "Weekly summary email", "value": "Disabled"},
        ],
    },
]
