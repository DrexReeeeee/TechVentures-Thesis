"""
Temporary data provider for the TechVenture dashboard.

All UI components should retrieve their data from this module.
Later, these functions can be replaced with API calls without
changing the UI components.
"""

# DROPDOWN OPTIONS

DROPDOWN_OPTIONS = {

    "countries": [
        "Philippines",
        "Singapore",
        "Indonesia",
        "Malaysia",
        "Thailand",
        "Vietnam",
    ],

    "regions": [
        "Southeast Asia",
        "East Asia",
        "South Asia",
        "North America",
        "Europe",
        "Africa",
        "Latin America"
    ],

    "industries": [
        "Artificial Intelligence",
        "FinTech",
        "HealthTech",
        "EdTech",
        "SaaS",
        "Cybersecurity",
        "E-Commerce",
        "ClimateTech",
    ],

    "funding_stages": [
        "Pre-Seed",
        "Seed",
        "Series A",
        "Series B",
        "Series C",
        "Series D+",
    ],

}


# PRESET MODELS

PRESET_MODELS = [

    {
        "title": "SEA SaaS",
        "subtitle": "Early-stage SaaS startup",
        "icon": "apartment",
    },

    {
        "title": "Africa FinTech",
        "subtitle": "Growth-stage FinTech startup",
        "icon": "public",
    },

    {
        "title": "North America AI",
        "subtitle": "Late-stage AI startup",
        "icon": "neurology",
    },

]


# SAMPLE PREDICTION

PREDICTION_RESULT = {

    "probability": 86.7,

    "confidence": 92.3,

    "label": "Likely Successful",

}


# FEATURE IMPORTANCE

FEATURE_IMPORTANCE = [

    {
        "feature": "Total Funding",
        "importance": 0.31,
    },

    {
        "feature": "Funding Stage",
        "importance": 0.24,
    },

    {
        "feature": "Industry",
        "importance": 0.19,
    },

    {
        "feature": "Country",
        "importance": 0.15,
    },

    {
        "feature": "Region",
        "importance": 0.11,
    },

]


# SYSTEM STATUS

SYSTEM_STATUS = {

    "dataset": "SEA Startup Dataset",

    "model": "Spatial Feature Transformation Neural Network",

    "version": "v1.0",

    "latency": "0.24 sec",

}


# GETTERS

def get_dropdown_options():

    return DROPDOWN_OPTIONS


def get_countries():

    return DROPDOWN_OPTIONS["countries"]


def get_regions():

    return DROPDOWN_OPTIONS["regions"]


def get_industries():

    return DROPDOWN_OPTIONS["industries"]


def get_funding_stages():

    return DROPDOWN_OPTIONS["funding_stages"]


def get_preset_models():

    return PRESET_MODELS


def get_prediction_result():

    return PREDICTION_RESULT


def get_feature_importance():

    return FEATURE_IMPORTANCE


def get_system_status():

    return SYSTEM_STATUS