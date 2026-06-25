# TechVenture-SFM Thesis

## Topic
Spatial Feature Transform Neural Network for Region-Aware Startup Success Prediction

## Thesis Overview
This project investigates whether startup success prediction can be improved by making the model region-aware. Instead of treating all startup ecosystems as if they follow the same patterns, the study introduces a Spatial Feature Modulation (SFM) layer that adjusts feature processing according to the startup’s geographic context.

The central hypothesis is that startup success is not shaped by the same feature dynamics in every region. A funding gap that is normal in one ecosystem may signal something very different in another. By allowing the network to adapt to regional context, the model is expected to produce better predictions for startups in less-represented or more heterogeneous regions.

## The New Architecture: Spatial Feature Transform (SFT)
Instead of building a gate that decides if a feature is “noisy,” the neural network uses a Spatial Feature Transform (SFT) layer.

The concept borrows from FiLM (Feature-wise Linear Modulation), a machine learning method used in computer vision and NLP, but rarely applied to tabular startup data.

The core idea is simple:

- a region embedding learns contextual information
- that embedding generates a scaling factor $\gamma$ and a shifting factor $\beta$
- the network applies these values to the tabular features dynamically

This yields:

$$y = \gamma_{region} \odot x + \beta_{region}$$

where $x$ is the input feature representation and the modulation parameters adapt it for the given region.

## Why This Fits the Thesis Goal
This approach is closely related to prior work while remaining conceptually distinct:

1. It uses the same family of deep learning methods as related studies, giving it a rigorous and professional foundation.
2. The research question is different: previous work focuses on filtering noisy features, while this thesis asks whether regional context should adapt feature interpretation.
3. It creates a strong baseline comparison: a standard MLP can be used as the baseline, and the SFM-enhanced model can be shown to improve performance in regions outside major Western hubs.

## Thesis Overview: TechVenture-SFM
### The Core Idea
We are building a machine learning model that predicts whether a tech startup will succeed or fail using startup data such as funding history, investor information, round patterns, and geographic location.

Existing studies treat all global startups the same. This thesis adds a custom layer—Spatial Feature Modulation (SFM)—that acts like a localized volume knob. It automatically amplifies or dampens the importance of startup features depending on where the startup is located.

### How It Works
1. The model takes standard startup data such as total funding, number of rounds, time between rounds, category, and location.
2. A global neural network processes the startup features to find universal success patterns.
3. The SFM layer uses the startup’s location to generate modulating factors $\gamma$ and $\beta$.
4. These factors adjust the feature representation before the final success/failure prediction is made.

### Why It Is Different from Existing Studies
- The baseline study focuses on feature gating to suppress noisy or missing data globally.
- This thesis focuses on spatial modulation, where regional context is treated as a meaningful signal rather than something to be filtered out.

## The Core Justification: Binary Shutoff vs. Contextual Scaling
Feature gating works like an on/off switch. Spatial Feature Modulation works like a volume knob or equalizer. It does not delete information; it recontextualizes it.

This is especially important for startup ecosystems because the same feature can carry different meaning in different regions.

## Why SFM Is Better for This Topic
### 1. Regional Dynamics Are Not Noise
A variable like time between funding rounds can mean something very different in Silicon Valley than in Southeast Asia. SFM allows the model to adapt to that context instead of suppressing the feature globally.

### 2. Preserving Weak Signals in Sparse Regions
In smaller or less-represented regions, the data may be sparse. A gating strategy may erase these signals entirely, while SFM can adjust how they are interpreted without discarding them.

### 3. Clearer Architectural Interpretation
A SHAP analysis of an SFM-based model is more informative than a simple gating model because it can show how the modulation parameters change feature importance across regions.

## The 30-Second Panel Pitch
“Existing approaches rely on feature gating, which treats regional variance as noise to be filtered out. Our approach, Spatial Feature Modulation (SFM), argues that regional context is a critical signal. By using linear modulation with $\gamma$ and $\beta$, we dynamically recalibrate feature importance for each region, preserving predictive accuracy in data-scarce and geographically diverse startup ecosystems.”

## Research Phases
### Phase 1: Problem Identification and Research Conceptualization
This phase establishes the foundation of the study by identifying the limitations of existing startup success prediction models and defining the research problem. The researcher examines how current approaches often assume that startup success factors are universally applicable across regions, despite substantial differences in entrepreneurial ecosystems worldwide.

### Phase 2: Data Collection and Preparation
This phase focuses on acquiring and preparing the startup dataset. Raw data is collected, cleaned, normalized, merged, and enriched with regional labels so it is suitable for machine learning.

### Phase 3: Development of the TechVenture-SFM Model
This phase introduces the baseline MLP and the proposed SFM architecture. Regional embeddings are used to generate modulation parameters that dynamically adjust feature representations inside the neural network.

### Phase 4: Model Training, Optimization, and Validation
Both the baseline model and the proposed model are trained under comparable conditions. Performance is evaluated using accuracy, precision, recall, F1-score, and ROC-AUC.

### Phase 5: Model Interpretation and Knowledge Discovery
This phase uses explainability methods such as SHAP and examines the modulation behavior to understand how feature importance shifts across regions.

### Phase 6: Research Documentation, Defense, and Dissemination
This phase covers the formal thesis writing, presentation, defense, and submission of the research outputs.

## Project Documentation
- Dataset documentation: [DATASET.md](DATASET.md)
- Africa folder guide: [AFRICA/README.md](AFRICA/README.md)
- Extra datasets guide: [EXTRA DATASETS/README.md](EXTRA%20DATASETS/README.md)
- Final dataset guide: [FINAL DATASET/README.md](FINAL%20DATASET/README.md)
- Latin America folder guide: [LATIN AMERICA/README.md](LATIN%20AMERICA/README.md)
- Scraping pipeline guide: [StartUp_FundingScrappingData/README.md](StartUp_FundingScrappingData/README.md)
