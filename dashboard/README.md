# TechVenture SFM Dashboard

The **TechVenture SFM Dashboard** is the frontend interface for the TechVenture Startup Forecasting Model (SFM) thesis project.

It is built using **Streamlit** and serves as the visualization layer for the machine learning pipeline. During development, the dashboard uses centralized fake data to prototype the interface before integrating the trained models.

---

# Objectives

- Provide a modern dashboard for startup prediction.
- Visualize model outputs and evaluation metrics.
- Display Explainable AI (Integrated Gradients) results.
- Present research findings interactively.
- Remain independent from the ML pipeline during frontend development.

---

# Technology Stack

- Python 3.11+
- Streamlit
- Plotly
- Pandas
- NumPy
- Custom CSS

---

# Folder Structure

```text
dashboard/
│
├── app.py                       # Dashboard entry point
├── README.md
│
├── assets/
│   ├── css/
│   │   ├── components.css       # Component-specific styles
│   │   ├── navigation.css       # Navigation styles
│   │   ├── root.css             # Global reusable styles
│   │   └── sidebar.css          # Sidebar styles
│   └── icons/
│       └── default_telp.png     # Default placeholder icon
│
├── components/                  # Reusable UI components
│   ├── common/
│   │   ├── icon.py              # Icon component
│   │   ├── startup_form.py      # Startup input form
│   │   └── upload_zone.py       # File upload zone
│   │
│   ├── display/
│   │   ├── card_header.py       # Card header component
│   │   ├── comparison_card.py   # Side-by-side comparison card
│   │   ├── model_result_card.py # Model prediction result card
│   │   ├── preset_card.py       # Preset scenario card
│   │   ├── explainability/
│   │   └── insights/            # (reserved for future insights)
│   │
│   ├── feedback/
│   │   ├── empty_state.py       # Empty state placeholder
│   │   └── loading_state.py     # Loading spinner / skeleton
│   │
│   ├── layout/
│   │   ├── page_header.py       # Page header component
│   │   ├── section_header.py    # Section header component
│   │   └── sidebar.py           # Sidebar navigation
│   │
│   └── navigation/
│       └── nav_item.py          # Single navigation item
│
├── data/
│   └── fake_data.py             # Centralized placeholder data
│
└── views/                       # Dashboard page views
    ├── about.py                 # About / project info page
    ├── predictor.py             # Startup prediction page
    └── research.py              # Research findings page

```

---

# Getting Started

## 1. Activate the Virtual Environment

# if venv not created yet
```powershell
python -m venv .venv
```
PowerShell
```powershell
.\.venv\Scripts\Activate.ps1
```

Command Prompt
```cmd
.venv\Scripts\activate.bat
```

# if it errors
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
---

## 2. Install Dependencies

From the project root:

```bash
pip install -r requirements.txt
```

---

## 3. Launch the Dashboard

From the project root:

```bash
streamlit run dashboard/app.py
```

The application will automatically open in your browser.

---

# Development Workflow

During frontend development:

```
Fake Data
      │
      ▼
Reusable Components
      │
      ▼
Pages
      │
      ▼
Dashboard
```

No backend or machine learning models are loaded during this phase.

All displayed values should originate from `fake_data.py`.

---