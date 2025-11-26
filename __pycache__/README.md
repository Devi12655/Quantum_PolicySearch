Quantum Policy Search System

A Quantum-Inspired Policy Analysis and Search Tool

1. Project Overview

The Quantum Policy Search System is a FastAPI application that performs intelligent policy search using quantum-inspired embeddings.
It includes a modern Glow UI, search ranking, interactive relevance charts, and export options such as PDF, Excel, and Text.

This project is fully local and does not require internet access for processing queries.

2. Folder Structure

Your project must follow this structure:

Quantum_AI_Glow_Final_Complete/
│
├── app.py
├── quantum_model.pkl
├── requirements.txt
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
│
└── templates/
    ├── index.html
    └── search.html

Explanation of Each File
app.py

Main FastAPI backend.
Handles:

Loading quantum model (quantum_model.pkl)

Generating embeddings for user queries

Ranking policy relevance

Returning results to frontend

Providing PDF, Excel, Text downloads

quantum_model.pkl

Serialized model containing:

TF-IDF vectorizer

Standard scaler

PCA reducer

Quantum-inspired embeddings

Original policy dataset

This file must remain in the project root.

requirements.txt

Contains all required Python packages.

static/css/style.css

Glow UI styling including:

Header gradient

Cards

Buttons

Input fields

Hover effects

Chart container styling

static/js/app.js

Optional JS logic.
Used for UI enhancements (can be extended later).

templates/index.html

User search input page.

templates/search.html

Displays:

Ranked policy results

Metadata (region, year, status)

Relevance score chart (Chart.js)

Download options for each item

3. How the Quantum Search Works

User enters a query.

Text is vectorized using the stored TF-IDF vectorizer.

Features are scaled.

Reduced using PCA.

Quantum-inspired encoding is applied:

tanh transformation

sin transformation

Embeddings are compared with cosine similarity.

Top-K results are returned with scores.

Chart.js visualizes relevance across results.

4. How to Run the Project
Step 1 — Install dependencies
pip install -r requirements.txt

Step 2 — Start FastAPI server
uvicorn app:app --reload

Step 3 — Open browser
http://127.0.0.1:8000

5. Features
✓ Quantum-Inspired Search

Advanced vector transformations for better semantic matching.

✓ Glow UI

Modern interface with smooth gradients and clean components.

✓ Multiple Export Formats

Each search result can be downloaded as:

PDF

Excel

Text

✓ Relevance Visualization

Dynamic Chart.js graph showing ranking strength.

6. Requirements

Python 3.9+
FastAPI
Uvicorn
Joblib
scikit-learn
ReportLab
Openpyxl
Chart.js (CDN inside search.html)

7. Notes

The quantum_model.pkl file must stay at the project root.

This project does not include login/registration.

The dataset used to train the model is not required at runtime.

Works offline once model is generated.