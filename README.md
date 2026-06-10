# 📈 Automated Financial Intelligence Dashboard

**Live Demo:** https://data-fin.streamlit.app/  
**Sample Data to Test:** Nestle India_AR FY 2025-26.pdf

### Overview
An end-to-end data pipeline that extracts unstructured financial metrics from complex annual report PDFs, processes the data, and generates AI-driven equity research summaries. Built to automate the manual data entry process for financial analysts.

<img width="1919" height="953" alt="Dashboard_Preview" src="https://github.com/user-attachments/assets/cb5a8e44-5220-47d4-99aa-cd67918481c6" />


### Key Features
* **LLM Data Extraction:** Utilizes the Groq API (Llama-3) to accurately pull Revenue, Net Profit, Debt, and EPS directly from dense PDF text, adapting to regional accounting terminology.
* **Automated Pandas Pipeline:** Cleanses extracted JSON, handles missing values (N/A), and computes YoY Revenue Growth and Profit Margins chronologically.
* **AI Analyst Summaries:** Orchestrates a secondary LLM call to synthesize the calculated dataframe into a professional, third-person executive summary highlighting CAGRs and margin stability.
* **Interactive UI:** Deployed via Streamlit with CSV export functionality, multi-tab chronological charting, and dynamic KPI metric cards.

### Tech Stack
* **Python** (Core logic)
* **Pandas** (Data manipulation & financial math)
* **PyMuPDF / fitz** (PDF document parsing)
* **OpenAI SDK / Groq** (LLM orchestration & prompt engineering)
* **Streamlit** (Frontend dashboard & cloud deployment)

### How to Run Locally

1. Clone the repository:
   ```bash
   git clone [https://github.com/vrushabh678/automated-finance-dashboard.git](https://github.com/vrushabh678/automated-finance-dashboard.git)
   cd automated-finance-dashboard
   Install dependencies:

2.Bash
pip install -r requirements.txt
3.Set up your Groq API key:

Windows: set GROQ_API_KEY=your_real_key_here
4.Run the app:

Bash
streamlit run app.py
