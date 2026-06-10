import streamlit as st
import pandas as pd
import fitz  # PyMuPDF for reading PDFs
from openai import OpenAI
import json

# 1. Page Configuration
st.set_page_config(page_title="Automated Financial Dashboard", page_icon="📈", layout="wide")

# 2. Initialize Groq API Client securely
# (Keep using your local hardcoded key for testing, or st.secrets for cloud!)
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    GROQ_API_KEY = "YOUR_KEY_REMOVED_FOR_SECURITY" # Safe to upload!

client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")


# 3. PDF Extraction Helper
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# 4. LLM Data Extraction Helper (FIX 1 & 2 APPLIED)
def analyze_financials(text):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an elite equity researcher. Extract financial data from the text. "
                        "Return a JSON object with a single key 'financials' containing a list of objects. "
                        "Each object must have these exact keys: 'year' (string), 'revenue' (number), "
                        "'net_profit' (number), 'total_debt' (number), 'eps' (number). "
                        "CRITICAL INSTRUCTIONS: "
                        "1. For 'total_debt', aggressively search for 'borrowings' or 'long-term liabilities'. "
                        "2. If ANY value is missing or not explicitly stated, return null (do NOT use 0). "
                        "Order from oldest year to newest year."
                    )
                },
                {
                    "role": "user",
                    "content": f"Extract the metrics from this report:\n\n{text[:20000]}"
                }
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)["financials"]
    except Exception as e:
        st.error(f"Extraction Error: {e}")
        return None


# 5. LLM Analyst Summary Helper (FIX 4 APPLIED)
def generate_executive_summary(df_string):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Chief Financial Officer summarizing financial performance. "
                        "Read the provided JSON data and write a highly professional, 3-sentence executive summary. "
                        "Highlight the revenue trend, profit margin stability, and EPS growth. "
                        "Do not use generic fluff; use the actual numbers provided."
                    )
                },
                {
                    "role": "user",
                    "content": f"Write an analyst summary for this data:\n{df_string}"
                }
            ]
        )
        return response.choices[0].message.content
    except Exception:
        return "Summary generation unavailable at this time."


# 6. Dashboard UI Layout
st.title("📈 Automated Financial Intelligence Dashboard")
st.markdown(
    "Upload a company earnings report (PDF) to extract metrics, calculate ratios, visualize trends, and generate an AI-driven summary.")
st.divider()

uploaded_pdf = st.file_uploader("Upload Annual Report / Earnings Release (PDF file)", type=["pdf"])

if uploaded_pdf:
    if st.button("Run Financial Pipeline", type="primary"):

        with st.spinner("Extracting raw text from PDF pages..."):
            raw_text = extract_text_from_pdf(uploaded_pdf)

        with st.spinner("AI is analyzing financial metrics..."):
            extracted_data = analyze_financials(raw_text)

        if extracted_data:
            # Step C: Pandas Calculation Layer
            df = pd.DataFrame(extracted_data)

            # Mathematical calculations (ignoring nulls temporarily)
            if len(df) > 1:
                df['Revenue Growth (%)'] = df['revenue'].pct_change() * 100
            else:
                df['Revenue Growth (%)'] = 0.0

            df['Profit Margin (%)'] = (df['net_profit'] / df['revenue']) * 100

            # Clean up numbers for the math
            df_math = df.copy()

            # Format the display dataframe (FIX 1 APPLIED: "N/A" instead of 0)
            df_display = df.fillna("N/A")
            for col in ['revenue', 'net_profit', 'total_debt', 'eps', 'Revenue Growth (%)', 'Profit Margin (%)']:
                # Only round numbers, leave "N/A" alone
                df_display[col] = df_display[col].apply(lambda x: round(x, 2) if isinstance(x, (int, float)) else x)

            # Generate GPT Summary
            with st.spinner("Generating AI Executive Summary..."):
                summary_text = generate_executive_summary(df_display.to_json(orient="records"))

            st.success("Analysis Complete!")

            # Display AI Summary (FIX 4 APPLIED)
            st.info(f"**🤖 AI Analyst Summary:**\n\n{summary_text}")

            # Dashboard Visualization
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Extracted & Calculated Data")
                st.dataframe(df_display, use_container_width=True, hide_index=True)

                csv = df_display.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Data (CSV)", data=csv, file_name="financial_analysis.csv",
                                   mime="text/csv")

            with col2:
                # Multiple Charts using Tabs (FIX 3 APPLIED)
                st.subheader("📈 Financial Trends")
                tab1, tab2, tab3 = st.tabs(["Revenue", "Profit Margin", "EPS"])

                with tab1:
                    st.bar_chart(data=df_math, x="year", y="revenue")
                with tab2:
                    st.line_chart(data=df_math, x="year", y="Profit Margin (%)")
                with tab3:
                    st.line_chart(data=df_math, x="year", y="eps")