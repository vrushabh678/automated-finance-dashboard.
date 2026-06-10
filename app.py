import streamlit as st
import pandas as pd
import fitz  # PyMuPDF for reading PDFs
from openai import OpenAI
import json

# 1. Page Configuration
st.set_page_config(page_title="Automated Financial Dashboard", page_icon="📈", layout="wide")

# 2. Initialize Groq API Client securely
# (Try cloud secrets first, fallback to hardcoded for local testing)
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    # REPLACE THIS WITH YOUR REAL KEY FOR LOCAL TESTING, BUT REMOVE BEFORE GITHUB UPLOAD
    GROQ_API_KEY = "YOUR_API_KEY_HERE"

client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")


# 3. PDF Extraction Helper
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# 4. LLM Data Extraction Helper
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
                        "1. For 'total_debt', Indian corporate reporting often omits the word 'debt'. You MUST aggressively search for 'borrowings', 'long-term borrowings', 'short-term borrowings', or 'financial liabilities'. "
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


# 5. LLM Analyst Summary Helper
def generate_executive_summary(df_string):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an independent equity analyst summarizing financial performance. "
                        "Read the provided JSON data and write a highly professional, 3-sentence executive summary. "
                        "Highlight the revenue trend, profit margin stability, and EPS growth. "
                        "Write entirely in the third person (refer to 'the company'). Do NOT use first-person pronouns like 'our', 'we', or 'my'. "
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
    "Upload a company earnings report (PDF) to instantly extract metrics, calculate ratios, and generate an AI-driven summary.")
st.divider()

# Drag and Drop PDF Uploader
uploaded_pdf = st.file_uploader("Upload Annual Report / Earnings Release (PDF file)", type=["pdf"])

if uploaded_pdf:
    if st.button("Run Financial Pipeline", type="primary"):

        # Step A: Read the PDF
        with st.spinner("Extracting raw text from PDF pages..."):
            raw_text = extract_text_from_pdf(uploaded_pdf)

        # Step B: AI Extraction
        with st.spinner("AI is analyzing financial metrics..."):
            extracted_data = analyze_financials(raw_text)

        if extracted_data:
            # Step C: Pandas Calculation Layer
            df = pd.DataFrame(extracted_data)

            # FIX: Force chronological order (oldest to newest) BEFORE doing math
            df = df.sort_values(by="year", ascending=True).reset_index(drop=True)

            # Calculate YoY Growth if we have multiple years
            if len(df) > 1:
                df['Revenue Growth (%)'] = df['revenue'].pct_change() * 100
            else:
                df['Revenue Growth (%)'] = 0.0

            # Calculate Profit Margin
            df['Profit Margin (%)'] = (df['net_profit'] / df['revenue']) * 100

            # Clean up numbers for the math (used for charts)
            df_math = df.copy()

            # Format the display dataframe (used for table and summary)
            df_display = df.fillna("N/A")
            for col in ['revenue', 'net_profit', 'total_debt', 'eps', 'Revenue Growth (%)', 'Profit Margin (%)']:
                df_display[col] = df_display[col].apply(lambda x: round(x, 2) if isinstance(x, (int, float)) else x)

            # Generate GPT Summary
            with st.spinner("Generating AI Executive Summary..."):
                summary_text = generate_executive_summary(df_display.to_json(orient="records"))

            st.success("Analysis Complete!")

            # Display AI Summary
            st.info(f"**🤖 AI Analyst Summary:**\n\n{summary_text}")

            # Step D: Dashboard Visualization
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Extracted & Calculated Data")
                # Sort descending just for the table view so newest is at the top
                st.dataframe(df_display.sort_values(by="year", ascending=False), use_container_width=True,
                             hide_index=True)

                # Export to CSV
                csv = df_display.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Dashboard Data (CSV)", data=csv, file_name="financial_analysis.csv",
                                   mime="text/csv")

            with col2:
                # Multiple Charts using Tabs
                st.subheader("📈 Financial Trends")
                tab1, tab2, tab3 = st.tabs(["Revenue", "Profit Margin", "EPS"])

                with tab1:
                    st.bar_chart(data=df_math, x="year", y="revenue")
                with tab2:
                    st.line_chart(data=df_math, x="year", y="Profit Margin (%)")
                with tab3:
                    st.line_chart(data=df_math, x="year", y="eps")