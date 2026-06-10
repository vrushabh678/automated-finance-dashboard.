import streamlit as st
import pandas as pd
import fitz  # PyMuPDF for reading PDFs
from openai import OpenAI
import json

# 1. Page Configuration
st.set_page_config(page_title="Automated Financial Dashboard", page_icon="📈", layout="wide")

# 2. Initialize Groq API Client securely
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")


# 3. PDF Extraction Helper
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# 4. LLM Analysis Helper
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
                        "If a value is missing, use 0. Order from oldest year to newest year."
                    )
                },
                {
                    "role": "user",
                    "content": f"Extract the metrics from this report:\n\n{text[:20000]}"
                    # Truncated to fit API limits safely
                }
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)["financials"]
    except Exception as e:
        st.error(f"Extraction Error: {e}")
        return None


# 5. Dashboard UI Layout
st.title("📈 Automated Financial Intelligence Dashboard")
st.markdown(
    "Upload a company earnings report (PDF) to instantly extract metrics, calculate ratios, and generate charts.")
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
            st.success("Analysis Complete!")

            # Step C: Pandas Calculation Layer
            df = pd.DataFrame(extracted_data)

            # Calculate YoY Growth if we have multiple years
            if len(df) > 1:
                df['Revenue Growth (%)'] = df['revenue'].pct_change() * 100
            else:
                df['Revenue Growth (%)'] = 0.0

            # Calculate Profit Margin
            df['Profit Margin (%)'] = (df['net_profit'] / df['revenue']) * 100
            df = df.fillna(0).round(2)  # Clean up the numbers

            # Step D: Dashboard Visualization
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Extracted & Calculated Data")
                st.dataframe(df, use_container_width=True, hide_index=True)

                # Export to CSV
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Dashboard Data (CSV)", data=csv, file_name="financial_analysis.csv",
                                   mime="text/csv")

            with col2:
                st.subheader("📈 Revenue Trend Chart")
                st.bar_chart(data=df, x="year", y="revenue")