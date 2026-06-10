import streamlit as st
import pandas as pd
import fitz  # PyMuPDF for reading PDFs
from openai import OpenAI
import json

# 1. Page Configuration
st.set_page_config(page_title="Automated Financial Dashboard", page_icon="📈", layout="wide")

# 2. Initialize Groq API Client securely
# Streamlit Cloud automatically pulls this from your Advanced Settings -> Secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")


# 3. PDF Extraction Helper
def extract_text_from_pdf(uploaded_file):
    uploaded_file.seek(0) # Reset file pointer for Streamlit re-runs
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# 4. LLM Data Extraction Helper
def analyze_financials(text):
    try:
        # Safe text slicing to avoid mid-number cuts
        safe_text = text[:20000].rsplit('\n', 1)[0]
        
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
                    "content": f"Extract the metrics from this report:\n\n{safe_text}"
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

            # Strict column enforcer
            expected_cols = ['year', 'revenue', 'net_profit', 'total_debt', 'eps']
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = None 
            df = df[expected_cols] 

            # Force chronological order (oldest to newest) BEFORE doing math
            df = df.sort_values(by="year", ascending=True).reset_index(drop=True)

            # Ensure numeric typing for math
            df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
            df['net_profit'] = pd.to_numeric(df['net_profit'], errors='coerce')

            # Calculate YoY Growth
            if len(df) > 1:
                df['Revenue Growth (%)'] = df['revenue'].pct_change() * 100
                df['Revenue Growth (%)'] = df['Revenue Growth (%)'].round(2)
            else:
                df['Revenue Growth (%)'] = 0.0

            # Safe Profit Margin Calculation (Avoid Division by Zero)
            df['Profit Margin (%)'] = df.apply(
                lambda row: (row['net_profit'] / row['revenue'] * 100) 
                if pd.notna(row['revenue']) and row['revenue'] != 0 
                else None, axis=1
            )
            
            # Clean up numbers for the math (used for charts)
            df_math = df.copy() 
            df_math['Revenue Growth (%)'] = df_math['Revenue Growth (%)'].fillna(0)
            
            # Format the display dataframe (used for table and summary)
            df_display = df.fillna("N/A")
            for col in ['revenue', 'net_profit', 'total_debt', 'eps', 'Revenue Growth (%)', 'Profit Margin (%)']:
                df_display[col] = df_display[col].apply(lambda x: round(x, 2) if isinstance(x, (int, float)) else x)

            # Generate GPT Summary
            with st.spinner("Generating AI Executive Summary..."):
                summary_text = generate_executive_summary(df_display.to_json(orient="records"))
                summary_text = summary_text.replace("`", "") # Sanitize markdown

            st.success("Analysis Complete!")

            # Key metrics summary cards (WITH NAN SAFETY AND DELTAS)
            st.subheader("💡 Key Performance Indicators")
            m1, m2, m3, m4 = st.columns(4)
            
            latest_rev = df_math['revenue'].iloc[-1] if not df_math.empty else None
            latest_eps = df_math['eps'].iloc[-1] if not df_math.empty else None
            avg_margin = df_math['Profit Margin (%)'].mean()
            
            rev_delta = None
            eps_delta = None
            
            if len(df_math) > 1:
                prev_rev = df_math['revenue'].iloc[-2]
                if pd.notna(latest_rev) and pd.notna(prev_rev):
                    rev_delta = latest_rev - prev_rev
                    
                prev_eps = df_math['eps'].iloc[-2]
                if pd.notna(latest_eps) and pd.notna(prev_eps):
                    eps_delta = latest_eps - prev_eps

            m1.metric(
                "Latest Revenue", 
                f"{latest_rev:,.0f}" if pd.notna(latest_rev) else "N/A",
                delta=f"{rev_delta:,.0f}" if pd.notna(rev_delta) else None
            )
            
            m2.metric(
                "Latest EPS", 
                f"{latest_eps:.2f}" if pd.notna(latest_eps) else "N/A",
                delta=f"{eps_delta:.2f}" if pd.notna(eps_delta) else None
            )
            
            m3.metric(
                "Avg Profit Margin", 
                f"{avg_margin:.1f}%" if pd.notna(avg_margin) else "N/A"
            )
            
            # Safe CAGR calculation
            if len(df_math) > 1 and pd.notna(df_math['revenue'].iloc[0]) and df_math['revenue'].iloc[0] > 0 and pd.notna(latest_rev):
                years = len(df_math) - 1
                cagr = ((latest_rev / df_math['revenue'].iloc[0]) ** (1/years) - 1) * 100
                m4.metric("Revenue CAGR", f"{cagr:.1f}%")
            else:
                m4.metric("Revenue CAGR", "N/A")

            # Display AI Summary
            st.info(f"**🤖 AI Analyst Summary:**\n\n{summary_text}")

            # Dashboard Visualization
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Extracted Data")
                st.dataframe(df_display, use_container_width=True, hide_index=True)

                csv = df_display.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Data (CSV)", data=csv, file_name="financial_analysis.csv", mime="text/csv")

            with col2:
                st.subheader("📈 Financial Trends")
                tab1, tab2, tab3, tab4 = st.tabs(["Revenue", "Profit Margin", "EPS", "Revenue Growth"])
                
                with tab1:
                    st.bar_chart(data=df_math, x="year", y="revenue")
                with tab2:
                    st.line_chart(data=df_math, x="year", y="Profit Margin (%)")
                with tab3:
                    st.line_chart(data=df_math, x="year", y="eps")
                with tab4:
                    st.bar_chart(data=df_math.dropna(subset=['Revenue Growth (%)']), x="year", y="Revenue Growth (%)")
