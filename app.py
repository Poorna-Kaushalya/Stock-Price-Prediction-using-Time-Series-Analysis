import streamlit as st
import pandas as pd
import joblib
import yfinance as yf
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile
import os

st.set_page_config(page_title="Apple Stock Predictor", page_icon="📈", layout="centered")

model = joblib.load('linear_regression_model.pkl')

st.markdown("<h1 style='text-align: center;'>Interactive Apple Stock Predictor</h1><center><h4 style='color:yellow; padding-top: 2px;'>Powered by Machine Learning</h4></center>", unsafe_allow_html=True)
st.markdown("---")

st.markdown("""
<p style="text-align: justify; padding-top: px;">
This web application is built to assist users in predicting Apple Inc. (AAPL) stock closing prices using machine learning models. It combines simplicity with powerful analytics, allowing investors, traders, and data science enthusiasts to explore stock trends and make informed decisions. Powered by historical data and models like Linear Regression and LSTM, this tool offers a user-friendly way to understand and anticipate market movements.
</p>
""", unsafe_allow_html=True)
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
with col2:
    close_lag1 = st.number_input("📌 Yesterday's Close", min_value=0.0, value=180.00)
with col3:
    ma7 = st.number_input("📊 Week Avg", min_value=0.0, value=178.00)
with col4:
    ma14 = st.number_input("📉 2 Week Avg", min_value=0.0, value=175.00)
with col1:
    Open_p = st.number_input("📌 Yesterday's Open", min_value=0.0, value=175.00)

if st.button("🔮 Predict Closing Price"):
    input_df = pd.DataFrame([[Open_p, close_lag1, ma7, ma14]], columns=['Open_p', 'Close_Lag1', 'MA7', 'MA14'])
    prediction = model.predict(input_df)[0]
    direction = "📈 Market Expected to Rise" if prediction > close_lag1 else "📉 Market Expected to Fall"
    color = "#34a853" if prediction > close_lag1 else "#ea4335"

    # Display prediction on web page
    st.markdown(f"""
    <div style="display: flex; justify-content: center; align-items: stretch; padding: 20px; background-color: #e6f4ea; border-radius: 10px; color: #1a1a1a; font-family: 'Segoe UI', sans-serif;">
        <div style="flex: 1; padding-right: 20px;">
            <h4>📊 Input Summary:</h4>
            <ul style="list-style-type: none; padding-left: 0;">
                <li>Yesterday Opening Price:<strong>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;${Open_p:.2f}</strong></li>
                <li>Yesterday Closing Price:<strong>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;${close_lag1:.2f}</strong></li>
                <li>Last Week Average Price:<strong>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;${ma7:.2f}</strong></li>
                <li>Last Two Week Average Price:<strong>&nbsp;&nbsp;&nbsp;&nbsp;${ma14:.2f}</strong></li>
            </ul>
        </div>
        <div style="flex: 1; padding-right: 0px; padding-top: 40px; border-left: 2px solid #999; text-align: center;">
            <h1 style="color:{color}; ">💵 ${prediction:.2f}</h1>
            <p><strong>{direction}</strong></p>
        </div>
    </div>
""", unsafe_allow_html=True)
    st.markdown("---")

    # --- Create PDF with styles ---
    class PDF(FPDF):
        def header(self):
            # Register fonts once here or before creating PDF
            self.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
            self.add_font('DejaVu', 'B', 'DejaVuSans.ttf', uni=True)

            self.set_font('DejaVu', 'B', 18)
            self.set_text_color(30, 30, 150)
            self.cell(0, 15, 'Apple Stock Prediction Report', 0, 1, 'C')
            self.set_draw_color(50, 50, 50)
            self.set_line_width(0.8)
            self.line(10, 30, 200, 30)
            self.ln(10)

        def footer(self):
            self.set_y(-20)
            self.set_font('DejaVu', '', 10)
            self.set_text_color(100, 100, 100)
            self.cell(0, 10, '🚀 Built with ❤️ using Streamlit | Model: Linear Regression | Data: Yahoo Finance', 0, 0, 'C')

    pdf = PDF()
    pdf.add_page()

    pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
    pdf.set_font('DejaVu', '', 12)

    # Description paragraph
    desc = ("This report provides a prediction for Apple Inc. (AAPL) stock closing prices "
            "based on historical data and machine learning models. Use this information "
            "to make informed decisions on stock trading and investment.")
    pdf.multi_cell(0, 10, desc)
    pdf.ln(5)

    # Input summary title
    pdf.set_font("DejaVu", 'B', 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, 'Input Summary:', ln=True)
    pdf.set_font("DejaVu", '', 12)
    pdf.ln(2)

    # Input values
    inputs = [
        f"Yesterday Opening Price: ${Open_p:.2f}",
        f"Yesterday Closing Price: ${close_lag1:.2f}",
        f"Last Week Average Price: ${ma7:.2f}",
        f"Last Two Week Average Price: ${ma14:.2f}",
    ]
    for line in inputs:
        pdf.cell(0, 8, line, ln=True)
    pdf.ln(10)

    # Add 30-day closing price plot
    stock_data = yf.download("AAPL", period="30d", interval="1d", auto_adjust=True)[['Close']]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(stock_data.index, stock_data['Close'], marker='o', linestyle='-')
    ax.set_title("AAPL Closing Prices (Last 30 Days)", fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Closing Price (USD)")
    ax.grid(True)
    plt.tight_layout()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        fig.savefig(tmpfile.name, dpi=150)
        plt.close(fig)

        pdf.image(tmpfile.name, x=30, w=150)
        pdf.ln(10)

    # Prediction result with styled box and colors centered horizontally
    if prediction > close_lag1:
        bg_color = (220, 255, 220)  # Light green background
        text_color = (52, 168, 83)  # Dark green text
    else:
        bg_color = (255, 220, 220)  # Light red background
        text_color = (234, 67, 53)  # Dark red text

    pdf.set_fill_color(*bg_color)
    pdf.set_text_color(*text_color)
    pdf.set_draw_color(*text_color)
    pdf.set_line_width(0.8)

    pdf.set_font("DejaVu", 'B', 16)
    pred_text = f"Predicted Closing Price: ${prediction:.2f}"
    pred_text_width = pdf.get_string_width(pred_text) + 20  # padding

    pdf.set_font("DejaVu", '', 14)
    dir_text = f"Market Direction: {direction.replace('📈 ', '').replace('📉 ', '')}"
    dir_text_width = pdf.get_string_width(dir_text) + 20

    box_width = max(pred_text_width, dir_text_width)
    box_height = 25  # approximate height for two lines

    # Calculate centered X position based on page width (210mm for A4)
    page_width = pdf.w
    x_start = (page_width - box_width) / 2

    y_start = pdf.get_y()

    # Draw filled rectangle with border
    pdf.rect(x_start, y_start - 3, box_width, box_height, style='FD')

    # Print texts centered inside the box
    pdf.set_xy(x_start, y_start)
    pdf.set_font("DejaVu", 'B', 16)
    pdf.cell(box_width, 12, pred_text, ln=2, align='C')

    pdf.set_font("DejaVu", '', 14)
    pdf.cell(box_width, 12, dir_text, ln=1, align='C')

    pdf.set_text_color(0, 0, 0)  # Reset color to black
    pdf.ln(10)


    os.unlink(tmpfile.name)

    # Output PDF bytes (no encode needed for fpdf2)
    pdf_bytes = bytes(pdf.output(dest='S'))

    st.download_button(
        label="📄 Download Prediction Report (PDF)",
        data=pdf_bytes,
        file_name="apple_stock_prediction_report.pdf",
        mime="application/pdf"
    )

st.markdown("---")
st.subheader("📉 Apple Stock - Last 30 Days Trend")

@st.cache_data
def fetch_data():
    df = yf.download("AAPL", period="30d", interval="1d", auto_adjust=True)
    return df[['Close']]

stock_data = fetch_data()
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(stock_data.index, stock_data['Close'], marker='o', linestyle='-')
ax.set_title("AAPL Closing Prices (Last 30 Days)", fontsize=14)
ax.set_xlabel("Date")
ax.set_ylabel("Closing Price (USD)")
ax.grid(True)
st.pyplot(fig)

st.markdown("""<hr style='margin-top:40px;'>""", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>🚀 Built with ❤️ using Streamlit | Model: Linear Regression | Data: Yahoo Finance</p>", unsafe_allow_html=True)
