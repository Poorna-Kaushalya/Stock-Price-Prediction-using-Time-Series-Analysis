
# Apple Stock Predictor

📈 **Interactive Apple Stock Predictor** — Predict Apple Inc. (AAPL) stock closing prices using Machine Learning and Deep Learning models based on historical data.

---

## Project Overview

This project forecasts Apple Inc. (AAPL) stock closing prices using historical stock data from 2020 to 2024. It implements multiple models including:

- Linear Regression  
- Random Forest Regressor  
- LSTM (Long Short-Term Memory neural network)  
- Ensemble model combining Linear Regression and LSTM  

The goal is to compare the effectiveness of classical machine learning methods and deep learning approaches for time series stock price prediction.

---

## Features

- Download historical stock data using Yahoo Finance API (`yfinance`)  
- Feature engineering with lagged closing prices and moving averages  
- Train/test split preserving time series order  
- Models implemented:  
  - Linear Regression  
  - Random Forest Regressor  
  - LSTM with PyTorch  
  - Ensemble of Linear Regression and LSTM  
- Performance evaluation with RMSE, MAE, and R² metrics  
- Visualization of actual vs predicted prices  
- Interactive web app built with Streamlit  
- PDF report generation with prediction summary and stock trend plots  

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Poorna-Kaushalya/Stock-Price-Prediction-using-Time-Series-Analysis.git
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Train the Models

Run the training script to download data, train models, evaluate, and save the Linear Regression model:

```bash
python daily_update.py
```

### Run the Streamlit App

Launch the interactive web app for prediction and PDF report generation:

```bash
streamlit run app.py
```

---

## Model Details

### Linear Regression

* Uses lagged opening price, lagged closing price, 7-day and 14-day moving averages as features
* Fast, interpretable, and best performance among tested models on this dataset

### Random Forest Regressor

* Ensemble tree-based model
* Did not perform well on this time series dataset likely due to temporal dependencies

### LSTM Neural Network

* Deep learning model for sequential data
* Uses scaled closing prices and sequences of 60 days for training

### Ensemble Model

* Averages predictions from Linear Regression and LSTM models
* Improves accuracy by combining strengths of both

---

## Evaluation Metrics
<div align="center">
  
| Model                | RMSE  | MAE   | R²   | Interpretation                                    |
| -------------------- | ----- | ----- | ---- | ------------------------------------------------- |
| Linear Regression    | 2.70  | 2.06  | 0.99 | Best performance among individual models          |
| Random Forest        | 25.51 | 18.87 | 0.00 | Poor performance; likely overfitting              |
| LSTM                 | 10.41 | 8.46  | 0.84 | Good performance leveraging sequential patterns   |
| Ensemble (LR + LSTM) | 5.85  | 4.73  | 0.95 | Combines strengths to improve prediction accuracy |

</div>

---

## Visualization

* Plots comparing actual vs predicted closing prices for each model

<div align="center">
  
<table>
  <tr>
    <td align="center">
      <strong>Linear Regression</strong><br>
      <img src="images/lr.png" width="400">
    </td>
    <td align="center">
      <strong>Random Forest</strong><br>
      <img src="images/rf.png" width="400">
    </td>
  </tr>
  <tr>
    <td align="center">
      <strong>LSTM</strong><br>
      <img src="images/lstm.png" width="400">
    </td>
    <td align="center">
      <strong>Ensemble (LR + LSTM)</strong><br>
      <img src="images/ensemble.png" width="400">
    </td>
  </tr>
</table>

</div>

* Interactive 30-day closing price trend graph in the Streamlit app

<div align="center">
  <img src="images/30day.png" width="400">
</div>

* User Interface
  
<div align="center">

<table>
  <tr>
    <td align="center">
      <strong>User Interface</strong><br>
      <img src="images/ui.png" width="400">
    </td>
    <td align="center">
      <strong>PDF Format</strong><br>
      <img src="images/pdf.png" width="400">
    </td>
  </tr>
</table>

</div>

---

<div align="center">
  
#### 👁 Streamlit Web App [Apple Stock Predictor](https://stock-price-prediction-using-time-series-analysis-dwve35gtszyf.streamlit.app/)

</div>

---

## 🛠️ Built With

<div align="center">

<table>
  <thead>
    <tr>
      <th style="width: 400px; text-align: center; padding-right: 20px;">Technology</th>
      <th style="text-align: center;">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center;">
        <img src="https://img.icons8.com/color/24/000000/python--v1.png" alt="Python" style="vertical-align: middle;"/> 
        <a href="https://www.python.org/">Python 3.8+</a>
      </td>
      <td style="text-align: center;">Programming language</td>
    </tr>
    <tr>
      <td style="text-align: center;">
        <img src="https://img.icons8.com/ios-filled/24/000000/money.png" alt="yfinance" style="vertical-align: middle;"/> 
        <a href="https://pypi.org/project/yfinance/">yfinance</a>
      </td>
      <td style="text-align: center;">Yahoo Finance API wrapper</td>
    </tr>
    <tr>
      <td style="text-align: center;">
        <img src="https://img.icons8.com/color/24/000000/pandas.png" alt="pandas" style="vertical-align: middle;"/> 
        <a href="https://pandas.pydata.org/">pandas</a>
      </td>
      <td style="text-align: center;">Data manipulation</td>
    </tr>
    <tr>
      <td style="text-align: center;">
        <img src="https://img.icons8.com/color/24/000000/numpy.png" alt="numpy" style="vertical-align: middle;"/> 
        <a href="https://numpy.org/">numpy</a>
      </td>
      <td style="text-align: center;">Numerical computing</td>
    </tr>
    <tr>
      <td style="text-align: center;">
        <img src="https://img.icons8.com/color/24/000000/graph.png" alt="matplotlib" style="vertical-align: middle;"/> 
        <a href="https://matplotlib.org/">matplotlib</a>
      </td>
      <td style="text-align: center;">Data visualization</td>
    </tr>
    <tr>
      <td style="text-align: center;">
        <img src="https://img.icons8.com/color/24/000000/artificial-intelligence.png" alt="scikit-learn" style="vertical-align: middle;"/> 
        <a href="https://scikit-learn.org/">scikit-learn</a>
      </td>
      <td style="text-align: center;">Machine learning library</td>
    </tr>
    <tr>
      <td style="text-align: center;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/9/96/Pytorch_logo.png" alt="PyTorch" width="24" height="24" style="vertical-align: middle;"/> 
        <a href="https://pytorch.org/">PyTorch</a>
      </td>
      <td style="text-align: center;">Deep learning framework</td>
    </tr>
    <tr>
      <td style="text-align: center;">
        <img src="https://img.icons8.com/color/24/000000/streamlit.png" alt="Streamlit" style="vertical-align: middle;"/> 
        <a href="https://streamlit.io/">Streamlit</a>
      </td>
      <td style="text-align: center;">Web app framework</td>
    </tr>
    <tr>
      <td style="text-align: center;">
        <img src="https://img.icons8.com/ios-filled/24/000000/pdf.png" alt="fpdf2" style="vertical-align: middle;"/> 
        <a href="https://pypi.org/project/fpdf2/">fpdf2</a>
      </td>
      <td style="text-align: center;">PDF generation</td>
    </tr>
  </tbody>
</table>

</div>



---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

* Yahoo Finance API (`yfinance`) for providing free financial data
* Open-source libraries such as Scikit-learn, PyTorch, and Streamlit
* Inspiration and examples from various machine learning and financial forecasting resources
---
#### Built with  by [Poorna Kaushalya](https://github.com/Poorna-Kaushalya)
If you want, I can also help you create the requirements.txt, daily_update.py, or app.py files. Just ask!
