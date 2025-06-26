# Time Series Forecasting: Microsoft Stock Price Prediction

A comprehensive time series analysis and forecasting project using multiple machine learning and statistical models to predict Microsoft (MSFT) stock prices from Yahoo Finance data.

## 🎯 Project Overview

This project implements and compares four different time series forecasting models to predict Microsoft stock prices:
- **ARIMA** (AutoRegressive Integrated Moving Average)
- **SARIMA** (Seasonal ARIMA)
- **Prophet** (Facebook's forecasting tool)
- **LSTM** (Long Short-Term Memory Neural Network)

## 🚀 Live Demo

**[View Live Application](https://timeseriesmodels-eedscq3mr6j2chkc7tl33r.streamlit.app/)**
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://timeseriesmodels-eedscq3mr6j2chkc7tl33r.streamlit.app/)

## 📊 Model Performance Comparison

| Model   | RMSE  | MAE  | MAPE | Training Time | Notes                        |
|---------|-------|------|------|---------------|------------------------------|
| SARIMA  | 7.07  | ...  | ...  | Medium        | Good baseline model          |
| Prophet | 23.32 | ...  | ...  | Fast          | Handles seasonality well     |
| LSTM    | 0.015 | ...  | ...  | Slow          | Needs more data, complex     |
| ARIMA   | 7.77  | 5.21 | 1.29%| Medium        | Classic statistical approach |

### 🏆 Key Findings
- **LSTM** achieved the lowest RMSE (0.015), demonstrating superior accuracy
- **ARIMA** showed balanced performance with MAPE of 1.29%
- **Prophet** excelled in handling seasonal patterns despite higher RMSE
- **SARIMA** provided a solid baseline with good interpretability

## 🛠️ Project Workflow

### 1. Data Collection
- **Data Source**: Yahoo Finance API
- **Stock**: Microsoft Corporation (MSFT)
- **Features**: Open, High, Low, Close, Volume, Adjusted Close

### 2. Data Cleaning & Preprocessing
- Handling missing values
- Date formatting and indexing
- Outlier detection and treatment
- Data validation and quality checks

### 3. Exploratory Data Analysis (EDA)
- Time series visualization
- Trend and seasonality analysis
- Statistical properties examination
- Correlation analysis
- Stationarity testing

### 4. Model Building & Implementation

#### ARIMA Model
- Auto-correlation and partial auto-correlation analysis
- Parameter selection using AIC/BIC criteria
- Model diagnostics and residual analysis

#### SARIMA Model
- Seasonal decomposition
- Seasonal parameter optimization
- Enhanced pattern recognition

#### Prophet Model
- Automatic seasonality detection
- Holiday effects incorporation
- Trend changepoint analysis

#### LSTM Model
- Sequential neural network architecture
- Time series data reshaping
- Hyperparameter tuning
- Early stopping and regularization

### 5. Model Evaluation & Comparison
- Multiple evaluation metrics (RMSE, MAE, MAPE)
- Cross-validation techniques
- Performance visualization
- Statistical significance testing

### 6. Deployment
- Interactive Streamlit web application
- Real-time predictions
- Model comparison dashboard
- User-friendly interface

## 📋 Requirements

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow
keras
statsmodels
prophet
yfinance
streamlit
plotly
```

## 🔧 Installation & Setup

1. Clone the repository:
```bash
git clone https://github.com/Gopalbhalani137/Time_Series_Models.git
cd Time_Series_Models
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit application:
```bash
streamlit run app.py
```

## 📁 Project Structure

```
Time_Series_Models/
│
├── .devcontainer/           # Development container configuration
├── app.py                   # Main Streamlit application
├── arima_model.pkl          # Trained ARIMA model
├── best_lstm_model.h5       # Trained LSTM model (Keras format)
├── cleaned (1).csv          # Cleaned Microsoft stock data
├── microsoft_stock_pred.ipynb # Jupyter notebook with full analysis
├── prophet_model.pkl        # Trained Prophet model
├── requirements.txt         # Python dependencies
├── runtime.txt             # Python runtime version
├── sarima_model.pkl        # Trained SARIMA model
└── README.md               # Project documentation
```

## 📈 Key Features

- **Multi-Model Comparison**: Side-by-side evaluation of four different forecasting approaches
- **Interactive Visualization**: Dynamic charts and plots for better insights
- **Real-time Predictions**: Live forecasting capabilities
- **Statistical Analysis**: Comprehensive model diagnostics
- **User-Friendly Interface**: Streamlit-based web application

## 🎯 Business Applications

- **Investment Decision Making**: Informed stock price predictions
- **Risk Management**: Volatility forecasting and assessment
- **Portfolio Optimization**: Strategic asset allocation
- **Market Analysis**: Trend identification and pattern recognition

## 🚀 Future Enhancements

- [ ] Add more stocks and comparison capabilities
- [ ] Implement ensemble methods
- [ ] Include sentiment analysis from news data
- [ ] Add real-time data streaming
- [ ] Expand to cryptocurrency forecasting
- [ ] Include confidence intervals and uncertainty quantification

## 📝 Methodology

### Data Preprocessing
1. Missing value imputation
2. Stationarity transformation
3. Feature engineering
4. Train-test split with temporal considerations

### Model Selection Criteria
- **Accuracy**: RMSE, MAE, MAPE metrics
- **Computational Efficiency**: Training and prediction time
- **Interpretability**: Model transparency and explainability
- **Robustness**: Performance across different market conditions

## 📊 Results & Insights

The LSTM model demonstrated superior performance in terms of RMSE, while ARIMA provided the best balance of accuracy and interpretability. Prophet excelled in capturing seasonal patterns, making it valuable for long-term forecasting scenarios.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.


Project Link: https://github.com/Gopalbhalani137/Time_Series_Models

---

