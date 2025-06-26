import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import with error handling for better compatibility
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    try:
        from fbprophet import Prophet  # Try older version
        PROPHET_AVAILABLE = True
    except ImportError:
        PROPHET_AVAILABLE = False
        print("⚠️ Prophet not available. Prophet model will be disabled.")

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    from tensorflow.keras.models import load_model
    from sklearn.preprocessing import MinMaxScaler
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available. LSTM model will be disabled.")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("❌ yfinance is required. Please install: pip install yfinance")

from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Microsoft Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .prediction-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .model-comparison {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Cache functions for model loading
@st.cache_resource
def load_models():
    """Load all trained models with error handling"""
    models = {}
    
    # Load Prophet model
    if PROPHET_AVAILABLE:
        try:
            with open('prophet_model.pkl', 'rb') as f:
                models['Prophet'] = pickle.load(f)
        except FileNotFoundError:
            st.warning("⚠️ prophet_model.pkl not found")
        except Exception as e:
            st.warning(f"⚠️ Error loading Prophet model: {str(e)}")
    
    # Load ARIMA model
    try:
        with open('arima_model.pkl', 'rb') as f:
            models['ARIMA'] = pickle.load(f)
    except FileNotFoundError:
        st.warning("⚠️ arima_model.pkl not found")
    except Exception as e:
        st.warning(f"⚠️ Error loading ARIMA model: {str(e)}")
    
    # Load SARIMA model
    try:
        with open('sarima_model.pkl', 'rb') as f:
            models['SARIMA'] = pickle.load(f)
    except FileNotFoundError:
        st.warning("⚠️ sarima_model.pkl not found")
    except Exception as e:
        st.warning(f"⚠️ Error loading SARIMA model: {str(e)}")
    
    # Load LSTM model
    if TENSORFLOW_AVAILABLE:
        try:
            models['LSTM'] = load_model('best_lstm_model.h5')
        except FileNotFoundError:
            st.warning("⚠️ best_lstm_model.h5 not found")
        except Exception as e:
            st.warning(f"⚠️ Error loading LSTM model: {str(e)}")
    
    return models if models else None

@st.cache_data
def get_stock_data(symbol="MSFT", period="2y"):
    """Fetch stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        return data
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        return None

def prepare_data_for_models(data):
    """Prepare data for different models"""
    # For Prophet
    prophet_data = pd.DataFrame({
        'ds': data.index,
        'y': data['Close']
    })
    
    # For LSTM (normalized data)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    return prophet_data, scaled_data, scaler

def create_lstm_sequences(data, seq_length=60):
    """Create sequences for LSTM prediction"""
    X = []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
    return np.array(X)

def make_predictions(models, data, days_ahead=30):
    """Make predictions using all models"""
    predictions = {}
    
    # Get the latest data
    prophet_data, scaled_data, scaler = prepare_data_for_models(data)
    
    # Prophet Predictions
    if 'Prophet' in models:
        try:
            future_dates = models['Prophet'].make_future_dataframe(periods=days_ahead)
            prophet_forecast = models['Prophet'].predict(future_dates)
            predictions['Prophet'] = prophet_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days_ahead)
        except Exception as e:
            st.warning(f"Prophet prediction failed: {str(e)}")
    
    # ARIMA Predictions
    if 'ARIMA' in models:
        try:
            arima_forecast = models['ARIMA'].forecast(steps=days_ahead)
            future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=days_ahead)
            predictions['ARIMA'] = pd.DataFrame({
                'ds': future_dates,
                'yhat': arima_forecast
            })
        except Exception as e:
            st.warning(f"ARIMA prediction failed: {str(e)}")
    
    # SARIMA Predictions
    if 'SARIMA' in models:
        try:
            sarima_forecast = models['SARIMA'].forecast(steps=days_ahead)
            future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=days_ahead)
            predictions['SARIMA'] = pd.DataFrame({
                'ds': future_dates,
                'yhat': sarima_forecast
            })
        except Exception as e:
            st.warning(f"SARIMA prediction failed: {str(e)}")
    
    # LSTM Predictions
    if 'LSTM' in models:
        try:
            # Prepare last 60 days of data for LSTM
            last_60_days = scaled_data[-60:]
            lstm_predictions = []
            
            for _ in range(days_ahead):
                lstm_input = last_60_days.reshape(1, 60, 1)
                pred = models['LSTM'].predict(lstm_input, verbose=0)
                lstm_predictions.append(pred[0, 0])
                last_60_days = np.append(last_60_days[1:], pred)
            
            # Inverse transform predictions
            lstm_predictions = scaler.inverse_transform(np.array(lstm_predictions).reshape(-1, 1))
            future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=days_ahead)
            predictions['LSTM'] = pd.DataFrame({
                'ds': future_dates,
                'yhat': lstm_predictions.flatten()
            })
        except Exception as e:
            st.warning(f"LSTM prediction failed: {str(e)}")
    
    return predictions

def create_prediction_chart(data, predictions, model_name):
    """Create interactive prediction chart"""
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=data.index[-90:],  # Last 90 days
        y=data['Close'].iloc[-90:],
        mode='lines',
        name='Historical Data',
        line=dict(color='blue', width=2)
    ))
    
    # Add predictions
    if model_name in predictions:
        pred_data = predictions[model_name]
        fig.add_trace(go.Scatter(
            x=pred_data['ds'],
            y=pred_data['yhat'],
            mode='lines+markers',
            name=f'{model_name} Predictions',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=6)
        ))
        
        # Add confidence intervals for Prophet
        if model_name == 'Prophet' and 'yhat_lower' in pred_data.columns:
            fig.add_trace(go.Scatter(
                x=pred_data['ds'],
                y=pred_data['yhat_upper'],
                fill=None,
                mode='lines',
                line_color='rgba(0,0,0,0)',
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=pred_data['ds'],
                y=pred_data['yhat_lower'],
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,0,0,0)',
                name='Confidence Interval',
                fillcolor='rgba(255,0,0,0.2)'
            ))
    
    fig.update_layout(
        title=f'{model_name} - Microsoft Stock Price Prediction',
        xaxis_title='Date',
        yaxis_title='Stock Price ($)',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig

def main():
    # Main header
    st.markdown('<h1 class="main-header">📈 Microsoft Stock Prediction Dashboard</h1>', unsafe_allow_html=True)
    
    # Show available packages status
    with st.expander("📦 Package Status", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write("✅ Streamlit" if True else "❌ Streamlit")
            st.write("✅ Pandas" if True else "❌ Pandas")
            st.write("✅ Numpy" if True else "❌ Numpy")
            st.write("✅ Plotly" if True else "❌ Plotly")
        with col2:
            st.write("✅ Prophet" if PROPHET_AVAILABLE else "❌ Prophet")
            st.write("✅ TensorFlow" if TENSORFLOW_AVAILABLE else "❌ TensorFlow")
            st.write("✅ YFinance" if YFINANCE_AVAILABLE else "❌ YFinance")
            st.write("✅ Joblib" if JOBLIB_AVAILABLE else "❌ Joblib")
    
    if not YFINANCE_AVAILABLE:
        st.error("❌ YFinance is required for this app. Please install it: `pip install yfinance`")
        return
    
    # Sidebar
    st.sidebar.header("🎛️ Controls")
    
    # Load models
    with st.spinner("Loading models..."):
        models = load_models()
    
    if models is None:
        st.error("Failed to load models. Please ensure all model files are in the correct directory.")
        return
    
    st.sidebar.success(f"✅ Loaded {len(models)} models successfully!")
    
    # Model selection
    available_models = list(models.keys())
    selected_models = st.sidebar.multiselect(
        "Select Models to Compare",
        available_models,
        default=available_models
    )
    
    # Prediction parameters
    days_ahead = st.sidebar.slider("Days to Predict", 1, 90, 30)
    
    # Fetch stock data
    with st.spinner("Fetching latest Microsoft stock data..."):
        stock_data = get_stock_data()
    
    if stock_data is None:
        st.error("Failed to fetch stock data.")
        return
    
    # Display current stock info
    current_price = stock_data['Close'].iloc[-1]
    previous_price = stock_data['Close'].iloc[-2]
    price_change = current_price - previous_price
    price_change_pct = (price_change / previous_price) * 100
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Current Price",
            value=f"${current_price:.2f}",
            delta=f"${price_change:.2f}"
        )
    
    with col2:
        st.metric(
            label="Change %",
            value=f"{price_change_pct:.2f}%"
        )
    
    with col3:
        st.metric(
            label="Volume",
            value=f"{stock_data['Volume'].iloc[-1]:,}"
        )
    
    with col4:
        st.metric(
            label="52W High",
            value=f"${stock_data['High'].max():.2f}"
        )
    
    # Make predictions
    if st.sidebar.button("🔮 Generate Predictions", type="primary"):
        with st.spinner("Generating predictions..."):
            predictions = make_predictions(models, stock_data, days_ahead)
        
        if predictions:
            st.markdown('<h2 class="sub-header">📊 Prediction Results</h2>', unsafe_allow_html=True)
            
            # Create tabs for different models
            tabs = st.tabs(selected_models)
            
            for i, model_name in enumerate(selected_models):
                with tabs[i]:
                    if model_name in predictions:
                        # Show prediction chart
                        fig = create_prediction_chart(stock_data, predictions, model_name)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show prediction summary
                        pred_data = predictions[model_name]
                        avg_prediction = pred_data['yhat'].mean()
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"""
                            <div class="prediction-box">
                                <h3>{model_name} Average Prediction</h3>
                                <h2>${avg_prediction:.2f}</h2>
                                <p>Next {days_ahead} days</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            final_prediction = pred_data['yhat'].iloc[-1]
                            change_from_current = final_prediction - current_price
                            change_pct = (change_from_current / current_price) * 100
                            
                            st.markdown(f"""
                            <div class="prediction-box">
                                <h3>Final Day Prediction</h3>
                                <h2>${final_prediction:.2f}</h2>
                                <p>Change: {change_pct:+.2f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Show detailed predictions table
                        with st.expander(f"📋 Detailed {model_name} Predictions"):
                            display_data = pred_data.copy()
                            display_data['ds'] = display_data['ds'].dt.strftime('%Y-%m-%d')
                            display_data['yhat'] = display_data['yhat'].round(2)
                            st.dataframe(display_data, use_container_width=True)
            
            # Model comparison
            if len(selected_models) > 1:
                st.markdown('<h2 class="sub-header">🏆 Model Comparison</h2>', unsafe_allow_html=True)
                
                # Create comparison chart
                fig_comp = go.Figure()
                
                # Add historical data
                fig_comp.add_trace(go.Scatter(
                    x=stock_data.index[-30:],
                    y=stock_data['Close'].iloc[-30:],
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue', width=3)
                ))
                
                colors = ['red', 'green', 'orange', 'purple']
                for i, model_name in enumerate(selected_models):
                    if model_name in predictions:
                        pred_data = predictions[model_name]
                        fig_comp.add_trace(go.Scatter(
                            x=pred_data['ds'],
                            y=pred_data['yhat'],
                            mode='lines+markers',
                            name=f'{model_name}',
                            line=dict(color=colors[i % len(colors)], width=2),
                            marker=dict(size=4)
                        ))
                
                fig_comp.update_layout(
                    title='Model Predictions Comparison',
                    xaxis_title='Date',
                    yaxis_title='Stock Price ($)',
                    hovermode='x unified',
                    template='plotly_white',
                    height=500
                )
                
                st.plotly_chart(fig_comp, use_container_width=True)
                
                # Comparison metrics
                comparison_data = []
                for model_name in selected_models:
                    if model_name in predictions:
                        pred_data = predictions[model_name]
                        avg_pred = pred_data['yhat'].mean()
                        final_pred = pred_data['yhat'].iloc[-1]
                        comparison_data.append({
                            'Model': model_name,
                            'Average Prediction': f"${avg_pred:.2f}",
                            'Final Day Prediction': f"${final_pred:.2f}",
                            'Change from Current': f"{((final_pred - current_price) / current_price * 100):+.2f}%"
                        })
                
                st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
    
    # Historical data visualization
    st.markdown('<h2 class="sub-header">📈 Historical Stock Data</h2>', unsafe_allow_html=True)
    
    # Create historical chart with multiple indicators
    fig_hist = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Stock Price', 'Volume'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Stock price
    fig_hist.add_trace(
        go.Candlestick(
            x=stock_data.index,
            open=stock_data['Open'],
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close'],
            name='MSFT'
        ),
        row=1, col=1
    )
    
    # Volume
    fig_hist.add_trace(
        go.Bar(
            x=stock_data.index,
            y=stock_data['Volume'],
            name='Volume',
            marker_color='lightblue'
        ),
        row=2, col=1
    )
    
    fig_hist.update_layout(
        title='Microsoft Stock - Historical Data',
        xaxis_rangeslider_visible=False,
        height=600,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit | Data from Yahoo Finance")

if __name__ == "__main__":
    main()