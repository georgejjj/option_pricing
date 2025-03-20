import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd

# Page configuration with custom styling
st.set_page_config(page_title="European Option Pricing Demo", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 1rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2563EB;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .info-text {
        font-size: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        background-color: #F3F4F6;
        border-radius: 4px 4px 0px 0px;
        gap: 1rem;
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #DBEAFE;
        border-bottom: 2px solid #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">European Option Pricing and Greeks</p>', unsafe_allow_html=True)

# Define Black-Scholes model function
def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Calculate European option price and Greeks
    
    Parameters:
    S: Current asset price
    K: Strike price
    T: Time to expiration (years)
    r: Risk-free rate
    sigma: Volatility
    option_type: 'call' or 'put'
    
    Returns:
    price: Option price
    delta: Delta value
    gamma: Gamma value
    theta: Theta value
    vega: Vega value
    rho: Rho value
    """
    # Prevent errors when T is close to 0
    T = max(T, 1e-10)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = -(S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:  # Put option
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        theta = -(S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
    
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * np.sqrt(T) * norm.pdf(d1) * 0.01  # Vega for 1% change in volatility
    rho = K * T * np.exp(-r * T) * norm.cdf(d2) * 0.01 if option_type == 'call' else -K * T * np.exp(-r * T) * norm.cdf(-d2) * 0.01
    
    return price, delta, gamma, theta / 365, vega, rho  # Convert theta to daily

# Interface design - Parameters on the left, charts on the right
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown('<p class="sub-header">Option Parameters</p>', unsafe_allow_html=True)
    
    option_type = st.selectbox("Option Type", ["Call Option", "Put Option"])
    option_type_value = "call" if option_type == "Call Option" else "put"
    
    S0 = st.number_input("Current Asset Price (S)", min_value=1.0, max_value=1000.0, value=100.0, step=1.0)
    K = st.number_input("Strike Price (K)", min_value=1.0, max_value=1000.0, value=100.0, step=1.0)
    
    col1_1, col1_2 = st.columns(2)
    with col1_1:
        T_max = st.number_input("Max Time to Expiration (years)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    with col1_2:
        T = st.slider("Time to Expiration (years)", min_value=0.01, max_value=T_max, value=T_max, step=0.01)
    
    r = st.slider("Risk-free Rate (%)", min_value=0.0, max_value=10.0, value=2.5, step=0.1) / 100
    sigma = st.slider("Volatility (%)", min_value=1.0, max_value=100.0, value=20.0, step=1.0) / 100
    
    # Calculate option price and Greeks with current parameters
    price, delta, gamma, theta, vega, rho = black_scholes(S0, K, T, r, sigma, option_type_value)
    
    st.markdown('<p class="sub-header">Option Value and Greeks</p>', unsafe_allow_html=True)
    
    # Create a cleaner display for metrics
    metrics_df = pd.DataFrame({
        'Metric': ['Option Price', 'Delta', 'Gamma', 'Theta (Daily)', 'Vega', 'Rho'],
        'Value': [f"{price:.4f}", f"{delta:.4f}", f"{gamma:.4f}", f"{theta:.4f}", f"{vega:.4f}", f"{rho:.4f}"]
    })
    
    st.dataframe(metrics_df, hide_index=True, use_container_width=True)

with col2:
    st.markdown('<p class="sub-header">Chart Analysis</p>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Value Curves", "Greeks"])
    
    # Price range for the asset
    price_range = np.linspace(max(1, S0 * 0.5), S0 * 1.5, 1000)
    
    with tab1:
        st.markdown('<p class="sub-header">Option Value Curves</p>', unsafe_allow_html=True)
        
        # Prepare chart data
        payoff = np.maximum(price_range - K, 0) if option_type_value == 'call' else np.maximum(K - price_range, 0)
        option_prices = np.array([black_scholes(s, K, T, r, sigma, option_type_value)[0] for s in price_range])
        
        # Create the chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(price_range, payoff, 'r--', label='Payoff at Expiration')
        ax.plot(price_range, option_prices, 'b-', label='Current Option Value')
        ax.axvline(x=S0, color='green', linestyle=':', label=f'Current Price ({S0})')
        ax.axvline(x=K, color='black', linestyle=':', label=f'Strike Price ({K})')
        ax.set_xlabel('Asset Price')
        ax.set_ylabel('Option Value')
        ax.set_title(f'{option_type} Value Curve (Time to Expiration: {T:.2f} years)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with tab2:
        st.markdown('<p class="sub-header">Greeks Curves</p>', unsafe_allow_html=True)
        
        # Greek selection
        greek_choice = st.selectbox("Select Greek", ["Delta", "Gamma", "Theta", "Vega", "Rho"])
        
        # Calculate the selected Greek across the price range
        if greek_choice == "Delta":
            greek_values = np.array([black_scholes(s, K, T, r, sigma, option_type_value)[1] for s in price_range])
            y_label = "Delta"
        elif greek_choice == "Gamma":
            greek_values = np.array([black_scholes(s, K, T, r, sigma, option_type_value)[2] for s in price_range])
            y_label = "Gamma"
        elif greek_choice == "Theta":
            greek_values = np.array([black_scholes(s, K, T, r, sigma, option_type_value)[3] for s in price_range])
            y_label = "Theta (Daily)"
        elif greek_choice == "Vega":
            greek_values = np.array([black_scholes(s, K, T, r, sigma, option_type_value)[4] for s in price_range])
            y_label = "Vega"
        else:  # Rho
            greek_values = np.array([black_scholes(s, K, T, r, sigma, option_type_value)[5] for s in price_range])
            y_label = "Rho"
        
        # Create the Greek curve chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(price_range, greek_values, 'g-')
        ax.axvline(x=S0, color='green', linestyle=':', label=f'Current Price ({S0})')
        ax.axvline(x=K, color='black', linestyle=':', label=f'Strike Price ({K})')
        ax.set_xlabel('Asset Price')
        ax.set_ylabel(y_label)
        ax.set_title(f'{option_type} {greek_choice} Curve (Time to Expiration: {T:.2f} years)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# Educational explanation
with st.expander("About European Options and Greeks"):
    st.markdown("""
    ## European Options
    A European option is a financial derivative that gives the holder the right, but not the obligation, to buy (call) or sell (put) an underlying asset at a specific price (strike price) on a specific date (expiration date).
    
    ### Black-Scholes Model
    This demo uses the Black-Scholes model to calculate European option prices based on the following assumptions:
    - No arbitrage opportunities
    - Market efficiency with continuous trading
    - Asset prices follow geometric Brownian motion
    - Constant risk-free rate and volatility
    
    ### Greeks
    Greeks measure the sensitivity of option prices to various factors:
    
    - **Delta (Δ)**: Sensitivity of option price to changes in the underlying asset price
    - **Gamma (Γ)**: Rate of change of Delta with respect to the underlying asset price
    - **Theta (Θ)**: Sensitivity of option price to the passage of time
    - **Vega (v)**: Sensitivity of option price to changes in volatility
    - **Rho (ρ)**: Sensitivity of option price to changes in the risk-free rate
    """)

st.caption("Note: This application is for educational purposes only and does not constitute investment advice.")