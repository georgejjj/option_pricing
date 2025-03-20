# European Option Pricing Demo Application

This is a Streamlit-based educational application for demonstrating European option pricing and Greeks calculation.

## Features

1. Configurable option parameters (type, price, strike, expiration time, interest rate, volatility)
2. Visual representation of option payoff at expiration and current option value curves
3. Display of option Greeks (Delta, Gamma, Theta, Vega, Rho) and their behavior

## Installation and Running

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Run the application:
```
streamlit run option_pricing_app.py
```

## Usage Guide

- Adjust option parameters in the left panel
- View value curves and Greek charts in the right panel

## Theoretical Foundation

This application uses the Black-Scholes model to calculate European option prices and Greeks. 