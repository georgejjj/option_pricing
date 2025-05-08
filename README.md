# Option Pricing Demo Application

This is a Streamlit-based educational application for demonstrating option pricing and Greeks calculation.

## Features

1. European Options:
   - Configurable option parameters (type, price, strike, expiration time, interest rate, volatility)
   - Visual representation of option payoff at expiration and current option value curves
   - Display of option Greeks (Delta, Gamma, Theta, Vega, Rho) and their behavior

2. Barrier Options:
   - Both knock-in and knock-out barrier options
   - Configurable barrier level
   - Comparative visualization with standard European options
   - Value curve and Delta analysis

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

- Switch between option types using the tabs at the top
- Adjust option parameters in the left panel
- View value curves and Greek charts in the right panel
- For barrier options, observe how the barrier level affects option pricing and Delta

## Theoretical Foundation

This application uses the Black-Scholes model and its extensions to calculate option prices and Greeks. 