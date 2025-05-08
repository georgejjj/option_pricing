import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
import time
from functools import lru_cache  # 添加缓存功能

# Page configuration with custom styling
st.set_page_config(page_title="Option Pricing Demo", layout="wide")

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
st.markdown('<p class="main-header">Option Pricing and Greeks</p>', unsafe_allow_html=True)

# Main tab selection
option_type_tab = st.tabs(["European Option", "Barrier Option"])

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

# 修改cached_barrier_option_price函数，移除delta相关参数
@lru_cache(maxsize=128)
def cached_barrier_option_price(S, K, T, r, sigma, H, option_type, barrier_type, cache_key=None):
    """缓存版本的障碍期权定价函数，使用lru_cache优化性能"""
    return _barrier_option_price(S, K, T, r, sigma, H, option_type, barrier_type)

def barrier_option_price(S, K, T, r, sigma, H, option_type, barrier_type, N_sims=10000, N_steps=100):
    """
    障碍期权定价主函数。根据不同情况选择最高效的计算方法
    """
    # 根据需求确定是否使用快速或高精度模式 - 界面显示曲线时使用轻量级计算
    need_high_precision = True
    
    # 生成缓存键 - 四舍五入，减少缓存键变化
    S_rounded = round(S, 2)
    K_rounded = round(K, 2)
    T_rounded = round(T, 3)
    r_rounded = round(r, 4)
    sigma_rounded = round(sigma, 4) 
    H_rounded = round(H, 2)
    
    cache_key = f"{S_rounded}_{K_rounded}_{T_rounded}_{r_rounded}_{sigma_rounded}_{H_rounded}_{option_type}_{barrier_type}"
    
    # 对于界面中的曲线计算等调用，使用轻量级模式
    if N_sims <= 1000 or N_steps <= 30:
        need_high_precision = False
    
    # 对于几乎在障碍上的情况，使用精确计算
    barrier_distance_factor = abs(S - H) / S
    is_at_barrier = barrier_distance_factor < 0.01
    
    # 使用缓存函数避免重复计算
    if need_high_precision and not is_at_barrier:
        return cached_barrier_option_price(S_rounded, K_rounded, T_rounded, r_rounded, sigma_rounded, H_rounded, 
                                          option_type, barrier_type, cache_key)
    else:
        # 某些情况下需要绕过缓存，直接计算
        return _barrier_option_price(S, K, T, r, sigma, H, option_type, barrier_type)

# 实际的障碍期权定价实现
def _barrier_option_price(S, K, T, r, sigma, H, option_type, barrier_type):
    """核心障碍期权定价实现"""
    S_orig = S

    # 快速计算辅助函数 - 使用解析公式：对于某些简单情况可快速计算
    def fast_barrier_approximation():
        """使用解析近似公式快速计算障碍期权，仅适用于某些特殊情况"""
        # 欧式期权价格作为基础
        d1 = (np.log(S_orig / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            euro_price = S_orig * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            euro_price = K * np.exp(-r * T) * norm.cdf(-d2) - S_orig * norm.cdf(-d1)
        
        # 障碍调整因子 - 简化版
        barrier_factor = 1.0
        
        # 根据障碍类型调整价格
        if "out" in barrier_type:
            # 已敲出情况
            if (barrier_type == "down-and-out" and S_orig <= H) or \
               (barrier_type == "up-and-out" and S_orig >= H):
                return 0.0
                
            # 远离障碍时，类似欧式期权
            if (barrier_type == "down-and-out" and S_orig > H * 1.5) or \
               (barrier_type == "up-and-out" and S_orig < H * 0.67):
                return euro_price
                
            # 接近障碍时的调整
            if barrier_type == "down-and-out":
                barrier_factor = min(1.0, (S_orig / H - 1.0) * 2)
            else:  # up-and-out
                barrier_factor = min(1.0, (H / S_orig - 1.0) * 2)
                
        else:  # "in" 类型
            # 已敲入情况
            if (barrier_type == "down-and-in" and S_orig <= H) or \
               (barrier_type == "up-and-in" and S_orig >= H):
                return euro_price
                
            # 远离障碍时，价值趋近于0
            if (barrier_type == "down-and-in" and S_orig > H * 1.5) or \
               (barrier_type == "up-and-in" and S_orig < H * 0.67):
                return 0.0
                
            # 接近障碍时的调整
            if barrier_type == "down-and-in":
                barrier_factor = max(0.0, 1.0 - (S_orig / H - 1.0) * 2)
            else:  # up-and-in
                barrier_factor = max(0.0, 1.0 - (H / S_orig - 1.0) * 2)
        
        # 应用调整因子
        adjusted_price = euro_price * barrier_factor
        
        return adjusted_price

    # Internal helper function to calculate price using Monte Carlo for a given spot S_val
    def _calculate_mc_price_raw(S_val, K_val, T_val, r_val, sigma_val, H_val, opt_type_val, barr_type_val, num_sims=5000, num_steps=50):
        if sigma_val <= 0: # Should be caught by the main function's check, but good for direct calls if any
            if opt_type_val == 'call': payoff = np.maximum(0, S_val - K_val)
            else: payoff = np.maximum(0, K_val - S_val)
            return payoff * np.exp(-r_val * T_val) if T_val > 0 else payoff

        # 初始检查 - T=0的情况
        if T_val == 0:
            if opt_type_val == 'call': payoff = np.maximum(0, S_val - K_val)
            else: payoff = np.maximum(0, K_val - S_val)
            
            # 对于敲出期权
            if "out" in barr_type_val:
                if (barr_type_val == "down-and-out" and S_val <= H_val) or \
                   (barr_type_val == "up-and-out" and S_val >= H_val):
                    return 0.0  # 已经触碰障碍，价值为0
                else:
                    return payoff  # 未触碰障碍，返回普通期权价值
            # 对于敲入期权
            else:
                if (barr_type_val == "down-and-in" and S_val <= H_val) or \
                   (barr_type_val == "up-and-in" and S_val >= H_val):
                    return payoff  # 已经触碰障碍，返回普通期权价值
                else:
                    return 0.0  # 未触碰障碍，价值为0

        # 对于已经触碰障碍的情况直接处理
        if ((barr_type_val == 'down-and-out' and S_val <= H_val) or 
            (barr_type_val == 'up-and-out' and S_val >= H_val)):
            return 0.0
            
        # 对于敲入期权，检查是否已经敲入
        is_initially_knocked_in = False
        if ((barr_type_val == 'down-and-in' and S_val <= H_val) or 
            (barr_type_val == 'up-and-in' and S_val >= H_val)):
            is_initially_knocked_in = True
            
        # 是否是下障碍
        is_down_barrier = "down" in barr_type_val
        
        # 准备蒙特卡洛模拟
        dt = T_val / num_steps
        
        # 使用向量化操作获得更好的性能
        # 为所有路径一次性生成随机数
        Z = np.random.normal(0, 1, (num_sims, num_steps))
        
        # 创建所有路径的起始价格数组
        S_paths = np.zeros((num_sims, num_steps + 1))
        S_paths[:, 0] = S_val
        
        # 记录敲入/敲出状态
        knocked_in = np.full(num_sims, is_initially_knocked_in)
        barrier_hit = np.zeros(num_sims, dtype=bool)
        
        # 高效率的Brownian bridge辅助函数
        def fast_prob_cross(S_now, S_next, barrier, is_down):
            """快速计算障碍穿越概率的向量化版本"""
            if is_down:
                # 对于下障碍
                below_mask = (S_now <= barrier) | (S_next <= barrier)
                h = np.log(S_now/barrier) * np.log(S_next/barrier)
                h[below_mask] = 0  # 已经在障碍下方的设为0
                p = np.exp(-2 * h / (sigma_val**2 * dt))
                p[below_mask] = 1.0  # 确保已在障碍下方的概率为1
            else:
                # 对于上障碍
                above_mask = (S_now >= barrier) | (S_next >= barrier)
                h = np.log(barrier/S_now) * np.log(barrier/S_next)
                h[above_mask] = 0  # 已经在障碍上方的设为0
                p = np.exp(-2 * h / (sigma_val**2 * dt))
                p[above_mask] = 1.0  # 确保已在障碍上方的概率为1
            
            return p
        
        # 生成所有路径 - 使用向量化操作大幅提高性能
        for t in range(num_steps):
            # 计算当前还未完成的路径
            active_paths = ~barrier_hit
            if not np.any(active_paths):
                break  # 如果所有路径都已确定，提前结束
                
            # 只更新活跃路径
            S_paths[active_paths, t+1] = S_paths[active_paths, t] * np.exp(
                (r_val - 0.5 * sigma_val**2) * dt + sigma_val * np.sqrt(dt) * Z[active_paths, t]
            )
            
            # 检查障碍条件
            if is_down_barrier:
                # 下障碍检查
                # 直接检查当前价格是否低于障碍
                direct_hit = S_paths[active_paths, t+1] <= H_val
                
                # 更新状态
                for idx, hit in zip(np.where(active_paths)[0], direct_hit):
                    if hit:
                        if "out" in barr_type_val:
                            barrier_hit[idx] = True  # 敲出
                        else:
                            knocked_in[idx] = True  # 敲入
            else:
                # 上障碍检查
                # 直接检查当前价格是否高于障碍
                direct_hit = S_paths[active_paths, t+1] >= H_val
                
                # 更新状态
                for idx, hit in zip(np.where(active_paths)[0], direct_hit):
                    if hit:
                        if "out" in barr_type_val:
                            barrier_hit[idx] = True  # 敲出
                        else:
                            knocked_in[idx] = True  # 敲入
        
        # 计算期权价值
        payoffs = np.zeros(num_sims)
        
        # 根据期权类型计算最终价值
        if opt_type_val == 'call':
            vanilla_payoffs = np.maximum(0, S_paths[:, -1] - K_val)
        else:  # put
            vanilla_payoffs = np.maximum(0, K_val - S_paths[:, -1])
        
        if "out" in barr_type_val:
            # 敲出期权：仅在未敲出时有价值
            payoffs[~barrier_hit] = vanilla_payoffs[~barrier_hit]
        else:
            # 敲入期权：仅在已敲入时有价值
            payoffs[knocked_in] = vanilla_payoffs[knocked_in]
        
        # 返回平均折现payoff
        return np.mean(payoffs) * np.exp(-r_val * T_val)

    # Edge cases not suitable for MC
    if T == 0:
        price = 0.0
        # 删除重复的内在价值计算
        if option_type == 'call':
            intrinsic_value = np.maximum(0, S_orig - K)
        else: # put
            intrinsic_value = np.maximum(0, K - S_orig)

        if barrier_type == 'down-and-out':
            price = 0.0 if S_orig <= H else intrinsic_value
        elif barrier_type == 'down-and-in':
            # At T=0, if S <= H, it's knocked in AND expires. If S > H, never knocked in, 0.
            price = intrinsic_value if S_orig <= H else 0.0
        elif barrier_type == 'up-and-out':
            price = 0.0 if S_orig >= H else intrinsic_value
        elif barrier_type == 'up-and-in':
            # At T=0, if S >= H, it's knocked in AND expires. If S < H, never knocked in, 0.
            price = intrinsic_value if S_orig >= H else 0.0
        
        return price

    if sigma <= 0 or T < 0: # T < 0 is invalid, sigma <=0 means no volatility (deterministic price)
        # 无波动性情况下障碍期权的处理
        price = 0.0
        # 计算无波动性情况下的终值
        S_T = S_orig * np.exp(r * T) if T > 0 else S_orig
        
        # 确定期权价值
        if option_type == 'call': 
            temp_price = np.maximum(0, S_T - K) 
        else: 
            temp_price = np.maximum(0, K - S_T)
            
        # 对于down-and-out和up-and-out期权，检查路径是否会穿越障碍
        if barrier_type == 'down-and-out':
            # 如果当前价格低于障碍 或 终值低于障碍（且r<0)，则期权失效
            min_path_value = min(S_orig, S_T) if r < 0 else S_orig
            price = 0.0 if min_path_value <= H else temp_price
        elif barrier_type == 'up-and-out':
            # 如果当前价格高于障碍 或 终值高于障碍（且r>0)，则期权失效
            max_path_value = max(S_orig, S_T) if r > 0 else S_orig
            price = 0.0 if max_path_value >= H else temp_price
        elif barrier_type == 'down-and-in':
            # 如果当前价格低于障碍 或 终值低于障碍（且r<0)，则期权有效
            min_path_value = min(S_orig, S_T) if r < 0 else S_orig
            price = temp_price if min_path_value <= H else 0.0
        elif barrier_type == 'up-and-in':
            # 如果当前价格高于障碍 或 终值高于障碍（且r>0)，则期权有效
            max_path_value = max(S_orig, S_T) if r > 0 else S_orig
            price = temp_price if max_path_value >= H else 0.0
            
        # 折现
        price = price * np.exp(-r * T) if T > 0 else price

        return price

    # 对于标准情况：首先检查当前价格是否已满足障碍条件
    # 对于 knock-out 期权, 如果当前价格已经触碰障碍, 期权价值为0
    if ((barrier_type == 'down-and-out' and S_orig <= H) or 
        (barrier_type == 'up-and-out' and S_orig >= H)):
        return 0.0
    
    # 判断是否可以使用快速近似方法
    barrier_distance_ratio = abs(S_orig - H) / S_orig
    use_fast_approx = False
    
    # 远离障碍时使用快速近似
    if (barrier_distance_ratio > 0.3 and T < 1) or \
       (S_orig > H * 2 and "down" in barrier_type) or \
       (S_orig < H * 0.5 and "up" in barrier_type):
        use_fast_approx = True
    
    # 使用快速近似方法
    if use_fast_approx:
        return fast_barrier_approximation()
    
    # 根据不同情况动态调整蒙特卡洛参数，平衡速度和精度
    num_sims = 5000  # 基准模拟路径数
    num_steps = 50   # 基准时间步数
    
    # 价格靠近障碍时增加精度
    if barrier_distance_ratio < 0.2:
        num_sims = 10000
        num_steps = 100
    
    # 特别接近障碍时，需要更高精度
    if barrier_distance_ratio < 0.1:
        num_sims = 20000
        num_steps = 100
    
    # 利率较高或波动率较大时，增加步数以减小离散误差
    if r > 0.05 or sigma > 0.3:
        num_steps = max(num_steps, 100)
    
    # 远期期权需要更多步数
    if T > 1:
        num_steps = max(num_steps, 100)
    
    # 计算标准价格
    price = _calculate_mc_price_raw(S_orig, K, T, r, sigma, H, option_type, barrier_type, num_sims, num_steps)
    
    return price

# European Option Tab
with option_type_tab[0]:
    # Interface design - Parameters on the left, charts on the right
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<p class="sub-header">Option Parameters</p>', unsafe_allow_html=True)
        
        option_type = st.selectbox("Option Type", ["Call Option", "Put Option"], key="euro_option_type")
        option_type_value = "call" if option_type == "Call Option" else "put"
        
        S0 = st.number_input("Current Asset Price (S)", min_value=1.0, max_value=1000.0, value=100.0, step=1.0, key="euro_S0")
        K = st.number_input("Strike Price (K)", min_value=1.0, max_value=1000.0, value=100.0, step=1.0, key="euro_K")
        
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            T_max = st.number_input("Max Time to Expiration (years)", min_value=0.1, max_value=5.0, value=1.0, step=0.1, key="euro_T_max")
        with col1_2:
            T = st.slider("Time to Expiration (years)", min_value=0.01, max_value=T_max, value=T_max, step=0.01, key="euro_T")
        
        r = st.slider("Risk-free Rate (%)", min_value=0.0, max_value=10.0, value=2.5, step=0.1, key="euro_r") / 100
        sigma = st.slider("Volatility (%)", min_value=1.0, max_value=100.0, value=20.0, step=1.0, key="euro_sigma") / 100
        
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
            greek_choice = st.selectbox("Select Greek", ["Delta", "Gamma", "Theta", "Vega", "Rho"], key="euro_greek")
            
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

# Barrier Option Tab
with option_type_tab[1]:
    # Interface design - Parameters on the left, charts on the right
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<p class="sub-header">Barrier Option Parameters</p>', unsafe_allow_html=True)
        
        barrier_option_type = st.selectbox("Option Type", ["Call Option", "Put Option"], key="barrier_option_type")
        barrier_option_type_value = "call" if barrier_option_type == "Call Option" else "put"
        
        barrier_type = st.selectbox("Barrier Type", ["Down-and-Out", "Down-and-In", "Up-and-Out", "Up-and-In"], key="barrier_type")
        barrier_type_value = barrier_type.lower().replace("-", "-and-")
        
        S0_barrier = st.number_input("Current Asset Price (S)", min_value=1.0, max_value=1000.0, value=100.0, step=1.0, key="barrier_S0")
        K_barrier = st.number_input("Strike Price (K)", min_value=1.0, max_value=1000.0, value=100.0, step=1.0, key="barrier_K")
        
        # Set appropriate default and range for barrier based on barrier type and current price
        if "down" in barrier_type_value:
            default_barrier = S0_barrier * 0.8
            max_barrier = S0_barrier * 0.99
        else:  # up barrier
            default_barrier = S0_barrier * 1.2
            max_barrier = S0_barrier * 2.0
            
        min_barrier = 1.0
        max_barrier = max(min_barrier + 0.1, max_barrier)
        
        H_barrier = st.number_input("Barrier Level (H)", 
                                    min_value=min_barrier, 
                                    max_value=S0_barrier * 2.0,
                                    value=default_barrier, 
                                    step=1.0, 
                                    key="barrier_H")
        
        T_barrier = st.slider("Time to Expiration (years)", min_value=0.01, max_value=5.0, value=1.0, step=0.01, key="barrier_T")
        r_barrier = st.slider("Risk-free Rate (%)", min_value=0.0, max_value=10.0, value=2.5, step=0.1, key="barrier_r") / 100
        sigma_barrier = st.slider("Volatility (%)", min_value=1.0, max_value=100.0, value=20.0, step=1.0, key="barrier_sigma") / 100
        
        # Calculate barrier option price with current parameters
        try:
            barrier_price = barrier_option_price(S0_barrier, K_barrier, T_barrier, r_barrier, sigma_barrier, H_barrier, 
                                                barrier_option_type_value, barrier_type_value)
            
            # Handle the case where the option is already knocked out
            knocked_out = ("down" in barrier_type_value and S0_barrier <= H_barrier and "out" in barrier_type_value) or \
                          ("up" in barrier_type_value and S0_barrier >= H_barrier and "out" in barrier_type_value)
            
            knocked_in = ("down" in barrier_type_value and S0_barrier <= H_barrier and "in" in barrier_type_value) or \
                         ("up" in barrier_type_value and S0_barrier >= H_barrier and "in" in barrier_type_value)
                         
            status_message = ""
            if knocked_out:
                status_message = "⚠️ Barrier crossed - option is knocked out"
            elif knocked_in:
                status_message = "✓ Barrier crossed - option is knocked in"
                
            if status_message:
                st.warning(status_message)
                
        except Exception as e:
            st.error(f"Error calculating barrier option: {e}")
            barrier_price = 0.0
        
        st.markdown('<p class="sub-header">Barrier Option Value</p>', unsafe_allow_html=True)
        
        # Create a display for metrics
        metrics_df = pd.DataFrame({
            'Metric': ['Option Price'],
            'Value': [f"{barrier_price:.4f}"]
        })
        
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)
        
        # Information about barrier options
        with st.expander("About Barrier Options"):
            st.markdown(f"""
            ## Barrier Options
            
            Barrier options are path-dependent options that are activated or extinguished when the underlying asset price reaches a predetermined barrier level.
            
            **Current settings:**
            - Option type: {barrier_option_type}
            - Barrier type: {barrier_type}
            - Barrier level: {H_barrier}
            
            **Types of barrier options:**
            - **Down-and-Out**: Option becomes worthless if the price falls below the barrier
            - **Down-and-In**: Option is activated only if the price falls below the barrier
            - **Up-and-Out**: Option becomes worthless if the price rises above the barrier
            - **Up-and-In**: Option is activated only if the price rises above the barrier
            """)

    with col2:
        st.markdown('<p class="sub-header">Barrier Option Analysis</p>', unsafe_allow_html=True)
        
        barrier_tab1 = st.tabs(["Value Curve"])
        
        # Price range for the asset - more points around barrier
        lower_range = min(H_barrier * 0.7, S0_barrier * 0.5)
        upper_range = max(H_barrier * 1.3, S0_barrier * 1.5)
        
        # Create dense price range
        price_range_barrier = np.linspace(max(1, lower_range), upper_range, 1000)
        
        with barrier_tab1[0]:
            st.markdown('<p class="sub-header">Barrier Option Value Curve</p>', unsafe_allow_html=True)
            
            # Calculate barrier option prices and European option prices for comparison
            barrier_option_prices = []
            euro_option_prices = []
            
            with st.spinner("Calculating option prices..."):
                # Use fewer points for faster calculation
                sample_points = np.linspace(max(1, lower_range), upper_range, 100)
                
                for s in sample_points:
                    try:
                        # Calculate barrier and European option prices
                        b_price = barrier_option_price(s, K_barrier, T_barrier, r_barrier, sigma_barrier, H_barrier, 
                                                      barrier_option_type_value, barrier_type_value,
                                                      N_sims=500, N_steps=30)
                        e_price = black_scholes(s, K_barrier, T_barrier, r_barrier, sigma_barrier, barrier_option_type_value)[0]
                        
                        barrier_option_prices.append(b_price)
                        euro_option_prices.append(e_price)
                    except Exception as e:
                        # Handle calculation errors
                        st.error(f"Error at price {s}: {str(e)}")
                        barrier_option_prices.append(0)
                        euro_option_prices.append(0)
            
            barrier_option_prices = np.array(barrier_option_prices)
            euro_option_prices = np.array(euro_option_prices)
            
            # Payoff at expiration (assuming European equivalent)
            payoff = np.maximum(sample_points - K_barrier, 0) if barrier_option_type_value == 'call' else np.maximum(K_barrier - sample_points, 0)
            
            # Create the chart
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(sample_points, barrier_option_prices, 'b-', label='Barrier Option Value')
            ax.plot(sample_points, euro_option_prices, 'g--', label='European Option Value', alpha=0.5)
            ax.axvline(x=S0_barrier, color='green', linestyle=':', label=f'Current Price ({S0_barrier})')
            ax.axvline(x=K_barrier, color='black', linestyle=':', label=f'Strike Price ({K_barrier})')
            ax.axvline(x=H_barrier, color='red', linestyle='-', label=f'Barrier Level ({H_barrier})')
            
            # Add shaded region based on barrier type
            if "down" in barrier_type_value:
                ax.axvspan(0, H_barrier, alpha=0.1, color='red')
            else:
                ax.axvspan(H_barrier, upper_range * 1.1, alpha=0.1, color='red')
                
            ax.set_xlabel('Asset Price')
            ax.set_ylabel('Option Value')
            ax.set_title(f'{barrier_type} {barrier_option_type} Value Curve')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

# Educational explanation
with st.expander("About Options and Pricing Models"):
    st.markdown("""
    ## Option Types
    
    ### European Options
    A European option is a financial derivative that gives the holder the right, but not the obligation, to buy (call) or sell (put) an underlying asset at a specific price (strike price) on a specific date (expiration date).
    
    ### Barrier Options
    Barrier options are path-dependent options where the payoff depends on whether the underlying asset price reaches a predetermined barrier level during the option's life:
    
    - **Knock-in options** become active only if the underlying price reaches the barrier
    - **Knock-out options** become worthless if the underlying price reaches the barrier
    
    ## Pricing Models
    
    ### Black-Scholes Model
    This demo uses the Black-Scholes model to calculate European option prices based on the following assumptions:
    - No arbitrage opportunities
    - Market efficiency with continuous trading
    - Asset prices follow geometric Brownian motion
    - Constant risk-free rate and volatility
    
    ### Barrier Option Pricing
    Barrier options are priced using extensions of the Black-Scholes model that account for the barrier feature. The calculation considers the probability of the barrier being reached during the option's life.
    
    ## Greeks
    Greeks measure the sensitivity of option prices to various factors:
    
    - **Delta (Δ)**: Sensitivity of option price to changes in the underlying asset price
    - **Gamma (Γ)**: Rate of change of Delta with respect to the underlying asset price
    - **Theta (Θ)**: Sensitivity of option price to the passage of time
    - **Vega (v)**: Sensitivity of option price to changes in volatility
    - **Rho (ρ)**: Sensitivity of option price to changes in the risk-free rate
    """)

st.caption("Note: This application is for educational purposes only and does not constitute investment advice.")