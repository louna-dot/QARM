import streamlit as st

# Note: st.set_page_config is handled in app.py

st.title(" Multi-Asset Risk Management Mandate")
st.caption("Institutional Investment Solutions | Portfolio Construction & Risk Budgeting")

st.divider()

# --- 1. Mandate Overview ---
st.header(" 🔹 Mandate Overview")
st.markdown(
    """
    **The Client:** Multi-Asset Pension Fund.
    
    **The Objective:** Construct a robust, diversified portfolio optimized for long-term capital preservation and stable compounding.
    
    **The Requirement:** The board demands a move away from traditional capital allocation (which ignores volatility nuances) toward a **Risk Budgeting** framework. The goal is to ensure no single asset class disproportionately drives the portfolio's volatility.
    """
)

# --- 2. The Structural Problem ---
st.info(
    """
    **⚠️ The Structural Inefficiency:**  
    In a traditional "60/40" portfolio (60% Equity, 40% Bonds), Equities often account for **over 90% of the total risk**. 
    
    This creates a "fair-weather" portfolio that appears diversified in capital terms but is actually highly concentrated in equity beta. This dashboard provides the tooling to correct this imbalance.
    """
)

# --- 3. Proposed Solutions ---
st.header(" 🔹 Implementation Strategy")
st.markdown(
    """
To fulfill this mandate, we deploy several complementary allocation frameworks:

### **Equal Risk Contribution (ERC)**
- **Strategy:** Allocates capital such that each asset contributes the same amount of risk.
- **Benefit:** Improves resilience during equity drawdowns by forcing diversification.

### **Most Diversified Portfolio (MDP)**
- **Strategy:** Maximises the diversification ratio by favouring low-correlation assets.
- **Benefit:** Extracts the maximum theoretical diversification “free lunch.”

### **Minimum Expected Shortfall (ES)**
- **Strategy:** Minimises Expected Shortfall (tail loss beyond VaR).
- **Benefit:** Provides downside protection during crisis periods.

### **Target Volatility (MVO)**
- **Strategy:** Adjusts weights so that annualised volatility matches a target.
- **Benefit:** Maintains stable risk exposure and adheres to volatility limits.

### **Equal Weight (Benchmark)**
- **Strategy:** Allocates the same weight to each asset.
- **Benefit:** Simple, robust, transparent reference allocation.
"""
)


st.caption("""
These five models cover the full spectrum of allocation philosophies: risk-balanced, 
diversification-based, risk-targeted, downside-protected, and neutral benchmarks.
""")

# --- 4. Dashboard Workflow ---
st.header("🔹 Operational Workflow")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Universe Definition")
    st.markdown(
        """
        Use the **Sidebar Settings** to define the investment universe.
        *   **Dynamic Tickers:** Input global asset classes (Equities, Sovereigns, Credit, Commodities).
        *   **Regime Selection:** Adjust the start date to stress-test allocations across different historical inflation and growth regimes.
        *   **Constraints:** Apply specific weight caps to adhere to investment policy statements (IPS).
        """
    )

with col2:
    st.subheader("2. Optimization & Analysis")
    st.markdown(
        """
        Navigate to the **"Portfolio Optimizer"** module.
        *   **Compare Models:** Instantly toggle between MVO, ERC, and MDP to visualize the allocation shift.
        *   **Stress Testing:** Use the Crisis Simulator to audit performance during specific market shocks (e.g., 2020 Liquidity Crisis, 2022 Inflation Shock).
        """
    )

st.divider()
st.markdown(
    """
    <div style="text-align: center;">
        <i>Proceed to the <b>Portfolio Optimizer</b> tab to begin the allocation process.</i>
    </div>
    """, 
    unsafe_allow_html=True

)






