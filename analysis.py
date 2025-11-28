import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yfinance as yf
from datetime import datetime
import altair as alt

# ==========================================
# 1. DATA & MATH FUNCTIONS
# ==========================================

@st.cache_data
def get_asset_info(tickers):
    """
    Classify assets into broad categories (Equity, Bond, Commodities, etc.)
    using a mix of a hard-coded ETF map and yfinance metadata.
    """
    known_etfs = {
        'SPY': 'Equities - US Large Cap', 
        'QQQ': 'Equities - US Tech', 
        'DIA': 'Equities - US Value',
        'IWM': 'Equities - US Small Cap', 
        'EEM': 'Equities - Emerging Markets', 
        'VTI': 'Equities - Total Market',
        'VGK': 'Equities - Europe', 
        'EWJ': 'Equities - Japan',

        'TLT': 'Bonds - US Long Term', 
        'IEF': 'Bonds - US Interm.', 
        'SHY': 'Bonds - US Short',
        'LQD': 'Bonds - Corporate', 
        'HYG': 'Bonds - High Yield', 
        'EMB': 'Bonds - Emerging Markets',
        'BND': 'Bonds - Total Market', 
        'AGG': 'Bonds - Aggregate',

        'GLD': 'Commodities - Gold', 
        'SLV': 'Commodities - Silver', 
        'USO': 'Commodities - Oil', 
        'DBC': 'Commodities - Broad',

        'VNQ': 'Real Estate (REITs)', 
        'BTC-USD': 'Crypto', 
        'ETH-USD': 'Crypto'
    }
    
    classification = {}
    for t in tickers:
        # 1) Known ETF / asset mapping
        if t in known_etfs:
            classification[t] = known_etfs[t]
            continue

        # 2) Try to infer from yfinance metadata
        try:
            info = yf.Ticker(t).info
            q_type = info.get('quoteType', 'UNKNOWN')

            if q_type == 'EQUITY':
                sector = info.get('sector', 'Equities - General')
                classification[t] = f"Stock - {sector}"

            elif q_type == 'CRYPTOCURRENCY':
                classification[t] = "Crypto"

            elif q_type == 'FUTURE':
                classification[t] = "Futures/Commodities"

            else:
                classification[t] = "ETF / Other"

        except Exception:
            classification[t] = "Unclassified"

    return classification

@st.cache_data
def get_asset_currency(tickers):
    """
    Return a dict {ticker: trading currency code} using yfinance metadata.
    """
    currencies = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            currencies[t] = info.get("currency", "Unknown")
        except Exception:
            currencies[t] = "Unknown"
    return currencies

@st.cache_data
def load_data(tickers, start_date, rebalance_freq="M"):
    """
    Download price data, align start date to data availability, and compute
    periodic returns & annualised mean/cov based on the rebalancing frequency.

    Parameters
    ----------
    tickers : list[str]
    start_date : date or str
        User-requested start date (investment horizon).
    rebalance_freq : str
        Pandas resample code for rebalancing frequency:
        'M' = Monthly, 'Q' = Quarterly, 'Y' = Yearly.

    Returns
    -------
    returns_df : pd.DataFrame or None
        Periodic returns (according to rebalance_freq).
    mu : pd.Series or None
        Annualised expected returns.
    Sigma : pd.DataFrame or None
        Annualised covariance matrix.
    first_valid_str : str or None
        Actual start date used (string format '%Y-%m-%d').
    warning_msg : str or None
        Warning if actual data start is significantly later than user start.
    """
    end_date = datetime.now().strftime('%Y-%m-%d')

    try:
        # Raw close prices
        data = yf.download(tickers, start=start_date, end=end_date)['Close']
        if data is None or data.empty:
            return None, None, None, None, None

        # Drop columns that are entirely NaN
        data = data.dropna(axis=1, how='all')
        if data.empty:
            return None, None, None, None, None

        # Ensure DataFrame even if single ticker
        if isinstance(data, pd.Series):
            data = data.to_frame()

        # Align user start date with actual data availability:
        # we take the latest first_valid_index across all tickers
        first_valid = data.apply(lambda col: col.first_valid_index()).max()

        user_start = pd.to_datetime(start_date).tz_localize(None)
        actual_start = pd.to_datetime(first_valid).tz_localize(None)

        warning_msg = None
        # Only warn if the gap is > 5 days
        if (actual_start - user_start).days > 5:
            warning_msg = (
                "⚠️ Data availability limited. "
                f"Start date adjusted from {user_start.date()} "
                f"to {actual_start.date()}."
            )

        # Align dataset to the actual start date and fill remaining small gaps
        data = data.loc[first_valid:].ffill().bfill()

        # Resample prices to the chosen rebalancing frequency
        returns_df = data.resample(rebalance_freq).last().pct_change().dropna()
        if returns_df.empty:
            return None, None, None, None, None

        # Annualisation factor consistent with rebalancing frequency
        if rebalance_freq == "M":
            ann_factor = 12      # monthly
        elif rebalance_freq == "Q":
            ann_factor = 4       # quarterly
        elif rebalance_freq == "Y":
            ann_factor = 1       # yearly
        else:
            ann_factor = 12      # default fallback

        mu = returns_df.mean() * ann_factor
        Sigma = returns_df.cov() * ann_factor

        return returns_df, mu, Sigma, first_valid.strftime('%Y-%m-%d'), warning_msg

    except Exception:
        # In case of any download / processing issue,
        # return the same structure filled with None.
        return None, None, None, None, None


def calculate_portfolio_vol(weights, cov_matrix):
    """
    Annualised portfolio volatility given weights and covariance matrix.
    """
    return float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))


def calculate_portfolio_es(weights, returns_df, alpha=0.95):
    """
    Expected Shortfall (ES) of a portfolio, at confidence level alpha,
    using historical returns.

    ES is the average loss conditional on being beyond the VaR threshold.
    """
    port_returns = returns_df.dot(weights)
    var_level = (1 - alpha) * 100  # e.g. alpha=0.95 → 5th percentile
    var_threshold = np.percentile(port_returns, var_level)
    tail_losses = port_returns[port_returns < var_threshold]

    if len(tail_losses) == 0:
        return 0.0

    # Negative sign: ES is a positive number representing a loss
    return float(-tail_losses.mean())


def calculate_risk_contributions(weights, cov_matrix):
    """
    Risk contributions of each asset to total volatility
    (used for risk budgeting / ERC).

    Returns the normalized contributions that sum to 1.
    """
    port_vol = calculate_portfolio_vol(weights, cov_matrix)
    if port_vol == 0:
        return np.zeros_like(weights)

    # Marginal contribution to risk: Σw / σ_p
    marginal_risk = np.dot(cov_matrix, weights) / port_vol
    # Total contribution of each asset: w_i * (∂σ_p/∂w_i)
    rc = weights * marginal_risk

    # Normalise so that sum(rc_normalised) = 1
    return rc / port_vol


def calculate_diversification_ratio(weights, cov_matrix, asset_vols):
    """
    Diversification Ratio = (w · σ_i) / σ_p

    - numerator: weighted average of individual asset volatilities
    - denominator: portfolio volatility
    """
    port_vol = calculate_portfolio_vol(weights, cov_matrix)
    if port_vol == 0:
        return 0.0

    weighted_avg_vol = np.dot(weights, asset_vols)
    return float(weighted_avg_vol / port_vol)
                

# ==========================================
# 2. OPTIMIZATION LOGIC
# ==========================================
def objective_erc(weights, sigma):
    n = len(weights)
    rc = calculate_risk_contributions(weights, sigma)
    target_rc = 1.0 / n
    return np.sum((rc - target_rc)**2)

def objective_mdp(weights, sigma, asset_vols):
    dr = calculate_diversification_ratio(weights, sigma, asset_vols)
    return -dr 

def objective_target_vol(weights, sigma, target_vol):
    port_vol = calculate_portfolio_vol(weights, sigma)
    return (port_vol - target_vol)**2

def objective_min_es(weights, returns_df, alpha):
    return calculate_portfolio_es(weights, returns_df, alpha)

def run_optimization(strategy, mu, sigma, returns_df, max_weight=1.0, target_vol=0.10, alpha=0.95):
    n_assets = len(mu)
    initial_weights = np.array([1/n_assets] * n_assets)
    asset_vols = np.sqrt(np.diag(sigma))
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0.0, max_weight) for _ in range(n_assets))
    
    if strategy == "Equal Risk Contribution (ERC)":
        res = minimize(objective_erc, initial_weights, args=(sigma,), method='SLSQP', bounds=bounds, constraints=constraints)
    elif strategy == "Most Diversified Portfolio (MDP)":
        res = minimize(objective_mdp, initial_weights, args=(sigma, asset_vols), method='SLSQP', bounds=bounds, constraints=constraints)
    elif strategy == "Target Volatility (MVO)":
        res = minimize(objective_target_vol, initial_weights, args=(sigma, target_vol), method='SLSQP', bounds=bounds, constraints=constraints)
    elif strategy == "Minimum Expected Shortfall (ES)":
        res = minimize(objective_min_es, initial_weights, args=(returns_df, alpha), method='SLSQP', bounds=bounds, constraints=constraints)
    elif strategy == "Equal Weight (Benchmark)":
        return initial_weights
    return res.x

@st.cache_data
def generate_efficient_frontier(mu, sigma, n_portfolios=2000):
    n_assets = len(mu)
    results = np.zeros((3, n_portfolios)) 
    for i in range(n_portfolios):
        w = np.random.random(n_assets)
        w /= np.sum(w)
        p_ret = np.dot(w, mu)
        p_vol = np.sqrt(np.dot(w.T, np.dot(sigma, w)))
        results[0,i] = p_vol
        results[1,i] = p_ret
        results[2,i] = p_ret / p_vol if p_vol > 0 else 0
    return pd.DataFrame({'Volatility': results[0], 'Return': results[1], 'Sharpe': results[2]})


# ==========================================
# 3. SIDEBAR CONTROLS – INVESTOR VIEW
# ==========================================

st.sidebar.header("1. Investor Profile")

# Montant investi
investment_amount = st.sidebar.number_input(
    "Investment amount (base currency)",
    min_value=0.0,
    value=1_000_000.0,
    step=50_000.0,
    format="%.0f"
)

# Profil de risque
risk_profile = st.sidebar.radio(
    "Risk level",
    options=["Conservative", "Balanced", "Aggressive"],
    index=1
)

# Horizon de placement
time_horizon_years = st.sidebar.slider(
    "Investment horizon (years)",
    min_value=1,
    max_value=40,
    value=15
)

# Mapping profil de risque → stratégie par défaut
strategy_options = [
    "Equal Risk Contribution (ERC)",
    "Most Diversified Portfolio (MDP)",
    "Minimum Expected Shortfall (ES)",
    "Target Volatility (MVO)",
    "Equal Weight (Benchmark)"
]
default_strategy_index = {
    "Conservative": 0,   # ERC
    "Balanced": 1,       # MDP
    "Aggressive": 3      # Target Vol
}[risk_profile]

st.sidebar.header("2. Strategy Selection")
strategy_choice = st.sidebar.selectbox(
    "Allocation model:",
    strategy_options,
    index=default_strategy_index
)

with st.sidebar.expander("🛠️ Portfolio & Data Settings", expanded=True):

    # Classes d’actifs haut niveau pour filtrer l’univers (optionnel)
    asset_class_choices = ["Equities", "Bonds", "Commodities", "Real Estate", "Crypto"]
    selected_asset_classes = st.multiselect(
        "Asset classes to include",
        options=asset_class_choices,
        default=asset_class_choices[:-1]  # par défaut sans Crypto
    )

    # Universe de tickers (toujours modifiable manuellement)
    default_tickers = "SPY, TLT, GLD, VNQ, QQQ, EEM, EMB"
    ticker_input = st.text_input("Assets (comma separated)", value=default_tickers)
    tickers = [x.strip().upper() for x in ticker_input.split(',') if x.strip()]

    # Fréquence de rebalancement
    rebalance_label = st.selectbox(
        "Rebalancing frequency",
        options=["Monthly", "Quarterly", "Yearly"],
        index=0
    )
    rebalance_map = {"Monthly": "M", "Quarterly": "Q", "Yearly": "Y"}
    rebalance_freq = rebalance_map[rebalance_label]

    # Horizon → période de backtest
    end_date = datetime.now().date()
    start_date = (pd.to_datetime(end_date) - pd.DateOffset(years=time_horizon_years)).date()
    st.caption(f"Backtest period: {start_date} → {end_date}")

    # Max weight par titre (plus strict si profil prudent)
    default_max_weight = {
        "Conservative": 0.25,
        "Balanced": 0.35,
        "Aggressive": 0.50
    }[risk_profile]
    max_weight = st.slider(
        "Max weight per asset",
        0.05, 1.0, float(default_max_weight), 0.05
    )

    # Paramètres spécifiques aux stratégies
    target_vol_input = 0.10
    alpha_input = 0.95

    if strategy_choice == "Target Volatility (MVO)":
        # cible de volatilité selon profil de risque
        default_tv = {
            "Conservative": 0.08,
            "Balanced": 0.12,
            "Aggressive": 0.18
        }[risk_profile]
        target_vol_input = st.slider(
            "Target volatility (annualised)",
            0.05, 0.30, float(default_tv), 0.01
        )

    elif strategy_choice == "Minimum Expected Shortfall (ES)":
        alpha_input = st.slider(
            "Confidence level (α)",
            0.90, 0.99, 0.95
        )


# ==========================================
# 4. MAIN EXECUTION
# ==========================================

st.title(f"Dashboard: {strategy_choice}")

if len(tickers) < 2:
    st.error("Please enter at least 2 assets to build a portfolio.")
    st.stop()

with st.spinner("Fetching data and optimizing..."):

    # 1. Filtrer par classes d’actifs
    asset_info_dict = get_asset_info(tickers)
    if selected_asset_classes:
        filtered_tickers = [
            t for t in tickers
            if any(cls in asset_info_dict.get(t, "") for cls in selected_asset_classes)
        ]
    else:
        filtered_tickers = tickers

    if len(filtered_tickers) < 2:
        st.error("Please select at least 2 assets in the chosen asset classes.")
        st.stop()

    # 2. Charger les données
    returns_df, mu, Sigma, _, date_warning = load_data(
        filtered_tickers,
        start_date,
        rebalance_freq=rebalance_freq
    )

    if returns_df is None:
        st.error("Could not download data. Please check tickers.")
        st.stop()

    if date_warning:
        st.warning(date_warning)

    # 3. Optimisation
    opt_weights = run_optimization(
        strategy_choice,
        mu,
        Sigma,
        returns_df,
        max_weight=max_weight,
        target_vol=target_vol_input,
        alpha=alpha_input,
    )

    # Currency information for each asset
    currency_dict = get_asset_currency(filtered_tickers)

    # Simple region mapping based on the high-level category
    regions = []
    for t in filtered_tickers:
        cat = asset_info_dict.get(t, "Unclassified")
        if "Emerging" in cat:
            regions.append("Emerging Markets")
        elif "Europe" in cat:
            regions.append("Europe")
        elif "Japan" in cat:
            regions.append("Japan")
        elif "US" in cat:
            regions.append("United States")
        else:
            regions.append("Global / Other")

    alloc_df = pd.DataFrame({
        "Asset": filtered_tickers,
        "Weight": opt_weights,
        "Category": [asset_info_dict.get(t, "Unclassified") for t in filtered_tickers],
        "Currency": [currency_dict.get(t, "Unknown") for t in filtered_tickers],
        "Region": regions,
    })

    alloc_df = alloc_df.sort_values("Weight", ascending=False)


    # 4. Calc Stats
    final_vol = calculate_portfolio_vol(opt_weights, Sigma)
    final_es_metric = calculate_portfolio_es(opt_weights, returns_df, alpha_input)
    final_rc = calculate_risk_contributions(opt_weights, Sigma)
    asset_vols = np.sqrt(np.diag(Sigma))
    div_ratio = calculate_diversification_ratio(opt_weights, Sigma, asset_vols)
    exp_ret = np.dot(opt_weights, mu)

# ============================================================
# 5. TOP BAR – PORTFOLIO OVERVIEW
# ============================================================

st.markdown("### Portfolio overview")

top_c1, top_c2, top_c3, top_c4 = st.columns(4)
top_c1.metric("Expected Return (p.a.)", f"{exp_ret*100:.1f}%")
top_c2.metric("Volatility (p.a.)", f"{final_vol*100:.1f}%")
top_c3.metric("Expected Shortfall (α)", f"{final_es_metric*100:.1f}%")
top_c4.metric("Diversification Ratio", f"{div_ratio:.2f}")

st.caption("These metrics provide a high-level snapshot of the portfolio. Use the tabs above to explore allocation, risk decomposition, scenario analysis, and implementation details.")

st.divider()

# ============================================================
# 6. TABS – INSTITUTIONAL STRUCTURE
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Investor Overview",
    "Portfolio Construction",
    "Risk Decomposition",
    "Scenario Analysis",
    "Implementation & Rebalancing"
])

# ============================================================
#  TAB 1 — INVESTOR OVERVIEW
# ============================================================
with tab1:
    st.header("Investor Overview")

    st.write("""
    This section summarises the **mandate** and key characteristics of the investor.
    It is designed for investment committees and governance bodies, rather than quants.
    """)

    c1, c2, c3 = st.columns(3)
    c1.metric("Investment Amount", f"{investment_amount:,.0f}")
    c2.metric("Risk Profile", risk_profile)
    c3.metric("Investment Horizon", f"{time_horizon_years} years")

    st.subheader("Mandate summary")
    st.markdown(f"""
    • **Objective:** Long-term capital preservation and growth  
    • **Risk tolerance:** {risk_profile}  
    • **Investment horizon:** {time_horizon_years} years  
    • **Rebalancing frequency:** {rebalance_label}  
    • **Allocation model:** *{strategy_choice}*  
    """)

# ============================================================
#  TAB 2 — PORTFOLIO CONSTRUCTION
# ============================================================
with tab2:
    st.header("Portfolio Construction")

    st.write("""
    This tab focuses on how the portfolio is constructed across asset classes 
    and instruments, based on the selected allocation model.
    """)

    # --- 1) Strategic Allocation (by Asset Class) ---
    st.subheader("Strategic Allocation (by Asset Class)")
    st.caption("Strategic weights aggregated at the asset-class level.")

    alloc_by_class = (
        alloc_df
        .groupby("Category", as_index=False)
        .agg({"Weight": "sum"})
        .assign(Weight_pct=lambda df: (df["Weight"] * 100).round(2))
        .rename(columns={"Weight_pct": "Weight (%)"})
        .sort_values("Weight", ascending=False)
    )
    st.dataframe(
        alloc_by_class[["Category", "Weight (%)"]],
        use_container_width=True
    )

    # --- 2) Pie chart by asset class ---
    st.subheader("Allocation by Asset Class (Pie Chart)")

    alloc_df_filtered = alloc_df[alloc_df["Weight"] > 0.001]

    pie = alt.Chart(alloc_df_filtered).mark_arc(innerRadius=60).encode(
        theta=alt.Theta(field="Weight", type="quantitative"),
        color=alt.Color(field="Category", type="nominal"),
        tooltip=["Category", "Asset", alt.Tooltip("Weight", format=".1%")]
    ).properties(title="Portfolio Exposure").interactive()

    st.altair_chart(pie, use_container_width=True)

    
    # --- 3) Geographic & asset-class mix ---
    st.subheader("Geographic exposure by asset class")

    st.caption("""
    This view shows how the portfolio's risk budget is distributed **by region** 
    and **by major asset class** (Equities, Bonds, Real Assets, etc.).
    """)

    # Helper: map detailed Category to a broad asset class label
    def classify_asset_class(cat: str) -> str:
        c = (cat or "").lower()
        if "equities" in c or "equity" in c or "stock" in c:
            return "Equities"
        if "bonds" in c or "fixed income" in c:
            return "Bonds"
        if "real estate" in c or "reit" in c:
            return "Real Assets"
        if "commodities" in c or "gold" in c or "silver" in c or "oil" in c:
            return "Commodities"
        if "crypto" in c or "bitcoin" in c or "ethereum" in c:
            return "Crypto"
        return "Other"

    # Add a broad asset-class label
    geo_ac_df = alloc_df.assign(
        AssetClass=lambda df: df["Category"].apply(classify_asset_class)
    )

    # Aggregate weights by Region × AssetClass
    region_ac = (
        geo_ac_df
        .groupby(["Region", "AssetClass"], as_index=False)
        .agg({"Weight": "sum"})
    )
    region_ac["Weight_pct"] = region_ac["Weight"] * 100

    # --- 3a) Matrix table (Region x Asset Class) ---
    matrix = (
        region_ac
        .pivot(index="Region", columns="AssetClass", values="Weight_pct")
        .fillna(0)
        .round(1)
    )
    matrix["Total (%)"] = matrix.sum(axis=1).round(1)
    matrix = matrix.reset_index()

    st.dataframe(matrix, use_container_width=True)

    st.caption("""
    Rows sum to the **total regional weight** in the portfolio. Columns show how 
    much of that regional exposure comes from Equities, Bonds, Real Assets, etc.
    """)

    # --- 3b) Stacked bar chart by Region & Asset Class ---
    stacked_chart = (
        alt.Chart(region_ac)
        .mark_bar()
        .encode(
            x=alt.X(
                "sum(Weight):Q",
                stack="normalize",
                axis=alt.Axis(format="%", title="Share of portfolio")
            ),
            y=alt.Y("Region:N", sort="-x"),
            color=alt.Color(
                "AssetClass:N",
                legend=alt.Legend(title="Asset class")
            ),
            tooltip=[
                alt.Tooltip("Region:N"),
                alt.Tooltip("AssetClass:N", title="Asset class"),
                alt.Tooltip("Weight_pct:Q", format=".1f", title="Weight (%)"),
            ],
        )
        .properties(height=260)
    )

    st.altair_chart(stacked_chart, use_container_width=True)

    st.caption("""
    Each bar represents a **region**; colours show how that region is split across 
    Equities, Bonds and other asset classes. This highlights, for example, 
    whether Emerging Markets exposure is primarily equity-driven or bond-driven.
    """)  

    # --- 4) Efficient Frontier view ---
    st.subheader("Efficient frontier positioning")

    st.caption("""
    The scatter plot below shows simulated portfolios in risk/return space, with 
    the optimised strategy highlighted. This is illustrative and based on the 
    estimated mean and covariance of the selected assets.
    """)

    sim_df = generate_efficient_frontier(mu, Sigma)
    sim_df["Type"] = "Other portfolios"

    opt_point = pd.DataFrame({
        "Volatility": [float(final_vol)],
        "Return": [float(exp_ret)],
        "Sharpe": [float(exp_ret / final_vol) if final_vol > 0 else 0.0],
        "Type": ["Optimised portfolio"],
    })

    combined_df = pd.concat([sim_df, opt_point], ignore_index=True)

    frontier_chart = (
        alt.Chart(combined_df)
        .mark_circle()
        .encode(
            x=alt.X("Volatility:Q", axis=alt.Axis(format="%", title="Annualised volatility")),
            y=alt.Y("Return:Q", axis=alt.Axis(format="%", title="Annualised return")),
            color=alt.Color("Type:N", legend=alt.Legend(title="Portfolio type")),
            size=alt.Size(
                "Type:N",
                legend=None,
                scale=alt.Scale(range=[40, 200])
            ),
            tooltip=[
                alt.Tooltip("Type:N"),
                alt.Tooltip("Volatility:Q", format=".1%", title="Volatility"),
                alt.Tooltip("Return:Q", format=".1%", title="Return"),
                alt.Tooltip("Sharpe:Q", format=".2f", title="Sharpe ratio"),
            ],
        )
        .properties(height=380)
        .interactive()
    )

    st.altair_chart(frontier_chart, use_container_width=True)

# ============================================================
#  TAB 3 — RISK DECOMPOSITION
# ============================================================
with tab3:
    st.header("Risk Decomposition")

    st.write("""
    This tab highlights how risk is distributed across assets and compares 
    capital allocation with risk contribution.
    """)

    # ---- 1) Risk contributions by asset (table only) ----
    rc_df = pd.DataFrame({"Asset": filtered_tickers, "Risk Contribution": final_rc})

    st.subheader("Risk Contribution by Asset")
    st.dataframe(rc_df, use_container_width=True)

    st.caption("""
    A well-balanced risk profile avoids concentration in a small number of assets 
    or asset classes.
    """)

    # ---- 2) Capital vs Risk: 2 bars per asset (small multiples) ----
    st.subheader("Capital vs Risk")

    alloc_df_full = pd.DataFrame({
        "Asset": filtered_tickers,
        "Weight": opt_weights,
        "Risk Contribution": final_rc
    })
    alloc_df_full = alloc_df_full[alloc_df_full["Weight"] > 0.001]

    # ordre des actifs (par exemple décroissant en Weight)
    alloc_df_full = alloc_df_full.sort_values("Weight", ascending=False)
    asset_order = list(alloc_df_full["Asset"])

    melted_df = alloc_df_full.melt(
        id_vars="Asset",
        value_vars=["Weight", "Risk Contribution"],
        var_name="Metric",
        value_name="Value"
    )

    base_chart = (
        alt.Chart(melted_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "Value:Q",
                axis=alt.Axis(format="%", title="Percentage of Portfolio"),
                scale=alt.Scale(domain=[0, float(melted_df["Value"].max()) * 1.1])
            ),
            y=alt.Y(
                "Metric:N",
                sort=["Risk Contribution", "Weight"],
                axis=None
            ),
            color=alt.Color(
                "Metric:N",
                scale=alt.Scale(
                    domain=["Weight", "Risk Contribution"],
                    range=["#87CEFA", "#1F77B4"]
                ),
                legend=alt.Legend(title="Metric")
            ),
            tooltip=[
                alt.Tooltip("Asset"),
                alt.Tooltip("Metric"),
                alt.Tooltip("Value", format=".1%")
            ]
        )
        .properties(height=40)
    )

    # une ligne (facette) par asset : deux barres l’une au-dessus de l’autre
    per_asset = base_chart.facet(
        row=alt.Row("Asset:N", sort=asset_order, title=None),
        spacing=8
    ).resolve_scale(x="shared")

    st.altair_chart(per_asset, use_container_width=True)

    st.caption("""
    For each asset, the top bar shows its Risk Contribution and the bottom bar 
    shows its Weight (capital allocation). This makes it easy to compare risk 
    vs capital asset by asset.
    """)



# ============================================================
#  TAB 4 — SCENARIO ANALYSIS
# ============================================================
with tab4:
    st.header("Scenario Analysis")

    st.write("""
    This tab illustrates how the portfolio may evolve under simple forward-looking 
    scenarios, and reviews the historical performance and risk profile.
    """)

    # ----------------------------------------
    # A. LONG-TERM RETURN SCENARIOS
    # ----------------------------------------
    st.subheader("A. Long-term return scenarios (μ, μ±σ)")

    exp_r = float(exp_ret)
    vol_r = float(final_vol)
    years = time_horizon_years

    central_growth = (1 + exp_r) ** years
    down_growth = max(1 + exp_r - vol_r, 0) ** years
    up_growth = (1 + exp_r + vol_r) ** years

    scenario_df = pd.DataFrame({
        "Scenario": ["Downside (μ − σ)", "Central (μ)", "Upside (μ + σ)"],
        "Annualised Return (%)": [
            (exp_r - vol_r) * 100,
            exp_r * 100,
            (exp_r + vol_r) * 100
        ],
        "Projected Value": [
            investment_amount * down_growth,
            investment_amount * central_growth,
            investment_amount * up_growth
        ]
    }).assign(
        **{"Projected Value": lambda df: df["Projected Value"].round(0)}
    )

    st.dataframe(scenario_df, use_container_width=True)
    st.caption("Scenarios are illustrative and not forecasts, but provide a range for potential long-term outcomes.")

    st.divider()

    # ----------------------------------------
    # B. HISTORICAL PERFORMANCE
    # ----------------------------------------
    st.subheader("B. Historical performance vs equal-weight benchmark")

    n_assets = len(filtered_tickers)
    eq_w = np.array([1 / n_assets] * n_assets)

    cum_opt = (1 + returns_df.dot(opt_weights)).cumprod()
    cum_eq = (1 + returns_df.dot(eq_w)).cumprod()

    hist_data = pd.DataFrame({
        "Date": returns_df.index,
        "Optimised strategy": cum_opt,
        "Equal-weight benchmark": cum_eq
    })

    hist_melted = hist_data.melt("Date", var_name="Strategy", value_name="Cumulative Return")

    perf_chart = (
        alt.Chart(hist_melted)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("Date:T", axis=alt.Axis(title="Date")),
            y=alt.Y("Cumulative Return:Q", axis=alt.Axis(title="Growth of 1")),
            color=alt.Color("Strategy:N"),
            tooltip=[
                alt.Tooltip("Date:T", format="%b %Y"),
                "Strategy:N",
                alt.Tooltip("Cumulative Return:Q", format=".2f")
            ],
        )
        .properties(height=350)
        .interactive()
    )

    st.altair_chart(perf_chart, use_container_width=True)

    st.divider()

    # ----------------------------------------
    # C. DYNAMIC RISK ANALYSIS (ROLLING VOL)
    # ----------------------------------------
    st.subheader("C. Dynamic Risk Analysis (Rolling Volatility)")

    st.write("""
Rolling volatility provides insight into how portfolio risk evolved over time.
A smoother profile indicates stable risk, whereas spikes reflect periods of stress.
    """)

    window = st.slider(
        "Rolling window (months)",
        min_value=3,
        max_value=36,
        value=12,
        step=1,
        key="scenario_rolling_win",
        help="Window length used to compute rolling volatility."
    )

    rolling_port_vol = (
        returns_df.dot(opt_weights)
        .rolling(window=window)
        .std()
        * np.sqrt(12)
    )

    asset_rolling_vol = returns_df.rolling(window=window).std() * np.sqrt(12)
    max_vol_asset = asset_rolling_vol.mean().idxmax()
    rolling_max_vol = asset_rolling_vol[max_vol_asset]

    vol_plot_df = pd.DataFrame({
        "Date": returns_df.index,
        "Portfolio rolling vol": rolling_port_vol,
        f"{max_vol_asset} rolling vol": rolling_max_vol,
    })

    vol_melted = vol_plot_df.melt("Date", var_name="Series", value_name="Volatility")

    unique_years = sorted(returns_df.index.year.unique())
    tick_values = [pd.Timestamp(f"{y}-01-01") for y in unique_years]

    dynamic_risk_chart = (
        alt.Chart(vol_melted)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X(
                "Date:T",
                axis=alt.Axis(
                    title="Year",
                    format="%Y",
                    values=tick_values,
                    grid=True
                )
            ),
            y=alt.Y(
                "Volatility:Q",
                axis=alt.Axis(format="%", title="Annualised volatility"),
            ),
            color=alt.Color(
                "Series:N",
                legend=alt.Legend(title="Series"),
                scale=alt.Scale(scheme="tableau10")
            ),
            tooltip=[
                alt.Tooltip("Date:T", format="%b %Y"),
                "Series:N",
                alt.Tooltip("Volatility:Q", format=".2%"),
            ],
        )
        .properties(height=350)
        .interactive()
    )

    st.altair_chart(dynamic_risk_chart, use_container_width=True)
    st.caption("The portfolio's rolling volatility is compared against the riskiest underlying asset over the same period.")

# ============================================================
#  TAB 5 — IMPLEMENTATION & REBALANCING
# ============================================================
with tab5:
    st.header("Implementation & Rebalancing")

    st.write("""
    This tab translates the optimised allocation into **practical rebalancing guidance**.
    The investor can provide a current portfolio via CSV; if no file is uploaded,
    an equal-weight portfolio is assumed as the current implementation.
    """)

    # --------------------------------------------------------
    # 0. Current portfolio input (optional CSV upload)
    # --------------------------------------------------------
    st.subheader("Current portfolio input")

    st.markdown("""
    Upload a CSV file with at least the following columns:
    - **Asset**: ticker (e.g. SPY, TLT, GLD)  
    - **Weight**: current portfolio weight in decimal (e.g. 0.25 for 25%)  

    Weights do not need to sum to 1. They will be normalised over the selected universe.
    """)

    uploaded_portfolio = st.file_uploader(
        "Upload current portfolio (optional)",
        type=["csv"],
        help="If no file is uploaded, an equal-weight portfolio is assumed."
    )

    n_assets_impl = len(filtered_tickers)
    if n_assets_impl == 0:
        st.warning("No assets selected for implementation.")
        st.stop()

    # Example CSV template
    example_df = pd.DataFrame({
        "Asset": filtered_tickers,
        "Weight": [round(1.0 / n_assets_impl, 4)] * n_assets_impl
    })
    example_csv = example_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download example CSV template",
        data=example_csv,
        file_name="example_current_portfolio.csv",
        mime="text/csv",
        help="Use this file as a template: adjust the Weight column to match the current portfolio."
    )

    # --- Build current_w from CSV if available, else equal-weight ---
    current_w_source = "equal-weight (fallback)"

    if uploaded_portfolio is not None:
        try:
            pf_df = pd.read_csv(uploaded_portfolio)
            pf_df.columns = [c.strip().lower() for c in pf_df.columns]

            if "asset" in pf_df.columns and "weight" in pf_df.columns:
                w_map = (
                    pf_df[["asset", "weight"]]
                    .copy()
                    .assign(asset=lambda df: df["asset"].astype(str).str.upper())
                    .groupby("asset", as_index=True)["weight"]
                    .sum()
                )
                w_list = [float(w_map.get(t, 0.0)) for t in filtered_tickers]
                current_w = np.array(w_list, dtype=float)

                if current_w.sum() > 0:
                    current_w = current_w / current_w.sum()
                    current_w_source = "uploaded CSV"
                else:
                    st.warning(
                        "Uploaded portfolio has zero total weight on the selected tickers. "
                        "Falling back to an equal-weight current portfolio."
                    )
                    current_w = np.array([1.0 / n_assets_impl] * n_assets_impl)
            else:
                st.warning(
                    "Uploaded CSV must contain at least 'Asset' and 'Weight' columns. "
                    "Falling back to an equal-weight current portfolio."
                )
                current_w = np.array([1.0 / n_assets_impl] * n_assets_impl)
        except Exception as e:
            st.warning(
                f"Could not read current portfolio from CSV ({e}). "
                "Falling back to an equal-weight current portfolio."
            )
            current_w = np.array([1.0 / n_assets_impl] * n_assets_impl)
    else:
        current_w = np.array([1.0 / n_assets_impl] * n_assets_impl)

    st.caption(f"Current portfolio source: **{current_w_source}**.")

    # --------------------------------------------------------
    # 1. Drift & risk comparison
    # --------------------------------------------------------
    target_w = opt_weights.copy()
    if not np.isclose(target_w.sum(), 1.0):
        target_w = target_w / target_w.sum()

    delta_w = target_w - current_w
    delta_pp = delta_w * 100.0

    abs_delta_pp = np.abs(delta_pp)
    max_drift_idx = int(abs_delta_pp.argmax())
    max_drift_asset = filtered_tickers[max_drift_idx]
    max_drift_pp = float(delta_pp[max_drift_idx])
    avg_drift_pp = float(abs_delta_pp.mean())
    drift_threshold_pp = 3.0
    n_breaches = int((abs_delta_pp > drift_threshold_pp).sum())

    vol_current_impl = calculate_portfolio_vol(current_w, Sigma)
    vol_target_impl = final_vol

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Max drift (pp)", f"{max_drift_pp:+.1f}")
    s2.metric("Average drift (pp)", f"{avg_drift_pp:.1f}")
    s3.metric("# assets above drift threshold", f"{n_breaches} / {n_assets_impl}")
    s4.metric(
        "Volatility: current → target",
        f"{vol_current_impl*100:.1f}% → {vol_target_impl*100:.1f}%"
    )

    if n_breaches > 0:
        st.warning(
            f"Rebalancing **recommended**: {n_breaches} asset(s) exceed the drift "
            f"threshold of {drift_threshold_pp:.1f}pp. "
            f"Largest drift: {max_drift_asset} at {max_drift_pp:+.1f}pp."
        )
    else:
        st.success(
            f"Rebalancing **not strictly required** based on the drift threshold "
            f"({drift_threshold_pp:.1f}pp). Largest drift: {max_drift_asset} "
            f"at {max_drift_pp:+.1f}pp."
        )

    st.divider()

    # --------------------------------------------------------
    # 2. Suggested trades & trade blotter
    # --------------------------------------------------------
    st.subheader("Suggested trades (from current to target allocation)")

    trade_amounts = delta_w * investment_amount

    trades_full = pd.DataFrame({
        "Asset": filtered_tickers,
        "Direction": np.where(trade_amounts > 0, "Buy", "Sell"),
        "Trade amount": trade_amounts.round(0).astype(int),
        "Δ weight (pp)": delta_pp.round(1),
        "Current weight (%)": (current_w * 100).round(1),
        "Target weight (%)": (target_w * 100).round(1),
    })

    trades_full["abs_drift"] = trades_full["Δ weight (pp)"].abs()
    trades_filtered = trades_full[trades_full["abs_drift"] > 0.1]

    if trades_filtered.empty:
        st.info("No meaningful trades: all assets are already very close to target weights.")
    else:
        top5_trades = (
            trades_filtered
            .sort_values("abs_drift", ascending=False)
            .head(5)
            .drop(columns=["abs_drift"])
        )

        st.markdown("**Top 5 trades by absolute drift**")
        st.dataframe(
            top5_trades[
                ["Asset", "Direction", "Trade amount",
                 "Current weight (%)", "Target weight (%)", "Δ weight (pp)"]
            ],
            use_container_width=True,
        )

        st.caption("""
        Positive trade amounts correspond to **buys**, negative amounts to **sells**.
        Changes in weights are expressed in **percentage points** relative to the
        current portfolio (or equal-weight fallback).
        """)

        st.markdown("**Full trade blotter**")
        csv_bytes = trades_filtered.drop(columns=["abs_drift"]).to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download full trade blotter (CSV)",
            data=csv_bytes,
            file_name="trade_blotter_rebalance.csv",
            mime="text/csv",
        )

    st.divider()

    # --------------------------------------------------------
    # 3. Turnover & estimated trading cost
    # --------------------------------------------------------
    st.subheader("Turnover & estimated trading cost")

    gross_turnover = float(np.sum(np.abs(trade_amounts))) / float(investment_amount)
    cost_bps = 20.0
    est_trading_cost = gross_turnover * (cost_bps / 10000.0) * investment_amount

    t1, t2, t3 = st.columns(3)
    t1.metric("Gross turnover", f"{gross_turnover*100:.1f}%")
    t2.metric("Cost assumption", f"{cost_bps:.0f} bps")
    t3.metric("Estimated trading cost", f"{est_trading_cost:,.0f}")

    st.caption("""
    Turnover is defined as the sum of absolute trade notionals divided by total portfolio value.
    The cost estimate is illustrative and based on a flat spread + impact assumption (20 bps total).
    """)

    st.divider()

    # --------------------------------------------------------
    # 4. Rebalancing calendar & policy reminder
    # --------------------------------------------------------
    st.subheader("Rebalancing calendar & policy reminder")

    today_ts = pd.to_datetime(end_date)

    if rebalance_label == "Monthly":
        last_reb = today_ts - pd.offsets.MonthEnd(1)
        next_reb = today_ts + pd.offsets.MonthEnd(1)
    elif rebalance_label == "Quarterly":
        last_reb = today_ts - pd.offsets.QuarterEnd(1)
        next_reb = today_ts + pd.offsets.QuarterEnd(1)
    else:  # Yearly
        last_reb = today_ts - pd.offsets.YearEnd(1)
        next_reb = today_ts + pd.offsets.YearEnd(1)

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Last rebalance (approx.)**")
        st.write(last_reb.date())
        st.markdown("**Next expected rebalance window**")
        st.write(next_reb.date())

    with d2:
        st.markdown("**Policy reminder**")
        st.markdown(f"""
        • **Frequency:** {rebalance_label}  
        • **Drift threshold:** {drift_threshold_pp:.1f} percentage points  
        • **Max weight per asset:** {max_weight:.0%}  
        • **Universe:** {', '.join(filtered_tickers)}  
        """)

    st.caption("""
    Dates are indicative and based on the selected backtest frequency. In practice, 
    the investment committee may also trigger off-cycle rebalancing when market moves 
    cause drifts or risk metrics to breach policy limits.
    """)
