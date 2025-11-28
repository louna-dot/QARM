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
#  TABS – CLIENT-FACING STRUCTURE
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Investor Summary", 
    "Portfolio Construction",
    "Risk Decomposition",
    "Scenario Analysis",
    "Implementation & Rebalancing"
])

# ============================================================
#  TAB 1 — INVESTOR SUMMARY
# ============================================================
with tab1:
    st.header("Investor Summary")

    st.write("""
    This section provides a high-level overview of the investor’s profile and 
    long-term mandate. The objective is to ensure that the portfolio construction 
    aligns with the risk tolerance, investment horizon, and strategic constraints.
    """)

    # Investor profile metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Investment Amount", f"{investment_amount:,.0f}")
    c2.metric("Risk Profile", risk_profile)
    c3.metric("Investment Horizon", f"{time_horizon_years} years")

    st.subheader("Mandate Overview")
    st.markdown(f"""
    • **Objective:** Long-term capital preservation and growth  
    • **Risk Tolerance:** {risk_profile}  
    • **Investment Horizon:** {time_horizon_years} years  
    • **Rebalancing Frequency:** {rebalance_label}  
    • **Selected Allocation Model:** *{strategy_choice}*  
    """)

    st.caption("""
    The selected allocation methodology reflects the investor's long-term objectives 
    and aims to maintain a stable and diversified risk profile.
    """)

    st.subheader("Key Portfolio Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected Return (p.a.)", f"{exp_ret*100:.1f}%")
    c2.metric("Expected Volatility", f"{final_vol*100:.1f}%")
    c3.metric("Expected Shortfall", f"{final_es_metric*100:.1f}%")
    c4.metric("Diversification Ratio", f"{div_ratio:.2f}")


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
    st.subheader("Efficient Frontier (Risk / Return Space)")

    sim_df = generate_efficient_frontier(mu, Sigma)
    sim_df["Type"] = "Other Portfolios"

    opt_point = pd.DataFrame({
        "Volatility": [float(final_vol)],
        "Return": [float(exp_ret)],
        "Sharpe": [float(exp_ret/final_vol) if final_vol > 0 else 0.0],
        "Type": ["Selected Strategy"]
    })

    combined_df = pd.concat([sim_df, opt_point], ignore_index=True)

    frontier_chart = alt.Chart(combined_df).mark_circle().encode(
        x=alt.X("Volatility", axis=alt.Axis(format="%", title="Annualised Volatility")),
        y=alt.Y("Return", axis=alt.Axis(format="%", title="Annualised Return")),
        color=alt.Color("Type", legend=alt.Legend(title="Portfolio Type")),
        size=alt.Size(
            "Type",
            legend=None,
            scale=alt.Scale(range=[40, 200])  # plus gros pour la stratégie choisie
        ),
        tooltip=[
            alt.Tooltip("Type"),
            alt.Tooltip("Volatility", format=".1%"),
            alt.Tooltip("Return", format=".1%"),
            alt.Tooltip("Sharpe", format=".2f")
        ]
    ).properties(height=400).interactive()

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
#  TAB 4 — SCENARIO & HISTORICAL ANALYSIS
# ============================================================
with tab4:
    st.header("Scenario & Historical Analysis")

    st.write("""
    This tab has two complementary parts:

    **A. Forward-looking scenarios** – simple what-if adjustments to the 
    current optimised allocation, showing how risk and return would change 
    *if we modified the portfolio today*.

    **B. Historical backtest & realised risk** – how the current strategy 
    would have behaved in the past, based on realised market data.  
    Historical metrics are not directly comparable to scenario metrics.
    """)

    # ========================================================
    #  A. FORWARD-LOOKING SCENARIOS (PROSPECTIVE VIEW)
    # ========================================================

    st.markdown("### A. Forward-looking scenarios")

    # Helper: map detailed categories to broad buckets
    def map_to_bucket(cat: str) -> str:
        if "Equities" in cat:
            return "Equities"
        elif "Bonds" in cat:
            return "Bonds"
        elif "Gold" in cat:
            return "Gold"
        elif "Real Estate" in cat:
            return "Real Estate"
        elif "Crypto" in cat:
            return "Crypto"
        else:
            return "Other"

    alloc_df["Bucket"] = alloc_df["Category"].apply(map_to_bucket)

    # ---------- A1. Scenario selection & construction ----------
    st.subheader("A1. Scenario selection")

    scenario_choice = st.selectbox(
        "Select scenario (what-if on the current optimised portfolio):",
        [
            "Base case (current optimisation)",
            "Risk-off: rotate 10% from Equities into Bonds & Gold",
            "Risk-on: rotate 10% from Bonds into Equities",
        ],
    )

    shift_share = 0.10  # 10% of the total portfolio
    base_weights = opt_weights.copy()
    scenario_weights = base_weights.copy()
    buckets = alloc_df["Bucket"].values

    if scenario_choice != "Base case (current optimisation)":
        w = base_weights.copy()
        eq_idx = np.where(buckets == "Equities")[0]
        bond_idx = np.where(buckets == "Bonds")[0]
        gold_idx = np.where(buckets == "Gold")[0]

        total_equity = w[eq_idx].sum() if len(eq_idx) > 0 else 0.0
        total_bond = w[bond_idx].sum() if len(bond_idx) > 0 else 0.0

        if scenario_choice.startswith("Risk-off") and total_equity > 0:
            # Move capital from Equities to Bonds & Gold
            shift = min(shift_share, total_equity)
            if len(eq_idx) > 0:
                w[eq_idx] *= (total_equity - shift) / total_equity
            dest_idx = np.concatenate([bond_idx, gold_idx])
            if len(dest_idx) > 0:
                dest_weights = w[dest_idx]
                dest_weights = dest_weights + shift * dest_weights / dest_weights.sum()
                w[dest_idx] = dest_weights

        elif scenario_choice.startswith("Risk-on") and total_bond > 0:
            # Move capital from Bonds to Equities
            shift = min(shift_share, total_bond)
            if len(bond_idx) > 0:
                w[bond_idx] *= (total_bond - shift) / total_bond
            if len(eq_idx) > 0:
                dest_weights = w[eq_idx]
                dest_weights = dest_weights + shift * dest_weights / dest_weights.sum()
                w[eq_idx] = dest_weights

        scenario_weights = w / w.sum()
    else:
        scenario_weights = base_weights.copy()

    # ---------- A2. Impact on key portfolio metrics ----------
    st.subheader("A2. Impact on key portfolio metrics (forward-looking)")

    base_ret = float(exp_ret)
    base_vol = float(final_vol)
    base_es = float(final_es_metric)

    scen_ret = float(np.dot(scenario_weights, mu))
    scen_vol = float(calculate_portfolio_vol(scenario_weights, Sigma))
    scen_es = float(calculate_portfolio_es(scenario_weights, returns_df, alpha_input))

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Expected Return (p.a.)",
        f"{scen_ret*100:.1f}%",
        delta=f"{(scen_ret - base_ret)*100:.1f} pp",
    )
    c2.metric(
        "Expected Volatility (model-based)",
        f"{scen_vol*100:.1f}%",
        delta=f"{(scen_vol - base_vol)*100:.1f} pp",
    )
    c3.metric(
        f"Expected Shortfall ({int(alpha_input*100)}%)",
        f"{scen_es*100:.1f}%",
        delta=f"{(scen_es - base_es)*100:.1f} pp",
    )

    st.caption("""
    Scenario metrics are **model-based and forward-looking**: they show how 
    risk and return would change if we adjusted the portfolio **today**, 
    holding the risk model constant.
    """)

    # ---------- A3. Key allocation changes: only Δ in pp ----------
    st.subheader("A3. Key allocation changes (Δ in percentage points)")

    changes_df = pd.DataFrame({
        "Asset": filtered_tickers,
        "Bucket": alloc_df["Bucket"].values,
        "Base Weight": base_weights,
        "Scenario Weight": scenario_weights,
    })
    # Δ en points de %
    changes_df["Change (pp)"] = (changes_df["Scenario Weight"] - changes_df["Base Weight"]) * 100
    changes_df["Abs Change"] = changes_df["Change (pp)"].abs()
    changes_df["Direction"] = np.where(changes_df["Change (pp)"] > 0, "Increase", "Decrease")
    # Notional indicatif associé à la variation
    changes_df["Approx. Trade Amount"] = (changes_df["Change (pp)"] / 100.0 * investment_amount)

    top_changes = changes_df.sort_values("Abs Change", ascending=False).head(6)

    st.caption("""
    Largest changes in allocation, expressed in **percentage points of portfolio weight**.  
    Detailed weights by asset are shown in the *Portfolio Construction* tab.
    """)

    display_changes = (
        top_changes[["Asset", "Bucket", "Direction", "Change (pp)", "Approx. Trade Amount"]]
        .assign(
            **{
                "Change (pp)":       lambda df: df["Change (pp)"].round(2),
                "Approx. Trade Amount": lambda df: df["Approx. Trade Amount"].round(0),
            }
        )
        .rename(columns={
            "Change (pp)": "Δ Weight (pp)",
            "Approx. Trade Amount": "Indicative Trade (Amount)",
        })
    )

    st.dataframe(display_changes, use_container_width=True)

    st.markdown("---")

    # ========================================================
    #  B. HISTORICAL BACKTEST & REALISED RISK (PAST VIEW)
    # ========================================================

    st.markdown("### B. Historical backtest & realised risk")

    # ---------- B1. Historical performance ----------
    st.subheader("B1. Historical performance of the current strategy")

    st.write("""
    The chart below shows how the **current optimised strategy** would have 
    performed historically versus a simple equal-weight benchmark constructed 
    on the same universe of assets.

    These results are based on realised past returns and are **not** affected 
    by the scenario selected above.
    """)

    n_assets = len(filtered_tickers)
    if n_assets > 0:
        eq_weights = np.array([1.0 / n_assets] * n_assets)
    else:
        eq_weights = np.array([])

    cum_opt = (1 + returns_df.dot(opt_weights)).cumprod()
    cum_eq = (1 + returns_df.dot(eq_weights)).cumprod()

    hist_data = pd.DataFrame({
        "Date": returns_df.index,
        "Selected Strategy": cum_opt,
        "Equal Weight Benchmark": cum_eq,
    })
    hist_melted = hist_data.melt("Date", var_name="Strategy", value_name="Cumulative Return")

    unique_years = sorted(returns_df.index.year.unique())
    tick_values = [pd.Timestamp(f"{y}-01-01") for y in unique_years]

    perf_chart = (
        alt.Chart(hist_melted)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("Date", axis=alt.Axis(values=tick_values, format="%Y", title="Year", grid=True)),
            y=alt.Y("Cumulative Return", title="Growth of 1"),
            color="Strategy",
            tooltip=[
                alt.Tooltip("Date", format="%b %Y"),
                alt.Tooltip("Strategy"),
                alt.Tooltip("Cumulative Return", format=".2f"),
            ],
        )
        .properties(height=380)
        .interactive()
    )

    st.altair_chart(perf_chart, use_container_width=True)

    st.caption("""
    Backtested performance based on the chosen investment universe and rebalancing frequency.  
    It reflects the **current strategy**, not the alternative scenarios defined above.
    """)

    # ---------- B2. Dynamic risk (rolling volatility) ----------
    st.subheader("B2. Realised risk over time (rolling volatility)")

    st.write("""
    Rolling annualised volatility measures how the realised risk of the strategy 
    has evolved over time, and how it compares to the riskiest single asset in 
    the universe.

    Unlike the scenario metrics, this is a **purely historical** measure.
    """)

    window = st.slider("Rolling window (months)", 3, 36, 12, key="rolling_window_tab4")

    rolling_asset_vol = returns_df.rolling(window=window).std() * np.sqrt(12)
    port_ret_series = returns_df.dot(opt_weights)
    rolling_port_vol = port_ret_series.rolling(window=window).std() * np.sqrt(12)

    max_vol_asset = asset_vols.argmax()
    riskiest_name = filtered_tickers[max_vol_asset]

    vol_data = pd.DataFrame({
        "Date": returns_df.index,
        "Strategy Risk": rolling_port_vol,
        f"{riskiest_name} (Riskiest Asset)": rolling_asset_vol.iloc[:, max_vol_asset],
    })
    vol_melted = vol_data.melt("Date", var_name="Metric", value_name="Volatility")

    vol_chart = (
        alt.Chart(vol_melted)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("Date", axis=alt.Axis(values=tick_values, format="%Y", title="Year", grid=True)),
            y=alt.Y("Volatility", axis=alt.Axis(format="%", title="Annualised volatility")),
            color="Metric",
            tooltip=[
                alt.Tooltip("Date", format="%b %Y"),
                alt.Tooltip("Metric"),
                alt.Tooltip("Volatility", format=".1%"),
            ],
        )
        .properties(height=340)
        .interactive()
    )

    st.altair_chart(vol_chart, use_container_width=True)

    st.caption(f"""
    Rolling {window}-month volatility of the **current optimised strategy** versus the 
    single riskiest asset in the universe.  
    This is based on realised past returns and is conceptually different from the 
    **forward-looking** scenario risk shown in section A.
    """)



# ============================================================
#  TAB 5 — IMPLEMENTATION & REBALANCING
# ============================================================
with tab5:
    st.header("Implementation & Rebalancing")

    st.write("""
    This tab summarises how the optimised portfolio can be implemented in practice 
    and which trades would be required to move from the **base strategy** to the 
    **scenario allocation** defined in the previous tab.
    """)

    # --- 1. Rebalancing policy summary ---
    st.subheader("1. Rebalancing policy")

    st.markdown(f"""
    • **Frequency:** {rebalance_label}  
    • **Objective:** Maintain the strategic risk profile and diversification  
    • **Max position size:** {max_weight:.0%} per single instrument  
    • **Investor profile:** {risk_profile} – {time_horizon_years}-year horizon  
    """)

    st.caption("""
    The portfolio is reviewed regularly according to the chosen rebalancing 
    frequency. Rebalancing aims to keep risk and allocation close to the 
    strategic targets, while limiting transaction costs.
    """)

    # --- 2. Trading instructions: Base -> Scenario ---
    st.subheader("2. Trading instructions (base strategy → scenario)")

    if len(filtered_tickers) == 0:
        st.info("No assets selected – trading instructions are not available.")
    else:
        # Current = optimised base strategy (Tab 2)
        current_weights = opt_weights.copy()

        # Target = scenario allocation (Tab 4).
        # If for some reason scenario_weights is not defined, fall back to base.
        try:
            target_weights = scenario_weights.copy()
        except NameError:
            target_weights = opt_weights.copy()

        # Amounts in investor's base currency
        current_amounts = current_weights * investment_amount
        target_amounts = target_weights * investment_amount
        delta_weights = target_weights - current_weights
        delta_amounts = target_amounts - current_amounts

        trades_df = pd.DataFrame({
            "Asset": filtered_tickers,
            "Category": alloc_df["Category"].values,
            "Current Weight": current_weights,
            "Target Weight": target_weights,
            "Current Amount": current_amounts,
            "Target Amount": target_amounts,
            "Delta Weight": delta_weights,
            "Trade Amount": delta_amounts,
        })

        trades_df["Direction"] = np.where(trades_df["Trade Amount"] > 0, "Buy", "Sell")
        trades_df["Abs Trade"] = trades_df["Trade Amount"].abs()

        # Turnover as a simple liquidity indicator
        turnover = trades_df["Abs Trade"].sum()

        # Format for display
        display_trades = (
            trades_df
            .assign(
                **{
                    "Current Weight (%)": lambda df: (df["Current Weight"] * 100).round(2),
                    "Target Weight (%)": lambda df: (df["Target Weight"] * 100).round(2),
                    "Δ Weight (pp)":     lambda df: (df["Delta Weight"] * 100).round(2),
                    "Trade (Amount)":    lambda df: df["Trade Amount"].round(0),
                }
            )[["Asset", "Category", "Direction",
               "Current Weight (%)", "Target Weight (%)", "Δ Weight (pp)", "Trade (Amount)"]]
            .sort_values("Trade (Amount)", ascending=False)
        )

        st.dataframe(display_trades, use_container_width=True)

        st.caption(
            f"Approximate one-off turnover to move from the base strategy to the "
            f"scenario allocation: **{turnover:,.0f}** in notional terms."
        )

    st.caption("""
    The trading instructions are indicative only. In practice, execution would 
    take into account liquidity, costs, and operational constraints.
    """)
