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
    Fetches meta-data to classify assets (Equity, Bond, etc.)
    """
    known_etfs = {
        'SPY': 'Equities - US Large Cap', 'QQQ': 'Equities - US Tech', 'DIA': 'Equities - US Value',
        'IWM': 'Equities - US Small Cap', 'EEM': 'Equities - Emerging Mkts', 'VTI': 'Equities - Total Market',
        'VGK': 'Equities - Europe', 'EWJ': 'Equities - Japan',
        'TLT': 'Bonds - US Long Term', 'IEF': 'Bonds - US Interm.', 'SHY': 'Bonds - US Short',
        'LQD': 'Bonds - Corporate', 'HYG': 'Bonds - High Yield', 'EMB': 'Bonds - Emerging Mkts',
        'BND': 'Bonds - Total Market', 'AGG': 'Bonds - Aggregate',
        'GLD': 'Commodities - Gold', 'SLV': 'Commodities - Silver', 
        'USO': 'Commodities - Oil', 'DBC': 'Commodities - Broad',
        'VNQ': 'Real Estate (REITs)', 'BTC-USD': 'Crypto', 'ETH-USD': 'Crypto'
    }
    
    classification = {}
    for t in tickers:
        if t in known_etfs:
            classification[t] = known_etfs[t]
            continue
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
        except:
            classification[t] = "Unclassified" 
    return classification

@st.cache_data
def load_data(tickers, start_date):
    end_date = datetime.now().strftime('%Y-%m-%d')
    try:
        data = yf.download(tickers, start=start_date, end=end_date)['Close']
        if data is None or data.empty: return None, None, None, None, None
        
        data = data.dropna(axis=1, how='all')
        if data.empty: return None, None, None, None, None

        if isinstance(data, pd.Series): data = data.to_frame()

        # Date Handling
        first_valid = data.apply(lambda col: col.first_valid_index()).max()
        user_start = pd.to_datetime(start_date).tz_localize(None)
        actual_start = pd.to_datetime(first_valid).tz_localize(None)
        
        # Smart Warning: Only warn if gap > 5 days
        warning_msg = None
        if (actual_start - user_start).days > 5:
            warning_msg = f"⚠️ Data availability limited. Start date adjusted from {user_start.date()} to {actual_start.date()}."

        def load_data(tickers, start_date, rebalance_freq="M"):
    end_date = datetime.now().strftime('%Y-%m-%d')
    try:
        data = yf.download(tickers, start=start_date, end=end_date)['Close']
        if data is None or data.empty: return None, None, None, None, None
        
        data = data.dropna(axis=1, how='all')
        if data.empty: return None, None, None, None, None

        if isinstance(data, pd.Series):
            data = data.to_frame()

        # Date Handling
        first_valid = data.apply(lambda col: col.first_valid_index()).max()
        user_start = pd.to_datetime(start_date).tz_localize(None)
        actual_start = pd.to_datetime(first_valid).tz_localize(None)
        
        warning_msg = None
        if (actual_start - user_start).days > 5:
            warning_msg = (
                f"⚠️ Data availability limited. "
                f"Start date adjusted from {user_start.date()} to {actual_start.date()}."
            )

        data = data.loc[first_valid:].ffill().bfill()

        # >>> rebalancing / fréquence d’échantillonnage <<<
        returns = data.resample(rebalance_freq).last().pct_change().dropna()
        if returns.empty: return None, None, None, None, None

        # Annualisation cohérente avec la fréquence
        if rebalance_freq == "M":    # monthly
            ann_factor = 12
        elif rebalance_freq == "Q":  # quarterly
            ann_factor = 4
        elif rebalance_freq == "Y":  # yearly
            ann_factor = 1
        else:
            ann_factor = 12  # fallback

        mu = returns.mean() * ann_factor
        Sigma = returns.cov() * ann_factor
        
        return returns, mu, Sigma, first_valid.strftime('%Y-%m-%d'), warning_msg

    except Exception as e:
        return None, None, None, None, None
        

def calculate_portfolio_vol(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def calculate_portfolio_es(weights, returns_df, alpha=0.95):
    port_returns = returns_df.dot(weights)
    var_level = (1 - alpha) * 100
    var_threshold = np.percentile(port_returns, var_level)
    tail_losses = port_returns[port_returns < var_threshold]
    if len(tail_losses) == 0: return 0.0
    return -tail_losses.mean()

def calculate_risk_contributions(weights, cov_matrix):
    port_vol = calculate_portfolio_vol(weights, cov_matrix)
    if port_vol == 0: return np.zeros_like(weights)
    marginal_risk = np.dot(cov_matrix, weights) / port_vol
    rc = weights * marginal_risk
    return rc / port_vol

def calculate_diversification_ratio(weights, sigma, asset_vols):
    port_vol = calculate_portfolio_vol(weights, sigma)
    if port_vol == 0: return 0
    weighted_avg_vol = np.dot(weights, asset_vols)
    return weighted_avg_vol / port_vol

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
# 3. SIDEBAR CONTROLS
# ==========================================

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
    # 1. Load Data
    returns_df, mu, Sigma, valid_start, date_warning = load_data(
        tickers, 
        start_date,
        rebalance_freq=rebalance_freq
    )

    if returns_df is None:
        st.error(f"Could not download data. Please check tickers.")
        st.stop()
        
    # 2. Asset Classification
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

# puis utiliser filtered_tickers partout à la place de tickers
returns_df, mu, Sigma, valid_start, date_warning = load_data(
    filtered_tickers, start_date, rebalance_freq=rebalance_freq
)
asset_info_dict = get_asset_info(filtered_tickers)
# ...
opt_weights = run_optimization(..., tickers=filtered_tickers, ...)
# et pour les DataFrames / graphes : filtered_tickers


    # 3. Date Warning
    if date_warning:
        st.warning(date_warning, icon="⚠️")

    # 4. Run Optimization
    opt_weights = run_optimization(strategy_choice, mu, Sigma, returns_df, max_weight=max_weight, target_vol=target_vol_input, alpha=alpha_input)
    
    # 5. Calc Stats
    final_vol = calculate_portfolio_vol(opt_weights, Sigma)
    final_es_metric = calculate_portfolio_es(opt_weights, returns_df, alpha_input)
    final_rc = calculate_risk_contributions(opt_weights, Sigma)
    asset_vols = np.sqrt(np.diag(Sigma))
    div_ratio = calculate_diversification_ratio(opt_weights, Sigma, asset_vols)
    exp_ret = np.dot(opt_weights, mu)

# ==========================================
# 5. VISUALIZATION
# ==========================================

col1, col2, col3, col4 = st.columns(4)
col1.metric("Annual Volatility", f"{final_vol*100:.2f}%")
col2.metric(f"Exp. Shortfall ({alpha_input*100:.0f}%)", f"{final_es_metric*100:.2f}%")
col3.metric("Diversification Ratio", f"{div_ratio:.2f}")
col4.metric("Sharpe Ratio (est.)", f"{(exp_ret/final_vol):.2f}")

st.divider()

# --- A. Asset Class Breakdown (Pie Chart) ---
st.subheader("1. Composition by Asset Class")
alloc_df = pd.DataFrame({
    'Asset': tickers,
    'Weight': opt_weights,
    'Notional': opt_weights * investment_amount,
    'Category': [asset_info_dict.get(t, "Other") for t in tickers]
})
alloc_df_filtered = alloc_df[alloc_df['Weight'] > 0.001]

# --- DEFINE COLORS & ADAPTIVE SCALE ---
category_colors = {
    'Equities - US Large Cap': '#1f77b4',  # Blue
    'Equities - US Tech': '#aec7e8',       # Light Blue
    'Equities - US Value': '#4c78a8',      # Dark Blue
    'Equities - US Small Cap': '#72b7b2',  # Teal
    'Equities - Emerging Mkts': '#9467bd', # Purple
    'Equities - Total Market': '#17becf',  # Cyan
    'Equities - Europe': '#9edae5',        # Pale Blue
    'Equities - Japan': '#dbdb8d',         # Yellow-Green
    'Bonds - US Long Term': '#ff7f0e',     # Orange
    'Bonds - US Interm.': '#ffbb78',       # Light Orange
    'Bonds - US Short': '#eeca3b',         # Gold
    'Bonds - Corporate': '#bcbd22',        # Olive
    'Bonds - High Yield': '#8c564b',       # Brown
    'Bonds - Emerging Mkts': '#d62728',    # Red
    'Bonds - Total Market': '#ff9896',     # Salmon
    'Bonds - Aggregate': '#f28e2b',        # Dark Orange
    'Commodities - Gold': '#e377c2',       # Pink
    'Commodities - Silver': '#f7b6d2',     # Light Pink
    'Commodities - Oil': '#7f7f7f',        # Grey
    'Commodities - Broad': '#c7c7c7',      # Light Grey
    'Real Estate (REITs)': '#2ca02c',      # Green
    'Crypto': '#000000',                   # Black
    'ETF / Other': '#bab0ac',              # Silver
    'Unclassified': '#bab0ac'
}

# Calculate Dynamic Domain & Range for the chart
present_categories = alloc_df_filtered['Category'].unique()
chart_domain = []
chart_range = []

for cat in present_categories:
    chart_domain.append(cat)
    # If specific color exists, use it. Else use Grey.
    if cat in category_colors:
        chart_range.append(category_colors[cat])
    else:
        chart_range.append("#808080") # Fallback Grey

pie = alt.Chart(alloc_df_filtered).mark_arc(innerRadius=60).encode(
    theta=alt.Theta(field="Weight", type="quantitative"),
    # Use the adaptive scale we just calculated
    color=alt.Color(field="Category", type="nominal", scale=alt.Scale(domain=chart_domain, range=chart_range)),
    tooltip=["Category", "Asset", alt.Tooltip("Weight", format=".1%")]
).properties(title="Portfolio Exposure").interactive()
st.altair_chart(pie, use_container_width=True)


st.subheader("2. Allocation in currency terms")
st.dataframe(
    alloc_df.sort_values('Weight', ascending=False).assign(
        Weight_pct=lambda df: (df['Weight'] * 100).round(2),
        Notional_rounded=lambda df: df['Notional'].round(0)
    )[["Asset", "Category", "Weight_pct", "Notional_rounded"]]
    .rename(columns={"Weight_pct": "Weight (%)", "Notional_rounded": "Amount"})
)

# --- B. Capital vs Risk ---
st.subheader("3. Allocation Detail (Capital vs Risk)")
col_chart1, col_chart2 = st.columns(2)

alloc_df_full = pd.DataFrame({'Asset': tickers, 'Weight': opt_weights, 'Risk Contribution': final_rc})
alloc_df_full = alloc_df_full[alloc_df_full['Weight'] > 0.001]
melted_df = alloc_df_full.melt('Asset', var_name='Metric', value_name='Value')

with col_chart1:
    chart = alt.Chart(melted_df).mark_bar().encode(
        x=alt.X('Asset', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('Value', axis=alt.Axis(format='%', title='Percentage')),
        color='Metric',
        column=alt.Column('Metric', title=None),
        tooltip=['Asset', 'Metric', alt.Tooltip('Value', format='.1%')]
    ).properties(width=300, height=300)
    st.altair_chart(chart)

with col_chart2:
    st.markdown("**Insight**")
    if strategy_choice == "Equal Risk Contribution (ERC)":
        st.info("Notice how the 'Risk Contribution' bars are nearly equal, even though the 'Weights' (Capital) are different. This is the definition of Risk Parity.")
    elif strategy_choice == "Equal Weight (Benchmark)":
        st.warning("Notice that while Weights are equal, Risk Contributions are uneven. Volatile assets dominate the risk profile.")
    else:
        st.info("The bar chart compares how much money is invested vs. how much risk that investment adds to the total portfolio.")

# --- C. Historical Performance ---
st.subheader("4. Historical Performance")
n_assets = len(tickers)
eq_weights = np.array([1/n_assets] * n_assets)
cum_opt = (1 + returns_df.dot(opt_weights)).cumprod()
cum_eq = (1 + returns_df.dot(eq_weights)).cumprod()

hist_data = pd.DataFrame({'Date': returns_df.index, 'Selected Strategy': cum_opt, 'Equal Weight Benchmark': cum_eq})
hist_melted = hist_data.melt('Date', var_name='Strategy', value_name='Cumulative Return')

unique_years = sorted(returns_df.index.year.unique())
tick_values = [pd.Timestamp(f"{y}-01-01") for y in unique_years]

perf_chart = alt.Chart(hist_melted).mark_line(strokeWidth=2).encode(
    x=alt.X('Date', axis=alt.Axis(values=tick_values, format='%Y', title='Year', grid=True)),
    y=alt.Y('Cumulative Return', title='Growth ($1 invested)'),
    color='Strategy',
    tooltip=[alt.Tooltip('Date', format='%b %Y'), alt.Tooltip('Strategy'), alt.Tooltip('Cumulative Return', format='.2f')]
).properties(height=400).interactive()
st.altair_chart(perf_chart, use_container_width=True)

# --- D. Rolling Risk Analysis ---
st.subheader("5. Dynamic Risk Analysis (Crisis Simulator)")
window = st.slider("Rolling Window (Months)", 3, 36, 12)

# Calculate Data
rolling_asset_vol = returns_df.rolling(window=window).std() * np.sqrt(12)
port_ret_series = returns_df.dot(opt_weights)
rolling_port_vol = port_ret_series.rolling(window=window).std() * np.sqrt(12)

# Identify Riskiest
max_vol_asset = asset_vols.argmax()
riskiest_name = tickers[max_vol_asset]

vol_data = pd.DataFrame({
    'Date': returns_df.index,
    'Strategy Risk': rolling_port_vol,
    f'{riskiest_name} (Riskiest Asset)': rolling_asset_vol.iloc[:, max_vol_asset]
})
vol_melted = vol_data.melt('Date', var_name='Metric', value_name='Volatility')

vol_chart = alt.Chart(vol_melted).mark_line(strokeWidth=2).encode(
    x=alt.X('Date', axis=alt.Axis(values=tick_values, format='%Y', title='Year', grid=True)),
    y=alt.Y('Volatility', axis=alt.Axis(format='%')),
    color=alt.Color('Metric', scale=alt.Scale(scheme='category10')), 
    tooltip=[alt.Tooltip('Date', format='%b %Y'), alt.Tooltip('Metric'), alt.Tooltip('Volatility', format='.1%')]
).properties(height=350).interactive()

st.altair_chart(vol_chart, use_container_width=True)
st.caption(f"Comparing Strategy Risk vs {riskiest_name} over time.")

# --- E. Efficient Frontier ---
st.subheader("6. The Efficient Frontier")
sim_df = generate_efficient_frontier(mu, Sigma)
sim_df['Type'] = 'Random'; sim_df['Size'] = 20; sim_df['Color'] = 'grey'

opt_point = pd.DataFrame({
    'Volatility': [float(final_vol)], 'Return': [float(exp_ret)], 
    'Sharpe': [float(exp_ret/final_vol) if final_vol > 0 else 0], 
    'Type': ['Strategy'], 'Size': [300], 'Color': ['red']
})
combined_df = pd.concat([sim_df, opt_point], ignore_index=True)

frontier_chart = alt.Chart(combined_df).mark_circle().encode(
    x=alt.X('Volatility', axis=alt.Axis(format='%', title='Annualized Volatility')),
    y=alt.Y('Return', axis=alt.Axis(format='%', title='Annualized Return')),
    color=alt.Color('Color', scale=None), size=alt.Size('Size', scale=None),
    tooltip=[alt.Tooltip('Type'), alt.Tooltip('Volatility', format='.1%'), alt.Tooltip('Return', format='.1%'), alt.Tooltip('Sharpe', format='.2f')],
    order=alt.Order('Size', sort='ascending') 
).interactive()

st.altair_chart(frontier_chart, use_container_width=True)
