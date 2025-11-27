import streamlit as st

st.title("Meet the Team 👥")

st.markdown(
    """
    Welcome to our project! We are a team of Master’s students passionate about 
    **quantitative finance**, **portfolio optimization**, and **data-driven investment strategies**.
    """
)

st.divider()

# --- ROW 1: 2 Columns (Edvin & Imen) ---
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("🎓 Edvin Avdija")
    st.caption("Quantitative Analyst & Project Lead")
    st.markdown(
        """
        **Role:**  
        Lead developer for the quantitative models, responsible for portfolio optimization logic and integration between risk-budgeting and Streamlit.

        **Key Contributions:**
        - Architected the full-stack Streamlit application and interactive dashboard.
        - Implemented core optimization algorithms (ERC, MDP, CVaR).
        - Integrated real-time data fetching and dynamic visualization modules.

        **Links:**  
        🔗 [LinkedIn](https://www.linkedin.com/in/edvinavdija/) | ✉️ [Email](mailto:edvinavdija@gmail.com)
        """
    )

with col2:
    st.subheader("🎓 Imen Hassine")
    st.caption("Senior Risk Modelling & Allocation Strategist")
    st.markdown(
        """
        **Role:**  
        Develops advanced risk models and enhances the engine with regime-aware and robust allocation techniques.

        **Key Contributions:**
        - Designs dynamic risk metrics & scenario-based stress testing.
        - Improves risk-budgeting and multi-regime allocation logic.
        - Expands the efficient frontier with advanced constraints.

        **Links:**  
        🔗 [LinkedIn](https://www.linkedin.com/in/imen-hassine-2aa265190/) | ✉️ [Email](mailto:imhassine@yahoo.com)
        """
    )

st.divider()

# --- ROW 2: 3 Columns (Marina, Louna, Ian) ---
col3, col4, col5 = st.columns(3, gap="medium")

with col3:
    st.subheader("🎓 Marina Bros De Puechredon")
    st.caption("Investment Solutions & Client Partnership Lead")
    st.markdown(
        """
        **Role:**  
        Bridges quantitative outputs with user-friendly insights, creating clear, client-oriented analyses.

        **Key Contributions:**
        - Builds intuitive Streamlit pages for reporting.
        - Translates model results into actionable scenarios.
        - Ensures platform accessibility.

        **Links:**  
        ✉️ [Email](mailto:Marina.brosdepuechredon@outlook.com)
        """
    )

with col4:
    st.subheader("🎓 Louna Seferdjeli")
    st.caption("Global Markets & Strategic Allocation Researcher")
    st.markdown(
        """
        **Role:**  
        Focuses on global market dynamics and long-term strategic asset allocation.

        **Key Contributions:**
        - Develops macro regimes & sector rotations.
        - Supports long-horizon allocation frameworks.
        - Integrates market insights into optimization.

        **Links:**  
        🔗 [LinkedIn](http://linkedin.com/in/louna-seferdjeli-373044204) | ✉️ [Email](mailto:louna.srdl@gmail.com)
        """
    )

with col5:
    st.subheader("🎓 Ian Monin")
    st.caption("Quantitative Research & Modelling Specialist")
    st.markdown(
        """
        **Role:**  
        Develops statistical models supporting forecasting and risk evaluation.

        **Key Contributions:**
        - Builds return dynamics & volatility models.
        - Performs empirical research on asset relationships.
        - Validates assumptions through simulations.

        **Links:**  
        🔗 [LinkedIn](https://www.linkedin.com/in/ian-monin-32a974231/) | ✉️ [Email](mailto:ian.monin@unil.ch)
        """
    )

st.divider()
st.caption("Thank you for exploring our Multi-Asset Portfolio Optimization project!")