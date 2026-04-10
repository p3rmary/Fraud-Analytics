import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from streamlit_gsheets import GSheetsConnection
import numpy as np
from datetime import datetime

st.set_page_config(
    page_title="FraudLens — Detection Analytics",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLES ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;500;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --red:       #E8392A;
    --red-light: #FDECEA;
    --red-mid:   #F08070;
    --amber:     #E8920A;
    --amber-light:#FEF3DC;
    --green:     #1A9E5F;
    --green-light:#E3F7EE;
    --blue:      #1B6FD8;
    --blue-light:#E8F0FD;
    --ink:       #0E0F10;
    --ink2:      #3A3C40;
    --ink3:      #72757C;
    --surface:   #F7F7F5;
    --surface2:  #EFEFED;
    --white:     #FFFFFF;
    --border:    rgba(0,0,0,0.08);
    --border2:   rgba(0,0,0,0.14);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: var(--surface) !important;
    color: var(--ink);
}

.stApp { background: var(--surface) !important; }
#MainMenu, footer, header { visibility: hidden; }

section[data-testid="stSidebar"] {
    background: var(--ink) !important;
    border-right: none;
}
section[data-testid="stSidebar"] * { color: #E8E8E4 !important; }

.dash-header {
    background: var(--ink);
    color: white;
    padding: 28px 36px;
    margin: -1rem -1rem 2rem -1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 3px solid var(--red);
}
.dash-header-title { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 26px; color: white; }
.dash-header-title span { color: var(--red); }
.dash-header-sub { font-size: 13px; color: rgba(255,255,255,0.5); font-family: 'DM Mono', monospace; }

.kpi-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
}
.kpi-accent { position: absolute; top: 0; left: 0; width: 100%; height: 3px; }
.kpi-label { font-size: 11px; font-weight: 500; color: var(--ink3); text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 10px; }
.kpi-value { font-family: 'Syne', sans-serif; font-size: 28px; font-weight: 700; color: var(--ink); line-height: 1; }

.chart-card { background: var(--white); border: 1px solid var(--border); border-radius: 12px; padding: 20px 22px; }
.section-title { font-family: 'Syne', sans-serif; font-size: 15px; font-weight: 700; color: var(--ink); margin-bottom: 2px; }
.section-sub { font-size: 12px; color: var(--ink3); margin-bottom: 14px; }
</style>
""", unsafe_allow_html=True)

# --- DATA LOADING ---

@st.cache_resource
def get_connection():
    return st.connection("gsheets", type=GSheetsConnection)

@st.cache_data(ttl=300)
def load_all_sheets(_conn):
    # 1. Totals Sheet (Aggregated)
    df_totals = _conn.read(worksheet="totals")
    df_totals['date'] = pd.to_datetime(df_totals['date'])
    
    # 2. Fraud Data (Granular)
    df_fraud = _conn.read(worksheet="fraud_data")
    df_fraud['date'] = pd.to_datetime(df_fraud['date'])
    df_fraud['datetime'] = pd.to_datetime(df_fraud['datetime'])
    
    # 3. Fraud Totals (Ranked)
    df_fraud_ranks = _conn.read(worksheet="fraud_totals")
    
    # 4. Hourly Data
    df_hourly = _conn.read(worksheet="hourly")
    
    return df_totals, df_fraud, df_fraud_ranks, df_hourly

# --- HELPERS ---
def fmt_number(n):
    if n >= 1_000_000_000: return f"${n/1_000_000_000:.2f}B"
    if n >= 1_000_000: return f"${n/1_000_000:.1f}M"
    return f"${n:,.0f}"

def kpi(label, value, accent="#E8392A"):
    return f"""
    <div class="kpi-card">
        <div class="kpi-accent" style="background:{accent}"></div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
    </div>"""

PLOT_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='DM Sans', color='#3A3C40', size=12),
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis=dict(showgrid=False),
    yaxis=dict(gridcolor='rgba(0,0,0,0.05)')
)

# --- MAIN ---
def main():
    try:
        conn = get_connection()
        df_t, df_f, df_fr, df_h = load_all_sheets(conn)
    except Exception as e:
        st.error(f"Failed to connect to Google Sheets: {e}")
        return

    # Sidebar
    with st.sidebar:
        st.markdown("<h2 style='color:white;'>FILTERS</h2>", unsafe_allow_html=True)
        types = ["All"] + sorted(df_t['type'].unique().tolist())
        sel_type = st.selectbox("Transaction Type", types)
        
        date_min, date_max = df_t['date'].min(), df_t['date'].max()
        sel_dates = st.date_input("Date Range", [date_min, date_max])

    # Apply Filters to Totals and Fraud data
    filt_t = df_t.copy()
    filt_f = df_f.copy()
    if sel_type != "All":
        filt_t = filt_t[filt_t['type'] == sel_type]
        filt_f = filt_f[filt_f['type'] == sel_type]
    if len(sel_dates) == 2:
        filt_t = filt_t[(filt_t['date'] >= pd.Timestamp(sel_dates[0])) & (filt_t['date'] <= pd.Timestamp(sel_dates[1]))]
        filt_f = filt_f[(filt_f['date'] >= pd.Timestamp(sel_dates[0])) & (filt_f['date'] <= pd.Timestamp(sel_dates[1]))]

    # Header
    st.markdown(f"""
    <div class="dash-header">
        <div>
            <div class="dash-header-title">FRAUD<span>LENS</span> — Dashboard</div>
            <div class="dash-header-sub">Source: Google Sheets · {datetime.now().strftime('%H:%M')} Sync</div>
        </div>
        <div class="dash-header-badge">● LIVE DATA</div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Fraud Deep Dive", "Accuracy", "Profiles"])

    with tab1:
        # Metrics from 'totals' sheet
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(kpi("Total Transactions", f"{filt_t['total_transactions'].sum():,}"), unsafe_allow_html=True)
        m2.markdown(kpi("Overall Volume", fmt_number(filt_t['overall_volume'].sum()), "#1B6FD8"), unsafe_allow_html=True)
        m3.markdown(kpi("Fraud Cases", f"{filt_t['fraud_tx'].sum():,}", "#E8392A"), unsafe_allow_html=True)
        m4.markdown(kpi("Fraud Volume", fmt_number(filt_t['fraud_volume'].sum()), "#E8920A"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Volume Chart
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Daily Transaction Volume</div>', unsafe_allow_html=True)
        daily_vol = filt_t.groupby('date')['overall_volume'].sum().reset_index()
        fig_vol = px.line(daily_vol, x='date', y='overall_volume', color_discrete_sequence=['#1B6FD8'])
        fig_vol.update_layout(**PLOT_LAYOUT, height=300)
        st.plotly_chart(fig_vol, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        # Peak hours from 'hourly' sheet
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Volume by Hour</div>', unsafe_allow_html=True)
            fig_h = px.bar(df_h, x='hour_of_day', y='total_volume', color_discrete_sequence=['#F08070'])
            fig_h.update_layout(**PLOT_LAYOUT, height=350)
            st.plotly_chart(fig_h, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with c2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Fraud by Type</div>', unsafe_allow_html=True)
            type_f = filt_f.groupby('type').size().reset_index(name='count')
            fig_p = px.pie(type_f, values='count', names='type', hole=0.6, color_discrete_sequence=['#E8392A', '#1B6FD8'])
            fig_p.update_layout(**PLOT_LAYOUT, height=350, showlegend=False)
            st.plotly_chart(fig_p, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        # Accuracy Metrics from 'fraud_data'
        # Assuming fraud_data contains isfraud and isflaggedfraud
        tp = int(filt_f[filt_f['isflaggedfraud'] == True].shape[0])
        fn = int(filt_f[filt_f['isflaggedfraud'] == False].shape[0])
        recall = (tp / (tp + fn)) * 100 if (tp+fn) > 0 else 0
        
        a1, a2 = st.columns(2)
        a1.markdown(kpi("True Positives (Correctly Flagged)", f"{tp:,}", "#1A9E5F"), unsafe_allow_html=True)
        a2.markdown(kpi("Recall Rate", f"{recall:.2f}%", "#E8392A"), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(filt_f[['nameorig', 'amount', 'type', 'isflaggedfraud', 'date']].head(10), use_container_width=True)

    with tab4:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Top 10 Fraud Originators</div>', unsafe_allow_html=True)
        # Using fraud_totals sheet
        top_senders = df_fr.nlargest(10, 'total_fraud_amount')
        fig_send = px.bar(top_senders, x='total_fraud_amount', y='nameorig', orientation='h', 
                          color='total_fraud_amount', color_continuous_scale='Reds')
        fig_send.update_layout(**PLOT_LAYOUT, height=400)
        st.plotly_chart(fig_send, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
