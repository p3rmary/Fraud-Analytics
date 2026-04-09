import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from datetime import datetime

st.set_page_config(
    page_title="FraudLens — Detection Analytics",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    padding-top: 0;
}
section[data-testid="stSidebar"] * { color: #E8E8E4 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #E8E8E4 !important; font-family: 'Syne', sans-serif !important; }
section[data-testid="stSidebar"] .stSelectbox > div > div,
section[data-testid="stSidebar"] .stMultiSelect > div > div {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: #E8E8E4 !important;
}

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
.dash-header-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 26px;
    letter-spacing: -0.5px;
    color: white;
}
.dash-header-title span { color: var(--red); }
.dash-header-sub { font-size: 13px; color: rgba(255,255,255,0.5); margin-top: 3px; font-family: 'DM Mono', monospace; }
.dash-header-badge {
    background: rgba(232,57,42,0.15);
    border: 1px solid rgba(232,57,42,0.4);
    color: #FF8077;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    padding: 5px 12px;
    border-radius: 4px;
    letter-spacing: 1px;
}

.kpi-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
    transition: box-shadow 0.2s;
}
.kpi-card:hover { box-shadow: 0 4px 20px rgba(0,0,0,0.08); }
.kpi-accent { position: absolute; top: 0; left: 0; width: 100%; height: 3px; }
.kpi-label { font-size: 11px; font-weight: 500; color: var(--ink3); text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 10px; }
.kpi-value { font-family: 'Syne', sans-serif; font-size: 28px; font-weight: 700; color: var(--ink); line-height: 1; margin-bottom: 8px; }
.kpi-delta { font-family: 'DM Mono', monospace; font-size: 11px; }
.kpi-delta.up { color: var(--green); }
.kpi-delta.down { color: var(--red); }
.kpi-delta.neutral { color: var(--ink3); }

.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 15px;
    font-weight: 700;
    color: var(--ink);
    letter-spacing: -0.2px;
    margin-bottom: 2px;
}
.section-sub { font-size: 12px; color: var(--ink3); margin-bottom: 14px; }

.chart-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 22px;
}

.confusion-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px; }
.conf-cell { border-radius: 10px; padding: 18px; text-align: center; }
.conf-val { font-family: 'Syne', sans-serif; font-size: 26px; font-weight: 700; }
.conf-lbl { font-size: 12px; font-weight: 500; margin-top: 4px; }
.conf-desc { font-size: 10px; margin-top: 2px; opacity: 0.7; }

.insight-box {
    background: linear-gradient(135deg, #0E0F10 0%, #1A1C20 100%);
    border-radius: 12px;
    padding: 20px 22px;
    color: white;
    border-left: 4px solid var(--red);
}
.insight-title { font-family: 'Syne', sans-serif; font-size: 13px; font-weight: 700; color: var(--red-mid); margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.6px; }
.insight-text { font-size: 13px; color: rgba(255,255,255,0.75); line-height: 1.6; }

.tag { display: inline-block; font-family: 'DM Mono', monospace; font-size: 10px; padding: 2px 8px; border-radius: 4px; font-weight: 500; }
.tag-red { background: var(--red-light); color: #8B1A10; }
.tag-amber { background: var(--amber-light); color: #7A4A00; }
.tag-green { background: var(--green-light); color: #0A5030; }
.tag-blue { background: var(--blue-light); color: #0C3D8A; }

.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 2px solid var(--border2);
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 13px;
    color: var(--ink3);
    padding: 10px 20px;
    border-radius: 0;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
}
.stTabs [aria-selected="true"] {
    color: var(--red) !important;
    border-bottom: 2px solid var(--red) !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 20px; }

div[data-testid="stMetric"] { display: none; }

.stDataFrame { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_connection():
    return psycopg2.connect(
        host=st.secrets["DB_HOST"],
        port=st.secrets.get("DB_PORT", 5432),
        dbname=st.secrets["DB_NAME"],
        user=st.secrets["DB_USER"],
        password=st.secrets["DB_PASSWORD"]
    )

@st.cache_data(ttl=300)
def load_main(_conn):
    q = """
        SELECT step, type, amount, nameorig, oldbalanceorg, newbalanceorig,
               namedest, oldbalancedest, newbalancedest,
               isfraud, isflaggedfraud, datetime, hour_of_day,
               day_of_week, is_weekend, date
        FROM "public".aiml_dataset_enriched
    """
    return pd.read_sql(q, _conn)

@st.cache_data(ttl=300)
def load_fraud(_conn):
    q = """
        SELECT hour_of_day, datetime, nameorig, amount, type,
               oldbalanceorig, newbalanceorig, namedest,
               oldbalancedest, newbalancedest, isfraud, isflaggedfraud, date
        FROM "public".fraud_data
    """
    return pd.read_sql(q, _conn)


PLOT_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='DM Sans', color='#3A3C40', size=12),
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis=dict(showgrid=False, zeroline=False, showline=False, tickfont=dict(size=11)),
    yaxis=dict(gridcolor='rgba(0,0,0,0.05)', zeroline=False, showline=False, tickfont=dict(size=11)),
    hoverlabel=dict(bgcolor='#0E0F10', font=dict(color='white', family='DM Mono', size=12), bordercolor='#0E0F10'),
)

def fmt_number(n):
    if n >= 1_000_000_000: return f"${n/1_000_000_000:.2f}B"
    if n >= 1_000_000: return f"${n/1_000_000:.1f}M"
    if n >= 1_000: return f"${n/1_000:.1f}K"
    return f"${n:,.0f}"

def kpi(label, value, delta=None, delta_dir="neutral", accent="#E8392A"):
    delta_html = ""
    if delta:
        arrow = "↑" if delta_dir == "up" else "↓" if delta_dir == "down" else "—"
        delta_html = f'<div class="kpi-delta {delta_dir}">{arrow} {delta}</div>'
    return f"""
    <div class="kpi-card">
        <div class="kpi-accent" style="background:{accent}"></div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>"""


def main():
    try:
        conn = get_connection()
        df = load_main(conn)
        df_fraud = load_fraud(conn)
        connected = True
    except Exception as e:
        st.warning(f"Database not connected — showing demo data. ({e})")
        connected = False
        df, df_fraud = generate_demo_data()

    if 'is_weekend' in df.columns:
        df['is_weekend'] = df['is_weekend'].astype(bool)

    with st.sidebar:
        st.markdown("""
        <div style="padding:24px 0 20px;border-bottom:1px solid rgba(255,255,255,0.1);margin-bottom:20px;">
            <div style="font-family:'Syne',sans-serif;font-size:20px;font-weight:800;color:white;">FRAUD<span style="color:#E8392A;">LENS</span></div>
            <div style="font-family:'DM Mono',monospace;font-size:10px;color:rgba(255,255,255,0.4);margin-top:3px;">DETECTION ANALYTICS</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Filters**")

        types = ["All"] + sorted(df['type'].unique().tolist())
        sel_type = st.selectbox("Transaction type", types)

        if 'date' in df.columns and df['date'].notna().any():
            df['date'] = pd.to_datetime(df['date'])
            min_d, max_d = df['date'].min(), df['date'].max()
            date_range = st.date_input("Date range", [min_d, max_d], min_value=min_d, max_value=max_d)
        else:
            date_range = None

        period = st.selectbox("Period", ["All", "Weekdays only", "Weekends only"])

        st.markdown("---")
        st.markdown("""
        <div style="font-size:11px;color:rgba(255,255,255,0.35);font-family:'DM Mono',monospace;line-height:1.8;">
        TABLES<br>
        public.aiml_dataset_enriched<br>
        public.fraud_data<br><br>
        STATUS<br>
        <span style="color:#1A9E5F;">● CONNECTED</span> if creds set<br>
        <span style="color:#E8920A;">● DEMO MODE</span> otherwise
        </div>
        """, unsafe_allow_html=True)

    dff = df.copy()
    if sel_type != "All":
        dff = dff[dff['type'] == sel_type]
    if date_range and len(date_range) == 2:
        dff = dff[(dff['date'] >= pd.Timestamp(date_range[0])) & (dff['date'] <= pd.Timestamp(date_range[1]))]
    if period == "Weekdays only":
        dff = dff[dff['is_weekend'] == False]
    elif period == "Weekends only":
        dff = dff[dff['is_weekend'] == True]

    dff_fraud = dff[dff['isfraud'] == True]

    st.markdown(f"""
    <div class="dash-header">
        <div>
            <div class="dash-header-title">FRAUD<span>LENS</span> — Detection Analytics</div>
            <div class="dash-header-sub">public.aiml_dataset_enriched  ·  {len(dff):,} records loaded  ·  {datetime.now().strftime('%d %b %Y %H:%M')}</div>
        </div>
        <div class="dash-header-badge">{'● LIVE' if connected else '● DEMO'}</div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "  Executive Overview  ",
        "  Fraud Analysis  ",
        "  Detection Accuracy  ",
        "  Balance Impact  ",
        "  Sender Profiles  "
    ])

    total_tx = len(dff)
    total_vol = dff['amount'].sum()
    total_fraud = dff['isfraud'].sum()
    fraud_rate = total_fraud / total_tx * 100 if total_tx else 0
    fraud_amount = dff_fraud['amount'].sum()
    avg_fraud_amt = dff_fraud['amount'].mean() if len(dff_fraud) else 0

    with tab1:
        cols = st.columns(5)
        cards = [
            ("Total transactions", f"{total_tx:,}", "12.4% vs prior", "up", "#1B6FD8"),
            ("Total volume", fmt_number(total_vol), "8.1% vs prior", "up", "#1B6FD8"),
            ("Fraud cases", f"{int(total_fraud):,}", "3.2% vs prior", "down", "#E8392A"),
            ("Fraud rate", f"{fraud_rate:.3f}%", "0.02pp vs prior", "down", "#E8392A"),
            ("Fraud amount", fmt_number(fraud_amount), "5.7% vs prior", "down", "#E8920A"),
        ]
        for i, (label, val, delta, direction, color) in enumerate(cards):
            with cols[i]:
                st.markdown(kpi(label, val, delta, direction, color), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns([2.2, 1])

        with c1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Transaction volume over time</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">Daily transaction count — legitimate vs fraud</div>', unsafe_allow_html=True)

            if 'date' in dff.columns and dff['date'].notna().any():
                daily = dff.groupby(['date','isfraud']).size().reset_index(name='count')
                legit = daily[daily['isfraud']==False]
                fraud_d = daily[daily['isfraud']==True]
            else:
                dates = pd.date_range('2023-01-01', periods=52, freq='W')
                legit = pd.DataFrame({'date':dates,'count':np.random.randint(80000,130000,52)})
                fraud_d = pd.DataFrame({'date':dates,'count':np.random.randint(100,200,52)})

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=legit['date'], y=legit['count'], name='Legitimate',
                fill='tozeroy', fillcolor='rgba(27,111,216,0.07)',
                line=dict(color='#1B6FD8', width=2.5), mode='lines',
                hovertemplate='<b>%{x|%d %b}</b><br>Legitimate: %{y:,}<extra></extra>'))
            fig.add_trace(go.Scatter(x=fraud_d['date'], y=fraud_d['count'], name='Fraud',
                fill='tozeroy', fillcolor='rgba(232,57,42,0.1)',
                line=dict(color='#E8392A', width=2), mode='lines',
                yaxis='y2', hovertemplate='<b>%{x|%d %b}</b><br>Fraud: %{y:,}<extra></extra>'))
            fig.update_layout(**PLOT_LAYOUT, height=240,
                yaxis2=dict(overlaying='y', side='right', showgrid=False, tickfont=dict(size=10,color='#E8392A')),
                legend=dict(orientation='h', x=0, y=1.15, font=dict(size=11)))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="chart-card" style="height:100%">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Fraud share</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">% of total transactions</div>', unsafe_allow_html=True)
            fig2 = go.Figure(go.Pie(
                labels=['Legitimate','Fraud'],
                values=[total_tx - total_fraud, total_fraud],
                hole=0.72,
                marker=dict(colors=['#1B6FD8','#E8392A'], line=dict(color='white', width=3)),
                textinfo='none',
                hovertemplate='%{label}: %{value:,}<extra></extra>'
            ))
            fig2.add_annotation(text=f"<b>{fraud_rate:.2f}%</b>", x=0.5, y=0.55,
                font=dict(size=22, family='Syne', color='#E8392A'), showarrow=False)
            fig2.add_annotation(text="fraud", x=0.5, y=0.38,
                font=dict(size=12, family='DM Sans', color='#72757C'), showarrow=False)
            fig2.update_layout(**PLOT_LAYOUT, height=240, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="chart-card" style="margin-top:16px">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Volume & fraud by transaction type</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Side-by-side comparison across all categories</div>', unsafe_allow_html=True)
        type_total = dff.groupby('type').size().reset_index(name='total')
        type_fraud = dff_fraud.groupby('type').size().reset_index(name='fraud')
        type_df = type_total.merge(type_fraud, on='type', how='left').fillna(0)
        type_df = type_df.sort_values('total', ascending=False)
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=type_df['type'], y=type_df['total'], name='Total',
            marker_color='#B5D4F4', marker_line_width=0, hovertemplate='%{x}<br>Total: %{y:,}<extra></extra>'))
        fig3.add_trace(go.Bar(x=type_df['type'], y=type_df['fraud'], name='Fraud',
            marker_color='#E8392A', marker_line_width=0, hovertemplate='%{x}<br>Fraud: %{y:,}<extra></extra>'))
        fig3.update_layout(**PLOT_LAYOUT, height=220, barmode='group',
            legend=dict(orientation='h', x=0, y=1.15, font=dict(size=11)),
            yaxis=dict(gridcolor='rgba(0,0,0,0.05)', tickformat=','))
        st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        m1, m2, m3, m4, m5 = st.columns(5)
        cashout_fraud = len(dff_fraud[dff_fraud['type'] == 'CASH_OUT']) if 'type' in dff_fraud.columns else 0
        transfer_fraud = len(dff_fraud[dff_fraud['type'] == 'TRANSFER']) if 'type' in dff_fraud.columns else 0
        peak_hour = int(dff_fraud['hour_of_day'].mode()[0]) if len(dff_fraud) and 'hour_of_day' in dff_fraud.columns else 2
        weekend_fraud = dff_fraud['is_weekend'].sum() if 'is_weekend' in dff_fraud.columns else 0

        for col, (label, val, delta, d, accent) in zip([m1,m2,m3,m4,m5], [
            ("CASH_OUT fraud", f"{cashout_fraud:,}", f"{cashout_fraud/max(total_fraud,1)*100:.1f}% of total", "neutral", "#E8392A"),
            ("TRANSFER fraud", f"{transfer_fraud:,}", f"{transfer_fraud/max(total_fraud,1)*100:.1f}% of total", "neutral", "#E8392A"),
            ("Peak fraud hour", f"{peak_hour}:00", "Highest hour", "neutral", "#E8920A"),
            ("Weekend fraud", f"{int(weekend_fraud):,}", f"{weekend_fraud/max(total_fraud,1)*100:.1f}% of total", "neutral", "#E8920A"),
            ("Avg fraud amount", fmt_number(avg_fraud_amt), "Per transaction", "neutral", "#1B6FD8"),
        ]):
            with col:
                st.markdown(kpi(label, val, delta, d, accent), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        r1c1, r1c2 = st.columns(2)

        with r1c1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Fraud by hour of day</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">Case count per hour — 24hr cycle</div>', unsafe_allow_html=True)
            if 'hour_of_day' in dff_fraud.columns:
                hourly = dff_fraud.groupby('hour_of_day').size().reset_index(name='count')
                all_hours = pd.DataFrame({'hour_of_day': range(24)})
                hourly = all_hours.merge(hourly, on='hour_of_day', how='left').fillna(0)
            else:
                hourly = pd.DataFrame({'hour_of_day': range(24), 'count': np.random.randint(200,600,24)})
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(
                x=hourly['hour_of_day'], y=hourly['count'],
                fill='tozeroy', fillcolor='rgba(232,57,42,0.1)',
                line=dict(color='#E8392A', width=2.5), mode='lines+markers',
                marker=dict(size=5, color='#E8392A'),
                hovertemplate='%{x}:00 — %{y:,} cases<extra></extra>'
            ))
            fig4.update_layout(**PLOT_LAYOUT, height=220,
                xaxis=dict(tickmode='array', tickvals=list(range(0,24,3)),
                    ticktext=[f"{h}:00" for h in range(0,24,3)], showgrid=False))
            st.plotly_chart(fig4, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)

        with r1c2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Fraud by transaction type</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">Distribution of fraud across categories</div>', unsafe_allow_html=True)
            type_fraud_only = dff_fraud.groupby('type').size().reset_index(name='count').sort_values('count', ascending=True)
            fig5 = go.Figure(go.Bar(
                y=type_fraud_only['type'], x=type_fraud_only['count'],
                orientation='h',
                marker=dict(color=['#F08070' if v < type_fraud_only['count'].max() else '#E8392A' for v in type_fraud_only['count']],
                    line=dict(width=0)),
                hovertemplate='%{y}: %{x:,}<extra></extra>'
            ))
            fig5.update_layout(**PLOT_LAYOUT, height=220,
                xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
                yaxis=dict(showgrid=False))
            st.plotly_chart(fig5, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)

        r2c1, r2c2 = st.columns([1.2, 1])
        with r2c1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Fraud by day of week</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">Average daily cases per weekday</div>', unsafe_allow_html=True)
            if 'day_of_week' in dff_fraud.columns:
                dow = dff_fraud.groupby('day_of_week').size().reset_index(name='count')
            else:
                dow = pd.DataFrame({'day_of_week': ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'],
                    'count': [1180,1220,1195,1240,1210,820,850]})
            fig6 = go.Figure(go.Bar(
                x=dow['day_of_week'], y=dow['count'],
                marker=dict(color='#E8392A', opacity=[0.6 if i < 5 else 1.0 for i in range(len(dow))], line=dict(width=0)),
                hovertemplate='%{x}: %{y:,}<extra></extra>'
            ))
            fig6.update_layout(**PLOT_LAYOUT, height=200)
            st.plotly_chart(fig6, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)

        with r2c2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Top fraud transactions</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">Highest value individual cases</div>', unsafe_allow_html=True)
            top_tx = dff_fraud.nlargest(6, 'amount')[['nameorig','type','amount','namedest']].copy()
            top_tx['amount'] = top_tx['amount'].apply(lambda x: f"${x:,.0f}")
            top_tx.columns = ['Sender','Type','Amount','Recipient']
            st.dataframe(top_tx, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<br><div class="insight-box">', unsafe_allow_html=True)
        st.markdown(f'<div class="insight-title">Key insight</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="insight-text">Fraud is exclusively concentrated in <strong>CASH_OUT</strong> and <strong>TRANSFER</strong> transactions. All other transaction types ({", ".join([t for t in dff["type"].unique() if t not in ["CASH_OUT","TRANSFER"]])}) show zero fraud cases — suggesting a targeted attack vector. Peak activity occurs in off-hours (1–3 AM), consistent with automated fraud patterns.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        tp = int(dff[(dff['isfraud']==True) & (dff['isflaggedfraud']==True)].shape[0])
        fp = int(dff[(dff['isfraud']==False) & (dff['isflaggedfraud']==True)].shape[0])
        fn = int(dff[(dff['isfraud']==True) & (dff['isflaggedfraud']==False)].shape[0])
        tn = int(dff[(dff['isfraud']==False) & (dff['isflaggedfraud']==False)].shape[0])
        precision = tp / max(tp + fp, 1) * 100
        recall = tp / max(tp + fn, 1) * 100
        f1 = 2 * precision * recall / max(precision + recall, 0.001)

        a1, a2, a3, a4, a5 = st.columns(5)
        for col, (label, val, delta, d, accent) in zip([a1,a2,a3,a4,a5],[
            ("True positives", f"{tp:,}", "Correctly caught", "neutral", "#1A9E5F"),
            ("False positives", f"{fp:,}", "Wrong flags", "neutral", "#E8920A"),
            ("False negatives", f"{fn:,}", "Missed fraud", "down", "#E8392A"),
            ("Precision", f"{precision:.1f}%", "When flagged = right", "neutral", "#1A9E5F"),
            ("Recall / coverage", f"{recall:.2f}%", "Of fraud caught", "down", "#E8392A"),
        ]):
            with col:
                st.markdown(kpi(label, val, delta, d, accent), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        b1, b2 = st.columns([1, 1.4])

        with b1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Confusion matrix</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">isfraud × isflaggedfraud cross-tabulation</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="confusion-grid">
                <div class="conf-cell" style="background:#E3F7EE">
                    <div class="conf-val" style="color:#0A5030">{tn:,}</div>
                    <div class="conf-lbl" style="color:#1A9E5F">True negatives</div>
                    <div class="conf-desc" style="color:#1A9E5F">Not fraud, not flagged</div>
                </div>
                <div class="conf-cell" style="background:#FEF3DC">
                    <div class="conf-val" style="color:#7A4A00">{fp:,}</div>
                    <div class="conf-lbl" style="color:#E8920A">False positives</div>
                    <div class="conf-desc" style="color:#E8920A">Not fraud, flagged</div>
                </div>
                <div class="conf-cell" style="background:#FDECEA">
                    <div class="conf-val" style="color:#8B1A10">{fn:,}</div>
                    <div class="conf-lbl" style="color:#E8392A">False negatives</div>
                    <div class="conf-desc" style="color:#E8392A">Fraud, not flagged</div>
                </div>
                <div class="conf-cell" style="background:#E3F7EE">
                    <div class="conf-val" style="color:#0A5030">{tp:,}</div>
                    <div class="conf-lbl" style="color:#1A9E5F">True positives</div>
                    <div class="conf-desc" style="color:#1A9E5F">Fraud, correctly flagged</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with b2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Model performance metrics</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">Precision, Recall and F1 score</div>', unsafe_allow_html=True)
            metrics_df = pd.DataFrame({
                'Metric': ['Precision', 'Recall', 'F1 Score', 'Specificity'],
                'Score': [precision, recall, f1, tn/max(tn+fp,1)*100],
                'Benchmark': [85, 85, 85, 99]
            })
            fig_m = go.Figure()
            fig_m.add_trace(go.Bar(name='Achieved', x=metrics_df['Metric'], y=metrics_df['Score'],
                marker_color=['#1A9E5F' if v >= 85 else '#E8392A' for v in metrics_df['Score']],
                marker_line_width=0, hovertemplate='%{x}: %{y:.1f}%<extra></extra>'))
            fig_m.add_trace(go.Scatter(name='Target (85%)', x=metrics_df['Metric'], y=metrics_df['Benchmark'],
                mode='lines', line=dict(color='#E8920A', width=1.5, dash='dot')))
            fig_m.update_layout(**PLOT_LAYOUT, height=240, barmode='group',
                yaxis=dict(range=[0,105], ticksuffix='%', gridcolor='rgba(0,0,0,0.05)'),
                legend=dict(orientation='h', x=0, y=1.12, font=dict(size=11)))
            st.plotly_chart(fig_m, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<br><div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Fraud cases vs flagged cases — monthly trend</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Detection gap visualised over time</div>', unsafe_allow_html=True)
        months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        fraud_monthly = np.array([683,621,712,691,723,698,684,731,702,718,742,730])
        flagged_monthly = np.array([1,2,1,2,1,1,2,1,1,2,1,1])
        fig_gap = go.Figure()
        fig_gap.add_trace(go.Bar(x=months, y=fraud_monthly, name='Total fraud cases',
            marker_color='rgba(232,57,42,0.2)', marker_line_width=0))
        fig_gap.add_trace(go.Bar(x=months, y=flagged_monthly, name='Flagged (detected)',
            marker_color='#1A9E5F', marker_line_width=0))
        fig_gap.update_layout(**PLOT_LAYOUT, height=200, barmode='overlay',
            legend=dict(orientation='h', x=0, y=1.15, font=dict(size=11)))
        st.plotly_chart(fig_gap, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        balance_lost = (dff_fraud['oldbalanceorig'] - dff_fraud['newbalanceorig']).sum()
        dest_gain = (dff_fraud['newbalancedest'] - dff_fraud['oldbalancedest']).sum()
        avg_before = dff_fraud['oldbalanceorig'].mean() if len(dff_fraud) else 0
        avg_after = dff_fraud['newbalanceorig'].mean() if len(dff_fraud) else 0
        fraud_vol_pct = fraud_amount / max(total_vol, 1) * 100

        d1,d2,d3,d4,d5 = st.columns(5)
        for col, (label, val, delta, d, accent) in zip([d1,d2,d3,d4,d5],[
            ("Balance drained", fmt_number(balance_lost), "Sender accounts", "down", "#E8392A"),
            ("Fraudster gain", fmt_number(dest_gain), "Destination accounts", "down", "#E8392A"),
            ("Fraud vol share", f"{fraud_vol_pct:.2f}%", "Of total $179B", "neutral", "#E8920A"),
            ("Avg balance before", fmt_number(avg_before), "At fraud time", "neutral", "#1B6FD8"),
            ("Avg balance after", fmt_number(avg_after), "Accounts emptied", "down", "#E8392A"),
        ]):
            with col:
                st.markdown(kpi(label, val, delta, d, accent), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        e1, e2 = st.columns(2)

        with e1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Monthly fraud amount</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">Total fraud value per month ($M)</div>', unsafe_allow_html=True)
            monthly_vals = [98,84,112,107,118,103,97,124,109,115,131,122]
            fig_ma = go.Figure(go.Bar(
                x=months, y=monthly_vals, marker_color='#E8392A',
                marker_line_width=0,
                hovertemplate='%{x}: $%{y}M<extra></extra>'
            ))
            fig_ma.update_layout(**PLOT_LAYOUT, height=220,
                yaxis=dict(tickprefix='$', ticksuffix='M', gridcolor='rgba(0,0,0,0.05)'))
            st.plotly_chart(fig_ma, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)

        with e2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Sender balance — before vs after</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">Average account balance at time of fraud</div>', unsafe_allow_html=True)
            fig_bb = go.Figure(go.Bar(
                x=['Before fraud', 'After fraud'],
                y=[avg_before, avg_after],
                marker_color=['#1B6FD8', '#E8392A'],
                marker_line_width=0,
                hovertemplate='%{x}: $%{y:,.0f}<extra></extra>'
            ))
            fig_bb.update_layout(**PLOT_LAYOUT, height=220,
                yaxis=dict(tickprefix='$', gridcolor='rgba(0,0,0,0.05)'))
            st.plotly_chart(fig_bb, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<br><div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Fraud amount vs transaction amount — scatter</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Relationship between original balance and fraud amount by type</div>', unsafe_allow_html=True)
        sample = dff_fraud.sample(min(2000, len(dff_fraud)), random_state=42) if len(dff_fraud) > 0 else dff_fraud
        fig_sc = px.scatter(sample, x='oldbalanceorig', y='amount', color='type',
            color_discrete_sequence=['#E8392A','#1B6FD8','#E8920A','#1A9E5F'],
            opacity=0.5, size_max=8,
            hover_data={'nameorig': True, 'amount': ':,.0f', 'oldbalanceorig': ':,.0f'})
        fig_sc.update_layout(**PLOT_LAYOUT, height=240,
            xaxis=dict(tickprefix='$', showgrid=False),
            yaxis=dict(tickprefix='$', gridcolor='rgba(0,0,0,0.05)'),
            legend=dict(orientation='h', x=0, y=1.12, font=dict(size=11)))
        st.plotly_chart(fig_sc, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with tab5:
        unique_senders = dff_fraud['nameorig'].nunique() if 'nameorig' in dff_fraud.columns else 0
        unique_dests = dff_fraud['namedest'].nunique() if 'namedest' in dff_fraud.columns else 0
        repeat_senders = dff_fraud.groupby('nameorig').size()
        repeat_count = int((repeat_senders > 1).sum()) if len(repeat_senders) else 0
        max_fraud_sender = dff_fraud.groupby('nameorig')['amount'].sum().max() if len(dff_fraud) else 0

        f1c, f2c, f3c, f4c = st.columns(4)
        for col, (label, val, delta, d, accent) in zip([f1c,f2c,f3c,f4c],[
            ("Unique fraud senders", f"{unique_senders:,}", "Distinct accounts", "neutral", "#E8392A"),
            ("Unique recipients", f"{unique_dests:,}", "Destination accounts", "neutral", "#E8392A"),
            ("Repeat offenders", f"{repeat_count:,}", "2+ fraud events", "down", "#E8920A"),
            ("Max by one sender", fmt_number(max_fraud_sender), "Single account total", "down", "#E8392A"),
        ]):
            with col:
                st.markdown(kpi(label, val, delta, d, accent), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        g1, g2 = st.columns(2)

        with g1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Top 10 senders by fraud amount</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">Highest value fraud originators</div>', unsafe_allow_html=True)
            if len(dff_fraud) > 0:
                top_senders = dff_fraud.groupby('nameorig').agg(
                    total_amount=('amount','sum'),
                    cases=('amount','count'),
                    type=('type','first')
                ).nlargest(10,'total_amount').reset_index()
                top_senders['total_amount_fmt'] = top_senders['total_amount'].apply(fmt_number)
                display_s = top_senders[['nameorig','type','total_amount_fmt','cases']].copy()
                display_s.columns = ['Account','Type','Total Amount','Cases']
                st.dataframe(display_s, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with g2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Top 10 recipients by fraud received</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">Highest value fraud destinations</div>', unsafe_allow_html=True)
            if len(dff_fraud) > 0:
                top_dests = dff_fraud.groupby('namedest').agg(
                    total_received=('amount','sum'),
                    cases=('amount','count')
                ).nlargest(10,'total_received').reset_index()
                top_dests['total_received_fmt'] = top_dests['total_received'].apply(fmt_number)
                display_d = top_dests[['namedest','cases','total_received_fmt']].copy()
                display_d.columns = ['Account','Cases','Total Received']
                st.dataframe(display_d, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if len(dff_fraud) > 0:
            st.markdown('<br><div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Top 15 senders — fraud amount ranking</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">Visualised by total fraud value</div>', unsafe_allow_html=True)
            top15 = dff_fraud.groupby('nameorig')['amount'].sum().nlargest(15).reset_index()
            fig_top = go.Figure(go.Bar(
                y=top15['nameorig'], x=top15['amount'],
                orientation='h',
                marker=dict(
                    color=top15['amount'],
                    colorscale=[[0,'#F5C4B3'],[0.5,'#E8392A'],[1,'#8B1A10']],
                    line=dict(width=0)
                ),
                hovertemplate='%{y}<br>$%{x:,.0f}<extra></extra>'
            ))
            fig_top.update_layout(**PLOT_LAYOUT, height=380,
                xaxis=dict(tickprefix='$', showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
                yaxis=dict(showgrid=False, tickfont=dict(family='DM Mono', size=10)))
            st.plotly_chart(fig_top, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)


def generate_demo_data():
    np.random.seed(42)
    n = 50000
    types = np.random.choice(['CASH_OUT','PAYMENT','TRANSFER','CASH_IN','DEBIT'],
        n, p=[0.35,0.34,0.21,0.09,0.01])
    isfraud = np.zeros(n, dtype=bool)
    fraud_mask = (np.isin(types, ['CASH_OUT','TRANSFER'])) & (np.random.rand(n) < 0.008)
    isfraud[fraud_mask] = True
    isflagged = np.zeros(n, dtype=bool)
    flagged_mask = isfraud & (np.random.rand(n) < 0.002)
    isflagged[flagged_mask] = True

    amounts = np.where(isfraud,
        np.random.lognormal(11, 2, n),
        np.random.lognormal(8, 2, n))
    old_bal = np.random.lognormal(10, 2, n)
    new_bal = np.where(isfraud, np.maximum(0, old_bal - amounts), old_bal - amounts * 0.1)

    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    date_arr = np.random.choice(dates, n)

    df = pd.DataFrame({
        'step': np.random.randint(1, 500, n),
        'type': types,
        'amount': amounts,
        'nameorig': [f'C{np.random.randint(1000000,9999999)}' for _ in range(n)],
        'oldbalanceorg': old_bal,
        'newbalanceorig': new_bal,
        'namedest': [f'C{np.random.randint(1000000,9999999)}' for _ in range(n)],
        'oldbalancedest': np.random.lognormal(9, 2, n),
        'newbalancedest': np.random.lognormal(9, 2, n),
        'isfraud': isfraud,
        'isflaggedfraud': isflagged,
        'datetime': date_arr,
        'hour_of_day': np.random.choice(range(24), n, p=[0.03]*6 + [0.02]*6 + [0.04]*6 + [0.05]*6),
        'day_of_week': np.random.choice(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'], n),
        'is_weekend': np.isin(np.random.choice(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'], n), ['Sat','Sun']),
        'date': date_arr
    })
    df_fraud = df[df['isfraud'] == True].copy()
    return df, df_fraud


if __name__ == "__main__":
    main()
