import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PCOS Analytics Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #f8f4f9; }
    .stApp { background-color: #f8f4f9; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #6b2d8b 0%, #9b59b6 100%);
    }
    section[data-testid="stSidebar"] * { color: white !important; }
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 2px 12px rgba(107,45,139,0.08);
        border-left: 5px solid #9b59b6;
        margin-bottom: 16px;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #6b2d8b; line-height: 1; }
    .metric-label { font-size: 0.85rem; color: #888; margin-top: 6px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-sub { font-size: 0.8rem; color: #bbb; margin-top: 4px; }
    .section-header { font-size: 1.4rem; font-weight: 700; color: #6b2d8b; margin: 24px 0 16px 0; padding-bottom: 8px; border-bottom: 2px solid #e8d5f5; }
    .info-box { background: linear-gradient(135deg, #f3e8ff, #fdf2ff); border-radius: 12px; padding: 20px; border: 1px solid #e0c4f5; margin: 12px 0; }
    .warning-box { background: linear-gradient(135deg, #fff3cd, #fff8e1); border-radius: 12px; padding: 20px; border: 1px solid #ffd54f; margin: 12px 0; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    div[data-testid="stDecoration"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ─── PASSWORD ──────────────────────────────────────────────────────────────────
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="background:white;border-radius:20px;padding:48px;box-shadow:0 8px 32px rgba(107,45,139,0.15);text-align:center;margin-top:80px;">
                <div style="font-size:3rem;margin-bottom:12px;">🩺</div>
                <div style="font-size:2rem;font-weight:700;color:#6b2d8b;margin-bottom:8px;">PCOS Analytics</div>
                <div style="color:#888;margin-bottom:32px;font-size:0.95rem;">MENA Region Healthcare Dashboard<br>MSBA382 – Healthcare Analytics</div>
            </div>
            """, unsafe_allow_html=True)
            password = st.text_input("Enter Dashboard Password", type="password", placeholder="Enter password...")
            if st.button("🔓 Access Dashboard", use_container_width=True):
                if password == "pcos2026":
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Incorrect password.")
            st.markdown("<p style='text-align:center;color:#ccc;font-size:0.75rem;margin-top:16px;'>Consultant Access Only</p>", unsafe_allow_html=True)
        return False
    return True

# ─── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_clinical():
    df = pd.read_excel("PCOS_data_without_infertility.xlsx", sheet_name="Full_new")
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        'PCOS (Y/N)':'PCOS',' Age (yrs)':'Age','Weight (Kg)':'Weight',
        'Height(Cm)':'Height','BMI':'BMI','Pulse rate(bpm)':'Pulse',
        'RR (breaths/min)':'RR','Hb(g/dl)':'Hb','Cycle(R/I)':'Cycle',
        'Cycle length(days)':'CycleLength','Marraige Status (Yrs)':'MarriageYrs',
        'Pregnant(Y/N)':'Pregnant','No. of aborptions':'Abortions',
        'I   beta-HCG(mIU/mL)':'betaHCG1','II    beta-HCG(mIU/mL)':'betaHCG2',
        'FSH(mIU/mL)':'FSH','LH(mIU/mL)':'LH','FSH/LH':'FSH_LH',
        'Hip(inch)':'Hip','Waist(inch)':'Waist','Waist:Hip Ratio':'WaistHip',
        'TSH (mIU/L)':'TSH','AMH(ng/mL)':'AMH','PRL(ng/mL)':'PRL',
        'Vit D3 (ng/mL)':'VitD3','PRG(ng/mL)':'PRG','RBS(mg/dl)':'RBS',
        'Weight gain(Y/N)':'WeightGain','hair growth(Y/N)':'HairGrowth',
        'Skin darkening (Y/N)':'SkinDarkening','Hair loss(Y/N)':'HairLoss',
        'Pimples(Y/N)':'Pimples','Fast food (Y/N)':'FastFood',
        'Reg.Exercise(Y/N)':'Exercise','BP _Systolic (mmHg)':'BPSystolic',
        'BP _Diastolic (mmHg)':'BPDiastolic','Follicle No. (L)':'FollicleL',
        'Follicle No. (R)':'FollicleR','Avg. F size (L) (mm)':'FollicleSizeL',
        'Avg. F size (R) (mm)':'FollicleSizeR','Endometrium (mm)':'Endometrium'
    })
    df['AMH'] = pd.to_numeric(df['AMH'], errors='coerce')
    df['betaHCG2'] = pd.to_numeric(df['betaHCG2'], errors='coerce')
    df['PCOS_Label'] = df['PCOS'].map({1:'PCOS Positive', 0:'PCOS Negative'})
    df['BMI_Category'] = pd.cut(df['BMI'], bins=[0,18.5,24.9,29.9,100],
        labels=['Underweight','Normal','Overweight','Obese'])
    df['Age_Group'] = pd.cut(df['Age'], bins=[18,24,29,34,39,50],
        labels=['20–24','25–29','30–34','35–39','40+'])
    return df

@st.cache_data
def load_gbd():
    df = pd.read_csv("IHME-GBD_2023_DATA-f76e773e-1.csv")
    # Standardize country names
    df['location_name'] = df['location_name'].replace({
        'Iran (Islamic Republic of)': 'Iran',
        'Syrian Arab Republic': 'Syria',
        'Türkiye': 'Turkey'
    })
    return df

@st.cache_data
def load_obesity():
    df = pd.read_csv("share-of-females-defined-as-obese.csv")
    df.columns = ['Entity', 'Code', 'Year', 'Obesity_Pct']
    df['Entity'] = df['Entity'].replace({
        'Iran': 'Iran',
        'Syria': 'Syria',
        'Turkey': 'Turkey'
    })
    return df

ISO_MAP = {
    'Afghanistan':'AFG','Algeria':'DZA','Bahrain':'BHR','Egypt':'EGY',
    'Iran':'IRN','Iraq':'IRQ','Jordan':'JOR','Kuwait':'KWT','Lebanon':'LBN',
    'Libya':'LBY','Morocco':'MAR','Oman':'OMN','Palestine':'PSE','Qatar':'QAT',
    'Saudi Arabia':'SAU','Sudan':'SDN','Syria':'SYR','Tunisia':'TUN',
    'Turkey':'TUR','United Arab Emirates':'ARE','Yemen':'YEM'
}

MENA_COUNTRIES = list(ISO_MAP.keys())

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("### 🩺 PCOS Analytics")
        st.markdown("**MENA Region Dashboard**")
        st.markdown("---")
        page = st.radio("Navigate", [
            "🏠 Overview",
            "🗺️ MENA Burden Map",
            "📈 Trends Over Time",
            "⚖️ Obesity & PCOS Link",
            "👩 Patient Profiles",
            "🧬 Hormonal Analysis",
            "⚕️ Symptom Explorer",
            "🤖 PCOS Predictor"
        ])
        st.markdown("---")
        st.markdown("**Data Sources**")
        st.markdown("📊 IHME GBD 2023 (6,426 rows)")
        st.markdown("📊 WHO Obesity Data (9,270 rows)")
        st.markdown("📊 Clinical Dataset (541 patients)")
        st.markdown("---")
        if st.button("🔒 Logout"):
            st.session_state.authenticated = False
            st.rerun()
    return page

# ─── PAGE 1: OVERVIEW ──────────────────────────────────────────────────────────
def page_overview(clinical, gbd):
    st.markdown("## 🏠 PCOS in the MENA Region — Overview")
    st.markdown("*Polycystic Ovary Syndrome (PCOS) is the most common endocrine disorder in women of reproductive age. This dashboard integrates clinical patient data with official IHME Global Burden of Disease 2023 statistics.*")

    # KPI from real GBD data
    prev_2023 = gbd[(gbd['measure_name']=='Prevalence') & (gbd['metric_name']=='Rate') & (gbd['year']==2023)]
    prev_1990 = gbd[(gbd['measure_name']=='Prevalence') & (gbd['metric_name']=='Rate') & (gbd['year']==1990)]
    avg_prev_2023 = prev_2023['val'].mean()
    avg_prev_1990 = prev_1990['val'].mean()
    pct_change = ((avg_prev_2023 - avg_prev_1990) / avg_prev_1990) * 100
    highest = prev_2023.loc[prev_2023['val'].idxmax(), 'location_name']
    highest_val = prev_2023['val'].max()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{avg_prev_2023:,.0f}</div>
            <div class="metric-label">Avg Prevalence Rate</div>
            <div class="metric-sub">Per 100,000 women (2023)</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">+{pct_change:.1f}%</div>
            <div class="metric-label">Rise Since 1990</div>
            <div class="metric-sub">IHME GBD 2023</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{highest}</div>
            <div class="metric-label">Highest Burden Country</div>
            <div class="metric-sub">{highest_val:,.0f} per 100,000</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">70%</div>
            <div class="metric-label">Remain Undiagnosed</div>
            <div class="metric-sub">WHO Estimate</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">PCOS Prevalence Rate by Country (2023)</div>', unsafe_allow_html=True)
        sorted_prev = prev_2023[prev_2023['location_name'].isin(MENA_COUNTRIES)].sort_values('val', ascending=True)
        fig = px.bar(sorted_prev, x='val', y='location_name', orientation='h',
                     color='val', color_continuous_scale=['#e8d5f5','#6b2d8b'],
                     labels={'val':'Prevalence Rate per 100,000','location_name':'Country'})
        fig.update_layout(paper_bgcolor='white', plot_bgcolor='white',
                          font=dict(family='Inter'), showlegend=False,
                          coloraxis_showscale=False, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">PCOS Burden: DALYs vs Prevalence (2023)</div>', unsafe_allow_html=True)
        dalys_2023 = gbd[(gbd['measure_name']=='DALYs (Disability-Adjusted Life Years)') &
                         (gbd['metric_name']=='Rate') & (gbd['year']==2023)]
        merged = prev_2023.merge(dalys_2023[['location_name','val']], on='location_name', suffixes=('_prev','_dalys'))
        fig2 = px.scatter(merged, x='val_prev', y='val_dalys', text='location_name',
                          color='val_prev', color_continuous_scale=['#e8d5f5','#6b2d8b'],
                          labels={'val_prev':'Prevalence Rate','val_dalys':'DALYs Rate'},
                          size='val_prev', size_max=30)
        fig2.update_traces(textposition='top center', textfont_size=9)
        fig2.update_layout(paper_bgcolor='white', plot_bgcolor='white',
                           font=dict(family='Inter'), showlegend=False,
                           coloraxis_showscale=False, margin=dict(t=10))
        st.plotly_chart(fig2, use_container_width=True)

    # Clinical dataset summary
    st.markdown("---")
    st.markdown('<div class="section-header">📋 Clinical Dataset Summary (541 Patients)</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    pcos_pct = round(clinical['PCOS'].mean()*100,1)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">541</div>
            <div class="metric-label">Total Patients</div>
            <div class="metric-sub">Clinical Records</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{pcos_pct}%</div>
            <div class="metric-label">PCOS Positive</div>
            <div class="metric-sub">177 patients</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{round(clinical[clinical['PCOS']==1]['Age'].mean(),1)}</div>
            <div class="metric-label">Avg Age (PCOS+)</div>
            <div class="metric-sub">Years</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{round(clinical[clinical['PCOS']==1]['BMI'].mean(),1)}</div>
            <div class="metric-label">Avg BMI (PCOS+)</div>
            <div class="metric-sub">kg/m²</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="info-box">
        📌 <b>Data Sources:</b> This dashboard integrates three official datasets:
        (1) <b>IHME Global Burden of Disease 2023</b> — prevalence, incidence and DALYs for 21 MENA countries from 1990–2023;
        (2) <b>WHO Global Health Observatory</b> via Our World in Data — female obesity rates by country 1980–2024;
        (3) <b>Clinical PCOS Dataset</b> (Kaggle/UCI) — 541 patients with 44 clinical variables for patient-level analysis.
    </div>""", unsafe_allow_html=True)

# ─── PAGE 2: MENA MAP ──────────────────────────────────────────────────────────
def page_map(gbd):
    st.markdown("## 🗺️ PCOS Burden Map — MENA Region")
    st.markdown("*Real IHME Global Burden of Disease 2023 data. Select measure, metric and year.*")

    col1, col2, col3 = st.columns(3)
    with col1:
        measure = st.selectbox("Measure", ['Prevalence','Incidence','DALYs (Disability-Adjusted Life Years)'],
                               format_func=lambda x: x.split('(')[0].strip())
    with col2:
        metric = st.selectbox("Metric", ['Rate','Number','Percent'])
    with col3:
        year = st.slider("Year", 1990, 2023, 2023)

    filtered = gbd[(gbd['measure_name']==measure) & (gbd['metric_name']==metric) & (gbd['year']==year)]
    filtered = filtered.copy()
    filtered['ISO'] = filtered['location_name'].map(ISO_MAP)

    label = f"{measure.split('(')[0].strip()} ({metric})"

    fig = px.choropleth(filtered, locations='ISO', color='val',
                        hover_name='location_name',
                        hover_data={'val':':.2f','ISO':False},
                        color_continuous_scale=['#f3e8ff','#9b59b6','#4a1a6e'],
                        title=f'PCOS {label} — MENA Region ({year})',
                        labels={'val': label})
    fig.update_geos(scope='world', center=dict(lat=26, lon=45),
                    projection_scale=3.5, showland=True, landcolor='#f9f9f9',
                    showocean=True, oceancolor='#e8f4fd', showframe=False)
    fig.update_layout(height=500, paper_bgcolor='white', font=dict(family='Inter'),
                      margin=dict(t=50,b=10,l=10,r=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">📋 Country Data Table</div>', unsafe_allow_html=True)
    table = filtered[['location_name','val','upper','lower']].sort_values('val', ascending=False).copy()
    table.columns = ['Country', f'{label}', 'Upper (95% UI)', 'Lower (95% UI)']
    table = table.reset_index(drop=True)
    table[f'{label}'] = table[f'{label}'].round(2)
    table['Upper (95% UI)'] = table['Upper (95% UI)'].round(2)
    table['Lower (95% UI)'] = table['Lower (95% UI)'].round(2)
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.markdown("""<div class="warning-box">
        📌 <b>Source:</b> Institute for Health Metrics and Evaluation (IHME). GBD Results. Seattle, WA: IHME, University of Washington, 2025.
        Available from vizhub.healthdata.org/gbd-results/. Values shown with 95% Uncertainty Intervals.
    </div>""", unsafe_allow_html=True)

# ─── PAGE 3: TRENDS ────────────────────────────────────────────────────────────
def page_trends(gbd):
    st.markdown("## 📈 PCOS Burden Trends 1990–2023")
    st.markdown("*Track how PCOS burden has evolved across MENA countries over 34 years using official IHME data.*")

    col1, col2 = st.columns(2)
    with col1:
        measure = st.selectbox("Measure", ['Prevalence','Incidence','DALYs (Disability-Adjusted Life Years)'],
                               format_func=lambda x: x.split('(')[0].strip(), key='trend_measure')
    with col2:
        countries = st.multiselect("Select Countries", MENA_COUNTRIES,
                                    default=['Qatar','Kuwait','Saudi Arabia','Egypt','Lebanon','Yemen'])

    filtered = gbd[(gbd['measure_name']==measure) &
                   (gbd['metric_name']=='Rate') &
                   (gbd['location_name'].isin(countries))]

    fig = px.line(filtered, x='year', y='val', color='location_name',
                  markers=False,
                  labels={'val':f'{measure.split("(")[0].strip()} Rate per 100,000','year':'Year','location_name':'Country'},
                  title=f'PCOS {measure.split("(")[0].strip()} Rate Trends 1990–2023')
    fig.update_layout(paper_bgcolor='white', plot_bgcolor='white',
                      font=dict(family='Inter'), legend_title='Country',
                      margin=dict(t=50), height=450)
    fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
    fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
    st.plotly_chart(fig, use_container_width=True)

    # % change table
    st.markdown("---")
    st.markdown('<div class="section-header">📊 Total Change 1990–2023 by Country</div>', unsafe_allow_html=True)
    all_rate = gbd[(gbd['measure_name']==measure) & (gbd['metric_name']=='Rate')]
    y1990 = all_rate[all_rate['year']==1990][['location_name','val']].rename(columns={'val':'val_1990'})
    y2023 = all_rate[all_rate['year']==2023][['location_name','val']].rename(columns={'val':'val_2023'})
    change = y1990.merge(y2023, on='location_name')
    change['Change (%)'] = ((change['val_2023'] - change['val_1990']) / change['val_1990'] * 100).round(1)
    change = change.sort_values('Change (%)', ascending=False)
    change.columns = ['Country', '1990 Rate', '2023 Rate', 'Change (%)']

    fig2 = px.bar(change, x='Change (%)', y='Country', orientation='h',
                  color='Change (%)', color_continuous_scale=['#e8d5f5','#6b2d8b'],
                  title=f'% Change in PCOS {measure.split("(")[0].strip()} Rate (1990–2023)')
    fig2.update_layout(paper_bgcolor='white', plot_bgcolor='white',
                       font=dict(family='Inter'), showlegend=False,
                       coloraxis_showscale=False, margin=dict(t=50), height=500)
    st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(change.reset_index(drop=True), use_container_width=True, hide_index=True)

# ─── PAGE 4: OBESITY & PCOS ────────────────────────────────────────────────────
def page_obesity(gbd, obesity):
    st.markdown("## ⚖️ Obesity & PCOS — The MENA Connection")
    st.markdown("*Obesity is a major risk factor for PCOS. This page cross-analyses WHO female obesity rates with IHME PCOS burden data across the MENA region.*")

    year = st.slider("Select Year", 1990, 2022, 2022)

    # Merge datasets
    pcos_yr = gbd[(gbd['measure_name']=='Prevalence') &
                  (gbd['metric_name']=='Rate') &
                  (gbd['year']==year)][['location_name','val']].rename(columns={'val':'PCOS_Rate'})

    obs_yr = obesity[obesity['Year']==year][['Entity','Obesity_Pct']].rename(columns={'Entity':'location_name'})
    obs_yr['location_name'] = obs_yr['location_name'].replace({
        'Iran':'Iran','Syria':'Syria','Turkey':'Turkey'
    })

    merged = pcos_yr.merge(obs_yr, on='location_name').dropna()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">PCOS Prevalence vs Female Obesity Rate</div>', unsafe_allow_html=True)
        fig = px.scatter(merged, x='Obesity_Pct', y='PCOS_Rate',
                         text='location_name', size='PCOS_Rate',
                         color='PCOS_Rate', color_continuous_scale=['#e8d5f5','#6b2d8b'],
                         labels={'Obesity_Pct':'Female Obesity Rate (%)','PCOS_Rate':'PCOS Prevalence Rate per 100,000'},
                         trendline='ols')
        fig.update_traces(textposition='top center', textfont_size=9)
        fig.update_layout(paper_bgcolor='white', plot_bgcolor='white',
                          font=dict(family='Inter'), showlegend=False,
                          coloraxis_showscale=False, margin=dict(t=10), height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Dual Ranking: Obesity vs PCOS Burden</div>', unsafe_allow_html=True)
        merged_sorted = merged.sort_values('Obesity_Pct', ascending=False)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name='Female Obesity (%)', x=merged_sorted['location_name'],
                              y=merged_sorted['Obesity_Pct'], marker_color='#9b59b6', opacity=0.8))
        fig2.add_trace(go.Scatter(name='PCOS Rate (per 100k)', x=merged_sorted['location_name'],
                                  y=merged_sorted['PCOS_Rate']/100, mode='lines+markers',
                                  line=dict(color='#e74c3c', width=2),
                                  yaxis='y2'))
        fig2.update_layout(
            paper_bgcolor='white', plot_bgcolor='white', font=dict(family='Inter'),
            yaxis=dict(title='Female Obesity (%)'),
            yaxis2=dict(title='PCOS Rate / 100 (per 100k)', overlaying='y', side='right'),
            legend=dict(orientation='h', y=1.1),
            xaxis_tickangle=-45, margin=dict(t=20,b=80), height=420
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Trend comparison
    st.markdown("---")
    st.markdown('<div class="section-header">📈 Obesity & PCOS Trends Over Time</div>', unsafe_allow_html=True)
    country = st.selectbox("Select Country", sorted(merged['location_name'].tolist()), index=0)

    pcos_trend = gbd[(gbd['measure_name']=='Prevalence') &
                     (gbd['metric_name']=='Rate') &
                     (gbd['location_name']==country)][['year','val']].rename(columns={'val':'PCOS_Rate'})
    obs_trend = obesity[obesity['Entity']==country][['Year','Obesity_Pct']].rename(columns={'Year':'year'})
    trend = pcos_trend.merge(obs_trend, on='year').dropna()

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=trend['year'], y=trend['PCOS_Rate'], name='PCOS Prevalence Rate',
                              line=dict(color='#9b59b6', width=2), mode='lines'))
    fig3.add_trace(go.Scatter(x=trend['year'], y=trend['Obesity_Pct']*30, name='Female Obesity (scaled)',
                              line=dict(color='#e74c3c', width=2, dash='dash'), mode='lines'))
    fig3.update_layout(
        paper_bgcolor='white', plot_bgcolor='white', font=dict(family='Inter'),
        title=f'PCOS Prevalence vs Female Obesity Over Time — {country}',
        xaxis_title='Year', yaxis_title='Rate',
        legend=dict(orientation='h', y=1.1),
        margin=dict(t=60), height=380
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""<div class="info-box">
        📌 <b>Key Insight:</b> A strong positive correlation exists between female obesity rates and PCOS prevalence 
        across the MENA region. Countries with the highest female obesity rates (Egypt 55.9%, Qatar 49.3%, Kuwait 47%) 
        also show among the highest PCOS burdens. This underscores obesity as a critical modifiable risk factor 
        for PCOS in the region, with major implications for prevention policy.
        <br><br>
        <b>Sources:</b> IHME GBD 2023; WHO Global Health Observatory via Our World in Data (NCD-RisC, Lancet 2024).
    </div>""", unsafe_allow_html=True)

# ─── PAGE 5: PATIENT PROFILES ──────────────────────────────────────────────────
def page_patients(clinical):
    st.markdown("## 👩 Patient Profiles")
    st.markdown("*Clinical-level analysis of 541 patients. Explore how physical characteristics differ between PCOS positive and negative patients.*")

    col1, col2, col3 = st.columns(3)
    with col1:
        age_range = st.slider("Age Range", int(clinical['Age'].min()), int(clinical['Age'].max()), (20, 48))
    with col2:
        bmi_filter = st.multiselect("BMI Category", ['Underweight','Normal','Overweight','Obese'],
                                     default=['Underweight','Normal','Overweight','Obese'])
    with col3:
        pcos_filter = st.multiselect("PCOS Status", ['PCOS Positive','PCOS Negative'],
                                      default=['PCOS Positive','PCOS Negative'])

    df = clinical[
        (clinical['Age'] >= age_range[0]) & (clinical['Age'] <= age_range[1]) &
        (clinical['BMI_Category'].isin(bmi_filter)) &
        (clinical['PCOS_Label'].isin(pcos_filter))
    ]
    st.markdown(f"**Showing {len(df)} patients**")
    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x='BMI', color='PCOS_Label', nbins=30,
                           color_discrete_map={'PCOS Positive':'#9b59b6','PCOS Negative':'#e8d5f5'},
                           barmode='overlay', opacity=0.75, title='BMI Distribution by PCOS Status')
        fig.update_layout(paper_bgcolor='white', plot_bgcolor='white',
                          font=dict(family='Inter'), legend_title='', margin=dict(t=50))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.box(df, x='PCOS_Label', y='WaistHip', color='PCOS_Label',
                      color_discrete_map={'PCOS Positive':'#9b59b6','PCOS Negative':'#e8d5f5'},
                      points='all', title='Waist:Hip Ratio by PCOS Status')
        fig2.update_layout(paper_bgcolor='white', plot_bgcolor='white',
                           font=dict(family='Inter'), showlegend=False, margin=dict(t=50))
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig3 = px.scatter(df, x='Age', y='BMI', color='PCOS_Label',
                          color_discrete_map={'PCOS Positive':'#9b59b6','PCOS Negative':'#c8a8e0'},
                          opacity=0.7, title='Age vs BMI')
        fig3.update_layout(paper_bgcolor='white', plot_bgcolor='white',
                           font=dict(family='Inter'), legend_title='', margin=dict(t=50))
        st.plotly_chart(fig3, use_container_width=True)
    with c4:
        bmi_group = df.groupby(['BMI_Category','PCOS_Label']).size().reset_index(name='Count')
        fig4 = px.bar(bmi_group, x='BMI_Category', y='Count', color='PCOS_Label',
                      barmode='group',
                      color_discrete_map={'PCOS Positive':'#9b59b6','PCOS Negative':'#e8d5f5'},
                      title='BMI Category vs PCOS Status')
        fig4.update_layout(paper_bgcolor='white', plot_bgcolor='white',
                           font=dict(family='Inter'), legend_title='', margin=dict(t=50))
        st.plotly_chart(fig4, use_container_width=True)

    # Lifestyle
    st.markdown("---")
    st.markdown('<div class="section-header">🍔 Lifestyle Factors</div>', unsafe_allow_html=True)
    lc1, lc2 = st.columns(2)
    with lc1:
        lifestyle = df.groupby('PCOS_Label')[['FastFood','Exercise']].mean().reset_index()
        lm = lifestyle.melt(id_vars='PCOS_Label', value_vars=['FastFood','Exercise'],
                            var_name='Factor', value_name='Proportion')
        lm['Factor'] = lm['Factor'].map({'FastFood':'Fast Food','Exercise':'Regular Exercise'})
        fig5 = px.bar(lm, x='Factor', y='Proportion', color='PCOS_Label', barmode='group',
                      color_discrete_map={'PCOS Positive':'#9b59b6','PCOS Negative':'#e8d5f5'})
        fig5.update_layout(paper_bgcolor='white', plot_bgcolor='white',
                           font=dict(family='Inter'), legend_title='', margin=dict(t=20))
        st.plotly_chart(fig5, use_container_width=True)
    with lc2:
        preg = df.groupby(['PCOS_Label','Pregnant']).size().reset_index(name='Count')
        preg['Pregnant'] = preg['Pregnant'].map({1:'Pregnant',0:'Not Pregnant'})
        fig6 = px.bar(preg, x='PCOS_Label', y='Count', color='Pregnant', barmode='stack',
                      color_discrete_sequence=['#9b59b6','#e8d5f5'])
        fig6.update_layout(paper_bgcolor='white', plot_bgcolor='white',
                           font=dict(family='Inter'), xaxis_title='', margin=dict(t=20))
        st.plotly_chart(fig6, use_container_width=True)

# ─── PAGE 6: HORMONAL ANALYSIS ─────────────────────────────────────────────────
def page_hormones(clinical):
    st.markdown("## 🧬 Hormonal Analysis")
    st.markdown("*Hormonal imbalances are the key diagnostic markers of PCOS.*")

    hormones = {'FSH':'FSH (mIU/mL)','LH':'LH (mIU/mL)','AMH':'AMH (ng/mL)',
                'TSH':'TSH (mIU/L)','PRL':'Prolactin (ng/mL)','VitD3':'Vitamin D3 (ng/mL)'}
    selected = st.selectbox("Select Hormone", list(hormones.keys()), format_func=lambda x: hormones[x])

    c1, c2 = st.columns(2)
    with c1:
        fig = px.violin(clinical, x='PCOS_Label', y=selected, color='PCOS_Label', box=True,
                        color_discrete_map={'PCOS Positive':'#9b59b6','PCOS Negative':'#e8d5f5'},
                        title=f'{hormones[selected]} Distribution')
        fig.update_layout(paper_bgcolor='white', plot_bgcolor='white',
                          font=dict(family='Inter'), showlegend=False, margin=dict(t=50))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.histogram(clinical, x=selected, color='PCOS_Label', nbins=40,
                            barmode='overlay', opacity=0.75,
                            color_discrete_map={'PCOS Positive':'#9b59b6','PCOS Negative':'#e8d5f5'},
                            title=f'{hormones[selected]} Frequency Distribution')
        fig2.update_layout(paper_bgcolor='white', plot_bgcolor='white',
                           font=dict(family='Inter'), legend_title='', margin=dict(t=50))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    hormone_cols = ['FSH','LH','FSH_LH','AMH','TSH','PRL','VitD3']
    stats = clinical.groupby('PCOS_Label')[hormone_cols].mean().round(2).T
    st.markdown('<div class="section-header">Hormone Summary Statistics</div>', unsafe_allow_html=True)
    st.dataframe(stats.style.highlight_max(axis=1, color='#e8d5f5'), use_container_width=True)

    st.markdown("---")
    corr_cols = ['FSH','LH','AMH','TSH','PRL','VitD3','BMI','PCOS']
    corr = clinical[corr_cols].dropna().corr().round(2)
    fig3 = px.imshow(corr, text_auto=True, aspect='auto',
                     color_continuous_scale=['#4a1a6e','white','#9b59b6'],
                     title='Correlation Matrix: Hormones & PCOS')
    fig3.update_layout(paper_bgcolor='white', font=dict(family='Inter'),
                       margin=dict(t=50), height=450)
    st.plotly_chart(fig3, use_container_width=True)

# ─── PAGE 7: SYMPTOMS ──────────────────────────────────────────────────────────
def page_symptoms(clinical):
    st.markdown("## ⚕️ Symptom Explorer")
    st.markdown("*Analyze clinical symptoms across PCOS positive and negative patients.*")

    symptoms = {'WeightGain':'Weight Gain','HairGrowth':'Excessive Hair Growth',
                'SkinDarkening':'Skin Darkening','HairLoss':'Hair Loss','Pimples':'Pimples/Acne'}

    sym_data = []
    for col, label in symptoms.items():
        for status in [0,1]:
            subset = clinical[clinical['PCOS']==status]
            pct = subset[col].mean()*100
            sym_data.append({'Symptom':label,'PCOS Status':'PCOS Positive' if status==1 else 'PCOS Negative','Prevalence (%)':round(pct,1)})
    sym_df = pd.DataFrame(sym_data)

    fig = px.bar(sym_df, x='Symptom', y='Prevalence (%)', color='PCOS Status', barmode='group',
                 color_discrete_map={'PCOS Positive':'#9b59b6','PCOS Negative':'#e8d5f5'},
                 text='Prevalence (%)', title='Symptom Prevalence: PCOS+ vs PCOS−')
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(paper_bgcolor='white', plot_bgcolor='white',
                      font=dict(family='Inter'), legend_title='',
                      legend=dict(orientation='h', y=1.1), height=420, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        cats = list(symptoms.values())
        pos_vals = [clinical[clinical['PCOS']==1][c].mean()*100 for c in symptoms.keys()]
        neg_vals = [clinical[clinical['PCOS']==0][c].mean()*100 for c in symptoms.keys()]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatterpolar(r=pos_vals+[pos_vals[0]], theta=cats+[cats[0]],
                                        fill='toself', name='PCOS Positive',
                                        line_color='#9b59b6', fillcolor='rgba(155,89,182,0.2)'))
        fig2.add_trace(go.Scatterpolar(r=neg_vals+[neg_vals[0]], theta=cats+[cats[0]],
                                        fill='toself', name='PCOS Negative',
                                        line_color='#e8d5f5', fillcolor='rgba(232,213,245,0.3)'))
        fig2.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])),
                           paper_bgcolor='white', font=dict(family='Inter'),
                           legend=dict(orientation='h', y=-0.1), height=380, margin=dict(t=20))
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        cycle = clinical.groupby(['PCOS_Label','Cycle']).size().reset_index(name='Count')
        cycle['Cycle'] = cycle['Cycle'].map({2:'Regular',4:'Irregular'})
        fig3 = px.bar(cycle, x='PCOS_Label', y='Count', color='Cycle', barmode='stack',
                      color_discrete_sequence=['#9b59b6','#e8d5f5'],
                      title='Menstrual Cycle Regularity by PCOS Status')
        fig3.update_layout(paper_bgcolor='white', plot_bgcolor='white',
                           font=dict(family='Inter'), xaxis_title='', margin=dict(t=50))
        st.plotly_chart(fig3, use_container_width=True)

# ─── PAGE 8: ML PREDICTOR ──────────────────────────────────────────────────────
def page_predictor(clinical):
    st.markdown("## 🤖 PCOS Risk Predictor")
    st.markdown("*Random Forest model trained on 541 clinical patients. Enter values to predict PCOS likelihood.*")

    @st.cache_resource
    def train_model(df):
        features = ['Age','BMI','FSH','LH','AMH','TSH','PRL','VitD3',
                    'WaistHip','WeightGain','HairGrowth','SkinDarkening',
                    'HairLoss','Pimples','FastFood','Exercise','FollicleL','FollicleR']
        mdf = df[features+['PCOS']].dropna()
        X, y = mdf[features], mdf['PCOS']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        return model, acc, features

    model, acc, features = train_model(clinical)
    st.markdown(f"""<div class="info-box">
        🤖 <b>Model:</b> Random Forest Classifier &nbsp;|&nbsp;
        🎯 <b>Accuracy:</b> {acc*100:.1f}% &nbsp;|&nbsp;
        📊 <b>Training Data:</b> 541 patients, 18 features
    </div>""", unsafe_allow_html=True)

    st.markdown("### Enter Patient Data")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age (years)", 18, 55, 28)
        bmi = st.number_input("BMI (kg/m²)", 15.0, 50.0, 24.0)
        fsh = st.number_input("FSH (mIU/mL)", 0.0, 30.0, 5.0)
        lh = st.number_input("LH (mIU/mL)", 0.0, 30.0, 3.0)
        amh = st.number_input("AMH (ng/mL)", 0.0, 20.0, 2.5)
        tsh = st.number_input("TSH (mIU/L)", 0.0, 20.0, 2.0)
    with c2:
        prl = st.number_input("Prolactin (ng/mL)", 0.0, 100.0, 20.0)
        vitd3 = st.number_input("Vitamin D3 (ng/mL)", 0.0, 100.0, 25.0)
        waisthip = st.number_input("Waist:Hip Ratio", 0.5, 1.5, 0.85)
        follicleL = st.number_input("Follicle Count (Left)", 0, 30, 5)
        follicleR = st.number_input("Follicle Count (Right)", 0, 30, 5)
    with c3:
        st.markdown("**Symptoms**")
        weight_gain = st.selectbox("Weight Gain", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
        hair_growth = st.selectbox("Excessive Hair Growth", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
        skin_dark = st.selectbox("Skin Darkening", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
        hair_loss = st.selectbox("Hair Loss", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
        pimples = st.selectbox("Pimples/Acne", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
        fastfood = st.selectbox("Fast Food", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
        exercise = st.selectbox("Regular Exercise", [0,1], format_func=lambda x: "Yes" if x==1 else "No")

    if st.button("🔍 Predict PCOS Risk", use_container_width=True):
        input_data = pd.DataFrame([[age,bmi,fsh,lh,amh,tsh,prl,vitd3,waisthip,
                                    weight_gain,hair_growth,skin_dark,hair_loss,
                                    pimples,fastfood,exercise,follicleL,follicleR]], columns=features)
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0]
        risk_pct = prob[1]*100

        if pred==1:
            st.error(f"⚠️ **PCOS Likely** — Risk Score: **{risk_pct:.1f}%**")
        else:
            st.success(f"✅ **PCOS Unlikely** — Risk Score: **{risk_pct:.1f}%**")

        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=risk_pct,
            title={'text':"PCOS Risk Score (%)"},
            gauge={'axis':{'range':[0,100]},'bar':{'color':'#9b59b6'},
                   'steps':[{'range':[0,33],'color':'#d5f5e3'},
                             {'range':[33,66],'color':'#fdebd0'},
                             {'range':[66,100],'color':'#f5b7b1'}]}
        ))
        fig.update_layout(paper_bgcolor='white', font=dict(family='Inter'), height=300, margin=dict(t=50))
        st.plotly_chart(fig, use_container_width=True)

        importance = pd.DataFrame({'Feature':features,'Importance':model.feature_importances_}).sort_values('Importance', ascending=True).tail(10)
        fig2 = px.bar(importance, x='Importance', y='Feature', orientation='h',
                      color='Importance', color_continuous_scale=['#e8d5f5','#6b2d8b'],
                      title='Top 10 Most Predictive Features')
        fig2.update_layout(paper_bgcolor='white', plot_bgcolor='white',
                           font=dict(family='Inter'), showlegend=False, margin=dict(t=50), height=350)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("""<div class="warning-box">
            ⚠️ <b>Disclaimer:</b> This tool is for educational and research purposes only and is not a substitute for clinical diagnosis.
        </div>""", unsafe_allow_html=True)

# ─── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    if not check_password():
        return
    clinical = load_clinical()
    gbd = load_gbd()
    obesity = load_obesity()
    page = render_sidebar()

    if page == "🏠 Overview":
        page_overview(clinical, gbd)
    elif page == "🗺️ MENA Burden Map":
        page_map(gbd)
    elif page == "📈 Trends Over Time":
        page_trends(gbd)
    elif page == "⚖️ Obesity & PCOS Link":
        page_obesity(gbd, obesity)
    elif page == "👩 Patient Profiles":
        page_patients(clinical)
    elif page == "🧬 Hormonal Analysis":
        page_hormones(clinical)
    elif page == "⚕️ Symptom Explorer":
        page_symptoms(clinical)
    elif page == "🤖 PCOS Predictor":
        page_predictor(clinical)

if __name__ == "__main__":
    main()
