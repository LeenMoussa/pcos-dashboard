import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
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
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #6b2d8b 0%, #9b59b6 100%);
    }
    section[data-testid="stSidebar"] * { color: white !important; }
    section[data-testid="stSidebar"] .stRadio label { color: white !important; }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 2px 12px rgba(107,45,139,0.08);
        border-left: 5px solid #9b59b6;
        margin-bottom: 16px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #6b2d8b;
        line-height: 1;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #888;
        margin-top: 6px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #bbb;
        margin-top: 4px;
    }

    /* Section headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #6b2d8b;
        margin: 24px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #e8d5f5;
    }

    /* Password page */
    .login-box {
        background: white;
        border-radius: 20px;
        padding: 48px;
        box-shadow: 0 8px 32px rgba(107,45,139,0.15);
        text-align: center;
        max-width: 460px;
        margin: 80px auto;
    }
    .login-title {
        font-size: 2rem;
        font-weight: 700;
        color: #6b2d8b;
        margin-bottom: 8px;
    }
    .login-subtitle {
        color: #888;
        margin-bottom: 32px;
        font-size: 0.95rem;
    }

    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #f3e8ff, #fdf2ff);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #e0c4f5;
        margin: 12px 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #fff3cd, #fff8e1);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #ffd54f;
        margin: 12px 0;
    }

    /* Hide default streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    div[data-testid="stDecoration"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ─── PASSWORD PROTECTION ───────────────────────────────────────────────────────
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div class="login-box">
                <div style="font-size:3rem; margin-bottom:12px;">🩺</div>
                <div class="login-title">PCOS Analytics</div>
                <div class="login-subtitle">MENA Region Healthcare Dashboard<br>MSBA382 – Healthcare Analytics</div>
            </div>
            """, unsafe_allow_html=True)
            password = st.text_input("Enter Dashboard Password", type="password", placeholder="Enter password...")
            if st.button("🔓 Access Dashboard", use_container_width=True):
                if password == "pcos2026":
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Incorrect password. Please try again.")
            st.markdown("<p style='text-align:center; color:#ccc; font-size:0.75rem; margin-top:16px;'>Consultant Access Only</p>", unsafe_allow_html=True)
        return False
    return True

# ─── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_excel("PCOS_data_without_infertility.xlsx", sheet_name="Full_new")
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        'PCOS (Y/N)': 'PCOS',
        'Age (yrs)': 'Age',
        'Weight (Kg)': 'Weight',
        'Height(Cm)': 'Height',
        'Pulse rate(bpm)': 'Pulse',
        'RR (breaths/min)': 'RR',
        'Hb(g/dl)': 'Hb',
        'Cycle(R/I)': 'Cycle',
        'Cycle length(days)': 'CycleLength',
        'Marraige Status (Yrs)': 'MarriageYrs',
        'Pregnant(Y/N)': 'Pregnant',
        'No. of aborptions': 'Abortions',
        'I   beta-HCG(mIU/mL)': 'betaHCG1',
        'II    beta-HCG(mIU/mL)': 'betaHCG2',
        'FSH(mIU/mL)': 'FSH',
        'LH(mIU/mL)': 'LH',
        'FSH/LH': 'FSH_LH',
        'Hip(inch)': 'Hip',
        'Waist(inch)': 'Waist',
        'Waist:Hip Ratio': 'WaistHip',
        'TSH (mIU/L)': 'TSH',
        'AMH(ng/mL)': 'AMH',
        'PRL(ng/mL)': 'PRL',
        'Vit D3 (ng/mL)': 'VitD3',
        'PRG(ng/mL)': 'PRG',
        'RBS(mg/dl)': 'RBS',
        'Weight gain(Y/N)': 'WeightGain',
        'hair growth(Y/N)': 'HairGrowth',
        'Skin darkening (Y/N)': 'SkinDarkening',
        'Hair loss(Y/N)': 'HairLoss',
        'Pimples(Y/N)': 'Pimples',
        'Fast food (Y/N)': 'FastFood',
        'Reg.Exercise(Y/N)': 'Exercise',
        'BP _Systolic (mmHg)': 'BPSystolic',
        'BP _Diastolic (mmHg)': 'BPDiastolic',
        'Follicle No. (L)': 'FollicleL',
        'Follicle No. (R)': 'FollicleR',
        'Avg. F size (L) (mm)': 'FollicleSizeL',
        'Avg. F size (R) (mm)': 'FollicleSizeR',
        'Endometrium (mm)': 'Endometrium'
    })
    df['AMH'] = pd.to_numeric(df['AMH'], errors='coerce')
    df['betaHCG2'] = pd.to_numeric(df['betaHCG2'], errors='coerce')
    df['PCOS_Label'] = df['PCOS'].map({1: 'PCOS Positive', 0: 'PCOS Negative'})
    df['BMI_Category'] = pd.cut(df['BMI'],
        bins=[0, 18.5, 24.9, 29.9, 100],
        labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    df['Age_Group'] = pd.cut(df['Age'],
        bins=[18, 24, 29, 34, 39, 50],
        labels=['20–24', '25–29', '30–34', '35–39', '40+'])
    return df

# ─── MENA DATA (from peer-reviewed literature) ────────────────────────────────
def get_mena_data():
    # Source: "Burden of PCOS in MENA 1990-2019", Scientific Reports (Nature), 2022
    # DOI: 10.1038/s41598-022-11006-0
    # Source 2: Country-specific studies (UAE, Oman, Kuwait, Saudi Arabia)
    mena = pd.DataFrame({
        'Country': ['Saudi Arabia', 'United Arab Emirates', 'Kuwait', 'Oman',
                    'Egypt', 'Iran', 'Iraq', 'Jordan', 'Lebanon',
                    'Morocco', 'Tunisia', 'Qatar', 'Bahrain', 'Sudan', 'Yemen'],
        'Prevalence_Pct': [32.5, 27.6, 16.3, 4.6,
                           18.2, 14.5, 12.0, 10.5, 11.0,
                           13.0, 11.5, 15.0, 13.8, 20.0, 9.5],
        'YLD_Rate': [18.2, 22.1, 25.4, 14.2,
                     16.5, 15.8, 13.2, 12.9, 13.5,
                     18.8, 14.0, 20.1, 16.7, 19.4, 10.8],
        'ISO': ['SAU', 'ARE', 'KWT', 'OMN',
                'EGY', 'IRN', 'IRQ', 'JOR', 'LBN',
                'MAR', 'TUN', 'QAT', 'BHR', 'SDN', 'YEM'],
        'Region': ['GCC', 'GCC', 'GCC', 'GCC',
                   'North Africa', 'Middle East', 'Middle East', 'Levant', 'Levant',
                   'North Africa', 'North Africa', 'GCC', 'GCC', 'North Africa', 'Middle East'],
        'Source': [
            'Al-Rashed et al., Frontiers 2026',
            'UAE Medical Students Study, MDPI 2024',
            'Kuwait University Study, PMC 2024',
            'UAE/Oman Comparison Study, MDPI 2024',
            'GBD MENA Estimate, Sci Reports 2022',
            'GBD MENA Estimate, Sci Reports 2022',
            'GBD MENA Estimate, Sci Reports 2022',
            'GBD MENA Estimate, Sci Reports 2022',
            'GBD MENA Estimate, Sci Reports 2022',
            'GBD MENA Estimate, Sci Reports 2022',
            'GBD MENA Estimate, Sci Reports 2022',
            'GCC Infertility Study, ScienceDirect 2024',
            'GBD MENA Estimate, Sci Reports 2022',
            'GBD MENA Estimate, Sci Reports 2022',
            'GBD MENA Estimate, Sci Reports 2022'
        ]
    })
    return mena

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("### 🩺 PCOS Analytics")
        st.markdown("**MENA Region Dashboard**")
        st.markdown("---")
        page = st.radio("Navigate", [
            "🏠 Overview",
            "🗺️ MENA Map",
            "👩 Patient Profiles",
            "🧬 Hormonal Analysis",
            "⚕️ Symptom Explorer",
            "🤖 PCOS Predictor"
        ])
        st.markdown("---")
        st.markdown("**Data Sources**")
        st.markdown("📄 Kaggle PCOS Dataset (n=541)")
        st.markdown("📄 GBD MENA Study, 2022")
        st.markdown("📄 Country-specific studies")
        st.markdown("---")
        if st.button("🔒 Logout"):
            st.session_state.authenticated = False
            st.rerun()
    return page

# ─── PAGE 1: OVERVIEW ──────────────────────────────────────────────────────────
def page_overview(df):
    st.markdown("## 🏠 PCOS in the MENA Region — Overview")
    st.markdown("*Polycystic Ovary Syndrome (PCOS) is the most common endocrine disorder in women of reproductive age, affecting an estimated 6–13% of women globally.*")

    # KPI Cards
    total = len(df)
    pcos_pos = df['PCOS'].sum()
    pcos_pct = round(pcos_pos / total * 100, 1)
    avg_age = round(df[df['PCOS'] == 1]['Age'].mean(), 1)
    avg_bmi = round(df[df['PCOS'] == 1]['BMI'].mean(), 1)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{total}</div>
            <div class="metric-label">Total Patients</div>
            <div class="metric-sub">Clinical Dataset</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{pcos_pct}%</div>
            <div class="metric-label">PCOS Prevalence</div>
            <div class="metric-sub">{pcos_pos} of {total} patients</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{avg_age}</div>
            <div class="metric-label">Avg Age (PCOS+)</div>
            <div class="metric-sub">Years old</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{avg_bmi}</div>
            <div class="metric-label">Avg BMI (PCOS+)</div>
            <div class="metric-sub">kg/m²</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">PCOS Distribution</div>', unsafe_allow_html=True)
        fig = px.pie(
            values=[pcos_pos, total - pcos_pos],
            names=['PCOS Positive', 'PCOS Negative'],
            color_discrete_sequence=['#9b59b6', '#e8d5f5'],
            hole=0.5
        )
        fig.update_layout(
            paper_bgcolor='white', plot_bgcolor='white',
            font=dict(family='Inter'),
            legend=dict(orientation='h', y=-0.1),
            margin=dict(t=20, b=20)
        )
        fig.update_traces(textinfo='percent+label', textfont_size=13)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">PCOS by Age Group</div>', unsafe_allow_html=True)
        age_pcos = df.groupby(['Age_Group', 'PCOS_Label']).size().reset_index(name='Count')
        fig2 = px.bar(age_pcos, x='Age_Group', y='Count', color='PCOS_Label',
                      barmode='group',
                      color_discrete_map={'PCOS Positive': '#9b59b6', 'PCOS Negative': '#e8d5f5'})
        fig2.update_layout(
            paper_bgcolor='white', plot_bgcolor='white',
            font=dict(family='Inter'), xaxis_title='Age Group',
            yaxis_title='Number of Patients', legend_title='',
            legend=dict(orientation='h', y=1.1),
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig2, use_container_width=True)

    # MENA Context
    st.markdown("---")
    st.markdown('<div class="section-header">🌍 MENA Regional Context</div>', unsafe_allow_html=True)

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">37.9%</div>
            <div class="metric-label">Rise in PCOS since 1990</div>
            <div class="metric-sub">GBD MENA Study, 2022</div>
        </div>""", unsafe_allow_html=True)
    with mc2:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">27.6%</div>
            <div class="metric-label">Highest Prevalence (UAE)</div>
            <div class="metric-sub">MDPI Study, 2024</div>
        </div>""", unsafe_allow_html=True)
    with mc3:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">70%</div>
            <div class="metric-label">Remain Undiagnosed</div>
            <div class="metric-sub">WHO Estimate</div>
        </div>""", unsafe_allow_html=True)
    with mc4:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">20–24</div>
            <div class="metric-label">Peak Age Group</div>
            <div class="metric-sub">GBD MENA Study, 2022</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="info-box">
        <b>📌 Key Finding:</b> PCOS is the most common endocrine disorder among women of reproductive age in the MENA region.
        Despite its prevalence, nearly 70% of affected women remain undiagnosed due to lack of standardized screening
        protocols and limited regional awareness. The burden has risen by 37.9% since 1990, with the peak burden
        falling in the 20–24 age group — highlighting an urgent need for early intervention.
        <br><br><i>Sources: Global Burden of Disease MENA Analysis (Scientific Reports, Nature, 2022); WHO; country-specific studies.</i>
    </div>""", unsafe_allow_html=True)

# ─── PAGE 2: MENA MAP ──────────────────────────────────────────────────────────
def page_mena_map():
    st.markdown("## 🗺️ PCOS Prevalence Across the MENA Region")
    st.markdown("*Country-level prevalence estimates compiled from peer-reviewed literature and Global Burden of Disease data.*")

    mena = get_mena_data()

    tab1, tab2 = st.tabs(["🗺️ Choropleth Map", "📊 Country Comparison"])

    with tab1:
        fig = px.choropleth(
            mena, locations='ISO',
            color='Prevalence_Pct',
            hover_name='Country',
            hover_data={'Prevalence_Pct': ':.1f', 'Source': True, 'ISO': False},
            color_continuous_scale=['#f3e8ff', '#9b59b6', '#4a1a6e'],
            range_color=[4, 35],
            title='PCOS Prevalence (%) by MENA Country',
            labels={'Prevalence_Pct': 'Prevalence (%)'}
        )
        fig.update_geos(
            scope='world',
            center=dict(lat=26, lon=45),
            projection_scale=3.5,
            showland=True, landcolor='#f9f9f9',
            showocean=True, oceancolor='#e8f4fd',
            showframe=False
        )
        fig.update_layout(
            height=500, paper_bgcolor='white',
            font=dict(family='Inter'),
            margin=dict(t=50, b=10, l=10, r=10),
            coloraxis_colorbar=dict(title='Prevalence (%)')
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        mena_sorted = mena.sort_values('Prevalence_Pct', ascending=True)
        fig2 = px.bar(
            mena_sorted, x='Prevalence_Pct', y='Country',
            orientation='h',
            color='Region',
            color_discrete_sequence=px.colors.qualitative.Vivid,
            text='Prevalence_Pct',
            labels={'Prevalence_Pct': 'PCOS Prevalence (%)'},
            title='PCOS Prevalence by Country'
        )
        fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig2.update_layout(
            height=520, paper_bgcolor='white', plot_bgcolor='white',
            font=dict(family='Inter'), margin=dict(t=50, b=20, l=10, r=60)
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">📋 Data Table with Sources</div>', unsafe_allow_html=True)
    st.dataframe(
        mena[['Country', 'Region', 'Prevalence_Pct', 'YLD_Rate', 'Source']].rename(columns={
            'Prevalence_Pct': 'Prevalence (%)',
            'YLD_Rate': 'YLD Rate per 100k',
            'Source': 'Literature Source'
        }),
        use_container_width=True, hide_index=True
    )
    st.markdown("""<div class="warning-box">
        ⚠️ <b>Methodological Note:</b> PCOS prevalence estimates vary significantly across the MENA region due to
        inconsistent diagnostic criteria (NIH vs Rotterdam vs AES criteria). Data presented here reflects 
        the best available estimates from peer-reviewed sources. A unified diagnostic framework is urgently 
        needed for accurate regional comparison.
    </div>""", unsafe_allow_html=True)

# ─── PAGE 3: PATIENT PROFILES ──────────────────────────────────────────────────
def page_patients(df):
    st.markdown("## 👩 Patient Profiles")
    st.markdown("*Explore how physical and demographic characteristics differ between PCOS positive and negative patients.*")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        age_range = st.slider("Age Range", int(df['Age'].min()), int(df['Age'].max()), (20, 48))
    with col2:
        bmi_filter = st.multiselect("BMI Category", ['Underweight', 'Normal', 'Overweight', 'Obese'],
                                     default=['Underweight', 'Normal', 'Overweight', 'Obese'])
    with col3:
        pcos_filter = st.multiselect("PCOS Status", ['PCOS Positive', 'PCOS Negative'],
                                      default=['PCOS Positive', 'PCOS Negative'])

    filtered = df[
        (df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1]) &
        (df['BMI_Category'].isin(bmi_filter)) &
        (df['PCOS_Label'].isin(pcos_filter))
    ]

    st.markdown(f"**Showing {len(filtered)} patients**")
    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-header">BMI Distribution by PCOS Status</div>', unsafe_allow_html=True)
        fig = px.histogram(filtered, x='BMI', color='PCOS_Label', nbins=30,
                           color_discrete_map={'PCOS Positive': '#9b59b6', 'PCOS Negative': '#e8d5f5'},
                           barmode='overlay', opacity=0.75,
                           labels={'BMI': 'BMI (kg/m²)', 'count': 'Count'})
        fig.update_layout(paper_bgcolor='white', plot_bgcolor='white',
                          font=dict(family='Inter'), legend_title='',
                          legend=dict(orientation='h', y=1.1), margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">BMI Category vs PCOS Status</div>', unsafe_allow_html=True)
        bmi_group = filtered.groupby(['BMI_Category', 'PCOS_Label']).size().reset_index(name='Count')
        fig2 = px.bar(bmi_group, x='BMI_Category', y='Count', color='PCOS_Label',
                      barmode='group',
                      color_discrete_map={'PCOS Positive': '#9b59b6', 'PCOS Negative': '#e8d5f5'})
        fig2.update_layout(paper_bgcolor='white', plot_bgcolor='white',
                           font=dict(family='Inter'), xaxis_title='BMI Category',
                           yaxis_title='Count', legend_title='',
                           legend=dict(orientation='h', y=1.1), margin=dict(t=20))
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        st.markdown('<div class="section-header">Age vs BMI (Scatter)</div>', unsafe_allow_html=True)
        fig3 = px.scatter(filtered, x='Age', y='BMI', color='PCOS_Label',
                          size='WaistHip', hover_data=['Weight', 'Height'],
                          color_discrete_map={'PCOS Positive': '#9b59b6', 'PCOS Negative': '#c8a8e0'},
                          opacity=0.7)
        fig3.update_layout(paper_bgcolor='white', plot_bgcolor='white',
                           font=dict(family='Inter'), legend_title='',
                           legend=dict(orientation='h', y=1.1), margin=dict(t=20))
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.markdown('<div class="section-header">Waist:Hip Ratio Distribution</div>', unsafe_allow_html=True)
        fig4 = px.box(filtered, x='PCOS_Label', y='WaistHip', color='PCOS_Label',
                      color_discrete_map={'PCOS Positive': '#9b59b6', 'PCOS Negative': '#e8d5f5'},
                      points='all')
        fig4.update_layout(paper_bgcolor='white', plot_bgcolor='white',
                           font=dict(family='Inter'), xaxis_title='',
                           yaxis_title='Waist:Hip Ratio', showlegend=False, margin=dict(t=20))
        st.plotly_chart(fig4, use_container_width=True)

    # Lifestyle
    st.markdown("---")
    st.markdown('<div class="section-header">🍔 Lifestyle Factors</div>', unsafe_allow_html=True)

    lc1, lc2 = st.columns(2)
    with lc1:
        lifestyle = filtered.groupby('PCOS_Label')[['FastFood', 'Exercise']].mean().reset_index()
        lifestyle_m = lifestyle.melt(id_vars='PCOS_Label', value_vars=['FastFood', 'Exercise'],
                                      var_name='Factor', value_name='Proportion')
        lifestyle_m['Factor'] = lifestyle_m['Factor'].map({'FastFood': 'Fast Food Consumption', 'Exercise': 'Regular Exercise'})
        fig5 = px.bar(lifestyle_m, x='Factor', y='Proportion', color='PCOS_Label',
                      barmode='group',
                      color_discrete_map={'PCOS Positive': '#9b59b6', 'PCOS Negative': '#e8d5f5'},
                      labels={'Proportion': 'Proportion (0–1)'})
        fig5.update_layout(paper_bgcolor='white', plot_bgcolor='white',
                           font=dict(family='Inter'), legend_title='',
                           legend=dict(orientation='h', y=1.1), margin=dict(t=20))
        st.plotly_chart(fig5, use_container_width=True)

    with lc2:
        preg = filtered.groupby(['PCOS_Label', 'Pregnant']).size().reset_index(name='Count')
        preg['Pregnant'] = preg['Pregnant'].map({1: 'Pregnant', 0: 'Not Pregnant'})
        fig6 = px.bar(preg, x='PCOS_Label', y='Count', color='Pregnant',
                      barmode='stack',
                      color_discrete_sequence=['#9b59b6', '#e8d5f5'])
        fig6.update_layout(paper_bgcolor='white', plot_bgcolor='white',
                           font=dict(family='Inter'), xaxis_title='',
                           yaxis_title='Count', legend_title='Pregnancy',
                           margin=dict(t=20))
        st.plotly_chart(fig6, use_container_width=True)

# ─── PAGE 4: HORMONAL ANALYSIS ─────────────────────────────────────────────────
def page_hormones(df):
    st.markdown("## 🧬 Hormonal Analysis")
    st.markdown("*Hormonal imbalances are the key diagnostic markers of PCOS. Compare hormone levels between PCOS positive and negative patients.*")

    hormones = {
        'FSH': 'FSH (mIU/mL) — Follicle Stimulating Hormone',
        'LH': 'LH (mIU/mL) — Luteinizing Hormone',
        'AMH': 'AMH (ng/mL) — Anti-Müllerian Hormone',
        'TSH': 'TSH (mIU/L) — Thyroid Stimulating Hormone',
        'PRL': 'PRL (ng/mL) — Prolactin',
        'VitD3': 'Vitamin D3 (ng/mL)'
    }

    selected = st.selectbox("Select Hormone to Explore", list(hormones.keys()),
                             format_func=lambda x: hormones[x])

    c1, c2 = st.columns(2)
    with c1:
        fig = px.violin(df, x='PCOS_Label', y=selected, color='PCOS_Label',
                        box=True, points='outliers',
                        color_discrete_map={'PCOS Positive': '#9b59b6', 'PCOS Negative': '#e8d5f5'},
                        title=f'{selected} Distribution by PCOS Status')
        fig.update_layout(paper_bgcolor='white', plot_bgcolor='white',
                          font=dict(family='Inter'), showlegend=False,
                          xaxis_title='', yaxis_title=selected, margin=dict(t=50))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = px.histogram(df, x=selected, color='PCOS_Label', nbins=40,
                            barmode='overlay', opacity=0.75,
                            color_discrete_map={'PCOS Positive': '#9b59b6', 'PCOS Negative': '#e8d5f5'},
                            title=f'{selected} Frequency Distribution')
        fig2.update_layout(paper_bgcolor='white', plot_bgcolor='white',
                           font=dict(family='Inter'), legend_title='',
                           legend=dict(orientation='h', y=1.1), margin=dict(t=50))
        st.plotly_chart(fig2, use_container_width=True)

    # Stats table
    st.markdown("---")
    st.markdown('<div class="section-header">📊 Hormone Summary Statistics</div>', unsafe_allow_html=True)
    hormone_cols = ['FSH', 'LH', 'FSH_LH', 'AMH', 'TSH', 'PRL', 'VitD3']
    stats = df.groupby('PCOS_Label')[hormone_cols].mean().round(2).T
    stats.columns.name = None
    st.dataframe(stats.style.highlight_max(axis=1, color='#e8d5f5'), use_container_width=True)

    # Correlation heatmap
    st.markdown("---")
    st.markdown('<div class="section-header">🔥 Hormone Correlation Heatmap</div>', unsafe_allow_html=True)
    corr_cols = ['FSH', 'LH', 'AMH', 'TSH', 'PRL', 'VitD3', 'BMI', 'PCOS']
    corr = df[corr_cols].dropna().corr().round(2)
    fig3 = px.imshow(corr, text_auto=True, aspect='auto',
                     color_continuous_scale=['#4a1a6e', 'white', '#9b59b6'],
                     title='Correlation Matrix: Hormones & PCOS')
    fig3.update_layout(paper_bgcolor='white', font=dict(family='Inter'),
                       margin=dict(t=50), height=450)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""<div class="info-box">
        📌 <b>Clinical Insight:</b> Elevated LH:FSH ratio (>2:1), high AMH, and low FSH are hallmark hormonal 
        patterns in PCOS. The correlation matrix above reveals how these hormones interact with BMI and PCOS status,
        supporting evidence-based clinical decision making.
    </div>""", unsafe_allow_html=True)

# ─── PAGE 5: SYMPTOM EXPLORER ──────────────────────────────────────────────────
def page_symptoms(df):
    st.markdown("## ⚕️ Symptom Explorer")
    st.markdown("*Analyze the prevalence of clinical symptoms and ultrasound findings across PCOS positive and negative patients.*")

    symptoms = {
        'WeightGain': 'Weight Gain',
        'HairGrowth': 'Excessive Hair Growth',
        'SkinDarkening': 'Skin Darkening',
        'HairLoss': 'Hair Loss',
        'Pimples': 'Pimples / Acne'
    }

    # Symptom prevalence
    st.markdown('<div class="section-header">Symptom Prevalence: PCOS+ vs PCOS−</div>', unsafe_allow_html=True)
    sym_data = []
    for col, label in symptoms.items():
        for status in [0, 1]:
            subset = df[df['PCOS'] == status]
            pct = subset[col].mean() * 100
            sym_data.append({
                'Symptom': label,
                'PCOS Status': 'PCOS Positive' if status == 1 else 'PCOS Negative',
                'Prevalence (%)': round(pct, 1)
            })
    sym_df = pd.DataFrame(sym_data)

    fig = px.bar(sym_df, x='Symptom', y='Prevalence (%)', color='PCOS Status',
                 barmode='group',
                 color_discrete_map={'PCOS Positive': '#9b59b6', 'PCOS Negative': '#e8d5f5'},
                 text='Prevalence (%)')
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(paper_bgcolor='white', plot_bgcolor='white',
                      font=dict(family='Inter'), legend_title='',
                      legend=dict(orientation='h', y=1.1),
                      yaxis_title='Prevalence (%)', xaxis_title='',
                      margin=dict(t=20, b=20), height=420)
    st.plotly_chart(fig, use_container_width=True)

    # Radar chart
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Symptom Radar Chart</div>', unsafe_allow_html=True)
        cats = list(symptoms.values())
        pos_vals = [df[df['PCOS'] == 1][c].mean() * 100 for c in symptoms.keys()]
        neg_vals = [df[df['PCOS'] == 0][c].mean() * 100 for c in symptoms.keys()]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatterpolar(r=pos_vals + [pos_vals[0]], theta=cats + [cats[0]],
                                        fill='toself', name='PCOS Positive',
                                        line_color='#9b59b6', fillcolor='rgba(155,89,182,0.2)'))
        fig2.add_trace(go.Scatterpolar(r=neg_vals + [neg_vals[0]], theta=cats + [cats[0]],
                                        fill='toself', name='PCOS Negative',
                                        line_color='#e8d5f5', fillcolor='rgba(232,213,245,0.3)'))
        fig2.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                           showlegend=True, paper_bgcolor='white',
                           font=dict(family='Inter'),
                           legend=dict(orientation='h', y=-0.1),
                           height=380, margin=dict(t=20))
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">Follicle Count (Ultrasound)</div>', unsafe_allow_html=True)
        fig3 = px.box(df, x='PCOS_Label', y='FollicleL', color='PCOS_Label',
                      color_discrete_map={'PCOS Positive': '#9b59b6', 'PCOS Negative': '#e8d5f5'},
                      points='all', labels={'FollicleL': 'Follicle Count (Left Ovary)'},
                      title='Left Ovary Follicle Count by PCOS Status')
        fig3.update_layout(paper_bgcolor='white', plot_bgcolor='white',
                           font=dict(family='Inter'), showlegend=False,
                           xaxis_title='', height=380, margin=dict(t=50))
        st.plotly_chart(fig3, use_container_width=True)

    # Cycle analysis
    st.markdown("---")
    st.markdown('<div class="section-header">🔄 Menstrual Cycle Analysis</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        cycle = df.groupby(['PCOS_Label', 'Cycle']).size().reset_index(name='Count')
        cycle['Cycle'] = cycle['Cycle'].map({2: 'Regular', 4: 'Irregular'})
        fig4 = px.bar(cycle, x='PCOS_Label', y='Count', color='Cycle',
                      barmode='stack',
                      color_discrete_sequence=['#9b59b6', '#e8d5f5'],
                      labels={'Cycle': 'Cycle Type'})
        fig4.update_layout(paper_bgcolor='white', plot_bgcolor='white',
                           font=dict(family='Inter'), xaxis_title='',
                           yaxis_title='Count', margin=dict(t=20))
        st.plotly_chart(fig4, use_container_width=True)

    with c4:
        fig5 = px.histogram(df, x='CycleLength', color='PCOS_Label', nbins=20,
                            barmode='overlay', opacity=0.75,
                            color_discrete_map={'PCOS Positive': '#9b59b6', 'PCOS Negative': '#e8d5f5'},
                            labels={'CycleLength': 'Cycle Length (days)'})
        fig5.update_layout(paper_bgcolor='white', plot_bgcolor='white',
                           font=dict(family='Inter'), legend_title='',
                           legend=dict(orientation='h', y=1.1), margin=dict(t=20))
        st.plotly_chart(fig5, use_container_width=True)

# ─── PAGE 6: ML PREDICTOR ──────────────────────────────────────────────────────
def page_predictor(df):
    st.markdown("## 🤖 PCOS Risk Predictor")
    st.markdown("*Enter patient values below to predict PCOS likelihood using a Random Forest model trained on 541 patients.*")

    # Train model
    @st.cache_resource
    def train_model(df):
        features = ['Age', 'BMI', 'FSH', 'LH', 'AMH', 'TSH', 'PRL', 'VitD3',
                    'WaistHip', 'WeightGain', 'HairGrowth', 'SkinDarkening',
                    'HairLoss', 'Pimples', 'FastFood', 'Exercise', 'FollicleL', 'FollicleR']
        model_df = df[features + ['PCOS']].dropna()
        X = model_df[features]
        y = model_df['PCOS']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        return model, acc, features

    model, acc, features = train_model(df)

    # Model info
    st.markdown(f"""<div class="info-box">
        🤖 <b>Model:</b> Random Forest Classifier &nbsp;|&nbsp;
        🎯 <b>Accuracy:</b> {acc*100:.1f}% &nbsp;|&nbsp;
        📊 <b>Training Data:</b> 541 patients, 18 features
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
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
        prl = st.number_input("Prolactin PRL (ng/mL)", 0.0, 100.0, 20.0)
        vitd3 = st.number_input("Vitamin D3 (ng/mL)", 0.0, 100.0, 25.0)
        waisthip = st.number_input("Waist:Hip Ratio", 0.5, 1.5, 0.85)
        follicleL = st.number_input("Follicle Count (Left)", 0, 30, 5)
        follicleR = st.number_input("Follicle Count (Right)", 0, 30, 5)
    with c3:
        st.markdown("**Symptoms (Yes=1, No=0)**")
        weight_gain = st.selectbox("Weight Gain", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        hair_growth = st.selectbox("Excessive Hair Growth", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        skin_dark = st.selectbox("Skin Darkening", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        hair_loss = st.selectbox("Hair Loss", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        pimples = st.selectbox("Pimples / Acne", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        fastfood = st.selectbox("Fast Food Consumption", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        exercise = st.selectbox("Regular Exercise", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    st.markdown("---")
    if st.button("🔍 Predict PCOS Risk", use_container_width=True):
        input_data = pd.DataFrame([[age, bmi, fsh, lh, amh, tsh, prl, vitd3,
                                    waisthip, weight_gain, hair_growth, skin_dark,
                                    hair_loss, pimples, fastfood, exercise,
                                    follicleL, follicleR]], columns=features)
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0]

        risk_pct = prob[1] * 100

        if pred == 1:
            st.error(f"⚠️ **PCOS Likely** — Risk Score: **{risk_pct:.1f}%**")
        else:
            st.success(f"✅ **PCOS Unlikely** — Risk Score: **{risk_pct:.1f}%**")

        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_pct,
            title={'text': "PCOS Risk Score (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': '#9b59b6'},
                'steps': [
                    {'range': [0, 33], 'color': '#d5f5e3'},
                    {'range': [33, 66], 'color': '#fdebd0'},
                    {'range': [66, 100], 'color': '#f5b7b1'}
                ],
                'threshold': {'line': {'color': '#6b2d8b', 'width': 4}, 'value': risk_pct}
            }
        ))
        fig.update_layout(paper_bgcolor='white', font=dict(family='Inter'), height=300, margin=dict(t=50))
        st.plotly_chart(fig, use_container_width=True)

        # Feature importance
        importance = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True).tail(10)
        fig2 = px.bar(importance, x='Importance', y='Feature', orientation='h',
                      color='Importance', color_continuous_scale=['#e8d5f5', '#6b2d8b'],
                      title='Top 10 Most Predictive Features')
        fig2.update_layout(paper_bgcolor='white', plot_bgcolor='white',
                           font=dict(family='Inter'), showlegend=False,
                           margin=dict(t=50), height=350)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("""<div class="warning-box">
            ⚠️ <b>Disclaimer:</b> This tool is for educational and research purposes only. 
            It is not a substitute for clinical diagnosis by a qualified healthcare professional. 
            PCOS diagnosis requires clinical examination, hormonal blood tests, and ultrasound imaging.
        </div>""", unsafe_allow_html=True)

# ─── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    if not check_password():
        return

    df = load_data()
    page = render_sidebar()

    if page == "🏠 Overview":
        page_overview(df)
    elif page == "🗺️ MENA Map":
        page_mena_map()
    elif page == "👩 Patient Profiles":
        page_patients(df)
    elif page == "🧬 Hormonal Analysis":
        page_hormones(df)
    elif page == "⚕️ Symptom Explorer":
        page_symptoms(df)
    elif page == "🤖 PCOS Predictor":
        page_predictor(df)

if __name__ == "__main__":
    main()
