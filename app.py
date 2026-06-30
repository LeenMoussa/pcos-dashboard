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
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #e9dcf0; }
    .stApp { background-color: #e9dcf0; }
    div[data-baseweb="popover"] * ,
    div[data-baseweb="menu"] *,
    ul[role="listbox"] * { color: #262730 !important; }
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 2px 12px rgba(107,45,139,0.08);
        border-left: 5px solid #9b59b6;
        margin-bottom: 12px;
    }
    .metric-value { font-size: 1.9rem; font-weight: 700; color: #6b2d8b; line-height: 1; }
    .metric-label { font-size: 0.78rem; color: #888; margin-top: 6px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-sub { font-size: 0.75rem; color: #bbb; margin-top: 4px; }
    .section-header { font-size: 1.25rem; font-weight: 700; color: #6b2d8b; margin: 8px 0 12px 0; padding-bottom: 6px; border-bottom: 2px solid #b48ec9; }
    .info-box { background: linear-gradient(135deg, #f3e8ff, #fdf2ff); border-radius: 12px; padding: 16px 20px; border: 1px solid #e0c4f5; margin: 10px 0; font-size: 0.92rem; }
    .warning-box { background: linear-gradient(135deg, #fff3cd, #fff8e1); border-radius: 12px; padding: 16px 20px; border: 1px solid #ffd54f; margin: 10px 0; font-size: 0.88rem; }
    .panel { background: white; border-radius: 16px; padding: 18px 20px; box-shadow: 0 2px 12px rgba(107,45,139,0.08); margin-bottom: 14px; }
    .filter-bar { background: #6b2d8b; border-radius: 14px; padding: 14px 22px; margin-bottom: 18px; }
    .filter-bar label, .filter-bar p, .filter-bar span, .filter-bar .stMarkdown { color: white !important; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    div[data-testid="stDecoration"] { display: none; }
    h1, h2, h3 { color: #2A1B33; }
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
                <div style="font-size:2rem;font-weight:700;color:#6b2d8b;margin-bottom:8px;">PCOS Analytics</div>
                <div style="color:#888;margin-bottom:32px;font-size:0.95rem;">MENA Region Healthcare Dashboard<br>MSBA382 \u2013 Healthcare Analytics</div>
            </div>
            """, unsafe_allow_html=True)
            password = st.text_input("Enter Dashboard Password", type="password", placeholder="Enter password...")
            if st.button("Access Dashboard", use_container_width=True):
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
        'PCOS (Y/N)':'PCOS','Age (yrs)':'Age','Weight (Kg)':'Weight',
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
    for col, upper_bound in [('FSH', 50), ('LH', 50), ('VitD3', 200)]:
        df.loc[df[col] > upper_bound, col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['PCOS_Label'] = df['PCOS'].map({1:'PCOS Positive', 0:'PCOS Negative'})
    df['BMI_Category'] = pd.cut(df['BMI'], bins=[0,18.5,24.9,29.9,100],
        labels=['Underweight','Normal','Overweight','Obese'])
    df['Age_Group'] = pd.cut(df['Age'], bins=[18,24,29,34,39,50],
        labels=['20\u201324','25\u201329','30\u201334','35\u201339','40+'])
    return df

@st.cache_data
def load_gbd():
    df = pd.read_csv("IHME-GBD_2023_DATA-f76e773e-1.csv")
    df['location_name'] = df['location_name'].replace({
        'Iran (Islamic Republic of)': 'Iran',
        'Syrian Arab Republic': 'Syria',
        'T\u00fcrkiye': 'Turkey'
    })
    return df

@st.cache_data
def load_obesity():
    df = pd.read_csv("share-of-females-defined-as-obese.csv")
    df.columns = ['Entity', 'Code', 'Year', 'Obesity_Pct']
    return df

ISO_MAP = {
    'Afghanistan':'AFG','Algeria':'DZA','Bahrain':'BHR','Egypt':'EGY',
    'Iran':'IRN','Iraq':'IRQ','Jordan':'JOR','Kuwait':'KWT','Lebanon':'LBN',
    'Libya':'LBY','Morocco':'MAR','Oman':'OMN','Palestine':'PSE','Qatar':'QAT',
    'Saudi Arabia':'SAU','Sudan':'SDN','Syria':'SYR','Tunisia':'TUN',
    'Turkey':'TUR','United Arab Emirates':'ARE','Yemen':'YEM'
}
MENA_COUNTRIES = list(ISO_MAP.keys())

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

# ─── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    if not check_password():
        return

    clinical = load_clinical()
    gbd = load_gbd()
    obesity = load_obesity()
    model, acc, features = train_model(clinical)

    st.markdown("# PCOS Analytics \u2014 MENA Region")
    st.markdown("*A single coordinated view of PCOS burden, regional risk factors, and patient-level clinical patterns. Adjust the filters below \u2014 every chart updates together.*")

    # ── GLOBAL FILTER BAR (coordinates the regional charts) ──────────────────
    st.markdown('<div class="filter-bar">', unsafe_allow_html=True)
    fc1, fc2, fc3 = st.columns([1, 1.3, 1.3])
    with fc1:
        sel_year = st.slider("Year", 1990, 2023, 2023)
    with fc2:
        sel_measure = st.selectbox("Regional Measure", ['Prevalence', 'Incidence', 'DALYs (Disability-Adjusted Life Years)'],
                                    format_func=lambda x: x.split('(')[0].strip())
    with fc3:
        sel_countries = st.multiselect("Highlight Countries (trend + map hover)", MENA_COUNTRIES,
                                        default=['Qatar', 'Kuwait', 'Saudi Arabia', 'Egypt', 'Lebanon', 'Yemen'])
    st.markdown('</div>', unsafe_allow_html=True)

    # ── ROW 1: KPI STRIP ──────────────────────────────────────────────────────
    prev_sel = gbd[(gbd['measure_name']==sel_measure) & (gbd['metric_name']=='Rate') & (gbd['year']==sel_year)]
    prev_1990 = gbd[(gbd['measure_name']==sel_measure) & (gbd['metric_name']=='Rate') & (gbd['year']==1990)]
    avg_sel = prev_sel['val'].mean()
    avg_1990 = prev_1990['val'].mean()
    pct_change = ((avg_sel - avg_1990) / avg_1990) * 100 if avg_1990 else 0
    highest = prev_sel.loc[prev_sel['val'].idxmax(), 'location_name'] if len(prev_sel) else "\u2014"
    pcos_pct = round(clinical['PCOS'].mean()*100, 1)

    k1, k2, k3, k4, k5 = st.columns(5)
    label_short = sel_measure.split('(')[0].strip()
    with k1:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{avg_sel:,.0f}</div>
            <div class="metric-label">Avg {label_short} Rate</div><div class="metric-sub">per 100,000 women, {sel_year}</div></div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{pct_change:+.0f}%</div>
            <div class="metric-label">Change vs 1990</div><div class="metric-sub">IHME GBD 2023</div></div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{highest}</div>
            <div class="metric-label">Highest Burden</div><div class="metric-sub">{sel_year}</div></div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{pcos_pct}%</div>
            <div class="metric-label">Clinical PCOS Rate</div><div class="metric-sub">541 patients</div></div>""", unsafe_allow_html=True)
    with k5:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{acc*100:.0f}%</div>
            <div class="metric-label">Predictor Accuracy</div><div class="metric-sub">Random Forest, 18 features</div></div>""", unsafe_allow_html=True)

    # ── ROW 2: MAP + TREND (both driven by global filters) ────────────────────
    st.markdown('<div class="section-header">Regional Burden \u2014 Map & Trend (coordinated by year/measure above)</div>', unsafe_allow_html=True)
    r2c1, r2c2 = st.columns([1.1, 1])
    with r2c1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        map_df = prev_sel.copy()
        map_df['ISO'] = map_df['location_name'].map(ISO_MAP)
        fig_map = px.choropleth(map_df, locations='ISO', color='val',
                                hover_name='location_name', hover_data={'val':':.0f','ISO':False},
                                color_continuous_scale=['#f3e8ff','#9b59b6','#4a1a6e'],
                                labels={'val': f'{label_short} Rate'})
        fig_map.update_geos(scope='world', center=dict(lat=26, lon=45), projection_scale=3.3,
                            showland=True, landcolor='#f9f9f9', showocean=True, oceancolor='#e8f4fd', showframe=False)
        fig_map.update_layout(height=380, paper_bgcolor='white', font=dict(family='Inter'),
                              margin=dict(t=10,b=10,l=10,r=10), coloraxis_colorbar=dict(title=label_short))
        st.plotly_chart(fig_map, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r2c2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        trend_df = gbd[(gbd['measure_name']==sel_measure) & (gbd['metric_name']=='Rate') &
                       (gbd['location_name'].isin(sel_countries))].sort_values(['location_name','year'])
        fig_trend = px.line(trend_df, x='year', y='val', color='location_name',
                            labels={'val': f'{label_short} Rate', 'year':'Year', 'location_name':'Country'})
        fig_trend.add_vline(x=sel_year, line_dash='dash', line_color='#b48ec9')
        fig_trend.update_layout(height=380, paper_bgcolor='white', plot_bgcolor='white',
                                font=dict(family='Inter'), legend_title='', margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig_trend, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── ROW 3: OBESITY LINK (uses sel_year too) ────────────────────────────────
    st.markdown('<div class="section-header">Obesity & PCOS Link \u2014 same year as above</div>', unsafe_allow_html=True)
    obesity_year = min(sel_year, 2022)
    pcos_yr = gbd[(gbd['measure_name']=='Prevalence') & (gbd['metric_name']=='Rate') & (gbd['year']==obesity_year)][['location_name','val']].rename(columns={'val':'PCOS_Rate'})
    obs_yr = obesity[obesity['Year']==obesity_year][['Entity','Obesity_Pct']].rename(columns={'Entity':'location_name'})
    merged = pcos_yr.merge(obs_yr, on='location_name').dropna()
    correlation = merged['PCOS_Rate'].corr(merged['Obesity_Pct']) if len(merged) > 2 else float('nan')

    r3c1, r3c2 = st.columns([1, 1])
    with r3c1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        fig_sc = px.scatter(merged, x='Obesity_Pct', y='PCOS_Rate', text='location_name', size='PCOS_Rate',
                            color='PCOS_Rate', color_continuous_scale=['#b48ec9','#6b2d8b'],
                            labels={'Obesity_Pct':'Female Obesity Rate (%)','PCOS_Rate':'PCOS Prevalence Rate'})
        fig_sc.update_traces(textposition='top center', textfont_size=8)
        fig_sc.update_layout(height=330, paper_bgcolor='white', plot_bgcolor='white', font=dict(family='Inter'),
                             showlegend=False, coloraxis_showscale=False, margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig_sc, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with r3c2:
        st.markdown(f"""<div class="info-box" style="height:330px; display:flex; flex-direction:column; justify-content:center;">
            <div style="font-size:2.2rem; font-weight:700; color:#6b2d8b;">r = {correlation:.2f}</div>
            <div style="font-weight:600; margin-top:4px;">Correlation: PCOS Rate vs Female Obesity ({obesity_year})</div>
            <div style="margin-top:10px;">A moderate-to-strong positive correlation links female obesity and PCOS prevalence across MENA \u2014
            GCC countries show both elevated obesity and high PCOS burden. Egypt is a notable outlier: highest female
            obesity (55.9%) yet only moderate PCOS prevalence, suggesting diagnostic access and genetics also matter.</div>
        </div>""", unsafe_allow_html=True)

    # ── ROW 4: PATIENT PROFILE (BMI / Symptoms) coordinated by a local PCOS filter ──
    st.markdown('<div class="section-header">Patient-Level Clinical Profile (541 patients)</div>', unsafe_allow_html=True)
    r4c1, r4c2, r4c3 = st.columns(3)
    with r4c1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**BMI Distribution by PCOS Status**")
        fig_bmi = px.histogram(clinical, x='BMI', color='PCOS_Label', nbins=25, barmode='overlay', opacity=0.75,
                               color_discrete_map={'PCOS Positive':'#9b59b6','PCOS Negative':'#b48ec9'})
        fig_bmi.update_layout(height=290, paper_bgcolor='white', plot_bgcolor='white', font=dict(family='Inter'),
                              legend_title='', legend=dict(orientation='h', y=1.15), margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig_bmi, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with r4c2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Symptom Prevalence: PCOS+ vs PCOS\u2212**")
        symptoms = {'WeightGain':'Weight Gain','HairGrowth':'Hair Growth','SkinDarkening':'Skin Darkening',
                    'HairLoss':'Hair Loss','Pimples':'Acne'}
        sym_data = []
        for col, label in symptoms.items():
            for status in [0,1]:
                pct = clinical[clinical['PCOS']==status][col].mean()*100
                sym_data.append({'Symptom':label,'Status':'PCOS+' if status==1 else 'PCOS-','Prevalence (%)':round(pct,1)})
        sym_df = pd.DataFrame(sym_data)
        fig_sym = px.bar(sym_df, x='Symptom', y='Prevalence (%)', color='Status', barmode='group',
                         color_discrete_map={'PCOS+':'#9b59b6','PCOS-':'#b48ec9'})
        fig_sym.update_layout(height=290, paper_bgcolor='white', plot_bgcolor='white', font=dict(family='Inter'),
                              legend_title='', legend=dict(orientation='h', y=1.15), margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig_sym, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with r4c3:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Hormone Correlation Matrix**")
        corr_cols = ['FSH','LH','AMH','TSH','PRL','VitD3','BMI','PCOS']
        corr = clinical[corr_cols].dropna().corr().round(2)
        fig_corr = px.imshow(corr, text_auto=True, aspect='auto',
                             color_continuous_scale=['#4a1a6e','white','#9b59b6'])
        fig_corr.update_layout(height=290, paper_bgcolor='white', font=dict(family='Inter', size=9),
                               margin=dict(t=10,b=10,l=10,r=10), coloraxis_showscale=False)
        st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── ROW 5: PREDICTOR ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">PCOS Risk Predictor (Quick Mode)</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    defaults = clinical[features].median(numeric_only=True)

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        age = st.number_input("Age", 18, 55, 28)
        bmi = st.number_input("BMI", 15.0, 50.0, 24.0)
    with p2:
        weight_gain = st.selectbox("Weight Gain", [0,1], format_func=lambda x: "Yes" if x else "No")
        hair_growth = st.selectbox("Hair Growth", [0,1], format_func=lambda x: "Yes" if x else "No")
    with p3:
        skin_dark = st.selectbox("Skin Darkening", [0,1], format_func=lambda x: "Yes" if x else "No")
        hair_loss = st.selectbox("Hair Loss", [0,1], format_func=lambda x: "Yes" if x else "No")
    with p4:
        pimples = st.selectbox("Acne", [0,1], format_func=lambda x: "Yes" if x else "No")
        irregular = st.selectbox("Irregular Periods", [0,1], format_func=lambda x: "Yes" if x else "No")

    fsh, lh, amh, tsh, prl, vitd3 = (defaults['FSH'], defaults['LH'], defaults['AMH'], defaults['TSH'], defaults['PRL'], defaults['VitD3'])
    waisthip = defaults['WaistHip']
    follicleL = defaults['FollicleL'] + (3 if irregular else 0)
    follicleR = defaults['FollicleR'] + (3 if irregular else 0)
    fastfood, exercise = int(defaults['FastFood']), int(defaults['Exercise'])

    if st.button("Predict PCOS Risk", use_container_width=True):
        input_data = pd.DataFrame([[age,bmi,fsh,lh,amh,tsh,prl,vitd3,waisthip,
                                    weight_gain,hair_growth,skin_dark,hair_loss,
                                    pimples,fastfood,exercise,follicleL,follicleR]], columns=features)
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0]
        risk_pct = prob[1]*100
        rc1, rc2 = st.columns([1,2])
        with rc1:
            if pred==1:
                st.error(f"**PCOS Likely** \u2014 Risk Score: **{risk_pct:.1f}%**")
            else:
                st.success(f"**PCOS Unlikely** \u2014 Risk Score: **{risk_pct:.1f}%**")
        with rc2:
            fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=risk_pct,
                gauge={'axis':{'range':[0,100]},'bar':{'color':'#9b59b6'},
                       'steps':[{'range':[0,33],'color':'#d5f5e3'},{'range':[33,66],'color':'#fdebd0'},{'range':[66,100],'color':'#f5b7b1'}]}))
            fig_gauge.update_layout(height=180, paper_bgcolor='white', font=dict(family='Inter'), margin=dict(t=10,b=10,l=10,r=10))
            st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown("""<div class="warning-box">Disclaimer: this tool is for educational and research purposes only and is not a substitute for clinical diagnosis.</div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── FOOTER: DATA SOURCES ────────────────────────────────────────────────────
    st.markdown("""<div class="info-box">
        <b>Data Sources:</b> IHME Global Burden of Disease 2023 (6,426 rows, 21 MENA countries, 1990\u20132023) \u00b7
        WHO Global Health Observatory via Our World in Data (9,270 rows, female obesity 1980\u20132024) \u00b7
        Clinical PCOS Dataset (541 patients, 44 variables).
    </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
