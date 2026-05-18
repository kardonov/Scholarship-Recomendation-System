import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
import lightgbm as lgb
import xgboost as xgb
import joblib
import io

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Beasiswa Recommendation System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

:root {
    --bg: #0b0f1a;
    --surface: #131929;
    --card: #1a2236;
    --border: #232f45;
    --accent: #3ecf8e;
    --accent2: #f7b731;
    --danger: #ff5e5e;
    --text: #e2e8f0;
    --muted: #7c8fa8;
}

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background: var(--bg); color: var(--text); }
.main { background: var(--bg); }
.block-container { padding: 1.5rem 2rem; max-width: 1400px; }

section[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] .block-container { padding: 1rem; }

.hero {
    background: linear-gradient(135deg, #0e1b30 0%, #12253d 50%, #0b1c2e 100%);
    border: 1px solid var(--border); border-radius: 16px;
    padding: 2rem 2.5rem; margin-bottom: 1.5rem; position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(62,207,142,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title { font-family: 'Syne', sans-serif; font-size: 2.1rem; font-weight: 800; color: #fff; margin: 0 0 0.35rem; }
.hero-sub { color: var(--muted); font-size: 0.92rem; margin: 0; }
.hero-badge {
    display: inline-block; background: rgba(62,207,142,0.12);
    border: 1px solid rgba(62,207,142,0.3); color: var(--accent);
    font-size: 0.7rem; font-weight: 600; padding: 0.2rem 0.65rem;
    border-radius: 50px; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 0.75rem;
}

.metric-row { display: flex; gap: 1rem; margin-bottom: 1.25rem; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 140px; background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.1rem 1.25rem; position: relative; overflow: hidden;
}
.metric-card::after { content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 3px; border-radius: 0 0 12px 12px; }
.metric-card.green::after { background: var(--accent); }
.metric-card.yellow::after { background: var(--accent2); }
.metric-card.red::after { background: var(--danger); }
.metric-card.blue::after { background: #4f91ff; }
.metric-label { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.07em; font-weight: 600; margin-bottom: 0.35rem; }
.metric-value { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 800; color: #fff; line-height: 1; }
.metric-sub { font-size: 0.72rem; color: var(--muted); margin-top: 0.2rem; }

.section-head {
    font-family: 'Syne', sans-serif; font-size: 1.2rem; font-weight: 700; color: #fff;
    margin: 1.5rem 0 0.75rem; padding-bottom: 0.5rem; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 0.5rem;
}

.model-tag {
    display: inline-block; font-size: 0.68rem; font-weight: 700; padding: 0.15rem 0.55rem;
    border-radius: 4px; text-transform: uppercase; letter-spacing: 0.06em;
}
.lgbm { background: rgba(62,207,142,0.15); color: var(--accent); border: 1px solid rgba(62,207,142,0.25); }
.xgb  { background: rgba(247,183,49,0.15); color: var(--accent2); border: 1px solid rgba(247,183,49,0.25); }

.pred-box { border-radius: 14px; padding: 1.5rem 2rem; text-align: center; margin: 1rem 0; border: 2px solid; }
.pred-layak  { background: rgba(62,207,142,0.08); border-color: rgba(62,207,142,0.4); }
.pred-tlayak { background: rgba(255,94,94,0.08);  border-color: rgba(255,94,94,0.4); }
.pred-icon   { font-size: 2.8rem; margin-bottom: 0.4rem; }
.pred-label  { font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 800; margin-bottom: 0.25rem; }
.pred-desc   { font-size: 0.85rem; color: var(--muted); }

.card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1.25rem 1.5rem; margin-bottom: 1rem; }

.stButton > button {
    background: var(--accent); color: #0b0f1a; font-weight: 700; border: none;
    border-radius: 8px; padding: 0.55rem 1.5rem; font-size: 0.9rem; transition: all 0.18s;
}
.stButton > button:hover { background: #2eb87a; transform: translateY(-1px); }

.stTabs [data-baseweb="tab-list"] { background: var(--surface); border-radius: 10px; padding: 0.25rem; gap: 0.25rem; }
.stTabs [data-baseweb="tab"] { background: transparent; color: var(--muted); border-radius: 8px; font-weight: 600; }
.stTabs [aria-selected="true"] { background: var(--card) !important; color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ─────────────────────────────────────────────────────────────────
DROP_COLS = [
    'Nama_Mahasiswa(2)', 'No_Pendaftaran(1)', 'NIK(3)', 'NISN(6)',
    'Jumlah_Tanggungan(28)', 'Tanggal_Lahir(15)', 'Jenis_Kelamin(16)',
    'Pendidikan Ayah', 'Pendidikan Ibu', 'Pekerjaan_Ibu(25)',
    'Pekerjaan_Ayah(21)', 'Status_Ayah(23)', 'Status_Ibu(27)',
    'Program_Studi_Pilihan 1(41)', 'Asal_Sekolah(11)', 'No_Handphone(18)',
    'Alamat_Email(19)', 'Program_Studi_Pilihan 2(42)'
]
CAT_COLS = [
    'Status_DTKS(7)', 'Status_P3KE(8)', 'Prestasi(50)',
    'Kepemilikan_Rumah(29)', 'Sumber_Daya_Listrik(31)',
    'Penghasilan_Ayah(22)', 'Penghasilan_Ibu(26)', 'Output_Class'
]

# Label deskriptif untuk kolom yang nilainya berupa angka di dataset.
# Key = nilai string yang muncul di dataset; Value = label ramah untuk user
LABEL_OVERRIDE = {
    'Prestasi(50)': {
        '0': 'Tidak Ada Prestasi',
        '1': 'Prestasi Tingkat Sekolah',
        '2': 'Prestasi Tingkat Kab/Kota',
        '3': 'Prestasi Tingkat Nasional',
    },
    'Kepemilikan_Rumah(29)': {
        '0': 'Milik Sendiri',
        '1': 'Sewa / Kontrak',
        '2': 'Numpang / Menumpang',
        '3': 'Lainnya',
    },
    'Sumber_Daya_Listrik(31)': {
        '0': 'Tidak Ada Listrik',
        '1': 'PLN (Listrik Negara)',
        '2': 'Genset / Generator',
        '3': 'Energi Solar / Panel Surya',
    },
}

PLOTLY_BASE = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='DM Sans', color='#e2e8f0'),
    margin=dict(l=20, r=20, t=45, b=20),
    xaxis=dict(gridcolor='#232f45', zerolinecolor='#232f45'),
    yaxis=dict(gridcolor='#232f45', zerolinecolor='#232f45'),
)

def pl(**kw):
    d = {**PLOTLY_BASE}
    d.update(kw)
    return d

# ─── DATA PROCESSING ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def process_data(raw_df):
    df = raw_df.copy()
    to_drop = [c for c in DROP_COLS if c in df.columns]
    df.drop(to_drop, axis=1, inplace=True)

    cate_val, cont_val = [], []
    for col in df.columns:
        if df[col].nunique() <= 20:
            cate_val.append(col)
        else:
            cont_val.append(col)

    # Simpan mapping label asli → encoded (sorted alphabetically by LabelEncoder)
    label_maps = {}   # col → {encoded_int: label_display_str}
    encode_maps = {}  # col → {label_display_str: encoded_int}
    le = LabelEncoder()
    for col in CAT_COLS:
        if col in df.columns:
            raw_vals = df[col].astype(str)
            classes = sorted(raw_vals.unique())   # urutan sama dengan LabelEncoder
            override = LABEL_OVERRIDE.get(col, {})
            # label_display: pakai override jika ada, fallback ke nilai asli dataset
            display_labels = [override.get(v, v) for v in classes]
            label_maps[col]  = {i: display_labels[i] for i in range(len(classes))}
            encode_maps[col] = {display_labels[i]: i  for i in range(len(classes))}
            df[col] = le.fit_transform(raw_vals)

    cont_no_target = [c for c in cont_val if c != 'Output_Class']
    if cont_no_target:
        sc = StandardScaler()
        df[cont_no_target] = sc.fit_transform(df[cont_no_target])

    feature = df.drop('Output_Class', axis=1)
    target  = df['Output_Class']
    X_train, X_test, y_train, y_test = train_test_split(
        feature, target, shuffle=True, test_size=0.3, random_state=42
    )
    return df, X_train, X_test, y_train, y_test, cate_val, cont_val, label_maps, encode_maps

# ─── MODEL TRAINING ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_lgbm_standard(_Xtr, _ytr):
    m = lgb.LGBMClassifier(verbose=-1, random_state=42)
    m.fit(_Xtr, _ytr); return m

@st.cache_resource(show_spinner=False)
def train_lgbm_goss(_Xtr, _ytr):
    m = lgb.LGBMClassifier(
        boosting_type='goss', verbose=-1, random_state=40,
        n_estimators=100, learning_rate=0.01, max_depth=3,
        num_leaves=20, feature_fraction=0.7
    )
    m.fit(_Xtr, _ytr); return m

@st.cache_resource(show_spinner=False)
def train_lgbm_dart(_Xtr, _ytr):
    m = lgb.LGBMClassifier(
        boosting_type='dart', verbose=-1, random_state=40,
        n_estimators=100, learning_rate=0.1, max_depth=5, num_leaves=31
    )
    m.fit(_Xtr, _ytr); return m

@st.cache_resource(show_spinner=False)
def train_lgbm_gbdt(_Xtr, _ytr):
    m = lgb.LGBMClassifier(
        boosting_type='gbdt', verbose=-1, random_state=40,
        n_estimators=100, learning_rate=0.1, max_depth=5, num_leaves=31
    )
    m.fit(_Xtr, _ytr); return m

@st.cache_resource(show_spinner=False)
def train_xgb_standard(_Xtr, _ytr):
    m = xgb.XGBClassifier(
        booster='gbtree', objective='binary:logistic',
        random_state=40, eval_metric='logloss', verbosity=0
    )
    m.fit(_Xtr, _ytr); return m

@st.cache_resource(show_spinner=False)
def train_xgb_gbtree(_Xtr, _ytr):
    m = xgb.XGBClassifier(
        booster='gbtree', objective='binary:logistic', random_state=40,
        eval_metric='logloss', verbosity=0,
        n_estimators=100, learning_rate=0.01, max_depth=3,
        subsample=0.7, colsample_bytree=0.7, min_child_weight=1,
        reg_alpha=0, reg_lambda=0
    )
    m.fit(_Xtr, _ytr); return m

@st.cache_resource(show_spinner=False)
def train_xgb_dart(_Xtr, _ytr):
    m = xgb.XGBClassifier(
        booster='dart', objective='binary:logistic', random_state=40,
        eval_metric='logloss', verbosity=0,
        n_estimators=100, learning_rate=0.1, max_depth=5
    )
    m.fit(_Xtr, _ytr); return m

def evaluate(model, X_test, y_test):
    yp = model.predict(X_test)
    ypr = model.predict_proba(X_test)[:, 1]
    return dict(
        accuracy=accuracy_score(y_test, yp),
        precision=precision_score(y_test, yp, zero_division=0),
        recall=recall_score(y_test, yp, zero_division=0),
        f1=f1_score(y_test, yp, zero_division=0),
        y_pred=yp, y_prob=ypr
    )

# ─── PLOT UTILS ────────────────────────────────────────────────────────────────
def plot_cm(y_test, y_pred, title):
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(
        cm, text_auto=True,
        color_continuous_scale=[[0,'#131929'],[0.5,'#1c4a6e'],[1,'#3ecf8e']],
        labels=dict(x='Predicted', y='Actual', color=''),
        x=['Tidak Layak','Layak'], y=['Tidak Layak','Layak'], title=title
    )
    fig.update_layout(**pl(title=dict(text=title, font=dict(size=14))))
    fig.update_coloraxes(showscale=False)
    return fig

def plot_roc_single(y_test, y_prob, title, color):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, line=dict(color=color, width=2.5), name=f'AUC={roc_auc:.4f}'))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], line=dict(color='#7c8fa8', dash='dash'), name='Baseline'))
    fig.update_layout(**pl(title=title, xaxis_title='FPR', yaxis_title='TPR'))
    return fig

def plot_fi(model, features, title, color):
    fi = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
    fi = fi.sort_values('Importance')
    fig = px.bar(fi, x='Importance', y='Feature', orientation='h', title=title,
                 color_discrete_sequence=[color])
    fig.update_layout(**pl(title=dict(text=title, font=dict(size=14)),
                           yaxis=dict(gridcolor='#232f45', zerolinecolor='#232f45',
                                      categoryorder='total ascending')))
    return fig

def metrics_html(acc, prec, rec, f1):
    return f"""
    <div class="metric-row">
        <div class="metric-card green"><div class="metric-label">Accuracy</div><div class="metric-value">{acc:.2%}</div></div>
        <div class="metric-card blue"><div class="metric-label">Precision</div><div class="metric-value">{prec:.2%}</div></div>
        <div class="metric-card yellow"><div class="metric-label">Recall</div><div class="metric-value">{rec:.2%}</div></div>
        <div class="metric-card red"><div class="metric-label">F1 Score</div><div class="metric-value">{f1:.2%}</div></div>
    </div>"""

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 1.5rem;">
        <div style="font-size:2.5rem;margin-bottom:0.5rem;">🎓</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;color:#fff;">Beasiswa AI</div>
        <div style="font-size:0.75rem;color:#7c8fa8;">Scholarship Recommendation System</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("### 📂 Upload Dataset")
    uploaded = st.file_uploader("Upload CSV dataset mahasiswa", type=['csv'],
                                help="CSV dengan kolom sesuai format dataset beasiswa")
    st.divider()
    st.markdown("### ⚙️ Navigasi")
    page = st.radio("Halaman", [
        "🏠 Dashboard", "📊 Data Exploration",
        "🤖 LightGBM Models", "⚡ XGBoost Models",
        "📈 Model Comparison", "🔮 Prediksi"
    ], label_visibility='collapsed')
    st.divider()
    st.markdown("""
    <div style="font-size:0.72rem;color:#7c8fa8;line-height:1.7;">
        <b style="color:#e2e8f0;">Model tersedia:</b><br>
        🟢 LGBM Standard · GOSS · DART · GBDT<br>
        🟡 XGB Standard · GBtree · DART<br><br>
        <b style="color:#e2e8f0;">Fitur:</b><br>
        ROC AUC · Confusion Matrix<br>
        Feature Importance · Live Prediction
    </div>""", unsafe_allow_html=True)

# ─── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">CRISP-DM · LightGBM & XGBoost · 7 Features</div>
    <div class="hero-title">🎓 Sistem Rekomendasi Beasiswa</div>
    <p class="hero-sub">Analisis kelayakan penerima beasiswa menggunakan Machine Learning — LightGBM dan XGBoost dengan fitur akademik dan sosial-ekonomi.</p>
</div>""", unsafe_allow_html=True)

# ─── GATE: Upload required ──────────────────────────────────────────────────────
if uploaded is None:
    st.info("⬅️ Upload dataset CSV terlebih dahulu di sidebar untuk memulai analisis.")
    st.markdown('<div class="section-head">📋 Format Kolom Dataset</div>', unsafe_allow_html=True)
    df_cols = pd.DataFrame({
        'Kolom': ['Status_DTKS(7)', 'Status_P3KE(8)', 'Penghasilan_Ayah(22)',
                  'Penghasilan_Ibu(26)', 'Kepemilikan_Rumah(29)',
                  'Sumber_Daya_Listrik(31)', 'Prestasi(50)',
                  'Rata-rata_Skor_Kuesioner', 'Nilai_Tes(51)', 'Output_Class'],
        'Tipe': ['Kategorikal']*7 + ['Numerik', 'Numerik', 'Target'],
        'Keterangan': [
            'Status Data Terpadu Kesejahteraan Sosial',
            'Status P3KE (Penghapusan Kemiskinan Ekstrem)',
            'Kategori penghasilan ayah',
            'Kategori penghasilan ibu',
            'Status kepemilikan rumah (Milik/Sewa/dll)',
            'Sumber daya listrik yang digunakan',
            'Prestasi akademik/non-akademik',
            'Rata-rata skor kuesioner (float)',
            'Nilai tes masuk (integer)',
            '0 = Layak Beasiswa, 1 = Tidak Layak'
        ]
    })
    st.dataframe(df_cols, use_container_width=True, hide_index=True)
    st.stop()

# ─── LOAD & PROCESS ────────────────────────────────────────────────────────────
try:
    raw_df = pd.read_csv(uploaded)
    df, X_train, X_test, y_train, y_test, cate_val, cont_val, label_maps, encode_maps = process_data(raw_df)
except Exception as e:
    st.error(f"❌ Error memproses data: {e}")
    st.stop()

# Sidebar info
with st.sidebar:
    st.markdown(f"""
    <div class="card" style="margin-top:0.5rem;">
        <div style="font-size:0.72rem;color:#7c8fa8;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.5rem;">Dataset Info</div>
        <div style="display:flex;justify-content:space-between;margin-bottom:0.3rem;">
            <span style="color:#7c8fa8;font-size:0.8rem;">Total Rows</span>
            <span style="color:#e2e8f0;font-weight:600;font-size:0.8rem;">{len(raw_df):,}</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:0.3rem;">
            <span style="color:#7c8fa8;font-size:0.8rem;">Fitur</span>
            <span style="color:#e2e8f0;font-weight:600;font-size:0.8rem;">{X_train.shape[1]}</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:0.3rem;">
            <span style="color:#7c8fa8;font-size:0.8rem;">Train</span>
            <span style="color:#3ecf8e;font-weight:600;font-size:0.8rem;">{len(X_train):,}</span>
        </div>
        <div style="display:flex;justify-content:space-between;">
            <span style="color:#7c8fa8;font-size:0.8rem;">Test</span>
            <span style="color:#f7b731;font-weight:600;font-size:0.8rem;">{len(X_test):,}</span>
        </div>
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
#  DASHBOARD
# ════════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    n_layak  = int((df['Output_Class'] == 0).sum()) if 'Output_Class' in df.columns else 0
    n_tlayak = int((df['Output_Class'] == 1).sum()) if 'Output_Class' in df.columns else 0

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card green">
            <div class="metric-label">Total Mahasiswa</div>
            <div class="metric-value">{len(raw_df):,}</div>
            <div class="metric-sub">Records dalam dataset</div>
        </div>
        <div class="metric-card blue">
            <div class="metric-label">Jumlah Fitur</div>
            <div class="metric-value">{X_train.shape[1]}</div>
            <div class="metric-sub">Setelah preprocessing</div>
        </div>
        <div class="metric-card green">
            <div class="metric-label">Layak Beasiswa</div>
            <div class="metric-value">{n_layak:,}</div>
            <div class="metric-sub">{n_layak/max(len(df),1)*100:.1f}% dari total</div>
        </div>
        <div class="metric-card red">
            <div class="metric-label">Tidak Layak</div>
            <div class="metric-value">{n_tlayak:,}</div>
            <div class="metric-sub">{n_tlayak/max(len(df),1)*100:.1f}% dari total</div>
        </div>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-head">📊 Distribusi Kelas Output</div>', unsafe_allow_html=True)
        vc = df['Output_Class'].value_counts().reset_index()
        vc.columns = ['Class', 'Count']
        vc['Label'] = vc['Class'].map({0: 'Layak', 1: 'Tidak Layak'})
        fig = px.bar(vc, x='Label', y='Count', color='Label',
                     color_discrete_map={'Layak': '#3ecf8e', 'Tidak Layak': '#ff5e5e'},
                     title='Distribusi Kelas')
        fig.update_layout(**pl())
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown('<div class="section-head">📋 Statistik Deskriptif</div>', unsafe_allow_html=True)
        st.dataframe(raw_df.describe().round(2), use_container_width=True)

    st.markdown('<div class="section-head">👁️ Sample Data (10 baris pertama)</div>', unsafe_allow_html=True)
    st.dataframe(raw_df.head(10), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
#  DATA EXPLORATION
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📊 Data Exploration":
    st.markdown('<div class="section-head">🔍 Data Exploration & Understanding</div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📋 Info Dataset", "📊 Distribusi Fitur", "🔥 Korelasi"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Missing Values**")
            miss = raw_df.isnull().sum().reset_index()
            miss.columns = ['Kolom', 'Missing']
            miss['%'] = (miss['Missing'] / len(raw_df) * 100).round(2)
            st.dataframe(miss, use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**Tipe Data & Unique Values**")
            info_df = pd.DataFrame({
                'Kolom': raw_df.columns,
                'Dtype': raw_df.dtypes.values,
                'Unique': [raw_df[c].nunique() for c in raw_df.columns]
            })
            st.dataframe(info_df, use_container_width=True, hide_index=True)
        st.markdown(f"**Duplikat:** `{raw_df.duplicated().sum()}` rows")
        if 'Output_Class' in raw_df.columns:
            st.markdown("**Distribusi Output_Class**")
            st.dataframe(raw_df['Output_Class'].value_counts().reset_index(), use_container_width=True, hide_index=True)

    with tab2:
        cat_opts = [c for c in CAT_COLS if c in raw_df.columns and c != 'Output_Class']
        if cat_opts:
            sel_cat = st.selectbox("Pilih Fitur Kategorikal", cat_opts)
            vc2 = raw_df[sel_cat].value_counts().reset_index()
            vc2.columns = ['Value', 'Count']
            fig = px.bar(vc2, x='Value', y='Count', title=f'Distribusi: {sel_cat}',
                         color_discrete_sequence=['#3ecf8e'])
            fig.update_layout(**pl())
            st.plotly_chart(fig, use_container_width=True)

        num_opts = [c for c in raw_df.columns if raw_df[c].dtype in [np.float64, np.int64] and c != 'Output_Class']
        if num_opts:
            sel_num = st.selectbox("Pilih Fitur Numerik", num_opts)
            fig2 = px.histogram(raw_df, x=sel_num, nbins=30, title=f'Histogram: {sel_num}',
                                color_discrete_sequence=['#4f91ff'])
            fig2.update_layout(**pl())
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        num_df = df.select_dtypes(include=[np.number])
        corr = num_df.corr()
        fig_c = px.imshow(corr.round(2), text_auto=True,
                          color_continuous_scale='RdBu_r', title='Correlation Matrix')
        fig_c.update_layout(**pl(title=dict(text='Correlation Matrix', font=dict(size=15))))
        st.plotly_chart(fig_c, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
#  LIGHTGBM
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🤖 LightGBM Models":
    st.markdown('<div class="section-head"><span class="model-tag lgbm">LightGBM</span> Model Training & Evaluation</div>', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["🟢 Standard", "🌀 GOSS", "🎯 DART", "📐 GBDT"])

    def lgbm_section(model_fn, model_label, caption, color, args):
        if caption:
            st.caption(caption)
        with st.spinner(f"Training {model_label}..."):
            m = model_fn(*args)
        ev = evaluate(m, X_test, y_test)
        st.markdown(metrics_html(ev['accuracy'], ev['precision'], ev['recall'], ev['f1']), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_cm(y_test, ev['y_pred'], f'Confusion Matrix — {model_label}'), use_container_width=True)
        with c2:
            st.plotly_chart(plot_roc_single(y_test, ev['y_prob'], f'ROC Curve — {model_label}', color), use_container_width=True)
        st.plotly_chart(plot_fi(m, X_train.columns.tolist(), f'Feature Importance — {model_label}', color), use_container_width=True)

    with tab1:
        st.markdown("#### LightGBM Standard (Without Hyperparameter Tuning)")
        lgbm_section(train_lgbm_standard, 'LGBM Standard', None, '#3ecf8e', (X_train, y_train))
    with tab2:
        st.markdown("#### LightGBM — Boosting Type: **GOSS**")
        lgbm_section(train_lgbm_goss, 'LGBM GOSS',
                     "Best params: n_estimators=100, lr=0.01, max_depth=3, num_leaves=20, feature_fraction=0.7",
                     '#3ecf8e', (X_train, y_train))
    with tab3:
        st.markdown("#### LightGBM — Boosting Type: **DART**")
        lgbm_section(train_lgbm_dart, 'LGBM DART',
                     "Best params: n_estimators=100, lr=0.1, max_depth=5, num_leaves=31",
                     '#b066ff', (X_train, y_train))
    with tab4:
        st.markdown("#### LightGBM — Boosting Type: **GBDT**")
        lgbm_section(train_lgbm_gbdt, 'LGBM GBDT',
                     "Best params: n_estimators=100, lr=0.1, max_depth=5, num_leaves=31",
                     '#4f91ff', (X_train, y_train))

# ════════════════════════════════════════════════════════════════════════════════
#  XGBOOST
# ════════════════════════════════════════════════════════════════════════════════
elif page == "⚡ XGBoost Models":
    st.markdown('<div class="section-head"><span class="model-tag xgb">XGBoost</span> Model Training & Evaluation</div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["🟡 Standard (GBtree)", "🌳 Tuned GBtree", "🎯 DART"])

    def xgb_section(model_fn, label, caption, color, args):
        if caption:
            st.caption(caption)
        with st.spinner(f"Training {label}..."):
            m = model_fn(*args)
        ev = evaluate(m, X_test, y_test)
        st.markdown(metrics_html(ev['accuracy'], ev['precision'], ev['recall'], ev['f1']), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_cm(y_test, ev['y_pred'], f'Confusion Matrix — {label}'), use_container_width=True)
        with c2:
            st.plotly_chart(plot_roc_single(y_test, ev['y_prob'], f'ROC Curve — {label}', color), use_container_width=True)
        st.plotly_chart(plot_fi(m, X_train.columns.tolist(), f'Feature Importance — {label}', color), use_container_width=True)

    with tab1:
        st.markdown("#### XGBoost Standard — Booster: **gbtree** (Without Tuning)")
        xgb_section(train_xgb_standard, 'XGB Standard', None, '#f7b731', (X_train, y_train))
    with tab2:
        st.markdown("#### XGBoost Tuned — Booster: **gbtree**")
        xgb_section(train_xgb_gbtree, 'XGB GBtree',
                    "Best params: n_estimators=100, lr=0.01, max_depth=3, subsample=0.7, colsample_bytree=0.7",
                    '#f7b731', (X_train, y_train))
    with tab3:
        st.markdown("#### XGBoost — Booster: **DART**")
        xgb_section(train_xgb_dart, 'XGB DART',
                    "Params: n_estimators=100, lr=0.1, max_depth=5",
                    '#ff9f40', (X_train, y_train))

# ════════════════════════════════════════════════════════════════════════════════
#  MODEL COMPARISON
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Comparison":
    st.markdown('<div class="section-head">📈 Perbandingan Semua Model</div>', unsafe_allow_html=True)

    with st.spinner("Training semua 7 model..."):
        all_models = {
            'LGBM Standard': train_lgbm_standard(X_train, y_train),
            'LGBM GOSS':     train_lgbm_goss(X_train, y_train),
            'LGBM DART':     train_lgbm_dart(X_train, y_train),
            'LGBM GBDT':     train_lgbm_gbdt(X_train, y_train),
            'XGB Standard':  train_xgb_standard(X_train, y_train),
            'XGB GBtree':    train_xgb_gbtree(X_train, y_train),
            'XGB DART':      train_xgb_dart(X_train, y_train),
        }

    rows = []
    for name, model in all_models.items():
        ev = evaluate(model, X_test, y_test)
        rows.append({'Model': name, 'Accuracy': ev['accuracy'],
                     'Precision': ev['precision'], 'Recall': ev['recall'], 'F1 Score': ev['f1']})
    comp = pd.DataFrame(rows).sort_values('F1 Score', ascending=False)

    fmt = {c: '{:.4f}' for c in ['Accuracy', 'Precision', 'Recall', 'F1 Score']}
    st.dataframe(comp.style.format(fmt).highlight_max(
        subset=['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        color='rgba(62,207,142,0.25)'), use_container_width=True, hide_index=True)

    # Grouped bar
    fig_bar = go.Figure()
    for m, c in zip(['Accuracy','Precision','Recall','F1 Score'], ['#3ecf8e','#4f91ff','#f7b731','#ff5e5e']):
        fig_bar.add_trace(go.Bar(name=m, x=comp['Model'], y=comp[m], marker_color=c, opacity=0.85))
    fig_bar.update_layout(**pl(title='Perbandingan Metrik', barmode='group',
                               legend=dict(bgcolor='rgba(0,0,0,0)')))
    st.plotly_chart(fig_bar, use_container_width=True)

    # ROC comparison key models
    st.markdown('<div class="section-head">📉 ROC Curve — LGBM GOSS vs XGB GBtree</div>', unsafe_allow_html=True)
    fig_roc = go.Figure()
    for name, color in [('LGBM GOSS','#3ecf8e'), ('XGB GBtree','#f7b731')]:
        m = all_models[name]
        y_prob = m.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f'{name} (AUC={roc_auc:.4f})',
                                     line=dict(color=color, width=2.5)))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], line=dict(color='#7c8fa8', dash='dash'), name='Baseline'))
    fig_roc.update_layout(**pl(xaxis_title='FPR', yaxis_title='TPR',
                               legend=dict(bgcolor='rgba(0,0,0,0)')))
    st.plotly_chart(fig_roc, use_container_width=True)

    # All models ROC
    st.markdown('<div class="section-head">📉 ROC Curve — Semua Model</div>', unsafe_allow_html=True)
    COLORS = ['#3ecf8e','#2ab574','#b066ff','#4f91ff','#f7b731','#ff9f40','#ff5e5e']
    fig_all = go.Figure()
    for (name, model), color in zip(all_models.items(), COLORS):
        y_prob = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        fig_all.add_trace(go.Scatter(x=fpr, y=tpr, name=f'{name} ({roc_auc:.4f})',
                                     line=dict(color=color, width=2)))
    fig_all.add_trace(go.Scatter(x=[0,1], y=[0,1], line=dict(color='#7c8fa8', dash='dash'), name='Baseline'))
    fig_all.update_layout(**pl(title='ROC Curve — All Models', xaxis_title='FPR', yaxis_title='TPR',
                               legend=dict(bgcolor='rgba(0,0,0,0)')))
    st.plotly_chart(fig_all, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
#  PREDIKSI
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Prediksi":
    st.markdown('<div class="section-head">🔮 Prediksi Kelayakan Beasiswa</div>', unsafe_allow_html=True)
    st.markdown("Masukkan data mahasiswa untuk memprediksi kelayakan penerimaan beasiswa secara real-time.")

    with st.spinner("Memuat model prediksi..."):
        pred_lgbm = train_lgbm_goss(X_train, y_train)
        pred_xgb  = train_xgb_gbtree(X_train, y_train)

    # ── Helper: bangun options selectbox dari label_maps (label asli dataset) ──
    def cat_options(col):
        """Return list of label strings dari dataset asli, sorted by encoded index."""
        if col in label_maps:
            return [label_maps[col][i] for i in sorted(label_maps[col].keys())]
        return []

    def to_encoded(col, label):
        """Konversi label string → integer encoded untuk model."""
        return encode_maps.get(col, {}).get(str(label), 0)

    # ── Form input ──
    with st.form("pred_form"):
        st.markdown("### 📝 Input Data Mahasiswa")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Sosial Ekonomi**")

            opts_dtks = cat_options('Status_DTKS(7)')
            sel_dtks = st.selectbox(
                "Status DTKS",
                options=opts_dtks if opts_dtks else list(range(6)),
                help="Pilih status DTKS mahasiswa sesuai data"
            )

            opts_p3ke = cat_options('Status_P3KE(8)')
            sel_p3ke = st.selectbox(
                "Status P3KE",
                options=opts_p3ke if opts_p3ke else list(range(10))
            )

            opts_rmh = cat_options('Kepemilikan_Rumah(29)')
            sel_rmh = st.selectbox(
                "Kepemilikan Rumah",
                options=opts_rmh if opts_rmh else ["Milik Sendiri","Sewa/Kontrak","Numpang","Lainnya"]
            )

        with c2:
            st.markdown("**Data Penghasilan**")

            opts_payah = cat_options('Penghasilan_Ayah(22)')
            sel_payah = st.selectbox(
                "Penghasilan Ayah",
                options=opts_payah if opts_payah else list(range(15))
            )

            opts_pibu = cat_options('Penghasilan_Ibu(26)')
            sel_pibu = st.selectbox(
                "Penghasilan Ibu",
                options=opts_pibu if opts_pibu else list(range(10))
            )

            opts_listrik = cat_options('Sumber_Daya_Listrik(31)')
            sel_listrik = st.selectbox(
                "Sumber Daya Listrik",
                options=opts_listrik if opts_listrik else ["Tidak Ada","PLN","Genset","Solar"]
            )

        with c3:
            st.markdown("**Akademik**")

            opts_prestasi = cat_options('Prestasi(50)')
            sel_prestasi = st.selectbox(
                "Prestasi",
                options=opts_prestasi if opts_prestasi else ["Tidak Ada","Sekolah","Kab/Kota","Nasional"]
            )

            skor_kuesioner = st.slider("Rata-rata Skor Kuesioner", 0.0, 10.0, 7.5, 0.1)
            nilai_tes      = st.slider("Nilai Tes", 0, 100, 75)

        st.markdown("---")
        sel_model = st.radio(
            "Pilih Model",
            ["LightGBM GOSS", "XGBoost GBtree", "Keduanya"],
            horizontal=True
        )
        submitted = st.form_submit_button("🔮 Prediksi Sekarang", use_container_width=True)

    if submitted:
        # Konversi pilihan label → nilai encoded untuk model
        enc_dtks    = to_encoded('Status_DTKS(7)',          sel_dtks)
        enc_p3ke    = to_encoded('Status_P3KE(8)',          sel_p3ke)
        enc_payah   = to_encoded('Penghasilan_Ayah(22)',    sel_payah)
        enc_pibu    = to_encoded('Penghasilan_Ibu(26)',     sel_pibu)
        enc_rmh     = to_encoded('Kepemilikan_Rumah(29)',   sel_rmh)
        enc_listrik = to_encoded('Sumber_Daya_Listrik(31)', sel_listrik)
        enc_prst    = to_encoded('Prestasi(50)',             sel_prestasi)

        new_data = pd.DataFrame({
            'Status_DTKS(7)':           [enc_dtks],
            'Status_P3KE(8)':           [enc_p3ke],
            'Penghasilan_Ayah(22)':     [enc_payah],
            'Penghasilan_Ibu(26)':      [enc_pibu],
            'Kepemilikan_Rumah(29)':    [enc_rmh],
            'Sumber_Daya_Listrik(31)':  [enc_listrik],
            'Prestasi(50)':             [enc_prst],
            'Rata-rata_Skor_Kuesioner': [skor_kuesioner],
            'Nilai_Tes(51)':            [nilai_tes]
        })
        for col in X_train.columns:
            if col not in new_data.columns:
                new_data[col] = 0
        new_data = new_data[X_train.columns]

        # ── Tampilkan hasil prediksi ──
        def show_pred(pred, prob_not_eligible, model_name):
            if pred == 0:
                st.markdown(f"""
                <div class="pred-box pred-layak">
                    <div class="pred-icon">✅</div>
                    <div class="pred-label" style="color:#3ecf8e;">LAYAK BEASISWA</div>
                    <div class="pred-desc">{model_name} — Prob. tidak layak: {prob_not_eligible:.2%}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pred-box pred-tlayak">
                    <div class="pred-icon">❌</div>
                    <div class="pred-label" style="color:#ff5e5e;">TIDAK LAYAK BEASISWA</div>
                    <div class="pred-desc">{model_name} — Prob. tidak layak: {prob_not_eligible:.2%}</div>
                </div>""", unsafe_allow_html=True)

        prob_lgbm, prob_xgb = None, None

        if sel_model in ["LightGBM GOSS", "Keduanya"]:
            p = pred_lgbm.predict(new_data)[0]
            prob_lgbm = pred_lgbm.predict_proba(new_data)[0][1]
            st.markdown('<div class="section-head"><span class="model-tag lgbm">LightGBM GOSS</span> Hasil</div>',
                        unsafe_allow_html=True)
            show_pred(p, prob_lgbm, "LightGBM GOSS")

        if sel_model in ["XGBoost GBtree", "Keduanya"]:
            p = pred_xgb.predict(new_data)[0]
            prob_xgb = pred_xgb.predict_proba(new_data)[0][1]
            st.markdown('<div class="section-head"><span class="model-tag xgb">XGBoost GBtree</span> Hasil</div>',
                        unsafe_allow_html=True)
            show_pred(p, prob_xgb, "XGBoost GBtree")

        if sel_model == "Keduanya" and prob_lgbm is not None and prob_xgb is not None:
            fig_g = make_subplots(rows=1, cols=2, specs=[[{"type":"indicator"},{"type":"indicator"}]])
            for gcol, val, gname, gcolor in [
                (1, prob_lgbm*100, "LGBM GOSS<br>Prob. Tidak Layak (%)", "#3ecf8e"),
                (2, prob_xgb*100,  "XGB GBtree<br>Prob. Tidak Layak (%)", "#f7b731")
            ]:
                fig_g.add_trace(go.Indicator(
                    mode="gauge+number", value=val,
                    title={'text': gname},
                    gauge=dict(axis=dict(range=[0,100]), bar=dict(color=gcolor),
                               bgcolor="#1a2236",
                               steps=[dict(range=[0,50],color="#132318"),
                                      dict(range=[50,100],color="#2d1212")])
                ), row=1, col=gcol)
            fig_g.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(family='DM Sans', color='#e2e8f0'), height=280)
            st.plotly_chart(fig_g, use_container_width=True)

        # ── Ringkasan input — tampilkan label asli (bukan angka) ──
        st.markdown('<div class="section-head">📋 Ringkasan Input</div>', unsafe_allow_html=True)
        input_disp = pd.DataFrame({
            'Fitur': [
                'Status DTKS', 'Status P3KE', 'Penghasilan Ayah', 'Penghasilan Ibu',
                'Kepemilikan Rumah', 'Sumber Daya Listrik', 'Prestasi',
                'Skor Kuesioner', 'Nilai Tes'
            ],
            'Label (Pilihan)': [
                sel_dtks, sel_p3ke, sel_payah, sel_pibu,
                sel_rmh, sel_listrik, sel_prestasi,
                skor_kuesioner, nilai_tes
            ],
            'Nilai Encoded (Model)': [
                enc_dtks, enc_p3ke, enc_payah, enc_pibu,
                enc_rmh, enc_listrik, enc_prst,
                skor_kuesioner, nilai_tes
            ]
        })
        st.dataframe(input_disp, use_container_width=True, hide_index=True)

        # ── Download model ──
        st.markdown('<div class="section-head">💾 Download Model</div>', unsafe_allow_html=True)
        dc1, dc2 = st.columns(2)
        with dc1:
            buf = io.BytesIO(); joblib.dump(pred_lgbm, buf)
            st.download_button("⬇️ Download LGBM Model", data=buf.getvalue(),
                               file_name="lgbm_scholarship.pkl", mime="application/octet-stream")
        with dc2:
            buf2 = io.BytesIO(); joblib.dump(pred_xgb, buf2)
            st.download_button("⬇️ Download XGB Model", data=buf2.getvalue(),
                               file_name="xgb_scholarship.pkl", mime="application/octet-stream")