from __future__ import annotations

import io
import time
import base64
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Nightwatch | Network Intelligence", page_icon="◈", layout="wide")

# Visual identity: a compact cyber-operations console with a violet/pink signal palette.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--ink:#f8efff;--muted:#a995b6;--line:#392941;--panel:#151020;--pink:#ff62c7;--mint:#65f1d0}
html,body,[class*="css"]{font-family:'Manrope',sans-serif}
.stApp{background:radial-gradient(ellipse at 72% 4%,rgba(121,47,132,.24),transparent 32%),radial-gradient(ellipse at 28% 47%,rgba(81,36,120,.15),transparent 36%),#0b0812;color:var(--ink)}
.block-container{padding-top:1.45rem;max-width:1500px}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#100b19,#0c0913);border-right:1px solid #32243b}
[data-testid="stSidebar"] *{color:#d8c8e1}
[data-testid="stMetric"]{background:linear-gradient(135deg,rgba(37,25,48,.96),rgba(20,15,31,.96));border:1px solid #46304f;padding:18px 20px;border-radius:17px;box-shadow:0 10px 28px #0002}
[data-testid="stMetricLabel"]{color:#ad9ab9!important;font-size:.78rem}
[data-testid="stMetricValue"]{color:#fff3ff!important}
[data-testid="stFileUploader"]{background:#151020;border:1px dashed #815092;border-radius:15px;padding:12px}
.stButton>button{background:linear-gradient(100deg,#e452b5,#9360f4);border:0;color:white;border-radius:11px;font-weight:800;padding:.55rem 1.2rem;box-shadow:0 5px 22px #e452b533}
.stButton>button:hover{border:0;color:white;filter:brightness(1.12)}
[data-testid="stDataFrame"]{border:1px solid #382940;border-radius:12px;overflow:hidden}
[data-testid="stAlert"]{border-radius:13px}
hr{border-color:#31243a}
.nw-kicker{color:#b68ac8;font-family:'DM Mono',monospace;font-size:11px;letter-spacing:2px;text-transform:uppercase}
.nw-title{font-size:37px;font-weight:800;letter-spacing:-1.5px;margin:0;color:#fff6ff}
.nw-sub{color:#aa99b4;margin-top:4px;font-size:14px}
.nw-chip{display:inline-block;border:1px solid #4c3655;background:#1c1424;border-radius:22px;padding:8px 13px;color:#d9c5e2;font:11px 'DM Mono',monospace}
.nw-card{background:linear-gradient(145deg,rgba(28,20,38,.96),rgba(17,13,25,.96));border:1px solid #3d2c46;border-radius:18px;padding:19px 20px}
.nw-note{color:#9d8ba8;font-size:12px;line-height:1.6}
.nw-pill{display:inline-block;border-radius:20px;padding:5px 10px;font-size:10px;font-weight:800;letter-spacing:.8px}
footer{visibility:hidden}
</style>
""", unsafe_allow_html=True)

# The home screen follows the reference's airy holographic-lab look.
st.markdown("""
<style>
.stApp{background:radial-gradient(ellipse at 77% 15%,rgba(208,184,245,.32),transparent 34%),linear-gradient(145deg,#f6f3fb,#eff2f9 55%,#f4eff8)!important;color:#352746!important}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#fdfbff,#f2eef8)!important;border-right:1px solid #e2d9ed!important}
[data-testid="stSidebar"] *{color:#55466b!important}
[data-testid="stMetric"]{background:rgba(255,255,255,.78)!important;border:1px solid #e5dced!important;box-shadow:0 12px 32px rgba(88,66,117,.08)!important}
[data-testid="stMetricLabel"]{color:#786889!important}[data-testid="stMetricValue"]{color:#352746!important}
[data-testid="stFileUploader"]{background:rgba(255,255,255,.72)!important;border:1px dashed #c4add9!important}
[data-testid="stDataFrame"]{border:1px solid #e4dced!important}
.nw-kicker{color:#8a63ac!important}.nw-title{color:#352746!important}.nw-sub{color:#776989!important}
.nw-chip{border-color:#e3d8ed!important;background:rgba(255,255,255,.6)!important;color:#68547d!important}
.nw-card{background:rgba(255,255,255,.7)!important;border-color:#e3d8ed!important}.nw-note{color:#786b84!important}
.nw-hero{display:grid;grid-template-columns:.86fr 1.14fr;gap:14px;align-items:center;margin:10px 0 24px;padding:28px 30px;border:1px solid rgba(255,255,255,.92);border-radius:28px;background:linear-gradient(125deg,rgba(255,255,255,.88),rgba(246,239,252,.72) 55%,rgba(230,239,250,.76));box-shadow:0 20px 65px rgba(90,69,123,.12);overflow:hidden}
.nw-hero-copy{padding:10px 0 12px 8px}.nw-hero-eyebrow{font:11px 'DM Mono',monospace;letter-spacing:2px;color:#936bac;text-transform:uppercase}
.nw-hero-title{font-size:39px;line-height:1.12;letter-spacing:-1.7px;font-weight:800;color:#38294c;margin:16px 0 14px}
.nw-hero-desc{font-size:14px;line-height:1.75;color:#786989;max-width:430px}
.nw-hero-points{display:flex;gap:8px;flex-wrap:wrap;margin-top:20px}.nw-hero-point{border:1px solid #e6dced;background:rgba(255,255,255,.58);border-radius:20px;padding:8px 11px;font-size:10px;color:#735d87}
.nw-hero-art{width:100%;height:auto;display:block;filter:drop-shadow(0 16px 30px rgba(102,77,143,.13));border-radius:22px}
@media(max-width:850px){.nw-hero{grid-template-columns:1fr;padding:20px}.nw-hero-title{font-size:31px}}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def make_demo_data(n: int = 1800) -> pd.DataFrame:
    """Create clearly labelled synthetic flows so the app works before a CSV is uploaded."""
    rng = np.random.default_rng(24)
    y = rng.binomial(1, .16, n)
    duration = np.where(y == 1, rng.lognormal(1.1, .9, n), rng.lognormal(.0, .7, n))
    packets = np.where(y == 1, rng.poisson(180, n), rng.poisson(42, n))
    byte_count = np.where(y == 1, rng.lognormal(10.7, 1.0, n), rng.lognormal(8.8, 1.0, n))
    rate = packets / np.maximum(duration, .01)
    return pd.DataFrame({"Flow Duration": duration, "Total Fwd Packets": packets,
                         "Flow Bytes/s": byte_count / np.maximum(duration, .01),
                         "Flow Packets/s": rate,
                         "Average Packet Size": byte_count / np.maximum(packets, 1),
                         "Label": np.where(y == 1, "Attack", "Benign")})

@st.cache_data(show_spinner=False)
def read_csv_bytes(data: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(data), low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(data), encoding="latin-1", low_memory=False)


def style_fig(fig, height=270):
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Manrope", color="#6f607f", size=11),
                      margin=dict(l=8, r=8, t=12, b=8), legend=dict(font=dict(color="#6f607f")))
    fig.update_xaxes(gridcolor="rgba(118,96,140,.14)", zerolinecolor="rgba(118,96,140,.14)")
    fig.update_yaxes(gridcolor="rgba(118,96,140,.14)", zerolinecolor="rgba(118,96,140,.14)")
    return fig

with st.sidebar:
    st.markdown("<div style='display:flex;align-items:center;gap:10px;margin:8px 2px 28px'><div style='background:linear-gradient(135deg,#ff62c7,#9462f6);border-radius:12px;width:36px;height:36px;display:grid;place-items:center;color:white;font-size:20px'>◈</div><div><b style='font-size:16px;letter-spacing:1px'>NIGHTWATCH</b><br><span style='font-size:9px;letter-spacing:1.5px;color:#9e85ac'>NETWORK INTELLIGENCE</span></div></div>", unsafe_allow_html=True)
    st.caption("МОНИТОРИНГ")
    page = st.radio("Навигация", ["Шолу", "Деректер зертханасы", "Жоба туралы"], label_visibility="collapsed")
    st.markdown("**Дерек файлы**")
    uploaded = st.file_uploader("CSV файлды жүктеу", type=["csv"], help="CIC-IDS2017 сияқты flow деректерінің CSV файлы", label_visibility="collapsed", key="traffic_csv")
    st.caption("Файл осы құрылғыда талданады.")
    st.divider()
    st.markdown("**◉ Қорғаныс белсенді**")
    st.caption("Оқу жобасының интерактивті демосы")

if uploaded:
    try:
        df = read_csv_bytes(uploaded.getvalue())
        df.columns = [str(c).strip() for c in df.columns]
        data_origin = f"Жүктелген файл · {uploaded.name}"
    except Exception as exc:
        st.error(f"CSV файлын оқу мүмкін болмады: {exc}")
        st.stop()
else:
    df = make_demo_data()
    data_origin = "Синтетикалық демо дерек"

# Drop empty and duplicate columns safely; preserve the unmodified row index for display.
df = df.loc[:, ~df.columns.duplicated()].copy()
df = df.dropna(axis=1, how="all")

st.markdown("<div class='nw-kicker'>КИБЕРҚАУІПСІЗДІК · MACHINE LEARNING</div>", unsafe_allow_html=True)
st.markdown("<div class='nw-title'>Қауіпті ерте байқа.</div><div class='nw-sub'>Желі белсенділігін бақыла · аномалияны анықта · әрекет ет</div>", unsafe_allow_html=True)
st.markdown(f"<div style='margin:14px 0 20px'><span class='nw-chip'>◉ {data_origin}</span>　<span class='nw-chip'>MODEL LAB · v1</span></div>", unsafe_allow_html=True)

if page == "Шолу":
    hero_path = Path(__file__).parent / "assets" / "orbit-hero.svg"
    hero_b64 = base64.b64encode(hero_path.read_bytes()).decode("ascii")
    st.markdown(f"""
    <section class="nw-hero">
      <div class="nw-hero-copy">
        <div class="nw-hero-eyebrow">NETWORK INTELLIGENCE · LIVE VIEW</div>
        <div class="nw-hero-title">Желі әлемін<br>қауіптен қорға.</div>
        <div class="nw-hero-desc">Nightwatch желі ағындарын зерттеп, әдеттен тыс белсенділікті байқатады. Деректі жүкте де, модельдің күмәнді деп тапқан қосылымдарын бір панельден қара.</div>
        <div class="nw-hero-points"><span class="nw-hero-point">✦ Аномалия радары</span><span class="nw-hero-point">◌ ML талдауы</span><span class="nw-hero-point">⌁ Интерактивті график</span></div>
      </div>
      <img class="nw-hero-art" alt="Голографиялық планета мен желі байланыстарын көрсететін иллюстрация" src="data:image/svg+xml;base64,{hero_b64}" />
    </section>
    """, unsafe_allow_html=True)

if page == "Жоба туралы":
    st.markdown("### Неге бұл құрал пайдалы?")
    st.write("Nightwatch желілік қосылымдардың сипаттамаларын қарап, күмәнді ағындарды маманның назарына ұсынады. Бұл оқу жобасының прототипі: ол трафикті автоматты түрде тоқтатпайды және нақты қауіпсіздік өнімі ретінде тексерілмеген.")
    st.markdown("#### Жоба ағыны")
    st.write("1. CSV трафик дерегін жүктеу.  2. Белгіленген деректе бірнеше классификаторды салыстыру.  3. Белгі жоқ деректе Isolation Forest арқылы ықтимал аномалияларды табу.")
    st.markdown("Дерек жиыны: [CIC-IDS2017 — Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html)")
    st.stop()

if page == "Деректер зертханасы":
    st.markdown("### Деректер зертханасы")
    st.caption("Деректердің бағандарын, бос мәндерін және сандық үлестірімін қараңыз.")
    left, right = st.columns([1.45, 1])
    with left:
        st.dataframe(df.head(12), use_container_width=True, height=350)
    with right:
        numeric = df.select_dtypes(include=np.number)
        st.metric("Жол саны", f"{len(df):,}")
        st.metric("Баған саны", f"{df.shape[1]:,}")
        st.metric("Бос ұяшық", f"{int(df.isna().sum().sum()):,}")
        if not numeric.empty:
            feature = st.selectbox("Үлестірімді көру", numeric.columns)
            fig = px.histogram(df, x=feature, nbins=35, color_discrete_sequence=["#da69d7"])
            st.plotly_chart(style_fig(fig, 260), use_container_width=True)
    st.stop()

# Main overview
numeric = df.select_dtypes(include=np.number).replace([np.inf, -np.inf], np.nan)
label_candidates = [c for c in df.columns if c.lower() in {"label", "class", "attack", "target", "category"}]
with st.expander("⚙️ Талдау параметрлері", expanded=False):
    label_col = st.selectbox("Нысана белгісі (егер деректе бар болса)", ["Аномалияны белгісіз табу"] + list(df.columns), index=(1 + df.columns.get_loc(label_candidates[0])) if label_candidates else 0)
    max_rows = st.slider("Оқытуға алынатын ең көп жол", 1000, 30000, 12000, 1000)

has_label = label_col != "Аномалияны белгісіз табу"
if has_label:
    n_classes = df[label_col].nunique(dropna=True)
    st.markdown("<div class='nw-note'>Белгіленген режим: бастапқы Dummy baseline және Random Forest / HistGradientBoosting салыстырылады. Белгі бағаны модельге feature ретінде берілмейді.</div>", unsafe_allow_html=True)
else:
    n_classes = 0
    st.markdown("<div class='nw-note'>Аномалия режимі: Isolation Forest қалыпты үлгіден ерекшеленетін қосылымдарды іздейді. Бұл режимде нақты шабуыл белгісі болмаса, метрикалар есептелмейді.</div>", unsafe_allow_html=True)

# Summary strip
s1,s2,s3,s4=st.columns(4)
s1.metric("ТАЛДАНҒАН ЖОЛ", f"{len(df):,}")
s2.metric("САНДЫҚ БЕЛГІ", f"{numeric.shape[1]:,}")
s3.metric("БЕЛГІ КЛАСЫ", f"{n_classes:,}" if has_label else "Жоқ")
s4.metric("БОС МӘН", f"{int(df.isna().sum().sum()):,}")

left, right = st.columns([1.45, 1], gap="large")
with left:
    st.markdown("#### ◉ Желі ағындарының шолуы")
    if has_label:
        counts = df[label_col].fillna("Бос").astype(str).value_counts().head(12).rename_axis("Класс").reset_index(name="Жазба")
        fig=px.bar(counts, x="Жазба", y="Класс", orientation="h", color="Жазба", color_continuous_scale=[[0,"#7045a7"],[1,"#ff62c7"]], text_auto=".2s")
        fig.update_layout(coloraxis_showscale=False, yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(style_fig(fig, 315), use_container_width=True)
    elif not numeric.empty:
        feat=numeric.columns[:min(5,len(numeric.columns))].tolist()
        trend=df[feat].head(120).reset_index(drop=True).reset_index(names="Ағын №")
        fig=px.line(trend, x="Ағын №", y=feat, color_discrete_sequence=["#f15fca","#8b69ff","#5de0c7","#ffba69","#7aaeff"])
        st.plotly_chart(style_fig(fig,315), use_container_width=True)
    else:
        st.info("График құру үшін сандық бағандар қажет.")
with right:
    st.markdown("#### ✦ Қауіп радары")
    if has_label:
        counts=df[label_col].fillna("Бос").astype(str).value_counts()
        pie=go.Figure(go.Pie(labels=counts.index[:8],values=counts.values[:8],hole=.72,marker_colors=["#ff62c7","#9567f5","#62e9d0","#ffbd69","#679aff","#bf71ff","#e8818f","#708295"],textinfo="none"))
        pie.add_annotation(text=f"{len(df):,}<br><span style='font-size:10px'>ағын</span>",x=.5,y=.5,showarrow=False,font=dict(size=23,color="#fff0ff"))
        st.plotly_chart(style_fig(pie,260), use_container_width=True)
    else:
        st.markdown("<div class='nw-card'><div class='nw-kicker'>АНОМАЛИЯНЫ АНЫҚТАУ</div><h2 style='margin:.6rem 0;color:#f5e9ff'>Isolation Forest</h2><div class='nw-note'>Белгіленбеген желі деректерінен сирек әрі ерекшеленетін үлгілерді табады. Нәтиже — тексеруге арналған белгі.</div></div>", unsafe_allow_html=True)

st.markdown("#### ⚡ Модель зертханасы")
run = st.button("▶　Трафикті талдау", type="primary", use_container_width=False)
if run:
    if numeric.shape[1] == 0:
        st.error("Модель үшін сандық feature бағаны табылмады.")
    else:
        clean = df.copy()
        clean[numeric.columns] = numeric
        clean = clean.dropna(subset=list(numeric.columns))
        if len(clean) > max_rows:
            clean = clean.sample(max_rows, random_state=42)
        X = clean.select_dtypes(include=np.number).replace([np.inf,-np.inf], np.nan)
        X = X.fillna(X.median()).clip(lower=-1e12, upper=1e12)
        if has_label:
            y_raw=clean[label_col].fillna("Unknown").astype(str)
            mask=y_raw.value_counts()[y_raw].values >= 2
            X=X.loc[mask]; y_raw=y_raw.loc[mask]
            if y_raw.nunique()<2:
                st.error("Салыстыру үшін нысана бағанында кемінде екі класс және әр класта екіден көп мысал болуы керек.")
            else:
                enc=LabelEncoder(); y=enc.fit_transform(y_raw)
                strat=y if pd.Series(y).value_counts().min() >= 2 else None
                Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=strat)
                models={"Baseline · көпшілік класы":DummyClassifier(strategy="most_frequent"),"Random Forest":RandomForestClassifier(n_estimators=120,max_depth=18,class_weight="balanced",n_jobs=-1,random_state=42),"Gradient Boosting":HistGradientBoostingClassifier(max_iter=100,learning_rate=.1,random_state=42)}
                results=[]; predictions={}; timings={}
                progress=st.progress(0)
                for i,(name,model) in enumerate(models.items()):
                    start=time.perf_counter(); model.fit(Xtr,ytr); pred=model.predict(Xte); timings[name]=time.perf_counter()-start
                    predictions[name]=pred
                    results.append({"Модель":name,"Accuracy":accuracy_score(yte,pred),"Precision (macro)":precision_score(yte,pred,average="macro",zero_division=0),"Recall (macro)":recall_score(yte,pred,average="macro",zero_division=0),"F1 (macro)":f1_score(yte,pred,average="macro",zero_division=0),"Оқыту, сек":round(timings[name],2)})
                    progress.progress((i+1)/len(models))
                res=pd.DataFrame(results).sort_values("F1 (macro)",ascending=False)
                st.success(f"Салыстыру аяқталды · {len(Xte):,} тест мысалы")
                st.dataframe(res.style.format({c:"{:.3f}" for c in ["Accuracy","Precision (macro)","Recall (macro)","F1 (macro)"]}),use_container_width=True,hide_index=True)
                chosen=st.selectbox("Қате болжамдарын қарау",list(models.keys()),index=list(models.keys()).index(res.iloc[0]["Модель"]))
                cm=confusion_matrix(yte,predictions[chosen])
                cf=px.imshow(cm,text_auto=True,x=[str(c) for c in enc.classes_],y=[str(c) for c in enc.classes_],color_continuous_scale=[[0,"#1a1423"],[1,"#ed57bd"]],labels=dict(x="Болжанған класс",y="Нақты класс",color="Саны"),aspect="auto")
                st.plotly_chart(style_fig(cf,330),use_container_width=True)
                st.caption("Ескерту: тест бөлінісі осы жүктелген файлдан алынған. Дерек жиыны уақыт бойынша бөлінбесе, нәтиже жаңа желідегі сапаны асыра көрсетуі мүмкін.")
        else:
            if len(X)<30:
                st.error("Аномалия моделін үйрету үшін кемінде 30 жарамды жол керек.")
            else:
                model=IsolationForest(n_estimators=180,contamination="auto",random_state=42,n_jobs=-1)
                start=time.perf_counter(); pred=model.fit_predict(X); elapsed=time.perf_counter()-start
                score=-model.score_samples(X)
                out=df.loc[X.index].copy(); out["Аномалия ұпайы"]=score; out["Модель нәтижесі"]=np.where(pred==-1,"КҮМӘНДІ","ҚАЛЫПТЫ")
                flagged=out[out["Модель нәтижесі"]=="КҮМӘНДІ"].sort_values("Аномалия ұпайы",ascending=False)
                st.success(f"Талдау аяқталды · {len(X):,} ағын · {elapsed:.2f} сек")
                a,b,c=st.columns(3); a.metric("Күмәнді белгі",f"{len(flagged):,}"); b.metric("Қалыпты белгі",f"{len(X)-len(flagged):,}"); c.metric("Үлес",f"{len(flagged)/len(X):.1%}")
                st.dataframe(flagged.head(100),use_container_width=True,hide_index=True)
                st.caption("Isolation Forest нәтижесі — аномалияға ұқсас үлгі. Бұл шабуылдың баталгаатай бар екенін білдірмейді.")

st.divider()
st.markdown("<div class='nw-note'>Оқу жобасының прототипі · Күмәнді белгі қосымша тексеруді қажет етеді. Сандық мәндер нақты файлды талдағаннан кейін есептеледі.</div>",unsafe_allow_html=True)


