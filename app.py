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

st.set_page_config(page_title="Nightwatch | Желі қауіпсіздігі", page_icon="◈", layout="wide")

# Nightwatch интерфейсі: голографиялық стиль, қазақша мәтін және оқуға жеңіл қаріптер.
st.markdown("""
<style>
:root{--ink:#f8efff;--muted:#a995b6;--line:#392941;--panel:#151020;--pink:#ff62c7;--mint:#65f1d0}
html,body,[class*="css"]{font-family:'Source Sans','Segoe UI',Arial,sans-serif}
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
.nw-kicker{color:#b68ac8;font-family:Consolas,monospace;font-size:12px;letter-spacing:1.5px}
.nw-title{font-family:Georgia,'Source Serif',serif;font-size:37px;font-weight:800;letter-spacing:-1.5px;margin:0;color:#fff6ff}
.nw-sub{color:#aa99b4;margin-top:4px;font-size:14px}
.nw-chip{display:inline-block;border:1px solid #4c3655;background:#1c1424;border-radius:22px;padding:8px 13px;color:#d9c5e2;font:12px Consolas,monospace}
.nw-card{background:linear-gradient(145deg,rgba(28,20,38,.96),rgba(17,13,25,.96));border:1px solid #3d2c46;border-radius:18px;padding:19px 20px}
.nw-note{color:#9d8ba8;font-size:12px;line-height:1.6}
.nw-pill{display:inline-block;border-radius:20px;padding:5px 10px;font-size:10px;font-weight:800;letter-spacing:.8px}
footer{visibility:hidden}
</style>
""", unsafe_allow_html=True)

# Басты бетте голографиялық зертхана стилі қолданылады.
st.markdown("""
<style>
.stApp{background:radial-gradient(ellipse at 77% 15%,rgba(208,184,245,.32),transparent 34%),linear-gradient(145deg,#f6f3fb,#eff2f9 55%,#f4eff8)!important;color:#352746!important}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#fdfbff,#f2eef8)!important;border-right:1px solid #e2d9ed!important}
[data-testid="stSidebar"] *{color:#55466b!important}
[data-testid="stMetric"]{background:rgba(255,255,255,.78)!important;border:1px solid #e5dced!important;box-shadow:0 12px 32px rgba(88,66,117,.08)!important}
[data-testid="stMetricLabel"]{color:#786889!important}[data-testid="stMetricValue"]{color:#352746!important}
[data-testid="stFileUploader"]{background:rgba(255,255,255,.72)!important;border:1px dashed #c4add9!important}
[data-testid="stFileUploaderDropzone"]{background:rgba(255,255,255,.82)!important;border:1px dashed #c4add9!important}
[data-testid="stFileUploaderDropzone"] *{color:#68547d!important}
[data-testid="stDataFrame"]{border:1px solid #e4dced!important}
[data-testid="stExpander"]{background:rgba(255,255,255,.78)!important;border:1px solid #e3d8ed!important;border-radius:16px!important;overflow:hidden}
[data-testid="stExpander"] summary,[data-testid="stExpander"] summary p{background:transparent!important;color:#49365f!important}
[data-testid="stExpander"] label,[data-testid="stExpander"] label p{color:#55466b!important}
[data-testid="stSelectbox"] div[data-baseweb="select"]>div{background:#fff!important;color:#49365f!important;border-color:#d9cbe5!important}
[data-testid="stSlider"] label,[data-testid="stSlider"] label p{color:#55466b!important}
header[data-testid="stHeader"]{background:transparent!important}
.nw-kicker{color:#8a63ac!important}.nw-title{color:#352746!important}.nw-sub{color:#776989!important}
.nw-chip{border-color:#e3d8ed!important;background:rgba(255,255,255,.6)!important;color:#68547d!important}
.nw-card{background:rgba(255,255,255,.7)!important;border-color:#e3d8ed!important}.nw-note{color:#786b84!important}
.nw-hero{display:grid;grid-template-columns:.86fr 1.14fr;gap:14px;align-items:center;margin:8px 0 16px;padding:18px 24px;border:1px solid rgba(255,255,255,.92);border-radius:24px;background:linear-gradient(125deg,rgba(255,255,255,.88),rgba(246,239,252,.72) 55%,rgba(230,239,250,.76));box-shadow:0 16px 42px rgba(90,69,123,.10);overflow:hidden}
.nw-hero-copy{padding:10px 0 12px 8px}.nw-hero-eyebrow{font:12px Consolas,monospace;letter-spacing:1.5px;color:#936bac}
.nw-hero-title{font-family:Georgia,'Source Serif',serif;font-size:33px;line-height:1.12;letter-spacing:-1.2px;font-weight:800;color:#38294c;margin:10px 0 9px}
.nw-hero-desc{font-size:14px;line-height:1.6;color:#786989;max-width:460px}
.nw-hero-points{display:flex;gap:8px;flex-wrap:wrap;margin-top:20px}.nw-hero-point{border:1px solid #e6dced;background:rgba(255,255,255,.58);border-radius:20px;padding:9px 12px;font-size:12px;color:#735d87}
.nw-hero-art{width:100%;height:auto;max-height:260px;object-fit:contain;display:block;filter:drop-shadow(0 16px 30px rgba(102,77,143,.13));border-radius:22px}
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
def read_csv_bytes(data: bytes, sample_limit: int = 50000) -> tuple[pd.DataFrame, int, bool]:
    """Read large CSVs in chunks and keep a reproducible sample for the interactive analysis."""
    if len(data) <= 50 * 1024 * 1024:
        try:
            frame = pd.read_csv(io.BytesIO(data), low_memory=False)
        except UnicodeDecodeError:
            frame = pd.read_csv(io.BytesIO(data), encoding="latin-1", low_memory=False)
        return frame, len(frame), False

    def sampled_chunks(encoding: str | None) -> tuple[pd.DataFrame, int]:
        pieces = []
        total_rows = 0
        for chunk in pd.read_csv(io.BytesIO(data), chunksize=100_000, low_memory=False, encoding=encoding):
            total_rows += len(chunk)
            pieces.append(chunk.sample(n=min(2000, len(chunk)), random_state=42))
        return pd.concat(pieces, ignore_index=True), total_rows

    try:
        frame, total_rows = sampled_chunks("utf-8")
    except UnicodeDecodeError:
        frame, total_rows = sampled_chunks("latin-1")
    sampled = len(frame) > sample_limit or total_rows > sample_limit
    if len(frame) > sample_limit:
        frame = frame.sample(n=sample_limit, random_state=42)
    return frame.reset_index(drop=True), total_rows, sampled


def style_fig(fig, height=270):
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Source Sans, Segoe UI, sans-serif", color="#6f607f", size=12),
                      margin=dict(l=8, r=8, t=12, b=8), legend=dict(font=dict(color="#6f607f")))
    fig.update_xaxes(gridcolor="rgba(118,96,140,.14)", zerolinecolor="rgba(118,96,140,.14)")
    fig.update_yaxes(gridcolor="rgba(118,96,140,.14)", zerolinecolor="rgba(118,96,140,.14)")
    return fig

with st.sidebar:
    st.markdown("<div style='display:flex;align-items:center;gap:10px;margin:8px 2px 28px'><div style='background:linear-gradient(135deg,#ff62c7,#9462f6);border-radius:12px;width:36px;height:36px;display:grid;place-items:center;color:white;font-size:20px'>◈</div><div><b style='font-size:16px;letter-spacing:1px'>NIGHTWATCH</b><br><span style='font-size:10px;letter-spacing:1px;color:#9e85ac'>ЖЕЛІ ҚАУІПСІЗДІГІ</span></div></div>", unsafe_allow_html=True)
    st.caption("БАСҚАРУ ПАНЕЛІ")
    page = st.radio("Бөлімді таңда", ["Басты бет", "Деректер зертханасы", "Жоба жайлы"], label_visibility="collapsed", key="nav_page")
    st.markdown("**📂 Трафик файлы**")
    uploaded = st.file_uploader("CSV файлын таңда", type=["csv"], help="CIC-IDS2017 сияқты желілік ағын деректерінің CSV файлы", label_visibility="collapsed", key="traffic_csv")
    st.caption("Жүктелген файл тек осы талдауда қолданылады.")
    st.divider()
    st.markdown("**◉ Талдау жүйесі дайын**")
    st.caption("Машиналық оқыту · интерактивті демо")

if uploaded:
    try:
        with st.spinner("Үлкен CSV файлы жадқа үнемді түрде оқылып жатыр…"):
            df, source_rows, was_sampled = read_csv_bytes(uploaded.getvalue())
        df.columns = [str(c).strip() for c in df.columns]
        data_origin = f"Жүктелген файл · {uploaded.name}"
    except Exception as exc:
        st.error(f"CSV файлын оқу мүмкін болмады: {exc}")
        st.stop()
else:
    df = make_demo_data()
    source_rows, was_sampled = len(df), False
    data_origin = "Сынақ дерегі · синтетикалық үлгі"

# Drop empty and duplicate columns safely; preserve the unmodified row index for display.
df = df.loc[:, ~df.columns.duplicated()].copy()
df = df.dropna(axis=1, how="all")

if page == "Басты бет":
    st.markdown("<div class='nw-kicker'>NIGHTWATCH · ЖЕЛІ ҚАУІПСІЗДІГІ</div>", unsafe_allow_html=True)
    st.markdown("<div class='nw-title'>Күмәнді желілік әрекеттер</div><div class='nw-sub'>CSV трафигін шолыңыз және модель тексеруге ұсынған қосылымдарды қараңыз.</div>", unsafe_allow_html=True)
elif page == "Деректер зертханасы":
    st.markdown("<div class='nw-kicker'>NIGHTWATCH · ДЕРЕКТЕР</div>", unsafe_allow_html=True)
    st.markdown("<div class='nw-title'>Деректер зертханасы</div><div class='nw-sub'>Файлдың жолдарын, бағандарын және бос ұяшықтарын тексеріңіз.</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='nw-kicker'>NIGHTWATCH · ЖОБА</div>", unsafe_allow_html=True)
    st.markdown("<div class='nw-title'>Жоба туралы</div><div class='nw-sub'>Жобаның мақсаты, дерек көзі және модельдердің шектеулері.</div>", unsafe_allow_html=True)
st.markdown(f"<div style='margin:12px 0 16px'><span class='nw-chip'>ДЕРЕК КӨЗІ · {data_origin}</span></div>", unsafe_allow_html=True)
if was_sampled:
    st.info(f"Файл көлемі үлкен болғандықтан, талдау үшін {len(df):,} жол кездейсоқ іріктелді (файлда барлығы {source_rows:,} жол). Метрикалар осы іріктемеге есептеледі.")

if page == "Басты бет":
    hero_path = Path(__file__).parent / "assets" / "orbit-hero.svg"
    hero_b64 = base64.b64encode(hero_path.read_bytes()).decode("ascii")
    st.markdown(f"""
    <section class="nw-hero">
      <div class="nw-hero-copy">
        <div class="nw-hero-eyebrow">ЖҮКТЕЛГЕН CSV · ОҚУ ДЕМОСЫ</div>
        <div class="nw-hero-title">Желі ағындарын<br>бір жерден шолыңыз.</div>
        <div class="nw-hero-desc">Жүйе файлдағы қосылымдарды талдайды. Белгіленген деректе үш модель салыстырылады; белгі жоқ болса, Isolation Forest ерекше ағындарды жеке көрсетеді. Бұл нақты уақыттағы мониторинг емес.</div>
        <div class="nw-hero-points"><span class="nw-hero-point">CSV дерегі</span><span class="nw-hero-point">3 модель салыстыруы</span><span class="nw-hero-point">Бөлек ескерту тізімі</span></div>
      </div>
      <img class="nw-hero-art" alt="Голографиялық планета мен желі байланыстарын көрсететін иллюстрация" src="data:image/svg+xml;base64,{hero_b64}" />
    </section>
    """, unsafe_allow_html=True)

if page == "Жоба жайлы":
    st.markdown("### Құралдың мақсаты")
    st.write("Nightwatch жүктелген CSV файлындағы желі ағындарын талдап, модель күмәнді деп белгілеген жолдарды көрсетеді. Жоба трафикті тоқтатпайды; әр ескертуді маман тексеруі керек.")
    st.markdown("#### Жұмыс тәртібі")
    st.markdown("1. **Дерек жүктеу:** CIC-IDS2017 CSV файлын таңда.\n2. **Деректі шолу:** кластарды, бос мәндерді және белгілерді қара.\n3. **Модельді іске қосу:** белгісі бар деректе қарапайым бастапқы үлгі, Random Forest және Gradient Boosting салыстырылады.\n4. **Күдікті ағындарды қарау:** аномалия режимінде Isolation Forest сирек үлгілерді бөлек көрсетеді.\n5. **Қорытынды жасау:** дәлдік, нақтылық, толықтық, F1 өлшемі және қате болжамдарын талда.")
    st.markdown("#### Кім қолдана алады?")
    st.write("Студенттер модельдерді салыстырып, аномалияны анықтау тәсілін үйрене алады. Желі маманы күмәнді деп белгіленген жолдарды бастапқы тексеруге пайдалана алады.")
    st.markdown("#### Дерек көзі")
    st.markdown("[CIC-IDS2017 ресми деректер беті — Канаданың киберқауіпсіздік институты](https://www.unb.ca/cic/datasets/ids-2017.html)")
    st.info("Шектеуі: дерек жиыны зертханалық ортада жиналған. Нәтиже нақты желідегі дәлдікті кепілдемейді; сынып теңгерімсіздігі мен қате дабылдарды міндетті түрде талдау керек.")
    st.stop()

if page == "Деректер зертханасы":
    st.markdown("### Файл мазмұны")
    st.caption("Дерек көлемін, бағандарды, бос мәндерді және сандық белгілердің үлестірімін тексер. Үлкен файлда талдау жылдамырақ болуы үшін үлгі қолданылады.")
    left, right = st.columns([1.45, 1])
    with left:
        st.dataframe(df.head(12), use_container_width=True, height=350)
    with right:
        numeric = df.select_dtypes(include=np.number)
        st.metric("Талдаудағы жол саны", f"{len(df):,}")
        st.metric("Баған саны", f"{df.shape[1]:,}")
        st.metric("Бос ұяшық саны", f"{int(df.isna().sum().sum()):,}")
        if not numeric.empty:
            feature = st.selectbox("Сандық бағанды таңдаңыз", numeric.columns)
            fig = px.histogram(df, x=feature, nbins=35, color_discrete_sequence=["#da69d7"])
            st.plotly_chart(style_fig(fig, 260), use_container_width=True)
    st.stop()

# Main overview
numeric = df.select_dtypes(include=np.number).replace([np.inf, -np.inf], np.nan)
label_candidates = [c for c in df.columns if c.lower() in {"label", "class", "attack", "target", "category"}]
with st.expander("Сүзгілер мен талдау параметрлері", expanded=True):
    label_col = st.selectbox("Класс немесе шабуыл белгісі бар баған", ["Белгісіз аномалияны іздеу"] + list(df.columns), index=(1 + df.columns.get_loc(label_candidates[0])) if label_candidates else 0)
    max_rows = st.slider("Модельге алынатын жол саны", 1000, 30000, 12000, 1000)
    normal_label = None
    if label_col != "Белгісіз аномалияны іздеу":
        label_values = sorted(df[label_col].dropna().astype(str).unique().tolist())
        normal_candidates = {"BENIGN", "BENIGN TRAFFIC", "NORMAL", "NORMAL TRAFFIC", "0"}
        detected_normal = next((v for v in label_values if v.strip().upper() in normal_candidates), None)
        normal_options = ["Автоматты анықтау"] + label_values
        default_index = normal_options.index(detected_normal) if detected_normal in normal_options else 0
        normal_label = st.selectbox("Қалыпты трафиктің белгісі", normal_options, index=default_index,
                                    help="Осы кластан бөлек болжанған нәтижелер күдікті жағдайлар тізімінде көрсетіледі.")

has_label = label_col != "Белгісіз аномалияны іздеу"
if has_label:
    n_classes = df[label_col].nunique(dropna=True)
    st.markdown("<div class='nw-note'>Режим: белгіленген дерек. Бастапқы үлгі, Random Forest және Gradient Boosting салыстырылады.</div>", unsafe_allow_html=True)
else:
    n_classes = 0
    st.markdown("<div class='nw-note'>Режим: белгісіз аномалияны іздеу. Isolation Forest ерекше ағындарды белгілейді; нақты жауап белгісі болмаса, сапа метрикалары есептелмейді.</div>", unsafe_allow_html=True)

# Summary strip
s1,s2,s3,s4=st.columns(4)
s1.metric("Талдаудағы жол", f"{len(df):,}")
s2.metric("Сандық баған", f"{numeric.shape[1]:,}")
s3.metric("Класс саны", f"{n_classes:,}" if has_label else "Белгі жоқ")
s4.metric("Бос ұяшық", f"{int(df.isna().sum().sum()):,}")

port_aliases = {"destination port", "dst port", "dest port", "destination_port", "dst_port"}
protocol_aliases = {"protocol", "proto"}
port_col = next((c for c in df.columns if c.strip().lower() in port_aliases), None)
protocol_col = next((c for c in df.columns if c.strip().lower() in protocol_aliases), None)

left, right = st.columns([1.45, 1], gap="large")
with left:
    with st.container(border=True):
        st.markdown("#### Трафик кластары")
        if has_label:
            counts = df[label_col].fillna("Бос мән").astype(str).value_counts().head(12).rename_axis("Класс атауы").reset_index(name="Қосылым саны")
            fig=px.bar(counts, x="Қосылым саны", y="Класс атауы", orientation="h", color="Қосылым саны", color_continuous_scale=[[0,"#7045a7"],[1,"#ff62c7"]], text_auto=".2s")
            fig.update_layout(coloraxis_showscale=False, yaxis={"categoryorder":"total ascending"})
            st.plotly_chart(style_fig(fig, 315), use_container_width=True)
        elif not numeric.empty:
            feat=numeric.columns[:min(5,len(numeric.columns))].tolist()
            trend=df[feat].head(120).reset_index(drop=True).reset_index(names="Жол реті")
            fig=px.line(trend, x="Жол реті", y=feat, color_discrete_sequence=["#f15fca","#8b69ff","#5de0c7","#ffba69","#7aaeff"])
            st.plotly_chart(style_fig(fig,315), use_container_width=True)
        else:
            st.info("График құру үшін сандық бағандар қажет.")
with right:
    with st.container(border=True):
        st.markdown("#### Желілік қызметтер")
        service_col = port_col or protocol_col
        if service_col:
            service_title = "Тағайындалған порттар" if service_col == port_col else "Протоколдар"
            services = df[service_col].fillna("Белгісіз").astype(str).value_counts().head(10).rename_axis("Қызмет").reset_index(name="Қосылым саны")
            service_fig = px.bar(services, x="Қосылым саны", y="Қызмет", orientation="h", color="Қосылым саны", color_continuous_scale=[[0,"#7045a7"],[1,"#ff62c7"]], text_auto=".2s")
            service_fig.update_layout(title=service_title, coloraxis_showscale=False, yaxis={"categoryorder":"total ascending"})
            st.plotly_chart(style_fig(service_fig,260), use_container_width=True)
            st.caption("Ең жиі кездесетін 10 мән.")
        elif has_label:
            counts=df[label_col].fillna("Белгісіз").astype(str).value_counts()
            pie=go.Figure(go.Pie(labels=counts.index[:8],values=counts.values[:8],hole=.72,marker_colors=["#ff62c7","#9567f5","#62e9d0","#ffbd69","#679aff","#bf71ff","#e8818f","#708295"],textinfo="label+percent"))
            pie.add_annotation(text=f"{len(df):,}<br><span style='font-size:10px'>жол</span>",x=.5,y=.5,showarrow=False,font=dict(size=23,color="#fff0ff"))
            st.plotly_chart(style_fig(pie,260), use_container_width=True)
            st.caption("Деректегі класс үлесі.")
        else:
            st.markdown("<div class='nw-card'><div class='nw-kicker'>ТАҢДАЛҒАН ӘДІС</div><h2 style='margin:.6rem 0;color:#49365f'>Isolation Forest</h2><div class='nw-note'>Белгі қойылмаған трафиктен өзгеше ағындарды анықтап, тексеру тізіміне қосады.</div></div>", unsafe_allow_html=True)

st.markdown("#### Модель нәтижелері")
run = st.button("Талдауды бастау", type="primary", use_container_width=False)
if run:
    if numeric.shape[1] == 0:
        st.error("Модельге жарайтын сандық белгі бағаны табылмады.")
    else:
        clean = df.copy()
        clean[numeric.columns] = numeric
        clean = clean.dropna(subset=list(numeric.columns))
        if len(clean) > max_rows:
            clean = clean.sample(max_rows, random_state=42)
        X = clean.select_dtypes(include=np.number).replace([np.inf,-np.inf], np.nan)
        X = X.fillna(X.median()).clip(lower=-1e12, upper=1e12)
        if has_label:
            y_raw=clean[label_col].fillna("Белгісіз").astype(str)
            mask=y_raw.value_counts()[y_raw].values >= 2
            X=X.loc[mask].drop(columns=[label_col], errors="ignore"); y_raw=y_raw.loc[mask]
            if y_raw.nunique()<2:
                st.error("Салыстыру үшін нысана бағанында кемінде екі класс және әр класта екіден көп мысал болуы керек.")
            else:
                enc=LabelEncoder(); y=enc.fit_transform(y_raw)
                strat=y if pd.Series(y).value_counts().min() >= 2 else None
                Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=strat)
                models={"Бастапқы үлгі · ең жиі класс":DummyClassifier(strategy="most_frequent"),"Random Forest":RandomForestClassifier(n_estimators=120,max_depth=18,class_weight="balanced",n_jobs=-1,random_state=42),"Gradient Boosting":HistGradientBoostingClassifier(max_iter=100,learning_rate=.1,random_state=42)}
                results=[]; predictions={}; timings={}
                progress=st.progress(0)
                for i,(name,model) in enumerate(models.items()):
                    start=time.perf_counter(); model.fit(Xtr,ytr); pred=model.predict(Xte); timings[name]=time.perf_counter()-start
                    predictions[name]=pred
                    results.append({"Модель":name,"Дәлдік":accuracy_score(yte,pred),"Нақтылық (Precision)":precision_score(yte,pred,average="macro",zero_division=0),"Толықтық (Recall)":recall_score(yte,pred,average="macro",zero_division=0),"F1 өлшемі":f1_score(yte,pred,average="macro",zero_division=0),"Оқыту, сек":round(timings[name],2)})
                    progress.progress((i+1)/len(models))
                res=pd.DataFrame(results).sort_values("F1 өлшемі",ascending=False)
                st.success(f"Модельдерді салыстыру аяқталды. Тексерілген жол: {len(Xte):,}.")
                st.dataframe(res.style.format({c:"{:.3f}" for c in ["Дәлдік","Нақтылық (Precision)","Толықтық (Recall)","F1 өлшемі"]}),use_container_width=True,hide_index=True)
                chosen=st.selectbox("Нәтижесін және күдікті болжамдарын қарайтын модель",list(models.keys()),index=list(models.keys()).index(res.iloc[0]["Модель"]))
                cm=confusion_matrix(yte,predictions[chosen])
                cf=px.imshow(cm,text_auto=True,x=[str(c) for c in enc.classes_],y=[str(c) for c in enc.classes_],color_continuous_scale=[[0,"#1a1423"],[1,"#ed57bd"]],labels=dict(x="Болжанған класс",y="Нақты класс",color="Саны"),aspect="auto")
                st.plotly_chart(style_fig(cf,330),use_container_width=True)
                predicted_labels=enc.inverse_transform(predictions[chosen]).astype(str)
                actual_labels=enc.inverse_transform(yte).astype(str)
                normal_value=normal_label if normal_label and normal_label != "Автоматты анықтау" else next((str(c) for c in enc.classes_ if str(c).strip().upper() in {"BENIGN","BENIGN TRAFFIC","NORMAL","NORMAL TRAFFIC","0"}), None)
                st.markdown("#### Күдікті деп белгіленген қосылымдар")
                if normal_value is None:
                    st.warning("Қалыпты трафик класы автоматты анықталмады. Сол жақтағы «Талдау параметрлері» бөлімінен қалыпты классты таңдасаң, қалған болжамдар осы тізімге шығады.")
                else:
                    alert_mask=predicted_labels != str(normal_value)
                    alerts=Xte.loc[alert_mask].copy()
                    alerts.insert(0,"Нақты класс",actual_labels[alert_mask])
                    alerts.insert(1,"Модель болжамы",predicted_labels[alert_mask])
                    if alerts.empty:
                        st.success("Тест жиынында күдікті деп белгіленген қосылым жоқ.")
                    else:
                        st.warning(f"Қалыпты емес деп болжанған қосылым саны: {len(alerts):,}. Әр жолды тексеріңіз.")
                        st.dataframe(alerts.head(100),use_container_width=True,hide_index=True)
                st.caption("Ескерту: тест бөлінісі жүктелген файлдан алынған. Жолдардың уақыты ескерілмесе, жаңа желідегі сапа бұдан төмен болуы мүмкін. Болжам шабуылдың дәлелі емес.")
        else:
            if len(X)<30:
                st.error("Аномалия моделін үйрету үшін кемінде 30 жарамды жол керек.")
            else:
                model=IsolationForest(n_estimators=180,contamination="auto",random_state=42,n_jobs=-1)
                start=time.perf_counter(); pred=model.fit_predict(X); elapsed=time.perf_counter()-start
                score=-model.score_samples(X)
                out=df.loc[X.index].copy(); out["Аномалия ұпайы"]=score; out["Модель нәтижесі"]=np.where(pred==-1,"КҮМӘНДІ","ҚАЛЫПТЫ")
                flagged=out[out["Модель нәтижесі"]=="КҮМӘНДІ"].sort_values("Аномалия ұпайы",ascending=False)
                if not flagged.empty:
                    high_cut=out["Аномалия ұпайы"].quantile(.99)
                    medium_cut=out["Аномалия ұпайы"].quantile(.95)
                    flagged["Тексеру басымдығы"]=np.select(
                        [flagged["Аномалия ұпайы"]>=high_cut,flagged["Аномалия ұпайы"]>=medium_cut],
                        ["Жоғары","Орташа"],default="Қарау керек")
                st.success(f"Талдау аяқталды · {len(X):,} ағын · {elapsed:.2f} сек")
                a,b,c=st.columns(3); a.metric("Күмәнді белгі",f"{len(flagged):,}"); b.metric("Қалыпты белгі",f"{len(X)-len(flagged):,}"); c.metric("Үлес",f"{len(flagged)/len(X):.1%}")
                st.markdown("#### Күдікті қосылымдар")
                st.caption("Тексеру басымдығы аномалия ұпайының осы файлдағы үлестіріміне сүйенеді. Бұл баға қауіптің нақты деңгейін немесе шабуылды дәлелдемейді.")
                st.dataframe(flagged.head(100),use_container_width=True,hide_index=True)

st.divider()
st.markdown("<div class='nw-note'>Оқу жобасының прототипі · Күмәнді белгі қосымша тексеруді қажет етеді. Сандық мәндер нақты файлды талдағаннан кейін есептеледі.</div>",unsafe_allow_html=True)


