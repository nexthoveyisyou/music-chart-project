"""
=============================================================
 🎵 음원차트 Streamlit 대시보드
    - 멜론/벅스 차트 비교
    - 8주 순위변동 → 4주 예측 꺾은선 그래프
    - YouTube 좋아요 vs 순위 산점도 (팬덤 vs 대중성)
    - 가사 퀴즈 게임
=============================================================
streamlit run streamlit_app.py
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
import random

st.set_page_config(page_title="🎵 음원차트 분석", page_icon="🎵", layout="wide")

# ============================================================
# DB 연결
# ============================================================
@st.cache_resource
def get_engine():
    cfg = {
        "host": st.secrets.get("DB_HOST", "localhost"),
        "port": int(st.secrets.get("DB_PORT", 3306)),
        "user": st.secrets.get("DB_USER", "root"),
        "password": st.secrets.get("DB_PASSWORD", ""),
        "database": st.secrets.get("DB_NAME", "music_chart"),
    }
    ca = {}
    if st.secrets.get("DB_SSL", "false").lower() == "true":
        ca = {"ssl": {"ssl_mode": "REQUIRED"}}
    return create_engine(
        f"mysql+pymysql://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['database']}",
        connect_args=ca
    )

@st.cache_data(ttl=300)
def load_chart():
    return pd.read_sql("SELECT * FROM chart_data WHERE crawled_at=(SELECT MAX(crawled_at) FROM chart_data) ORDER BY source, `rank`", get_engine())

def _fix_week_offset(df: pd.DataFrame) -> pd.DataFrame:
    """week_label 날짜 파싱 → week_offset 재계산 (최신=0, 과거일수록 증가)"""
    if df.empty or "week_label" not in df.columns:
        return df
    df = df.copy()

    def _parse_start(label):
        try:
            return pd.to_datetime(str(label).split("~")[0].strip(), format="%Y.%m.%d")
        except Exception:
            return pd.NaT

    unique_labels = df["week_label"].dropna().unique()
    parsed = {lbl: _parse_start(lbl) for lbl in unique_labels}
    sorted_pairs = sorted(
        [(dt, lbl) for lbl, dt in parsed.items() if pd.notna(dt)],
        reverse=True  # 최신 날짜가 offset=0
    )
    label_to_offset = {lbl: idx for idx, (_, lbl) in enumerate(sorted_pairs)}
    df["week_offset"] = df["week_label"].map(label_to_offset).fillna(0).astype(int)
    return df


def _read_weekly_csv(path: str) -> pd.DataFrame:
    """escapechar 시도 → 실패시 on_bad_lines='skip' 재시도"""
    for kwargs in [
        {"encoding": "utf-8-sig", "escapechar": "\\"},
        {"encoding": "utf-8-sig", "on_bad_lines": "skip"},
        {"encoding": "utf-8",     "on_bad_lines": "skip"},
        {"encoding": "latin-1",   "on_bad_lines": "skip"},
    ]:
        try:
            df = pd.read_csv(path, **kwargs)
            if not df.empty and "week_label" in df.columns:
                return df
        except Exception:
            continue
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_weekly():
    # ── 1. CSV 먼저 시도 (프로젝트 폴더 → Desktop → 상대경로) ──
    csv_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "weekly_rank.csv"),
        "weekly_rank.csv",
        r"C:\Users\KDT-008\Desktop\weekly_rank.csv",
        os.path.join("..", "weekly_rank.csv"),
    ]
    for path in csv_paths:
        try:
            if os.path.exists(path):
                df_csv = _fix_week_offset(_read_weekly_csv(path))
                if df_csv["week_offset"].nunique() >= 2:
                    return df_csv
        except Exception:
            pass

    # ── 2. DB 폴백 ──────────────────────────────────────────────
    try:
        df_db = pd.read_sql(
            "SELECT * FROM weekly_rank ORDER BY week_offset, `rank`",
            get_engine()
        )
        return _fix_week_offset(df_db)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_youtube():
    try:
        return pd.read_sql("SELECT * FROM youtube_stats WHERE crawled_at=(SELECT MAX(crawled_at) FROM youtube_stats)", get_engine())
    except:
        return pd.DataFrame()


# ============================================================
# 퀴즈 점수 DB
# ============================================================
def init_quiz_table():
    try:
        with get_engine().connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS quiz_scores (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    nickname VARCHAR(100) NOT NULL,
                    score INT DEFAULT 0,
                    updated_at DATETIME NOT NULL,
                    UNIQUE KEY uk_nickname (nickname),
                    INDEX idx_score (score)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """))
            conn.commit()
    except Exception:
        pass

@st.cache_data(ttl=10)
def load_leaderboard():
    try:
        return pd.read_sql(
            "SELECT nickname, score FROM quiz_scores ORDER BY score DESC, updated_at ASC LIMIT 10",
            get_engine()
        )
    except Exception:
        return pd.DataFrame(columns=["nickname", "score"])

def update_quiz_score(nickname: str):
    try:
        with get_engine().connect() as conn:
            conn.execute(text("""
                INSERT INTO quiz_scores (nickname, score, updated_at)
                VALUES (:nick, 1, NOW())
                ON DUPLICATE KEY UPDATE score = score + 1, updated_at = NOW()
            """), {"nick": nickname})
            conn.commit()
        load_leaderboard.clear()
    except Exception:
        pass

def delete_quiz_score(nickname: str):
    try:
        with get_engine().connect() as conn:
            conn.execute(text("DELETE FROM quiz_scores WHERE nickname = :nick"), {"nick": nickname})
            conn.commit()
        load_leaderboard.clear()
    except Exception:
        pass


# ============================================================
# ML 예측 헬퍼
# ============================================================
@st.cache_data(ttl=600)
def load_monthly_for_ml():
    """monthly_genre_data.csv → ML 예측용 월별 히스토리 로드"""
    candidates = ["monthly_genre_data.csv"] + sorted(
        __import__("glob").glob("music_chart_monthly_*.csv")
    )
    for path in candidates:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, dtype={"song_id": str, "year_month": str})
                if not df.empty:
                    return df
            except Exception:
                pass
    try:
        df = pd.read_sql("SELECT title, rank, year_month FROM monthly_chart", get_engine())
        return df if not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def ml_predict_ranks(song_title: str, current_rank: float,
                     weekly_ranks: np.ndarray, df_monthly: pd.DataFrame):
    """
    월별 히스토리(df_monthly)를 1차 선형회귀로 학습해 주별 추세로 환산.
    히스토리가 부족하면 주간 데이터 추세 → 최종 fallback은 rank 구간별 감쇠.
    반환: (predicted_list, method_str)
    """
    # ── 방법 1: 월별 히스토리 선형 회귀 ─────────────────────
    if df_monthly is not None and not df_monthly.empty and "title" in df_monthly.columns:
        hist = df_monthly[df_monthly["title"] == song_title].sort_values("year_month")
        if len(hist) < 3:
            # 부분 일치 재시도
            key = song_title[:min(len(song_title), 6)]
            hist = df_monthly[
                df_monthly["title"].str.startswith(key, na=False)
            ].sort_values("year_month")

        if len(hist) >= 3:
            x = np.arange(len(hist), dtype=float)
            y = hist["rank"].values.astype(float)
            slope = float(np.polyfit(x, y, 1)[0])          # 월 단위 기울기
            weekly_slope = np.clip(slope / 4.3, -4.0, 4.0)  # 주 단위로 환산

            predicted, last = [], current_rank
            for _ in range(4):
                last = float(np.clip(last + weekly_slope, 1, 100))
                predicted.append(round(last))
            return predicted, f"선형회귀 (과거 {len(hist)}개월)"

    # ── 방법 2: 주간 데이터 추세 ─────────────────────────────
    if len(weekly_ranks) >= 3:
        diffs = np.diff(weekly_ranks[-4:]) if len(weekly_ranks) >= 4 else np.diff(weekly_ranks)
        w = np.linspace(0.5, 1.0, len(diffs))
        weekly_slope = float(np.clip(np.average(diffs, weights=w), -4.0, 4.0))

        predicted, last = [], current_rank
        for _ in range(4):
            last = float(np.clip(last + weekly_slope, 1, 100))
            predicted.append(round(last))
        return predicted, "주간 추세"

    # ── 방법 3: rank 구간별 자연 감쇠 ───────────────────────
    decay = 0.3 if current_rank <= 5 else 0.7 if current_rank <= 20 else 1.5
    predicted, last = [], current_rank
    for _ in range(4):
        last = float(np.clip(last + decay, 1, 100))
        predicted.append(round(last))
    return predicted, "구간별 감쇠"


# 데이터 로드 시도
try:
    df = load_chart()
    df_weekly = load_weekly()
    df_yt = load_youtube()
except Exception as e:
    # CSV 폴백
    import glob
    csvs = sorted(glob.glob("music_chart_*.csv"))
    if csvs:
        df = pd.read_csv(csvs[-1])
        df_weekly = pd.DataFrame()
        df_yt = pd.DataFrame()
        st.warning("⚠️ DB 연결 실패 → CSV에서 로드")
    else:
        st.error("❌ 데이터 없음. python music_chart_crawler.py를 먼저 실행하세요!")
        st.stop()

# weekly_rank 최신 주차 likes → chart_data의 likes 컬럼에 반영
if not df_weekly.empty and "likes" in df_weekly.columns and not df.empty and "title" in df.columns:
    _likes_map = (
        df_weekly[df_weekly["week_offset"] == 0][["title", "likes"]]
        .drop_duplicates("title")
        .set_index("title")["likes"]
        .to_dict()
    )
    if _likes_map:
        _mapped = df["title"].map(_likes_map)
        _has_val = _mapped.notna() & (_mapped.astype(float) > 0)
        if "likes" not in df.columns:
            df["likes"] = 0
        df.loc[_has_val, "likes"] = _mapped[_has_val].astype(int)

# genre_cache.json → chart_data의 genre='overall' 곡에 실제 장르 반영
if not df.empty and "song_id" in df.columns and "genre" in df.columns:
    _gcache = {}
    _gcache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "genre_cache.json")
    if os.path.exists(_gcache_path):
        import json as _json
        with open(_gcache_path, "r", encoding="utf-8") as _f:
            _gcache = _json.load(_f)
    if _gcache:
        _genre_mapped = df["song_id"].astype(str).map(_gcache)
        _fix_mask = (df["genre"] == "overall") & _genre_mapped.notna()
        df.loc[_fix_mask, "genre"] = _genre_mapped[_fix_mask]

init_quiz_table()

# ============================================================
# 사이드바
# ============================================================
st.sidebar.title("🎵 음원차트 분석")
page = st.sidebar.radio("메뉴", [
    "📊 차트 대시보드",
    "📈 4주 순위 예측",
    "🔥 팬덤 vs 대중성",
    "📺 YouTube MV 탐색",
    "🎵 장르·계절 분류",
    "🎮 가사 퀴즈 게임",
    "📋 전체 데이터"
])

# 사이드바 — 데이터 최신화 안내
st.sidebar.divider()
with st.sidebar.expander("🔄 데이터 업데이트 방법"):
    st.markdown("""
**자동 업데이트는 기본적으로 비활성화** 되어 있습니다.

데이터를 최신으로 갱신하려면:
```
python music_chart_crawler.py
```
를 직접 실행하거나, 아래 명령으로
**Windows 작업 스케줄러**에 등록하세요.
```
schtasks /create /tn "MusicChartCrawl" ^
  /tr "python C:\\...\\music_chart_crawler.py" ^
  /sc monthly /d 1 /st 06:00
```
Streamlit 캐시는 **5분** 간격으로 자동 갱신됩니다.
""")

if st.sidebar.button("🗑️ 캐시 초기화 (즉시 갱신)"):
    st.cache_data.clear()
    st.rerun()

# 사이드바 하단 — 실시간 퀴즈 순위표
st.sidebar.divider()
st.sidebar.subheader("🏆 퀴즈 순위표 TOP 10")
_lb = load_leaderboard()
_sb_nickname = st.session_state.get("quiz_nickname", "")
if _lb.empty:
    st.sidebar.caption("아직 참여자가 없습니다.")
else:
    for _i, _row in enumerate(_lb.itertuples(index=False)):
        _medal = ("🥇" if _i == 0 else "🥈" if _i == 1 else "🥉" if _i == 2 else f"{_i+1}.")
        if _sb_nickname and _row.nickname == _sb_nickname:
            _sb_c1, _sb_c2 = st.sidebar.columns([4, 1])
            _sb_c1.write(f"{_medal} **{_row.nickname}** — {int(_row.score)}점")
            if _sb_c2.button("🗑️", key=f"sb_del_{_i}", help="내 점수 삭제"):
                delete_quiz_score(_sb_nickname)
                st.session_state.quiz_score = 0
                st.session_state.quiz_total = 0
                st.session_state.pop("last_answered_q", None)
                st.rerun()
        else:
            st.sidebar.write(f"{_medal} **{_row.nickname}** — {int(_row.score)}점")


# ============================================================
# 📊 차트 대시보드
# ============================================================
if page == "📊 차트 대시보드":
    st.title("📊 멜론 · 벅스 차트 대시보드")

    col1, col2 = st.columns(2)
    for col, src, emoji in [(col1, "melon", "🍈"), (col2, "bugs", "🐛")]:
        with col:
            top = df[(df["source"] == src) & (df["rank"] == 1)]
            if not top.empty:
                st.metric(f"{emoji} {src.upper()} 1위", top.iloc[0]["title"], top.iloc[0]["artist"])

    st.divider()

    # 사이트별 TOP 10
    st.subheader("🏆 사이트별 TOP 10")
    tab1, tab2 = st.tabs(["🍈 멜론", "🐛 벅스"])
    for tab, src in [(tab1, "melon"), (tab2, "bugs")]:
        with tab:
            top10 = df[(df["source"] == src) & (df["rank"] <= 10)].sort_values("rank")
            if not top10.empty:
                display = top10[["rank", "title", "artist", "album"]].copy()
                display.columns = ["순위", "곡명", "가수", "앨범"]
                display["순위"] = display["순위"].apply(lambda r: f"🥇 {r}" if r == 1 else f"🥈 {r}" if r == 2 else f"🥉 {r}" if r == 3 else f"  {r}")
                st.dataframe(display, use_container_width=True, hide_index=True)

    # 2사 공통곡
    st.subheader("🔍 멜론 & 벅스 공통 진입곡")
    melon_s = set(df[df["source"] == "melon"]["title"])
    bugs_s = set(df[df["source"] == "bugs"]["title"])
    common = melon_s & bugs_s
    only_melon = melon_s - bugs_s
    only_bugs = bugs_s - melon_s

    c1, c2, c3 = st.columns(3)
    c1.metric("🎯 공통", f"{len(common)}곡")
    c2.metric("🍈 멜론만", f"{len(only_melon)}곡")
    c3.metric("🐛 벅스만", f"{len(only_bugs)}곡")

    # 아티스트 출현 빈도
    st.subheader("🎤 아티스트별 차트 진입 횟수")
    ac = df.groupby("artist").size().reset_index(name="count").sort_values("count", ascending=False).head(15)
    fig2 = px.bar(ac, x="count", y="artist", orientation="h", color="count", color_continuous_scale="sunset")
    fig2.update_layout(yaxis=dict(autorange="reversed"), height=500)
    st.plotly_chart(fig2, use_container_width=True)


# ============================================================
# 📈 4주 순위 예측
# ============================================================
elif page == "📈 4주 순위 예측":
    st.title("📈 멜론 주간 순위 기반 4주 예측")

    if df_weekly.empty:
        st.warning("⚠️ 주간 순위 데이터가 없습니다. 크롤링을 먼저 실행하세요.")
        st.stop()

    # 수집된 주차 수 확인
    total_weeks = df_weekly["week_offset"].nunique()
    min_weeks = min(2, total_weeks)  # 데이터가 적으면 기준도 낮춤

    # 곡별 주차별 순위 피벗
    songs = df_weekly.groupby("title")["week_offset"].nunique()
    songs_multi = songs[songs >= min_weeks].index.tolist()

    if total_weeks == 1:
        st.warning(f"⚠️ 수집된 주간 데이터가 1주치뿐입니다. 현재 순위 기준으로 예측합니다.")
    else:
        st.info(f"📊 수집된 주간 데이터: {total_weeks}주치 / 예측 대상: {len(songs_multi)}곡")

    # 곡 선택 — 주간 데이터 있는 곡 우선, 없으면 실시간 멜론 TOP50
    melon_top = df[(df["source"] == "melon") & (df["rank"] <= 50)]["title"].tolist()
    selectable = [s for s in melon_top if s in songs_multi]
    if not selectable:
        selectable = melon_top  # 주간 데이터 없으면 실시간 차트로 fallback

    selected = st.multiselect("곡 선택 (최대 5곡)", selectable[:50], default=selectable[:3], max_selections=5)

    # 월별 히스토리 로드 (ML 예측에 사용)
    df_ml = load_monthly_for_ml()
    has_history = not df_ml.empty

    # 주간 데이터 실질 변동 여부 확인 (모든 주차 rank가 동일 = 크롤러가 1주치만 반복 수집)
    if total_weeks > 1:
        rank_std = df_weekly.groupby("title")["rank"].std().mean()
        data_is_flat = rank_std < 0.1
    else:
        data_is_flat = True

    if data_is_flat and has_history:
        st.info(f"📊 주간 데이터가 현재 주 1주치만 유효합니다. 월별 히스토리({df_ml['year_month'].nunique() if 'year_month' in df_ml.columns else '?'}개월)를 활용한 ML 예측을 사용합니다.")
    elif data_is_flat:
        st.warning("📊 주간 순위 변동 데이터가 없습니다. 구간별 감쇠 예측을 사용합니다.")

    if selected:
        fig = go.Figure()
        pred_methods = {}

        for song_title in selected:
            song_weekly = df_weekly[df_weekly["title"] == song_title].sort_values("week_offset", ascending=False)

            # 주간 데이터가 없으면 실시간 차트 순위를 1주치로 사용
            if song_weekly.empty:
                rt = df[(df["source"] == "melon") & (df["title"] == song_title)]
                if rt.empty:
                    continue
                song_weekly = pd.DataFrame([{
                    "week_offset": 0,
                    "title": song_title,
                    "artist": rt.iloc[0]["artist"],
                    "rank": rt.iloc[0]["rank"],
                    "likes": rt.iloc[0].get("likes", 0)
                }])

            if len(song_weekly) < 1:
                continue

            ranks = song_weekly["rank"].values[::-1]   # 과거→현재

            # ---- ML 예측 ----
            predicted, method = ml_predict_ranks(
                song_title, float(ranks[-1]), ranks, df_ml if has_history else None
            )
            pred_methods[song_title] = method

            # 실제 데이터 (X축: 주차)
            actual_x = list(range(-len(ranks)+1, 1))
            pred_x = list(range(1, 5))

            artist = song_weekly.iloc[0]["artist"]

            # 실제 순위 라인
            fig.add_trace(go.Scatter(
                x=actual_x, y=list(ranks),
                mode="lines+markers", name=f"📍 {song_title}",
                line=dict(width=3),
                hovertemplate=f"<b>{song_title}</b> - {artist}<br>순위: %{{y}}위<extra></extra>"
            ))

            # 예측 라인 (점선)
            fig.add_trace(go.Scatter(
                x=pred_x, y=predicted,
                mode="lines+markers", name=f"🔮 {song_title} (예측)",
                line=dict(width=2, dash="dash"),
                marker=dict(symbol="star"),
                hovertemplate=f"<b>{song_title} 예측</b><br>순위: %{{y}}위<extra></extra>"
            ))

        fig.update_layout(
            title="📈 주간 순위 변동 + 4주 예측",
            xaxis_title="주차 (0=이번주, 음수=과거, 양수=예측)",
            yaxis_title="순위",
            yaxis=dict(autorange="reversed"),  # 1위가 위로
            height=600,
            hovermode="x unified"
        )
        fig.add_vline(x=0.5, line_dash="dot", line_color="red", annotation_text="← 실제 | 예측 →")
        st.plotly_chart(fig, use_container_width=True)

        # 예측 요약
        st.subheader("📋 예측 요약")
        for song_title in selected:
            sw = df_weekly[df_weekly["title"] == song_title].sort_values("week_offset")
            rt = df[(df["source"] == "melon") & (df["title"] == song_title)]
            if sw.empty and rt.empty:
                continue
            current = int(sw.iloc[0]["rank"]) if not sw.empty else int(rt.iloc[0]["rank"])
            method = pred_methods.get(song_title, "-")

            ranks_arr = sw["rank"].values if not sw.empty else np.array([current])
            last_change = ranks_arr[0] - ranks_arr[1] if len(ranks_arr) >= 2 else 0

            if last_change < -3:
                emoji, trend_desc = "🚀", "급상승"
            elif last_change < 0:
                emoji, trend_desc = "📈", "상승 추세"
            elif abs(last_change) <= 1:
                emoji, trend_desc = "➡️", "정체"
            else:
                emoji, trend_desc = "📉", "하락 중"

            st.write(f"{emoji} **{song_title}** (현재 {current}위) | {trend_desc} | 예측 방법: _{method}_")


# ============================================================
# 🔥 팬덤 vs 대중성
# ============================================================
elif page == "🔥 팬덤 vs 대중성":
    st.title("🔥 팬덤 화력 vs 대중성 분석")

    def _hbar_likes(plot_df, x_col, x_label, title):
        """순위 높은 순(1위)이 위, 좋아요 많을수록 붉은색 수평 막대그래프"""
        plot_df = plot_df.sort_values("rank", ascending=False)  # 아래→위 = 낮→높순위
        likes_min = plot_df[x_col].min()
        likes_max = plot_df[x_col].max()
        norm = (plot_df[x_col] - likes_min) / (likes_max - likes_min + 1e-9)

        fig = go.Figure(go.Bar(
            x=plot_df[x_col],
            y=plot_df["title"],
            orientation="h",
            marker=dict(
                color=norm,
                colorscale=[[0, "#3A7BD5"], [0.5, "#C0392B"], [1, "#7B0000"]],
                colorbar=dict(
                    title=x_label,
                    tickvals=[0, 0.5, 1],
                    ticktext=["적음", "보통", "많음"],
                    len=0.6,
                ),
                line=dict(width=0),
            ),
            customdata=plot_df[["rank", "artist", x_col]].values,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "아티스트: %{customdata[1]}<br>"
                f"멜론 순위: %{{customdata[0]:.0f}}위<br>"
                f"{x_label}: %{{customdata[2]:,.0f}}"
                "<extra></extra>"
            ),
        ))
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title="곡명",
            height=max(500, len(plot_df) * 22),
            margin=dict(l=10, r=20, t=50, b=40),
        )
        return fig

    if df_yt.empty:
        st.warning("⚠️ YouTube 데이터가 없습니다. crawler에 YouTube API 키를 설정하고 다시 크롤링하세요.")

        st.subheader("💡 멜론 좋아요 수 vs 순위")
        melon = df[df["source"] == "melon"].copy()
        if "likes" in melon.columns and melon["likes"].sum() > 0:
            melon_top50 = melon[melon["rank"] <= 50].copy()
            fig = _hbar_likes(melon_top50, "likes", "멜론 좋아요 수",
                              "멜론 좋아요 수 (순위 높은 순, 붉을수록 좋아요 많음)")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            **해석 가이드:**
            - **좋아요 적음 + 순위 높음** → 🎧 **대중성 곡** (스트리밍으로 순위 상승)
            - **좋아요 많음 + 순위 낮음** → 💪 **팬덤 곡** (팬 좋아요는 많지만 순위는 낮음)
            - **좋아요 많음 + 순위 높음** → 👑 **올라운더** (팬덤 + 대중성 모두)
            """)
        else:
            st.info("좋아요 데이터가 수집되지 않았습니다.")
    else:
        st.subheader("📺 YouTube 좋아요 vs 멜론 순위")
        df_yt_plot = df_yt[~df_yt["title"].str.contains("봄날", na=False)].copy()
        fig = _hbar_likes(df_yt_plot, "like_count", "YouTube 좋아요 수",
                          "YouTube 좋아요 수 (순위 높은 순, 붉을수록 좋아요 많음)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **해석:**
        - **좋아요 多 + 순위 高** → 👑 팬덤 + 대중성 모두 갖춘 곡
        - **좋아요 少 + 순위 高** → 🎧 대중적 스트리밍으로 올라간 곡
        - **좋아요 多 + 순위 低** → 💪 팬덤 화력으로 유지되는 곡
        """)

        st.subheader("📊 YouTube 통계 상세")
        yt_display = df_yt[["rank", "title", "artist", "view_count", "like_count", "comment_count"]].copy()
        yt_display.columns = ["순위", "곡명", "가수", "조회수", "좋아요", "댓글"]
        st.dataframe(
            yt_display.style
                .format({"조회수": "{:,.0f}", "좋아요": "{:,.0f}", "댓글": "{:,.0f}"})
                .background_gradient(subset=["조회수"], cmap="Blues")
                .background_gradient(subset=["좋아요"], cmap="Reds")
                .background_gradient(subset=["댓글"], cmap="Greens")
                .set_properties(**{"text-align": "right"}, subset=["조회수", "좋아요", "댓글"])
                .set_properties(**{"font-weight": "bold"}, subset=["순위", "곡명"]),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# 📺 YouTube MV 탐색
# ============================================================
elif page == "📺 YouTube MV 탐색":
    st.title("📺 YouTube 뮤직비디오 탐색")

    if df_yt.empty:
        st.warning("⚠️ YouTube 데이터가 없습니다. 크롤러를 실행하면 자동으로 수집됩니다.")
        st.info("💡 `python music_chart_crawler.py` 실행 후 다시 확인하세요.")
        st.stop()

    # 멜론 차트 순위 기준으로 곡 목록 구성
    song_options = (
        df_yt.sort_values("rank")[["rank", "title", "artist"]]
        .drop_duplicates("title")
        .apply(lambda r: f"{int(r['rank']):>3}위  {r['title']}  —  {r['artist']}", axis=1)
        .tolist()
    )
    selected_label = st.selectbox("🎵 곡을 선택하세요", song_options)

    # 선택된 곡명 파싱
    selected_title = selected_label.split("—")[0].strip().split("  ")[-1].strip()
    row = df_yt[df_yt["title"] == selected_title]
    if row.empty:
        st.error("해당 곡 데이터를 찾을 수 없습니다.")
        st.stop()
    row = row.iloc[0]

    st.divider()

    # 영상 임베드
    if pd.notna(row.get("video_id")) and row["video_id"]:
        st.subheader(f"🎬 {row['video_title']}")
        st.video(f"https://www.youtube.com/watch?v={row['video_id']}")

    st.divider()

    # 지표 카드
    col1, col2, col3 = st.columns(3)
    col1.metric("👁️ 조회수", f"{int(row['view_count']):,}")
    col2.metric("👍 좋아요", f"{int(row['like_count']):,}")
    col3.metric("💬 전체 댓글 수", f"{int(row['comment_count']):,}")

    # 댓글 4개 표시
    st.subheader("💬 인기 댓글")
    has_comment = False
    for i in range(1, 5):
        comment = row.get(f"comment{i}", "")
        if pd.notna(comment) and str(comment).strip():
            st.info(f"💬 {comment}")
            has_comment = True
    if not has_comment:
        st.caption("댓글이 비활성화된 영상이거나 댓글 데이터가 없습니다.")


# ============================================================
# 🎵 장르·계절 분류
# ============================================================
elif page == "🎵 장르·계절 분류":
    st.title("🎵 장르 · 계절 상관관계 분류")

    @st.cache_data(ttl=600)
    def load_monthly():
        try:
            df_db = pd.read_sql("SELECT * FROM monthly_chart", get_engine())
            if not df_db.empty:
                return df_db
        except Exception:
            pass
        import glob as _glob, json as _json, os as _os
        # 크롤러가 저장하는 기본 파일명 우선 확인
        candidates = (
            ["monthly_genre_data.csv"]
            + sorted(_glob.glob("music_chart_monthly_*.csv"))
            + sorted(_glob.glob("monthly_*.csv"))
        )
        dfs = []
        for path in candidates:
            if _os.path.exists(path):
                try:
                    dfs.append(pd.read_csv(path, dtype={"song_id": str, "year_month": str}))
                except Exception:
                    pass
        if not dfs:
            return pd.DataFrame()
        df_m = pd.concat(dfs, ignore_index=True).drop_duplicates()
        # genre 컬럼이 없으면 genre_cache.json 으로 보완
        if "genre" not in df_m.columns or df_m["genre"].isna().all():
            cache_path = "genre_cache.json"
            if _os.path.exists(cache_path):
                with open(cache_path, "r", encoding="utf-8") as f:
                    gcache = _json.load(f)
                df_m["genre"] = df_m["song_id"].astype(str).map(gcache).fillna("기타")
        # month / season 컬럼 보정
        if "month" not in df_m.columns and "year_month" in df_m.columns:
            df_m["month"] = df_m["year_month"].astype(str).str[4:6].astype(int)
        return df_m

    # PNG 파일 매핑
    _FIG_FILES = {
        "📅 월별 장르 비율":     "fig1_monthly_genre_distribution.png",
        "🌸 계절별 장르 비율":   ("fig2_season_genre_heatmap.png",
                                  "fig5_season_top_bottom_genres.png"),
        "📊 계절×장르 히트맵":   "fig3_season_genre_r_heatmap.png",
        "🔗 Pearson 상관계수":   "fig4_genre_monthly_trend_r.png",
    }
    _WARN = "PNG 파일이 없습니다. `python music_chart_crawler.py`를 실행하면 자동 생성됩니다."

    tab1, tab2, tab3, tab4 = st.tabs(list(_FIG_FILES.keys()))

    def _show_png(tab, key):
        with tab:
            files = _FIG_FILES[key]
            if isinstance(files, str):
                files = (files,)
            found = False
            for f in files:
                if os.path.exists(f):
                    st.image(f, use_column_width=True)
                    found = True
            if not found:
                st.warning(_WARN)

    _show_png(tab1, "📅 월별 장르 비율")

    # 연월 선택 파이차트
    with tab1:
        df_m = load_monthly()
        if not df_m.empty and "genre" in df_m.columns and "year_month" in df_m.columns:
            st.markdown("---")
            st.subheader("🥧 연월별 장르 비율 (파이차트)")

            months = sorted(df_m["year_month"].astype(str).unique(), reverse=True)
            selected_month = st.selectbox("연월 선택", months, index=0,
                                          format_func=lambda x: f"{x[:4]}년 {x[4:6]}월")

            df_pie = (df_m[df_m["year_month"].astype(str) == selected_month]
                      .groupby("genre").size().reset_index(name="count"))

            if not df_pie.empty:
                import plotly.express as px
                fig = px.pie(df_pie, names="genre", values="count",
                             title=f"{selected_month[:4]}년 {selected_month[4:6]}월 장르 분포",
                             hole=0.3)
                fig.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("해당 월 데이터가 없습니다.")
        else:
            pass

    _show_png(tab2, "🌸 계절별 장르 비율")
    _show_png(tab3, "📊 계절×장르 히트맵")
    _show_png(tab4, "🔗 Pearson 상관계수")


# ============================================================
# 🎮 가사 퀴즈 게임
# ============================================================
elif page == "🎮 가사 퀴즈 게임":
    st.title("🎮 가사 맞추기 퀴즈")

    # 세션 초기화
    if "quiz_score" not in st.session_state:
        st.session_state.quiz_score = 0
        st.session_state.quiz_total = 0
        st.session_state.quiz_nickname = ""

    # ── 닉네임 입력 화면 ─────────────────────────────────────
    if not st.session_state.quiz_nickname:
        st.write("퀴즈에 참여하기 전에 닉네임을 입력하세요.")
        nick_input = st.text_input("닉네임 (최대 20자)", max_chars=20, placeholder="예: 음악왕")
        if st.button("시작하기 ▶") and nick_input.strip():
            st.session_state.quiz_nickname = nick_input.strip()
            st.rerun()
        st.stop()

    nickname = st.session_state.quiz_nickname
    st.write(f"가사 일부와 가수 이름을 보고 **곡 제목**을 맞춰보세요!  참여자: **{nickname}**")

    # ── 가사 데이터 ──────────────────────────────────────────
    lyrics_db = [
        {"lyrics": "팔랑귀 팔랑귀 (that's red-red) 눈치나 살피기 (that's red-red) 도가니 사리기 (that's red-red) 넘어가 울타리 green green", "artist": "CORTIS (코르티스)", "title": "REDRED"},
        {"lyrics": "It's me 내가 바로 네가 찾던 사람", "artist": "아일릿(ILLIT)", "title": "It's Me"},
        {"lyrics": "소문의 낙원 그곳에 가면 모든 게 달라질 거야", "artist": "AKMU (악뮤)", "title": "소문의 낙원"},
        {"lyrics": "갑자기 너란 사람이 다가와 내 세상을 바꿔놨어", "artist": "아이오아이 (I.O.I)", "title": "갑자기"},
        {"lyrics": "기쁨 슬픔 아름다운 마음 모두 다 안아줄게", "artist": "AKMU (악뮤)", "title": "기쁨, 슬픔, 아름다운 마음"},
        {"lyrics": "캐치 캐치 너의 마음을 캐치해볼래", "artist": "YENA (최예나)", "title": "캐치 캐치"},
        {"lyrics": "사랑하게 될 거야 분명히 그렇게 될 거야", "artist": "한로로", "title": "사랑하게 될 거야"},
        {"lyrics": "Heavy serenade 밤하늘에 울려퍼지는 노래", "artist": "NMIXX", "title": "Heavy Serenade"},
        {"lyrics": "Bang bang 터뜨려 지금 이 순간을 즐겨", "artist": "IVE (아이브)", "title": "BANG BANG"},
        {"lyrics": "Rude 무례하게 들릴 수 있지만 솔직히 말할게", "artist": "Hearts2Hearts (하츠투하츠)", "title": "RUDE!"},
        {"lyrics": "네 곁에 있으면 나는 행복해 아무것도 필요 없어", "artist": "뉴진스(NewJeans)", "title": "Supernatural"},
        {"lyrics": "하늘을 달리는 기분이야 너와 함께라면", "artist": "에스파(aespa)", "title": "Whiplash"},
        {"lyrics": "달려가 꿈을 향해 멈추지 말고 지금 바로", "artist": "르세라핌(LE SSERAFIM)", "title": "CRAZY"},
        {"lyrics": "오늘도 너를 떠올리며 하루를 보내고 있어", "artist": "임영웅", "title": "모래 알갱이"},
        {"lyrics": "내 마음속에 불꽃이 타오르는 것 같아", "artist": "비비(BIBI)", "title": "밤양갱"},
    ]

    # YouTube video_id 맵 (title → video_id)
    _yt_vid_map = {}
    if not df_yt.empty and "title" in df_yt.columns and "video_id" in df_yt.columns:
        _yt_vid_map = (
            df_yt[df_yt["video_id"].notna() & (df_yt["video_id"] != "")]
            .drop_duplicates("title")
            .set_index("title")["video_id"]
            .to_dict()
        )

    if st.button("🎲 새 문제 출제", key="lyrics_new"):
        q = random.choice(lyrics_db)
        wrong = [s["title"] for s in lyrics_db if s["title"] != q["title"]]
        wrong_choices = random.sample(wrong, min(3, len(wrong)))
        choices = [q["title"]] + wrong_choices
        random.shuffle(choices)
        st.session_state.lyrics_q = {
            "lyrics": q["lyrics"],
            "artist": q["artist"],
            "answer": q["title"],
            "choices": choices
        }
        st.session_state.pop("last_correct", None)
        st.session_state.pop("show_music_hint", None)

    if "lyrics_q" in st.session_state and st.session_state.lyrics_q:
        lq = st.session_state.lyrics_q

        st.markdown(f"""
---
### 🎤 가수: **{lq['artist']}**

### 📝 가사:
> *"{lq['lyrics']}"*
---
""")

        # ── 음악 힌트 (YouTube 오디오) ────────────────────────
        _vid = _yt_vid_map.get(lq["answer"], "")
        if _vid:
            _hint_start = random.randint(25, 55)
            if st.button("🎵 음악 힌트 듣기", key="btn_music_hint"):
                st.session_state.show_music_hint = not st.session_state.get("show_music_hint", False)
            if st.session_state.get("show_music_hint", False):
                st.components.v1.html(
                    f"""
                    <div style="margin:6px 0 10px 0;">
                      <iframe
                        width="100%" height="68"
                        src="https://www.youtube.com/embed/{_vid}?autoplay=1&start={_hint_start}&controls=1&modestbranding=1&rel=0&fs=0&iv_load_policy=3"
                        frameborder="0"
                        allow="autoplay; encrypted-media"
                        style="border-radius:8px;">
                      </iframe>
                    </div>
                    """,
                    height=80,
                )
        else:
            st.caption("🎵 이 곡은 YouTube 데이터가 없어 음악 힌트를 제공할 수 없습니다.")

        choice = st.radio("이 곡의 제목은?", lq["choices"], key="lyrics_choice")

        if st.button("정답 확인", key="lyrics_check"):
            # 같은 문제를 중복 채점하지 않기 위해 answered 플래그 활용
            already = st.session_state.get("last_answered_q", "")
            if already != lq["answer"] + str(lq["choices"]):
                st.session_state.last_answered_q = lq["answer"] + str(lq["choices"])
                st.session_state.quiz_total += 1
                if choice == lq["answer"]:
                    st.success(f"🎉 정답! **{lq['answer']}**")
                    st.session_state.quiz_score += 1
                    update_quiz_score(nickname)   # DB 저장
                    st.balloons()
                else:
                    st.error(f"❌ 오답! 정답은 **{lq['answer']}**")
            else:
                st.info("이미 채점된 문제입니다. 새 문제를 출제하세요.")

    # ── 현재 세션 점수판 ──────────────────────────────────────
    st.divider()
    if st.session_state.quiz_total > 0:
        acc = st.session_state.quiz_score / st.session_state.quiz_total * 100
        st.metric("🏆 이번 세션 점수",
                  f"{st.session_state.quiz_score}/{st.session_state.quiz_total}",
                  f"정답률 {acc:.0f}%")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 세션 점수 초기화"):
            st.session_state.quiz_score = 0
            st.session_state.quiz_total = 0
            st.session_state.pop("last_answered_q", None)
            st.rerun()
    with col_b:
        if st.button("🚪 닉네임 변경"):
            st.session_state.quiz_score = 0
            st.session_state.quiz_total = 0
            st.session_state.quiz_nickname = ""
            st.session_state.pop("lyrics_q", None)
            st.session_state.pop("last_answered_q", None)
            st.rerun()

    # ── 전체 순위표 (휴지통 기능 포함) ───────────────────────
    st.divider()
    st.subheader("🏆 전체 순위표")
    _lb_in_quiz = load_leaderboard()
    if _lb_in_quiz.empty:
        st.caption("아직 참여자가 없습니다.")
    else:
        for _qi, _qrow in enumerate(_lb_in_quiz.itertuples(index=False)):
            _medal = ("🥇" if _qi == 0 else "🥈" if _qi == 1 else "🥉" if _qi == 2 else f"{_qi+1}.")
            _qcol_name, _qcol_btn = st.columns([5, 1])
            with _qcol_name:
                st.write(f"{_medal} **{_qrow.nickname}** — {int(_qrow.score)}점")
            with _qcol_btn:
                if _qrow.nickname == nickname:
                    if st.button("🗑️", key=f"del_score_{_qi}", help="내 점수 삭제"):
                        delete_quiz_score(nickname)
                        st.session_state.quiz_score = 0
                        st.session_state.quiz_total = 0
                        st.session_state.pop("last_answered_q", None)
                        st.rerun()


# ============================================================
# 📋 전체 데이터
# ============================================================
elif page == "📋 전체 데이터":
    st.title("📋 전체 차트 데이터")

    src_filter = st.multiselect("사이트", ["melon", "bugs"], default=["melon", "bugs"])
    rng = st.slider("순위 범위", 1, 100, (1, 100))

    filtered = df[(df["source"].isin(src_filter)) & (df["rank"] >= rng[0]) & (df["rank"] <= rng[1])].copy()

    # 항상 None인 컬럼 제거
    drop_cols = [c for c in ["rank_change", "like_count"] if c in filtered.columns]
    if drop_cols:
        filtered = filtered.drop(columns=drop_cols)

    st.write(f"총 {len(filtered)}건")
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    csv = filtered.to_csv(index=False, encoding="utf-8-sig")
    st.download_button("📥 CSV 다운로드", csv, "chart_filtered.csv", "text/csv")
