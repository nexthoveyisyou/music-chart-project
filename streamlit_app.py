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
        df = pd.read_sql(
            "SELECT * FROM youtube_stats ORDER BY `rank`, id DESC",
            get_engine()
        )
        return df.drop_duplicates(subset=["rank"], keep="first")
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
        return predicted, "주간 변화량 가중평균"

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
        st.error("❌ 데이터가 없습니다. 잠시 후 다시 시도해 주세요.")
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

st.sidebar.divider()

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

    # 사이트별 TOP 100
    st.subheader("🏆 사이트별 TOP 100")
    tab1, tab2 = st.tabs(["🍈 멜론", "🐛 벅스"])
    for tab, src in [(tab1, "melon"), (tab2, "bugs")]:
        with tab:
            top100 = df[df["source"] == src].sort_values("rank")
            if not top100.empty:
                display = top100[["rank", "title", "artist", "album"]].copy()
                display.columns = ["순위", "곡명", "가수", "앨범"]
                display["순위"] = display["순위"].apply(lambda r: f"🥇 {r}" if r == 1 else f"🥈 {r}" if r == 2 else f"🥉 {r}" if r == 3 else f"  {r}")
                st.dataframe(display, use_container_width=True, hide_index=True, height=500)

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
    st.caption(
        "멜론·벅스 두 차트에 동시 진입한 곡은 특정 팬층이 아닌 **폭넓은 청취층**을 확보한 곡입니다. "
        "반대로 한 플랫폼에만 등장하는 곡은 해당 플랫폼 이용자 성향(연령대·장르 선호)에 맞는 곡일 가능성이 높습니다."
    )

    # 아티스트 출현 빈도
    st.subheader("🎤 아티스트별 차트 진입 횟수")
    ac = df.groupby("artist").size().reset_index(name="count").sort_values("count", ascending=False).head(15)
    fig2 = px.bar(ac, x="count", y="artist", orientation="h", color="count", color_continuous_scale="sunset")
    fig2.update_layout(yaxis=dict(autorange="reversed"), height=500)
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "멜론·벅스 차트를 합산한 수치입니다. 횟수가 2 이상이면 두 플랫폼 모두에 진입했거나 "
        "여러 곡이 동시에 차트에 올라있다는 의미로, **크로스플랫폼 영향력**이 큰 아티스트를 가늠할 수 있습니다."
    )


# ============================================================
# 📈 4주 순위 예측
# ============================================================
elif page == "📈 4주 순위 예측":
    st.title("📈 멜론 주간 순위 기반 4주 예측")

    if df_weekly.empty:
        st.warning("⚠️ 주간 순위 데이터가 없습니다.")
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

            ranks = song_weekly["rank"].values   # ascending=False 정렬 → 이미 과거→현재

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
        st.caption(
            "점선(🔮)은 과거 주간 순위 추세를 선형 회귀로 연장한 예측값입니다. "
            "주간 데이터가 부족할 경우 월별 히스토리 또는 순위 구간별 감쇠 모델을 사용하며, "
            "**급격한 이슈·컴백·음방 출연 등 외부 요인은 반영되지 않으므로** 참고용으로만 활용하세요."
        )

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
            # 최근 4주 평균 기울기로 판단 (1주 변화만 보면 평탄 구간에서 오류 발생)
            n = min(len(ranks_arr), 4)
            if n >= 2:
                recent = ranks_arr[:n][::-1]  # 오래된→최근 순
                avg_change = float(np.mean(np.diff(recent)))
            else:
                avg_change = 0

            if avg_change > 8:
                emoji, trend_desc = "📉", "급락"
            elif avg_change > 1:
                emoji, trend_desc = "📉", "하락 중"
            elif abs(avg_change) <= 1:
                emoji, trend_desc = "➡️", "정체"
            elif avg_change > -5:
                emoji, trend_desc = "📈", "상승 추세"
            else:
                emoji, trend_desc = "🚀", "급상승"

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
        st.warning("⚠️ YouTube 데이터가 없습니다.")

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
        df_yt_plot = (
            df_yt[~df_yt["title"].str.contains("Dynamite", na=False)]
            .nlargest(10, "like_count")
            .copy()
        )
        fig = _hbar_likes(df_yt_plot, "like_count", "YouTube 좋아요 수",
                          "YouTube 좋아요 수 TOP 10 (붉을수록 좋아요 많음)")
        st.plotly_chart(fig, use_container_width=True)

        dyn_row = df_yt[df_yt["title"].str.contains("Dynamite", na=False)]
        if not dyn_row.empty:
            dyn_likes = int(dyn_row.iloc[0]["like_count"])
            dyn_rank  = int(dyn_row.iloc[0]["rank"])
            st.caption(
                f"※ BTS - Dynamite ({dyn_rank}위): 좋아요 {dyn_likes:,}개 "
                "— 다른 곡들과 차이가 너무 커 차트에서 제외"
            )

        st.markdown("""
        **해석:**
        - **좋아요 多 + 순위 高** → 👑 팬덤 + 대중성 모두 갖춘 곡
        - **좋아요 少 + 순위 高** → 🎧 대중적 스트리밍으로 올라간 곡
        - **좋아요 多 + 순위 低** → 💪 팬덤 화력으로 유지되는 곡
        """)

        st.subheader("📊 YouTube 통계 상세 (멜론 TOP 100 기준)")
        melon_all = df[df["source"] == "melon"][["rank", "title", "artist"]].sort_values("rank")
        yt_cols = df_yt[["rank", "view_count", "like_count", "comment_count"]].drop_duplicates("rank")
        yt_merged = melon_all.merge(yt_cols, on="rank", how="left")
        yt_merged[["view_count", "like_count", "comment_count"]] = (
            yt_merged[["view_count", "like_count", "comment_count"]].fillna(0).astype(int)
        )
        yt_merged.columns = ["순위", "곡명", "가수", "조회수", "좋아요", "댓글"]
        st.dataframe(
            yt_merged.style
                .format({"조회수": "{:,.0f}", "좋아요": "{:,.0f}", "댓글": "{:,.0f}"})
                .background_gradient(subset=["조회수"], cmap="Blues")
                .background_gradient(subset=["좋아요"], cmap="Reds")
                .background_gradient(subset=["댓글"], cmap="Greens")
                .set_properties(**{"text-align": "right"}, subset=["조회수", "좋아요", "댓글"])
                .set_properties(**{"font-weight": "bold"}, subset=["순위", "곡명"]),
            use_container_width=True,
            hide_index=True,
            height=600,
        )
        st.caption(
            "YouTube 통계는 멜론 TOP 10 기준으로 수집됩니다. "
            "11위 이하 곡은 YouTube 데이터 없이 멜론 순위만 표시됩니다. "
            "**조회수가 높아도 좋아요 비율이 낮은 곡**은 알고리즘 추천으로 유입된 비팬 시청이 많은 경우일 수 있습니다."
        )


# ============================================================
# 📺 YouTube MV 탐색
# ============================================================
elif page == "📺 YouTube MV 탐색":
    st.title("📺 YouTube 뮤직비디오 탐색")

    if df_yt.empty:
        st.warning("⚠️ YouTube 데이터가 없습니다.")
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
        "🔗 Pearson 상관계수":   "fig4_genre_monthly_trend_r.png",
    }
    _WARN = "그래프 데이터가 없습니다."

    tab1, tab2, tab3 = st.tabs(list(_FIG_FILES.keys()))

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

    # 연월 선택 파이차트 (tab1)
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

    _show_png(tab2, "🌸 계절별 장르 비율")

    # 계절별 통합 파이차트 (tab2)
    with tab2:
        df_m2 = load_monthly()
        if not df_m2.empty and "genre" in df_m2.columns and "year_month" in df_m2.columns:
            st.markdown("---")
            st.subheader("🌸 계절별 장르 비율 (파이차트)")

            def _month_to_season(m):
                m = int(m)
                if m in (3, 4, 5):   return "봄 🌸"
                if m in (6, 7, 8):   return "여름 ☀️"
                if m in (9, 10, 11): return "가을 🍂"
                return "겨울 ❄️"

            df_s = df_m2.copy()
            df_s["month"] = df_s["year_month"].astype(str).str[4:6].astype(int)
            df_s["season"] = df_s["month"].apply(_month_to_season)

            season_order = ["봄 🌸", "여름 ☀️", "가을 🍂", "겨울 ❄️"]
            from plotly.subplots import make_subplots

            fig_s = make_subplots(
                rows=2, cols=2,
                subplot_titles=season_order,
                specs=[[{"type": "pie"}, {"type": "pie"}],
                       [{"type": "pie"}, {"type": "pie"}]]
            )
            positions = [(1,1),(1,2),(2,1),(2,2)]
            for i, season in enumerate(season_order):
                df_season = (df_s[df_s["season"] == season]
                             .groupby("genre").size().reset_index(name="count"))
                r, c = positions[i]
                fig_s.add_trace(
                    go.Pie(labels=df_season["genre"], values=df_season["count"],
                           hole=0.3, textinfo="percent+label",
                           textposition="inside", showlegend=(i == 0)),
                    row=r, col=c
                )
            fig_s.update_layout(height=700, title_text="계절별 장르 분포 (전체 기간 합산)")
            st.plotly_chart(fig_s, use_container_width=True)
            st.caption(
                "계절마다 선호 장르가 다를 수 있습니다. "
                "예를 들어 여름에 댄스·팝 비중이 높아지고, 겨울에 발라드 비중이 높아지는 패턴이 나타난다면 "
                "계절이 음원 소비 취향에 영향을 준다는 근거가 됩니다."
            )

    _show_png(tab3, "🔗 Pearson 상관계수")

    with tab3:
        st.markdown("---")
        st.markdown(
            """
**📖 Pearson 상관계수 해석 방법**

Pearson 상관계수(r)는 두 변수 간의 **선형 관계 강도**를 −1 ~ +1 사이 값으로 나타냅니다.

| r 값 범위 | 해석 |
|---|---|
| 0.7 ~ 1.0 | 강한 양의 상관 — 한 쪽이 늘면 다른 쪽도 확실히 늘어남 |
| 0.3 ~ 0.7 | 중간 양의 상관 — 어느 정도 함께 움직이는 경향 |
| −0.3 ~ 0.3 | 거의 상관 없음 — 두 변수가 독립적으로 움직임 |
| −0.7 ~ −0.3 | 중간 음의 상관 — 한 쪽이 늘면 다른 쪽은 줄어드는 경향 |
| −1.0 ~ −0.7 | 강한 음의 상관 — 한 쪽이 늘면 다른 쪽은 확실히 줄어듦 |

위 그래프에서 **각 장르의 r 값**을 확인하면, 특정 장르가 월이 지날수록 꾸준히 증가/감소하는지 파악할 수 있습니다.
예를 들어 발라드의 r이 −0.6이라면 "시간이 지날수록 발라드 비중이 낮아지는 추세"로 읽을 수 있고,
댄스/팝의 r이 +0.7 이상이라면 "최근으로 올수록 댄스·팝이 차트를 점유하는 비중이 커지고 있다"는 의미입니다.
            """
        )


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
    st.write(f"가사와 가수 초성을 보고 **가수와 곡 제목**을 맞춰보세요!  참여자: **{nickname}**")

    # ── 초성 추출 유틸 ────────────────────────────────────────
    def _get_chosung(text):
        CHOSUNG = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
        import re
        # 괄호 안 한글 우선 추출
        m = re.search(r'[((]([가-힣]+)[))]', text)
        korean = m.group(1) if m else re.sub(r'[^가-힣]', '', text)
        if not korean:
            return ' '.join(w[0].upper() for w in text.split() if w)
        return ' '.join(CHOSUNG[(ord(c) - 0xAC00) // 588] for c in korean)

    # ── 가사 데이터 ──────────────────────────────────────────
    lyrics_db = [
        {"lyrics": "팔랑귀 팔랑귀 (that's red-red) 눈치나 살피기 (that's red-red)", "artist": "CORTIS (코르티스)", "title": "REDRED"},
        {"lyrics": "Who's your bias? I'm your bias!", "artist": "아일릿(ILLIT)", "title": "It's Me"},
        {"lyrics": "지치고 병든 나그네여 우 외톨이 나그네여", "artist": "AKMU (악동뮤지션)", "title": "소문의 낙원"},
        {"lyrics": "Till the morning 그렇게 아침이 밝아오네 잊으려 누웠는데", "artist": "아이오아이 (I.O.I)", "title": "갑자기"},
        {"lyrics": "햇빛 뒤에 그늘이 있는 건 사랑스러운 모습이야", "artist": "AKMU (악동뮤지션)", "title": "기쁨, 슬픔, 아름다운 마음"},
        {"lyrics": "Oh oh 살짝쿵 Oh oh 느낌 왔지", "artist": "YENA (최예나)", "title": "캐치 캐치"},
        {"lyrics": "아 뭐가 그리 샘이 났길래 그토록 휘몰아쳤던가", "artist": "한로로", "title": "사랑하게 될 거야"},
        {"lyrics": "커진 심장 소릴 들어봐 영원히 기억될 이 순간", "artist": "NMIXX", "title": "Heavy Serenade"},
        {"lyrics": "I don't give a 쉿! What you say", "artist": "IVE (아이브)", "title": "BANG BANG"},
        {"lyrics": "You can't make me act right", "artist": "Hearts2Hearts (하츠투하츠)", "title": "RUDE!"},
        {"lyrics": "I don't know what we've done 되돌아가긴 싫어 もう知っている", "artist": "뉴진스(NewJeans)", "title": "Supernatural"},
        {"lyrics": "집중해 좀 더 Think fast 이유 넌 이해 못 해", "artist": "에스파(aespa)", "title": "Whiplash"},
        {"lyrics": "넌 CPR같이 손대면 like 피카츄 백만 볼트 전기 it's pumping", "artist": "르세라핌(LE SSERAFIM)", "title": "CRAZY"},
        {"lyrics": "그대 이 모래에 작은 발자국을 내어요 깊게 패이지 않을 만큼 가볍게", "artist": "임영웅", "title": "모래 알갱이"},
        {"lyrics": "떠나는 길에 네가 내게 말했지 너는 바라는 게 너무나 많아", "artist": "비비(BIBI)", "title": "밤양갱"},
        {"lyrics": "다 알면서 눈 감은 넌 왜 다정한 말로 나를 죽여놓고", "artist": "WOODZ", "title": "Drowning"},
        {"lyrics": "난 널 버리지 않아 너도 같은 생각이지?", "artist": "한로로", "title": "0+0"},
        {"lyrics": "나 잡아봐라 off the Wi-Fi 말을 안 해도 I'm already on this vibe", "artist": "KiiiKiii (키키)", "title": "404 (New Era)"},
        {"lyrics": "안녕은 우릴 아프게 하지만 우아할 거야 (oh)", "artist": "화사 (HWASA)", "title": "Good Goodbye"},
        {"lyrics": "첨 그날처럼, 그댈 보면, 행복해져", "artist": "도경수(D.O.)", "title": "Popcorn"},
        {"lyrics": "어디까지 갔나요? 또 어떤 어른이 됐나요?", "artist": "다비치", "title": "타임캡슐"},
        {"lyrics": "I just wanna dive, I just wanna dive", "artist": "방탄소년단", "title": "SWIM"},
        {"lyrics": "It is what it is, they'd be like \"Who's she?\"", "artist": "aespa", "title": "WDA (Whole Different Animal)"},
        {"lyrics": "천진난만한 이런 기분도 신이 나서 날아갈 정도로 웃었던 날도", "artist": "10CM", "title": "너에게 닿기를"},
        {"lyrics": "Back in the day 한 사람당 하나의 사랑이 있었대", "artist": "이찬혁", "title": "멸종위기사랑"},
        {"lyrics": "If this love is over 다시 뛰어들어 난", "artist": "NMIXX", "title": "Blue Valentine"},
        {"lyrics": "사랑이라는 이유로 서로를 포기하고 찢어질 것 같이 아파할 수 없어", "artist": "AKMU (악동뮤지션)", "title": "어떻게 이별까지 사랑하겠어, 널 사랑하는 거지"},
        {"lyrics": "넌 괜찮니? 지금도 나는 실감 나지 않는다", "artist": "우디 (Woody)", "title": "어제보다 슬픈 오늘"},
        {"lyrics": "만약 이게 나의 착각이래도 그대 오늘만은 품이 돼 주오", "artist": "카더가든", "title": "그대 작은 나의 세상이 되어"},
        {"lyrics": "두 눈을 감지 않을 이 밤 솟구치는 겨레의 마음", "artist": "방탄소년단", "title": "Body to Body"},
        {"lyrics": "네가 숨 쉬면, 따스한 바람이 불어와 네가 웃으면, 눈부신 햇살이 비춰", "artist": "성시경", "title": "너의 모든 순간"},
        {"lyrics": "Whoa, think you're runnin' that? Guess we gunnin' back", "artist": "BLACKPINK", "title": "뛰어(JUMP)"},
        {"lyrics": "Back then, when I was running out of your place", "artist": "로제 (ROSÉ)", "title": "toxic till the end"},
        {"lyrics": "때론 맘 같지 않아도 포기하지 않고", "artist": "로이킴", "title": "내게 사랑이 뭐냐고 물어본다면"},
        {"lyrics": "We're goin' up, up, up, it's our moment", "artist": "KPop Demon Hunters Cast", "title": "Golden"},
        {"lyrics": "찬바람 불어오니 그대 생각에 눈물짓네", "artist": "조째즈", "title": "모르시나요"},
        {"lyrics": "그대는 선물입니다, 하늘이 내려준", "artist": "이클립스 (ECLIPSE)", "title": "소나기"},
        {"lyrics": "하늘이 우리를 갈라 놓지만", "artist": "이창섭", "title": "천상연"},
        {"lyrics": "Golden days are still alive 외롭다는 말하지 마", "artist": "G-DRAGON", "title": "HOME SWEET HOME"},
        {"lyrics": "따사로운 온기가 닿을 구름을 향하는 비행이 망설여지기도 하겠지만", "artist": "이무진", "title": "청춘만화"},
        {"lyrics": "Keep on running 위험을 겁내지 마", "artist": "IVE (아이브)", "title": "BLACKHOLE"},
        {"lyrics": "매일 웃고 싶어요 걱정 없고 싶어요", "artist": "DAY6 (데이식스)", "title": "HAPPY"},
        {"lyrics": "You know how I do do do do do do", "artist": "방탄소년단", "title": "2.0"},
        {"lyrics": "All about you and I 다른 건 다 제쳐 두고 Now come with me, take my hand", "artist": "DAY6 (데이식스)", "title": "한 페이지가 될 수 있게"},
        {"lyrics": "난 real한 거만 뱉어서 개명 신청했어 김진짜", "artist": "김하온 (HAON)", "title": "TICK TOCK"},
        {"lyrics": "Like JENNIE JENNIE JENNIE JENNIE JENNIE", "artist": "제니 (JENNIE)", "title": "like JENNIE"},
        {"lyrics": "날 불러봐 눈을 뜬 순간 It's a comeback", "artist": "태양", "title": "LIVE FAST DIE SLOW"},
        {"lyrics": "이대로 내 곁에 있어야 해요 나를 떠나면 안 돼요", "artist": "너드커넥션 (Nerd Connection)", "title": "그대만 있다면"},
        {"lyrics": "Why this bassline slappin' so rude?", "artist": "방탄소년단", "title": "Hooligan"},
        {"lyrics": "사랑을 봄비처럼 내 마음 적시고", "artist": "임현정", "title": "사랑은 봄비처럼...이별은 겨울비처럼..."},
        {"lyrics": "나는 읽기 쉬운 마음이야 당신도 쓱 훑고 가셔요", "artist": "잔나비", "title": "주저하는 연인들을 위해"},
        {"lyrics": "밤 열두시 술 취해 지친 목소리 새벽 두시 차갑게 꺼진 전화기", "artist": "에픽하이 (EPIK HIGH)", "title": "Love Love Love"},
        {"lyrics": "아직도 가끔 네 생각이 나 어렵게 전화를 걸어볼까?", "artist": "DAY6 (데이식스)", "title": "예뻤어"},
        {"lyrics": "눈꽃이 떨어져요 또 조금씩 멀어져요", "artist": "방탄소년단", "title": "봄날"},
        {"lyrics": "Monday Tuesday Wednesday Thursday Friday Saturday Sunday (a week)", "artist": "정국", "title": "Seven (feat. Latto)"},
        {"lyrics": "Eat it up eat it eat it up", "artist": "LE SSERAFIM (르세라핌)", "title": "SPAGHETTI"},
        {"lyrics": "kissin' with somebody", "artist": "NOWIMYOUNG (나우아임영)", "title": "KISS KISS KISS"},
        {"lyrics": "나는 너 하나로 충분해 긴 말 안 해도 눈빛으로 다 아니깐", "artist": "폴킴", "title": "모든 날 모든 순간"},
        {"lyrics": "너와 함께 하고 싶은 일들을 상상하는 게 요즘 내 일상이 되고", "artist": "멜로망스", "title": "사랑인가 봐"},
        {"lyrics": "Don't you try me I want some more Don't you play me we on the floor", "artist": "ALLDAY PROJECT", "title": "FAMOUS"},
        {"lyrics": "사랑한다고 그대 하나만 바라본다고 그대 얼굴만", "artist": "순순희(지환)", "title": "눈을 감아도"},
        {"lyrics": "수백 번 연습하며 오늘을 기다려왔어", "artist": "이무진", "title": "청혼하지 않을 이유를 못 찾았어"},
        {"lyrics": "몰랐어요 난 내가 벌레라는 것을", "artist": "황가람", "title": "나는 반딧불"},
        {"lyrics": "아직 너를 너를 그리워해", "artist": "김나영", "title": "봄 내음보다 너를"},
        {"lyrics": "이것만큼은 맹세할게 내 전부를 다 바칠게", "artist": "DAY6 (데이식스)", "title": "Welcome to the Show"},
        {"lyrics": "그날 이후로 난 이렇게 살고 더는 기타 한 번도 들지 못 하고", "artist": "BOYNEXTDOOR", "title": "오늘만 I LOVE YOU"},
        {"lyrics": "수없이 많은 날들과 수없이 많은 사람 중에", "artist": "로이킴", "title": "달리 표현할 수 없어요"},
        {"lyrics": "사랑이란 게 참 쓰린 거더라", "artist": "임영웅", "title": "사랑은 늘 도망가"},
        {"lyrics": "Ooh 문 앞에서 셋을 세어본다 yeah 셋 둘 하나", "artist": "TWS (투어스)", "title": "첫 만남은 계획대로 되지 않아"},
        {"lyrics": "Oh you falling from the sky 언젠가 바라왔던 별 같은 기적", "artist": "오반(OVAN)", "title": "Flower"},
        {"lyrics": "happy in your smile 더 크게 웃어봐", "artist": "로이킴", "title": "Smile Boy"},
        {"lyrics": "시작의 푸름에 모든 이름에 니가 새겨져 있을 뿐", "artist": "박다혜, 마크툽 (MAKTUB)", "title": "시작의 아이 ❍"},
        {"lyrics": "Baby you make me smile", "artist": "Hearts2Hearts (하츠투하츠)", "title": "STYLE"},
        {"lyrics": "Don't you want me like I want you baby?", "artist": "로제 (ROSÉ)", "title": "APT."},
        {"lyrics": "You're makin' my heart 쿵쿵 (쿵!)", "artist": "TWS (투어스)", "title": "OVERDRIVE"},
        {"lyrics": "Uh boompala boompala boompala yeah (uh)", "artist": "LE SSERAFIM (르세라핌)", "title": "BOOMPALA"},
        {"lyrics": "느껴봐 Something in the air tonight", "artist": "ALLDAY PROJECT", "title": "ONE MORE TIME"},
        {"lyrics": "잠들지 않는 바다 위를 너와 함께 걷는 거 같아", "artist": "경서예지, 전건호", "title": "다정히 내 이름을 부르면"},
        {"lyrics": "So you can love me hate me You will never be never be never be me", "artist": "IVE (아이브)", "title": "REBEL HEART"},
        {"lyrics": "Everything lit it's fire", "artist": "방탄소년단", "title": "FYA"},
        {"lyrics": "I cannot focus on anything but you baby", "artist": "Hearts2Hearts (하츠투하츠)", "title": "FOCUS"},
        {"lyrics": "나와 저 끝까지 가줘 my lover", "artist": "아이유", "title": "Love wins all"},
        {"lyrics": "If the world was ending I'd wanna be next to you", "artist": "Lady Gaga, Bruno Mars", "title": "Die With A Smile"},
        {"lyrics": "숨기고 싶지 않아 자석 같은 my heart", "artist": "아일릿(ILLIT)", "title": "Magnetic"},
        {"lyrics": "언젠가 네가 말하던 그 사랑 얘기에 착각 속 살았던 날 다시 돌아본대도", "artist": "엔플라잉 (N.Flying)", "title": "Flashback"},
        {"lyrics": "'Cause I, I, I'm in the stars tonight", "artist": "방탄소년단", "title": "Dynamite"},
        {"lyrics": "손을 잡고 늘 걷던 거리에 첫눈을 보다가 문득 고백했던 그 순간", "artist": "박재정", "title": "헤어지자 말해요"},
        {"lyrics": "헤어질 수 없어 I'll just stick with you", "artist": "투모로우바이투게더", "title": "하루에 하루만 더"},
        {"lyrics": "근데 있잖아, 별 소용없다? 생각만 해도 행복한 순간들은 말야", "artist": "이무진", "title": "에피소드"},
        {"lyrics": "이 밤을 보내기엔 아쉽잖아요", "artist": "PLAVE", "title": "이 밤을 빌려 말해요"},
        {"lyrics": "Every night, every day 뭐든 더 빠르게", "artist": "방탄소년단", "title": "Aliens"},
        {"lyrics": "If you wanna be animals Baby we can be animals", "artist": "방탄소년단", "title": "Like Animals"},
        {"lyrics": "강아지보단 난 느슨한 해파리가 좋아", "artist": "아일릿(ILLIT)", "title": "NOT CUTE ANYMORE"},
        {"lyrics": "그리워하면 언젠간 만나게 되는", "artist": "아이유", "title": "Never Ending Story"},
        {"lyrics": "We sing for love Listen up 세상 모든 다정함", "artist": "NCT WISH", "title": "Ode to Love"},
        {"lyrics": "'Cause I know what you like boy (ah-ah)", "artist": "NewJeans", "title": "Hype Boy"},
        {"lyrics": "Life is 아름다운 galaxy Be a writer 장르로는 fantasy", "artist": "IVE (아이브)", "title": "I AM"},
        {"lyrics": "You're my soda pop my little soda pop", "artist": "KPop Demon Hunters Cast", "title": "Soda Pop"},
        {"lyrics": "And I can't get off of this ride", "artist": "방탄소년단", "title": "Merry Go Round"},
        {"lyrics": "시작의 푸름에 모든 이름에 니가 새겨져 있을 뿐", "artist": "마크툽 (MAKTUB)", "title": "시작의 아이"},
        {"lyrics": "흔하게 살자 아프지 말고", "artist": "임영웅", "title": "순간을 영원처럼"},
        {"lyrics": "Baby, oh please 세상이 우릴 갈라놓을 때", "artist": "방탄소년단", "title": "Please"},
        {"lyrics": "슬픔이 짙어질 때면 위로해 줄 그 한 사람이 될게요", "artist": "임영웅", "title": "우리들의 블루스"},
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
        wrong_pool = [s for s in lyrics_db if s["title"] != q["title"]]
        wrong_titles = [s["title"] for s in random.sample(wrong_pool, min(3, len(wrong_pool)))]
        choices = [q["title"]] + wrong_titles
        random.shuffle(choices)
        st.session_state.lyrics_q = {
            "lyrics": q["lyrics"],
            "artist_chosung": _get_chosung(q["artist"]),
            "answer_title": q["title"],
            "answer": f"{q['artist']} - {q['title']}",
            "choices": choices
        }
        st.session_state.pop("last_correct", None)
        st.session_state.pop("show_music_hint", None)

    if "lyrics_q" in st.session_state and st.session_state.lyrics_q:
        lq = st.session_state.lyrics_q

        st.markdown(f"""
---
### 🎤 가수 초성: **{lq['artist_chosung']}**

### 📝 가사:
> *"{lq['lyrics']}"*
---
""")

        # ── 음악 힌트 (YouTube 오디오) ────────────────────────
        _vid = _yt_vid_map.get(lq["answer_title"], "")
        if _vid:
            _hint_start = random.randint(25, 55)
            if st.button("🎵 음악 힌트 듣기", key="btn_music_hint"):
                st.session_state.show_music_hint = not st.session_state.get("show_music_hint", False)
            if st.session_state.get("show_music_hint", False):
                st.components.v1.html(
                    f"""
                    <div style="position:relative; margin:6px 0 10px 0; background:#000; border-radius:8px; overflow:hidden;">
                      <iframe
                        width="100%" height="68"
                        src="https://www.youtube.com/embed/{_vid}?autoplay=1&start={_hint_start}&controls=1&modestbranding=1&rel=0&fs=0&iv_load_policy=3"
                        frameborder="0"
                        allow="autoplay; encrypted-media"
                        style="display:block; border-radius:8px;">
                      </iframe>
                      <div style="
                        position:absolute; top:0; left:0;
                        width:50%; height:100%;
                        background:linear-gradient(to right, #000 70%, transparent 100%);
                        z-index:10; pointer-events:none;">
                      </div>
                    </div>
                    """,
                    height=80,
                )
        else:
            st.caption("🎵 이 곡은 YouTube 데이터가 없어 음악 힌트를 제공할 수 없습니다.")

        choice = st.radio("곡 제목은?", lq["choices"], key="lyrics_choice")

        if st.button("정답 확인", key="lyrics_check"):
            # 같은 문제를 중복 채점하지 않기 위해 answered 플래그 활용
            already = st.session_state.get("last_answered_q", "")
            if already != lq["answer_title"] + str(lq["choices"]):
                st.session_state.last_answered_q = lq["answer_title"] + str(lq["choices"])
                st.session_state.quiz_total += 1
                if choice == lq["answer_title"]:
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

    csv = filtered.to_csv(index=False).encode("utf-8-sig")
    st.download_button("📥 CSV 다운로드", csv, "chart_filtered.csv", "text/csv; charset=utf-8-sig")
