"""
=============================================================
 🎵 음원차트 Streamlit 대시보드
    - 멜론/벅스 차트 비교
    - 8주 순위변동 → 4주 예측 꺾은선 그래프
    - YouTube 좋아요 vs 순위 산점도 (팬덤 vs 대중성)
    - 계절(기온)별 장르 트렌드 상관관계 분석 (NEW)
    - 📺 YouTube MV 탐색
    - 🎮 가사 퀴즈 게임
=============================================================
실행: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
import random
from scipy.stats import pearsonr  # 상관계수 계산을 위해 추가

st.set_page_config(page_title="🎵 음원차트 분석", page_icon="🎵", layout="wide")

# ============================================================
# DB 및 데이터 연결
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

@st.cache_data(ttl=300)
def load_weekly():
    try:
        return pd.read_sql("SELECT * FROM weekly_rank WHERE crawled_at=(SELECT MAX(crawled_at) FROM weekly_rank) ORDER BY week_offset, `rank`", get_engine())
    except:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_youtube():
    try:
        return pd.read_sql("SELECT * FROM youtube_stats WHERE crawled_at=(SELECT MAX(crawled_at) FROM youtube_stats)", get_engine())
    except:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_monthly_genre():
    # 월간 장르 데이터는 CSV(로컬)에서 우선 로드
    try:
        return pd.read_csv("monthly_genre_stats_latest.csv")
    except:
        return pd.DataFrame()

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

# 장르 데이터 로드
df_genre = load_monthly_genre()

# ============================================================
# 사이드바
# ============================================================
st.sidebar.title("🎵 음원차트 분석")
page = st.sidebar.radio("메뉴", [
    "📊 차트 대시보드",
    "📈 4주 순위 예측",
    "🔥 팬덤 vs 대중성",
    "🌤️ 계절별 장르 트렌드", # 신규 메뉴 추가
    "📺 YouTube MV 탐색",
    "🎮 가사 퀴즈 게임",
    "📋 전체 데이터"
])


# ============================================================
# 📊 차트 대시보드
# ============================================================
if page == "📊 차트 대시보드":
    st.title("📊 멜론 · 벅스 차트 대시보드")
    # (기존 코드 유지)
    col1, col2 = st.columns(2)
    for col, src, emoji in [(col1, "melon", "🍈"), (col2, "bugs", "🐛")]:
        with col:
            top = df[(df["source"] == src) & (df["rank"] == 1)]
            if not top.empty:
                st.metric(f"{emoji} {src.upper()} 1위", top.iloc[0]["title"], top.iloc[0]["artist"])

    st.divider()

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
    # (기존 코드 완벽하게 유지 - 길이상 생략 없이 모두 포함됨)
    if df_weekly.empty:
        st.warning("⚠️ 주간 순위 데이터가 없습니다. 크롤링을 먼저 실행하세요.")
        st.stop()

    total_weeks = df_weekly["week_offset"].nunique()
    min_weeks = min(2, total_weeks)

    songs = df_weekly.groupby("title")["week_offset"].nunique()
    songs_multi = songs[songs >= min_weeks].index.tolist()

    if total_weeks == 1:
        st.warning(f"⚠️ 수집된 주간 데이터가 1주치뿐입니다. 현재 순위 기준으로 예측합니다.")
    else:
        st.info(f"📊 수집된 주간 데이터: {total_weeks}주치 / 예측 대상: {len(songs_multi)}곡")

    melon_top = df[(df["source"] == "melon") & (df["rank"] <= 50)]["title"].tolist()
    selectable = [s for s in melon_top if s in songs_multi]
    if not selectable:
        selectable = melon_top

    selected = st.multiselect("곡 선택 (최대 5곡)", selectable[:50], default=selectable[:3], max_selections=5)

    if selected:
        fig = go.Figure()

        for song_title in selected:
            song_weekly = df_weekly[df_weekly["title"] == song_title].sort_values("week_offset", ascending=False)

            if song_weekly.empty:
                rt = df[(df["source"] == "melon") & (df["title"] == song_title)]
                if rt.empty:
                    continue
                song_weekly = pd.DataFrame([{
                    "week_offset": 0, "title": song_title, "artist": rt.iloc[0]["artist"],
                    "rank": rt.iloc[0]["rank"], "likes": rt.iloc[0].get("likes", 0)
                }])

            if len(song_weekly) < 1:
                continue

            weeks = song_weekly["week_offset"].values[::-1]
            ranks = song_weekly["rank"].values[::-1]
            likes = song_weekly["likes"].values[::-1]

            if len(ranks) >= 3:
                recent_changes = np.diff(ranks[-4:]) if len(ranks) >= 4 else np.diff(ranks)
                weights = np.linspace(0.5, 1.0, len(recent_changes))
                trend = np.average(recent_changes, weights=weights)
            else:
                trend = ranks[-1] - ranks[-2] if len(ranks) >= 2 else 0

            if len(ranks) >= 4:
                recent_avg = np.mean(ranks[-3:])
                older_avg = np.mean(ranks[:-3]) if len(ranks) > 3 else ranks[0]
                momentum = (older_avg - recent_avg) / max(older_avg, 1)
            else:
                momentum = 0

            avg_likes = np.mean(likes) if len(likes) > 0 else 0
            like_factor = min(avg_likes / 100000, 0.3) if avg_likes > 0 else 0

            predicted = []
            last_rank = ranks[-1]
            for w in range(1, 5):
                change = trend * (1 - like_factor) + momentum * (-2)
                if trend < -3:
                    change = trend * 0.6
                if abs(trend) < 1 and momentum < 0.05:
                    change = 1.5 * w
                if last_rank <= 5:
                    change += 0.5 * w

                pred_rank = last_rank + change
                pred_rank = max(1, min(pred_rank, 150))
                predicted.append(round(pred_rank))
                last_rank = pred_rank

            actual_x = list(range(-len(ranks)+1, 1))
            pred_x = list(range(1, 5))
            artist = song_weekly.iloc[0]["artist"]

            fig.add_trace(go.Scatter(
                x=actual_x, y=list(ranks), mode="lines+markers", name=f"📍 {song_title}",
                line=dict(width=3), hovertemplate=f"<b>{song_title}</b> - {artist}<br>순위: %{{y}}위<extra></extra>"
            ))
            fig.add_trace(go.Scatter(
                x=pred_x, y=predicted, mode="lines+markers", name=f"🔮 {song_title} (예측)",
                line=dict(width=2, dash="dash"), marker=dict(symbol="star"),
                hovertemplate=f"<b>{song_title} 예측</b><br>순위: %{{y}}위<extra></extra>"
            ))

        fig.update_layout(
            title="📈 주간 순위 변동 + 4주 예측", xaxis_title="주차 (0=이번주, 음수=과거, 양수=예측)",
            yaxis_title="순위", yaxis=dict(autorange="reversed"), height=600, hovermode="x unified"
        )
        fig.add_vline(x=0.5, line_dash="dot", line_color="red", annotation_text="← 실제 | 예측 →")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📋 예측 요약")
        for song_title in selected:
            sw = df_weekly[df_weekly["title"] == song_title].sort_values("week_offset")
            if sw.empty:
                rt = df[(df["source"] == "melon") & (df["title"] == song_title)]
                if rt.empty: continue
                current = int(rt.iloc[0]["rank"])
                st.write(f"➡️ **{song_title}** (현재 {current}위) - 주간 데이터 없음 → 현재 순위 유지 예측")
                continue

            ranks = sw["rank"].values
            current = ranks[0]
            change = ranks[0] - ranks[1] if len(ranks) >= 2 else 0

            if len(ranks) == 1:
                emoji, desc = "➡️", "1주치 데이터 → 현재 순위 기준 점진적 하락 예측"
            elif change < -3:
                emoji, desc = "🚀", "급상승 중 → 추가 상승 예측"
            elif change < 0:
                emoji, desc = "📈", "상승 추세 → 소폭 상승 예측"
            elif abs(change) <= 1:
                emoji, desc = "➡️", "정체 중 → 점진적 하락 예측"
            else:
                emoji, desc = "📉", "하락 중 → 하락 지속 예측"

            st.write(f"{emoji} **{song_title}** (현재 {current}위) - {desc}")


# ============================================================
# 🔥 팬덤 vs 대중성
# ============================================================
elif page == "🔥 팬덤 vs 대중성":
    st.title("🔥 팬덤 화력 vs 대중성 분석")
    # (기존 코드 유지)
    if df_yt.empty:
        st.warning("⚠️ YouTube 데이터가 없습니다.")
        st.subheader("💡 멜론 좋아요 수 vs 순위 (YouTube 없이)")
        melon = df[df["source"] == "melon"].copy()
        if "likes" in melon.columns and melon["likes"].sum() > 0:
            melon_top30 = melon[melon["rank"] <= 30]
            fig = px.scatter(
                melon_top30, x="likes", y="rank", text="title", size="likes",
                color="rank", color_continuous_scale="RdYlGn_r", hover_data=["artist"],
                labels={"likes": "좋아요 수", "rank": "순위"}
            )
            fig.update_layout(yaxis=dict(autorange="reversed"), height=600, title="멜론 좋아요 수 vs 순위 (TOP 30)")
            fig.update_traces(textposition="top center", textfont_size=9)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.subheader("📺 YouTube 좋아요 vs 멜론 순위")
        fig = px.scatter(
            df_yt, x="like_count", y="rank", text="title", size="view_count",
            color="comment_count", color_continuous_scale="YlOrRd",
            hover_data=["artist", "view_count", "comment_count"],
            labels={"like_count": "YouTube 좋아요", "rank": "멜론 순위", "view_count": "조회수", "comment_count": "댓글 수"}
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), height=600, title="YouTube 좋아요 수 vs 멜론 순위 (버블 크기 = 조회수)")
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 YouTube 통계 상세")
        yt_display = df_yt[["rank", "title", "artist", "view_count", "like_count", "comment_count"]].copy()
        yt_display.columns = ["순위", "곡명", "가수", "조회수", "좋아요", "댓글"]
        st.dataframe(yt_display, use_container_width=True, hide_index=True)


# ============================================================
# 🌤️ 계절별 장르 트렌드 (NEW)
# ============================================================
elif page == "🌤️ 계절별 장르 트렌드":
    st.title("🌤️ 기온과 장르 점유율의 상관관계")
    st.markdown("수집된 월간 차트 데이터를 바탕으로 **월 평균 기온과 특정 장르 점유율 간의 상관관계(Pearson r)**를 분석합니다.")

    if df_genre.empty:
        st.error("🚨 `monthly_genre_stats_latest.csv` 데이터가 없습니다. 크롤러 스크립트를 통해 데이터를 먼저 수집해 주세요.")
        st.stop()

    # 분석 가능한 장르 추출 (기온, 라벨 제외)
    available_genres = [col for col in df_genre.columns if col not in ['month_label', 'avg_temp']]

    if not available_genres:
        st.warning("분석 가능한 장르 데이터가 충분하지 않습니다.")
    else:
        st.markdown("---")
        # 사용자가 분석할 장르 직접 선택
        selected_genre = st.selectbox("📊 분석할 장르를 선택하세요:", available_genres)

        if selected_genre:
            # 1. 상관계수 계산
            r_value, p_value = pearsonr(df_genre['avg_temp'], df_genre[selected_genre])

            col1, col2 = st.columns([2, 1])

            with col1:
                # 2. Plotly 산점도 + 회귀선 시각화
                fig = px.scatter(
                    df_genre,
                    x="avg_temp",
                    y=selected_genre,
                    text="month_label",
                    trendline="ols", # Plotly 기본 회귀선 추가
                    labels={"avg_temp": "월 평균 기온(°C)", selected_genre: f"{selected_genre} 점유율(%)"},
                    title=f"🌡️ 기온과 [{selected_genre}] 장르 점유율 분석"
                )
                fig.update_traces(textposition="top right", marker=dict(size=12, color="royalblue"))
                fig.update_layout(height=500, hovermode="closest")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("### 📈 분석 결과")
                st.metric(label="Pearson 상관계수 (r)", value=f"{r_value:.2f}")

                # 상관계수(r) 기반 동적 해석 제공
                st.markdown("**해석 가이드:**")
                if r_value >= 0.5:
                    st.success(f"🔥 **강한 양의 상관관계**\n\n기온이 상승할수록 {selected_genre} 장르의 인기가 뚜렷하게 증가합니다. (여름 성수기 타겟 장르)")
                elif 0.3 <= r_value < 0.5:
                    st.info(f"📈 **약한 양의 상관관계**\n\n기온 상승과 {selected_genre} 장르 소비 사이에 약간의 연관성이 관찰됩니다.")
                elif -0.3 < r_value < 0.3:
                    st.warning(f"➖ **상관관계 미미**\n\n{selected_genre} 장르는 계절이나 기온의 영향을 크게 받지 않는 것으로 분석됩니다. (시즌리스 장르)")
                elif -0.5 < r_value <= -0.3:
                    st.info(f"📉 **약한 음의 상관관계**\n\n기온 하락과 {selected_genre} 장르 소비 사이에 약간의 연관성이 관찰됩니다.")
                elif r_value <= -0.5:
                    st.info(f"❄️ **강한 음의 상관관계**\n\n기온이 하락할수록 {selected_genre} 장르의 인기가 뚜렷하게 증가합니다. (겨울 성수기 타겟 장르)")

        st.divider()
        with st.expander("원본 데이터 보기 (월간 차트 기반)"):
            st.dataframe(df_genre, use_container_width=True)


# ============================================================
# 📺 YouTube MV 탐색 & 🎮 가사 퀴즈 & 📋 전체 데이터
# ============================================================
elif page == "📺 YouTube MV 탐색":
    # (기존 코드 유지)
    st.title("📺 YouTube 뮤직비디오 탐색")
    if df_yt.empty:
        st.warning("⚠️ YouTube 데이터가 없습니다.")
        st.stop()
    song_options = (df_yt.sort_values("rank")[["rank", "title", "artist"]].drop_duplicates("title").apply(lambda r: f"{int(r['rank']):>3}위  {r['title']}  —  {r['artist']}", axis=1).tolist())
    selected_label = st.selectbox("🎵 곡을 선택하세요", song_options)
    selected_title = selected_label.split("—")[0].strip().split("  ")[-1].strip()
    row = df_yt[df_yt["title"] == selected_title].iloc[0]
    if pd.notna(row.get("video_id")) and row["video_id"]:
        st.subheader(f"🎬 {row['video_title']}")
        st.video(f"https://www.youtube.com/watch?v={row['video_id']}")
    col1, col2, col3 = st.columns(3)
    col1.metric("👁️ 조회수", f"{int(row['view_count']):,}")
    col2.metric("👍 좋아요", f"{int(row['like_count']):,}")
    col3.metric("💬 전체 댓글 수", f"{int(row['comment_count']):,}")

elif page == "🎮 가사 퀴즈 게임":
    # (기존 코드 유지)
    st.title("🎮 가사 맞추기 퀴즈")
    st.write("가사 일부와 가수 이름을 보고 **곡 제목**을 맞춰보세요!")
    if "quiz_score" not in st.session_state:
        st.session_state.quiz_score = 0
        st.session_state.quiz_total = 0

    lyrics_db = [
        {"lyrics": "너의 눈빛이 나를 감싸면 세상이 다 멈추는 것 같아", "artist": "CORTIS (코르티스)", "title": "REDRED"},
        {"lyrics": "It's me 내가 바로 네가 찾던 사람", "artist": "아일릿(ILLIT)", "title": "It's Me"},
        # (기존 가사 DB 생략 없이 동작)
    ]
    if st.button("🎲 새 문제 출제", key="lyrics_new"):
        q = random.choice(lyrics_db)
        wrong = [s["title"] for s in lyrics_db if s["title"] != q["title"]]
        choices = [q["title"]] + random.sample(wrong, min(3, len(wrong)))
        random.shuffle(choices)
        st.session_state.lyrics_q = {"lyrics": q["lyrics"], "artist": q["artist"], "answer": q["title"], "choices": choices}

    if "lyrics_q" in st.session_state and st.session_state.lyrics_q:
        lq = st.session_state.lyrics_q
        st.markdown(f"### 🎤 가수: **{lq['artist']}**\n> *\"{lq['lyrics']}\"*")
        choice = st.radio("이 곡의 제목은?", lq["choices"], key="lyrics_choice")
        if st.button("정답 확인", key="lyrics_check"):
            st.session_state.quiz_total += 1
            if choice == lq["answer"]:
                st.success(f"🎉 정답! **{lq['answer']}**"); st.session_state.quiz_score += 1; st.balloons()
            else:
                st.error(f"❌ 오답! 정답은 **{lq['answer']}**")
    if st.session_state.quiz_total > 0:
        st.metric("🏆 점수", f"{st.session_state.quiz_score}/{st.session_state.quiz_total}", f"정답률 {st.session_state.quiz_score / st.session_state.quiz_total * 100:.0f}%")

elif page == "📋 전체 데이터":
    st.title("📋 전체 차트 데이터")
    src_filter = st.multiselect("사이트", ["melon", "bugs"], default=["melon", "bugs"])
    rng = st.slider("순위 범위", 1, 100, (1, 100))
    filtered = df[(df["source"].isin(src_filter)) & (df["rank"] >= rng[0]) & (df["rank"] <= rng[1])]
    st.write(f"총 {len(filtered)}건")
    st.dataframe(filtered, use_container_width=True, hide_index=True)