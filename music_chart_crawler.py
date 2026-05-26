"""
=============================================================
 멜론 / 벅스 음원차트 크롤링
    + 멜론 8주간 주간 순위 변동 & 좋아요 수집
    + 멜론 월간 TOP100 + 장르 수집
    + 계절별 장르 상관관계 분석 및 시각화
    + YouTube Data API (TOP10 뮤직비디오 통계)
    + MariaDB(로컬 + Aiven Cloud) 저장
    + Streamlit 대시보드 연동
=============================================================

[사전 설치]
pip install selenium beautifulsoup4 pandas numpy matplotlib seaborn scipy
       openpyxl google-api-python-client pymysql sqlalchemy
"""

import time
import re
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup


# ============================================================
# 1. 크롬 드라이버 설정
# ============================================================
def get_driver():
    chrome_options = Options()
    # chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
    driver = webdriver.Chrome(options=chrome_options)
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"}
    )
    return driver


# ============================================================
# 2. 멜론 차트 크롤링 (실시간 TOP100)
# ============================================================
def crawl_melon(driver):
    print("=" * 50)
    print("🎵 멜론 차트 크롤링 시작...")
    print("=" * 50)

    url = "https://www.melon.com/chart/index.htm"
    driver.get(url)
    time.sleep(3)

    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    song_data = []

    for class_name in ["lst50", "lst100"]:
        rows = soup.select(f"table tbody tr.{class_name}")
        for row in rows:
            try:
                rank = row.select_one(".rank").text.strip()
                title = row.select_one(".rank01 span a").text.strip()
                artist = row.select_one(".rank02 span").text.strip()
                album_tag = row.select_one(".rank03 a")
                album = album_tag.text.strip() if album_tag else ""

                like_tag = row.select_one("button.like span.cnt")
                likes = 0
                if like_tag:
                    like_text = like_tag.text.strip().replace(",", "")
                    likes = int(like_text) if like_text.isdigit() else 0

                song_id = ""
                input_tag = row.select_one("input[name='songId']") or row.select_one("input.input_check")
                if input_tag:
                    song_id = input_tag.get("value", "")

                song_data.append({
                    "rank": int(rank),
                    "title": title,
                    "artist": artist,
                    "album": album,
                    "likes": likes,
                    "song_id": song_id,
                    "source": "melon"
                })
            except:
                continue

    if len(song_data) == 0:
        print("  ⚠️  기본 셀렉터 실패, 대체 셀렉터 시도...")
        rows = soup.select("table tbody tr")
        rank_num = 1
        for row in rows:
            try:
                title_tag = row.select_one("div.ellipsis.rank01 span a") or row.select_one(".rank01 a")
                artist_tag = row.select_one("div.ellipsis.rank02 a") or row.select_one(".rank02 a")
                if title_tag and artist_tag:
                    song_data.append({
                        "rank": rank_num,
                        "title": title_tag.text.strip(),
                        "artist": artist_tag.text.strip(),
                        "album": "", "likes": 0, "song_id": "",
                        "source": "melon"
                    })
                    rank_num += 1
            except:
                continue

    print(f"  ✅ 멜론: {len(song_data)}곡 수집 완료")
    return song_data



# ============================================================
# 4. 멜론 월간 TOP100 크롤링
# ============================================================
def _parse_chart_rows(soup):
    """lst50/lst100 행에서 rank, title, artist, song_id 추출"""
    songs = []
    for cls in ["lst50", "lst100"]:
        for row in soup.select(f"table tbody tr.{cls}"):
            try:
                rank = int(row.select_one(".rank").text.strip())
                title = row.select_one(".rank01 span a").text.strip()
                artist = row.select_one(".rank02 span").text.strip()
                song_id = ""
                inp = (row.select_one("input[name='songId']") or
                       row.select_one("input.input_check"))
                if inp:
                    song_id = inp.get("value", "")
                songs.append({"rank": rank, "title": title,
                               "artist": artist, "song_id": song_id})
            except Exception:
                continue
    return songs


def _subtract_months(dt, n):
    total = dt.year * 12 + dt.month - n
    y = (total - 1) // 12
    m = (total - 1) % 12 + 1
    return dt.replace(year=y, month=m, day=1)


def crawl_melon_monthly(driver, year_month):
    """year_month: 'YYYYMM'  예) '202501'"""
    url = f"https://www.melon.com/chart/month/index.htm?rankMonth={year_month}"
    driver.get(url)
    time.sleep(3)
    songs = _parse_chart_rows(BeautifulSoup(driver.page_source, "html.parser"))
    for s in songs:
        s["year_month"] = year_month
    print(f"  [월간 {year_month}] {len(songs)}곡")
    return songs


# ============================================================
# 5. 장르 수집 (곡 상세페이지 + 캐시)
# ============================================================
GENRE_CACHE_FILE = "genre_cache.json"

GENRE_MAP = {
    "발라드": "발라드", "댄스": "댄스", "힙합": "힙합",
    "R&B": "R&B", "소울": "R&B", "팝": "팝",
    "인디음악": "인디", "인디": "인디",
    "록": "록", "메탈": "록",
    "포크": "포크", "블루스": "포크",
    "트로트": "트로트", "OST": "OST",
    "일렉트로닉": "일렉트로닉", "어반팝": "댄스",
}


def _load_genre_cache():
    if os.path.exists(GENRE_CACHE_FILE):
        with open(GENRE_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_genre_cache(cache):
    with open(GENRE_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _normalize_genre(raw):
    if not raw or raw == "기타":
        return "기타"
    for key, val in GENRE_MAP.items():
        if key in raw:
            return val
    return "기타"


def get_song_genre(driver, song_id, cache):
    if not song_id:
        return "기타"
    if song_id in cache:
        return cache[song_id]

    url = f"https://www.melon.com/song/detail.htm?songId={song_id}"
    try:
        driver.get(url)
        time.sleep(1.5)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        genre = "기타"
        # div.section_info 안의 dl.list 에서만 장르 파싱 (더 정확한 셀렉터)
        for dl in soup.select("div.section_info dl.list"):
            for dt, dd in zip(dl.select("dt"), dl.select("dd")):
                if "장르" in dt.text:
                    # 예) "인디음악, 록/메탈" → 첫 번째 장르만 사용
                    raw = dd.text.strip().split(",")[0].split("/")[0].strip()
                    genre = _normalize_genre(raw)
                    break
            if genre != "기타":   # 장르를 찾으면 외부 루프도 종료
                break
    except Exception:
        genre = "기타"

    cache[song_id] = genre
    return genre


# ============================================================
# 6. 벅스 차트 크롤링
# ============================================================
def crawl_bugs(driver):
    print("=" * 50)
    print("🐛 벅스 차트 크롤링 시작...")
    print("=" * 50)

    url = "https://music.bugs.co.kr/chart"
    driver.get(url)
    time.sleep(3)

    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    song_data = []
    rows = soup.select("table.list tbody tr")

    for row in rows:
        try:
            rank_tag = row.select_one("div.ranking strong") or row.select_one(".ranking strong")
            rank = rank_tag.text.strip() if rank_tag else None

            title_tag = row.select_one("p.title a") or row.select_one("th p.title a")
            title = title_tag.text.strip() if title_tag else None

            artist_tag = row.select_one("p.artist a") or row.select_one("td p.artist a")
            artist = artist_tag.text.strip() if artist_tag else None

            album_tag = row.select_one("a.album")
            album = album_tag.text.strip() if album_tag else ""

            if rank and title and artist:
                song_data.append({
                    "rank": int(rank),
                    "title": title,
                    "artist": artist,
                    "album": album,
                    "likes": 0,
                    "song_id": "",
                    "source": "bugs"
                })
        except:
            continue

    print(f"  ✅ 벅스: {len(song_data)}곡 수집 완료")
    return song_data


# ============================================================
# 7. YouTube Data API - TOP10 뮤직비디오 통계
# ============================================================
def get_youtube_stats(song_list, api_key):
    from googleapiclient.discovery import build

    print("=" * 50)
    print(f"📺 YouTube 뮤직비디오 통계 수집 시작 (총 {len(song_list)}곡)...")
    print("=" * 50)

    youtube = build("youtube", "v3", developerKey=api_key)
    yt_data = []

    seen = set()
    unique_songs = []
    for song in song_list:
        key = (song["title"], song["artist"])
        if key not in seen:
            seen.add(key)
            unique_songs.append(song)

    for song in unique_songs:
        query = f"{song['title']} {song['artist']} MV"
        try:
            search = youtube.search().list(
                q=query, part="id,snippet", maxResults=1, type="video"
            ).execute()

            if not search["items"]:
                print(f"  ⏭️  {song['title']} - 검색 결과 없음")
                continue

            vid = search["items"][0]["id"]["videoId"]
            vtitle = search["items"][0]["snippet"]["title"]

            mv_keywords = ["mv", "m/v", "music video", "뮤직비디오", "official video"]
            if not any(kw in vtitle.lower() for kw in mv_keywords):
                print(f"  ⏭️  {song['title']} - MV 없음 (제외): {vtitle[:40]}")
                continue

            stats_res = youtube.videos().list(part="statistics", id=vid).execute()
            if not stats_res["items"]:
                continue

            s = stats_res["items"][0]["statistics"]

            comments = ["", "", "", ""]
            try:
                cres = youtube.commentThreads().list(
                    part="snippet", videoId=vid,
                    maxResults=4, textFormat="plainText"
                ).execute()
                for i, item in enumerate(cres.get("items", [])[:4]):
                    comments[i] = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            except Exception:
                pass

            yt_data.append({
                "rank": song["rank"],
                "title": song["title"],
                "artist": song["artist"],
                "video_title": vtitle,
                "video_id": vid,
                "view_count": int(s.get("viewCount", 0)),
                "like_count": int(s.get("likeCount", 0)),
                "comment_count": int(s.get("commentCount", 0)),
                "comment1": comments[0],
                "comment2": comments[1],
                "comment3": comments[2],
                "comment4": comments[3],
            })
            print(f"  ✅ {song['rank']:>3}위 {song['title'][:20]:<20} 조회수: {int(s.get('viewCount', 0)):>12,}")
            time.sleep(0.3)
        except Exception as e:
            print(f"  ❌ {song['title']} 실패: {e}")
            continue

    print(f"  ✅ YouTube 데이터 {len(yt_data)}곡 수집 완료")
    return yt_data


# ============================================================
# 8. 실시간 전체 크롤링 (멜론 + 벅스 + YouTube)
# ============================================================
def crawl_all(youtube_api_key=None):
    driver = get_driver()
    try:
        melon_data = crawl_melon(driver)
        bugs_data = crawl_bugs(driver)
    finally:
        driver.quit()

    all_chart = melon_data + bugs_data
    df_chart = pd.DataFrame(all_chart)
    df_chart["crawled_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df_weekly = pd.DataFrame()  # 주간 데이터는 load_weekly_from_csv()로 별도 로드

    df_youtube = pd.DataFrame()
    if youtube_api_key:
        yt_data = get_youtube_stats(melon_data, youtube_api_key)
        if yt_data:
            df_youtube = pd.DataFrame(yt_data)
            df_youtube["crawled_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("\n" + "=" * 50)
    print(f"📊 전체 수집 결과:")
    print(f"   - 멜론 실시간: {len(melon_data)}곡")
    print(f"   - 벅스 실시간: {len(bugs_data)}곡")
    print(f"   - 멜론 주간: weekly_rank.csv 에서 별도 로드")
    print(f"   - YouTube: {len(df_youtube)}곡")
    print("=" * 50)

    return df_chart, df_weekly, df_youtube


# ============================================================
# 9. 월간 TOP100 + 장르 크롤링
# ============================================================
def crawl_all_months(months_back=12, start_offset=1):
    """최근 months_back 개월치 월간 TOP100 + 곡 상세페이지 장르 수집"""
    driver = get_driver()
    genre_cache = _load_genre_cache()
    now = datetime.now()

    year_months = []
    for i in range(start_offset, start_offset + months_back):
        d = _subtract_months(now, i)
        year_months.append(d.strftime("%Y%m"))

    print("=" * 50)
    print(f"📅 월간 차트 수집: {year_months[-1]} ~ {year_months[0]}")
    print("=" * 50)

    all_songs = []
    try:
        print("\n[Step 1] 월간 TOP100 크롤링")
        for ym in year_months:
            songs = crawl_melon_monthly(driver, ym)
            all_songs.extend(songs)
            time.sleep(2)

        df_tmp = pd.DataFrame(all_songs)
        unique_ids = [s for s in df_tmp[df_tmp["song_id"] != ""]["song_id"].unique()
                      if s not in genre_cache]
        total = len(unique_ids)
        print(f"\n[Step 2] 장르 수집: 신규 {total}곡 (캐시 {len(genre_cache)}건 재사용)")
        for i, sid in enumerate(unique_ids):
            get_song_genre(driver, sid, genre_cache)
            if (i + 1) % 30 == 0 or (i + 1) == total:
                _save_genre_cache(genre_cache)
                print(f"  {i+1}/{total} ({(i+1)/total*100:.0f}%)")

    finally:
        driver.quit()

    df = pd.DataFrame(all_songs)
    df["genre"] = df["song_id"].map(lambda x: genre_cache.get(x, "기타"))
    df["month"] = df["year_month"].map(lambda x: int(x[4:6]))
    df["year"] = df["year_month"].map(lambda x: int(x[:4]))
    df["season"] = df["month"].map(_month_to_season)
    return df


# ============================================================
# 10. 계절/장르 분석 헬퍼
# ============================================================
SEASON_ORDER = ["봄", "여름", "가을", "겨울"]
SEASON_COLORS = {"봄": "#A8D5A2", "여름": "#87CEEB", "가을": "#E8A87C", "겨울": "#B0C4DE"}


def _month_to_season(month):
    if month in [3, 4, 5]:   return "봄"
    if month in [6, 7, 8]:   return "여름"
    if month in [9, 10, 11]: return "가을"
    return "겨울"


def _set_korean_font():
    for path in [
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/NanumGothic.ttf",
        "/System/Library/Fonts/AppleGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ]:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            name = fm.FontProperties(fname=path).get_name()
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return
    plt.rcParams["axes.unicode_minus"] = False


def _calc_monthly_pct(df, top_n=8):
    top_genres = df["genre"].value_counts().index[:top_n].tolist()
    mg = df.groupby(["year_month", "genre"]).size().reset_index(name="count")
    mg = mg.merge(df.groupby("year_month").size().reset_index(name="total"), on="year_month")
    mg["pct"] = mg["count"] / mg["total"] * 100
    mg["month"] = mg["year_month"].map(lambda x: int(x[4:6]))
    mg["season"] = mg["month"].map(_month_to_season)
    ym_sorted = sorted(mg["year_month"].unique())
    mg["month_idx"] = mg["year_month"].map({ym: i + 1 for i, ym in enumerate(ym_sorted)})
    return mg, top_genres


def _calc_season_pct(df):
    sg = df.groupby(["season", "genre"]).size().reset_index(name="count")
    sg = sg.merge(df.groupby("season").size().reset_index(name="total"), on="season")
    sg["pct"] = sg["count"] / sg["total"] * 100
    return sg


def _calc_season_r(mg, genres):
    records = []
    for genre in genres:
        gdata = mg[mg["genre"] == genre].copy()
        if len(gdata) < 4:
            continue
        for season in SEASON_ORDER:
            ind = (gdata["season"] == season).astype(int).values
            pct = gdata["pct"].values
            if ind.sum() in (0, len(ind)):
                r, p = np.nan, np.nan
            else:
                r, p = stats.pearsonr(ind, pct)
            records.append({"genre": genre, "season": season, "r": r, "p": p})
    return pd.DataFrame(records)


# ============================================================
# 11. 시각화 (그래프 5종)
# ============================================================
def plot_genre_season(df, top_n=8):
    _set_korean_font()
    mg, top_genres = _calc_monthly_pct(df, top_n)
    sg = _calc_season_pct(df)
    r_df = _calc_season_r(mg, top_genres)
    saved = []

    # ── Fig 1: 월별 장르 비율 누적 막대 ──────────────────
    _, ax1 = plt.subplots(figsize=(14, 6))
    pivot1 = mg[mg["genre"].isin(top_genres)].pivot_table(
        index="year_month", columns="genre", values="pct", fill_value=0
    ).reindex(columns=[g for g in top_genres if g in mg["genre"].unique()])
    pivot1.plot(kind="bar", stacked=True, ax=ax1, colormap="tab10", width=0.8)
    ax1.set_title("월별 TOP100 장르 비율 분포", fontsize=15, fontweight="bold", pad=12)
    ax1.set_xlabel("연월", fontsize=11)
    ax1.set_ylabel("비율 (%)", fontsize=11)
    ax1.legend(title="장르", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax1.set_xticklabels(
        [f"{ym[:4]}-{ym[4:]}" for ym in pivot1.index],
        rotation=45, ha="right"
    )
    plt.tight_layout()
    fn = "fig1_monthly_genre_distribution.png"
    plt.savefig(fn, dpi=150, bbox_inches="tight"); saved.append(fn); plt.show()

    # ── Fig 2: 계절별 장르 비율 히트맵 ──────────────────
    _, ax2 = plt.subplots(figsize=(11, 5))
    hmap = sg[sg["genre"].isin(top_genres)].pivot_table(
        index="season", columns="genre", values="pct", fill_value=0
    ).reindex(SEASON_ORDER).reindex(
        columns=[g for g in top_genres if g in sg["genre"].unique()]
    ).fillna(0)
    sns.heatmap(hmap, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=0.5,
                ax=ax2, cbar_kws={"label": "비율 (%)"}, annot_kws={"size": 11})
    ax2.set_title("계절별 장르 평균 비율 히트맵 (%)", fontsize=15, fontweight="bold", pad=12)
    ax2.set_xlabel("장르", fontsize=11); ax2.set_ylabel("계절", fontsize=11)
    plt.tight_layout()
    fn = "fig2_season_genre_heatmap.png"
    plt.savefig(fn, dpi=150, bbox_inches="tight"); saved.append(fn); plt.show()

    # ── Fig 3: 계절-장르 r 히트맵 (값 / 유의수준) ───────
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(16, 6))
    vg = [g for g in top_genres if g in r_df["genre"].unique()]
    r_pivot = r_df[r_df["genre"].isin(vg)].pivot_table(
        index="season", columns="genre", values="r"
    ).reindex(SEASON_ORDER).reindex(columns=vg)
    p_pivot = r_df[r_df["genre"].isin(vg)].pivot_table(
        index="season", columns="genre", values="p"
    ).reindex(SEASON_ORDER).reindex(columns=vg)

    r_annot = r_pivot.copy().astype(object)
    for s in SEASON_ORDER:
        for g in vg:
            rv = r_pivot.loc[s, g] if (s in r_pivot.index and g in r_pivot.columns) else np.nan
            r_annot.loc[s, g] = f"{rv:.2f}" if not pd.isna(rv) else ""
    sns.heatmap(r_pivot.fillna(0), annot=r_annot, fmt="", cmap="coolwarm", center=0,
                vmin=-1, vmax=1, linewidths=0.5, ax=ax3a,
                annot_kws={"size": 11, "fontweight": "bold"})
    ax3a.set_title("계절-장르 상관계수 r", fontsize=13, fontweight="bold")
    ax3a.set_xlabel("장르"); ax3a.set_ylabel("계절")

    sig_annot = r_pivot.copy().astype(object)
    for s in SEASON_ORDER:
        for g in vg:
            rv = r_pivot.loc[s, g] if (s in r_pivot.index and g in r_pivot.columns) else np.nan
            pv = p_pivot.loc[s, g] if (s in p_pivot.index and g in p_pivot.columns) else np.nan
            if pd.isna(rv):
                sig_annot.loc[s, g] = "N/A"
            else:
                stars = ("***" if not pd.isna(pv) and pv < 0.001 else
                         "**"  if not pd.isna(pv) and pv < 0.01  else
                         "*"   if not pd.isna(pv) and pv < 0.05  else "")
                sig_annot.loc[s, g] = f"{rv:.2f}{stars}"
    sns.heatmap(r_pivot, annot=sig_annot, fmt="", cmap="coolwarm", center=0,
                vmin=-1, vmax=1, linewidths=0.5, ax=ax3b, annot_kws={"size": 10})
    ax3b.set_title("계절-장르 r  (유의수준)\n*p<.05  **p<.01  ***p<.001",
                   fontsize=12, fontweight="bold")
    ax3b.set_xlabel("장르"); ax3b.set_ylabel("계절")
    fig3.suptitle("계절과 장르 비율의 상관관계 분석 (Pearson r)",
                  fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    fn = "fig3_season_genre_r_heatmap.png"
    plt.savefig(fn, dpi=150, bbox_inches="tight"); saved.append(fn); plt.show()

    # ── Fig 4: 장르별 월별 추세 + r 값 ──────────────────
    ncols = 2
    nrows = (len(top_genres) + 1) // 2
    fig4, axes4 = plt.subplots(nrows, ncols, figsize=(15, nrows * 4))
    axes4 = np.array(axes4).flatten()
    i = -1
    for i, genre in enumerate(top_genres):
        ax = axes4[i]
        gdata = mg[mg["genre"] == genre].sort_values("year_month")
        if len(gdata) < 2:
            ax.set_visible(False); continue
        x, y = gdata["month_idx"].values, gdata["pct"].values
        r_val, p_val = stats.pearsonr(x, y)
        for _, row in gdata.iterrows():
            ax.scatter(row["month_idx"], row["pct"],
                       color=SEASON_COLORS.get(row["season"], "gray"),
                       s=90, zorder=5, edgecolors="gray", linewidths=0.5)
        ax.plot(x, y, "-", color="gray", linewidth=1.2, alpha=0.4, zorder=3)
        x_line = np.linspace(x.min(), x.max(), 200)
        ax.plot(x_line, np.poly1d(np.polyfit(x, y, 1))(x_line),
                "r--", linewidth=1.8, alpha=0.8, zorder=4)
        p_txt = "< 0.05" if p_val < 0.05 else f"= {p_val:.3f}"
        r_color = "#C0392B" if abs(r_val) >= 0.4 else "#2C3E50"
        ax.text(0.97, 0.95, f"r = {r_val:.3f}\np {p_txt}",
                transform=ax.transAxes, ha="right", va="top", fontsize=10, color=r_color,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFFDE7",
                          alpha=0.9, edgecolor=r_color))
        if i == 0:
            from matplotlib.patches import Patch
            ax.legend(handles=[Patch(facecolor=c, label=s, edgecolor="gray")
                                for s, c in SEASON_COLORS.items()],
                      title="계절", loc="upper left", fontsize=8, title_fontsize=8)
        ym_labels = gdata.sort_values("month_idx")["year_month"].tolist()
        ax.set_xticks(range(1, len(ym_labels) + 1))
        ax.set_xticklabels([f"{ym[2:4]}.{ym[4:]}" for ym in ym_labels], rotation=45, fontsize=7)
        ax.set_title(f"【{genre}】 월별 비율 추세", fontsize=12, fontweight="bold")
        ax.set_ylabel("비율 (%)", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    for j in range(i + 1, len(axes4)):
        axes4[j].set_visible(False)
    fig4.suptitle("장르별 월별 비율 추세 및 상관계수 r  (빨간 점선: 추세선, 점 색상: 계절)",
                  fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fn = "fig4_genre_monthly_trend_r.png"
    plt.savefig(fn, dpi=150, bbox_inches="tight"); saved.append(fn); plt.show()

    # ── Fig 5: 계절별 상위/하위 장르 비교 ───────────────
    _, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(14, 6))
    x_pos = np.arange(len(SEASON_ORDER))
    width = 0.22
    palette = ["#E74C3C", "#E67E22", "#2ECC71"]
    for ax, is_top in [(ax5a, True), (ax5b, False)]:
        for k in range(3):
            vals, labels = [], []
            for s in SEASON_ORDER:
                sub = sg[(sg["season"] == s) & (sg["genre"].isin(top_genres))].copy()
                sub = sub.nlargest(3, "pct") if is_top else sub.nsmallest(3, "pct")
                if k < len(sub):
                    vals.append(sub.iloc[k]["pct"]); labels.append(sub.iloc[k]["genre"])
                else:
                    vals.append(0); labels.append("")
            bars = ax.bar(x_pos + k * width, vals, width,
                          label=f"{'Top' if is_top else 'Bottom'} {k+1}",
                          color=palette[k], alpha=0.85, edgecolor="white")
            for bar, label in zip(bars, labels):
                if label and bar.get_height() > 0.5:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                            label, ha="center", va="bottom", fontsize=8, rotation=15)
        ax.set_xticks(x_pos + width)
        ax.set_xticklabels(SEASON_ORDER, fontsize=12)
        ax.set_title("계절별 상위 3 장르" if is_top else "계절별 하위 3 장르",
                     fontsize=13, fontweight="bold")
        ax.set_ylabel("비율 (%)", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.suptitle("계절별 장르 점유율 상위/하위 비교", fontsize=15, fontweight="bold")
    plt.tight_layout()
    fn = "fig5_season_top_bottom_genres.png"
    plt.savefig(fn, dpi=150, bbox_inches="tight"); saved.append(fn); plt.show()

    # ── 콘솔 요약 ────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  계절별 장르 점유율 요약")
    print("=" * 55)
    for season in SEASON_ORDER:
        sub = sg[sg["season"] == season].nlargest(3, "pct")
        ranks = "  |  ".join([f"{r['genre']} {r['pct']:.1f}%" for _, r in sub.iterrows()])
        print(f"  {season} 상위: {ranks}")
    print("\n  계절-장르 r 값 (|r| >= 0.3)")
    print("-" * 55)
    for _, row in r_df[r_df["r"].abs() >= 0.3].sort_values("r", ascending=False).iterrows():
        direction = "▲ 많음" if row["r"] > 0 else "▼ 적음"
        star = " *" if (not pd.isna(row["p"]) and row["p"] < 0.05) else ""
        print(f"  {row['season']} × {row['genre']:8s}  r={row['r']:+.3f}  {direction}{star}")
    print("=" * 55)
    print("\n✅ 저장 완료:", ", ".join(saved))


# ============================================================
# 12. 주간 CSV 로드 & ML 예측 & 팬덤 차트
# ============================================================
def load_weekly_from_csv(csv_path="weekly_rank.csv"):
    if not os.path.exists(csv_path):
        print(f"  ⚠️ {csv_path} 파일이 없습니다. 빈 DataFrame 반환.")
        return pd.DataFrame()
    try:
        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding="cp949")
        print(f"  ✅ {csv_path} 로드 완료: {len(df)}건 / {df['week_label'].nunique()}주")
        return df
    except Exception as e:
        print(f"  ❌ CSV 로드 실패: {e}")
        return pd.DataFrame()


def analyze_and_predict_weekly(df_weekly, output_path="weekly_prediction.png"):
    _set_korean_font()
    if df_weekly.empty:
        print("  ⚠️ 주간 데이터가 없습니다.")
        return

    # 주차 라벨을 시간순으로 정렬 (week_offset 내림차순 = 오래된 순)
    week_order = (df_weekly.groupby("week_offset")["week_label"]
                  .first()
                  .sort_index(ascending=False))  # offset 클수록 오래된 주
    week_labels = week_order.values
    offset_order = week_order.index.tolist()
    n_actual = len(week_labels)

    # 피벗: 행=week_offset(시간순), 열=곡명, 값=순위
    pivot = df_weekly.pivot_table(
        index="week_offset", columns="title", values="rank", aggfunc="min"
    ).reindex(offset_order)

    # 최소 절반 이상 등장한 곡만 사용
    pivot = pivot.dropna(thresh=max(3, n_actual // 2), axis=1).fillna(101)

    # 변동성 점수 = std(ranks) + |slope|*3 + std(residuals)
    x = np.arange(n_actual, dtype=float)
    volatility = {}
    for song in pivot.columns:
        y = pivot[song].values.astype(float)
        coef = np.polyfit(x, y, 1)
        resid_std = np.std(y - np.polyval(coef, x))
        volatility[song] = np.std(y) + abs(coef[0]) * 3 + resid_std

    top10 = sorted(volatility, key=volatility.get, reverse=True)[:10]

    n_pred = 4
    fig, ax = plt.subplots(figsize=(16, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, song in enumerate(top10):
        y_actual = pivot[song].values.astype(float)
        coef = np.polyfit(x, y_actual, 1)
        x_pred = np.arange(n_actual, n_actual + n_pred, dtype=float)
        y_pred = np.clip(np.polyval(coef, x_pred), 1, 100)
        color = colors[i]

        ax.plot(x, y_actual, "-o", color=color, linewidth=2, markersize=5, label=song)
        ax.plot(x_pred, y_pred, "--", color=color, linewidth=2, alpha=0.75)
        ax.annotate(
            song,
            xy=(x_pred[-1], y_pred[-1]),
            xytext=(x_pred[-1] + 0.15, y_pred[-1]),
            fontsize=7.5, color=color, va="center",
            arrowprops=dict(arrowstyle="-", color=color, lw=0.6, alpha=0.5),
        )

    # 실제/예측 구분선
    divider_x = n_actual - 0.5
    ax.axvline(x=divider_x, color="gray", linestyle="--", linewidth=1.5, alpha=0.5)
    ylim = ax.get_ylim()
    ax.text(divider_x - 0.15, ylim[0] + 2, "← 실제", fontsize=9, color="gray",
            ha="right", va="bottom")
    ax.text(divider_x + 0.15, ylim[0] + 2, "예측 →", fontsize=9, color="gray",
            ha="left", va="bottom")

    # X축 레이블 생성 (실제 8주 + 예측 4주)
    all_labels = list(week_labels)
    try:
        m = re.search(r"(\d{4})\.(\d{2})\.(\d{2})\s*~\s*\d{4}\.(\d{2})\.(\d{2})",
                      str(week_labels[-1]))
        if m:
            last_end = datetime(int(m.group(1)), int(m.group(4)), int(m.group(5)))
            for j in range(1, n_pred + 1):
                s = last_end + timedelta(days=1 + (j - 1) * 7)
                e = s + timedelta(days=6)
                all_labels.append(f"{s.strftime('%m.%d')}~{e.strftime('%m.%d')}")
        else:
            for j in range(1, n_pred + 1):
                all_labels.append(f"예측+{j}주")
    except Exception:
        for j in range(1, n_pred + 1):
            all_labels.append(f"예측+{j}주")

    ax.set_xticks(range(n_actual + n_pred))
    ax.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=8)
    ax.invert_yaxis()
    ax.set_ylim(100.5, 0.5)
    ax.set_title("주간 순위 변동 예측 — 변동폭 상위 10곡\n실선: 실제 8주  |  점선: ML 예측 4주",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("주차", fontsize=11)
    ax.set_ylabel("순위", fontsize=11)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, title="곡명")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ 주간 예측 그래프 저장: {output_path}")
    print(f"  📊 변동폭 상위 10곡: {', '.join(top10)}")


def plot_fandom_chart(df_chart, df_youtube, output_path="fandom_chart.png"):
    _set_korean_font()
    if df_youtube is None or df_youtube.empty:
        print("  ⚠️ YouTube 데이터 없음 — 팬덤 차트 생략")
        return

    OUTLIERS = {"봄날", "Never Ending Story", "never ending story"}

    melon = df_chart[df_chart["source"] == "melon"][["rank", "title", "artist"]].copy()
    yt = df_youtube[["title", "like_count"]].copy()
    df_plot = (melon.merge(yt, on="title", how="inner")
               .query("title not in @OUTLIERS")
               .dropna(subset=["like_count"]))

    if df_plot.empty:
        print("  ⚠️ 병합 후 데이터 없음 — 팬덤 차트 생략")
        return

    fig, ax = plt.subplots(figsize=(14, 9))

    sc = ax.scatter(
        df_plot["like_count"], df_plot["rank"],
        s=100, alpha=0.75,
        c=df_plot["rank"], cmap="RdYlGn_r",
        edgecolors="gray", linewidths=0.5, zorder=5,
    )
    plt.colorbar(sc, ax=ax, label="멜론 순위")

    x_vals = df_plot["like_count"].values.astype(float)
    y_vals = df_plot["rank"].values.astype(float)
    titles = df_plot["title"].values
    cx, cy = x_vals.mean(), y_vals.mean()
    x_range = max(x_vals.max() - x_vals.min(), 1)
    y_range = max(y_vals.max() - y_vals.min(), 1)

    for xi, yi, title in zip(x_vals, y_vals, titles):
        dx, dy = xi - cx, yi - cy
        norm = max(np.sqrt(dx ** 2 + dy ** 2), 1e-9)
        text_x = xi + dx / norm * x_range * 0.12
        text_y = yi + dy / norm * y_range * 0.12
        ax.annotate(
            title,
            xy=(xi, yi),
            xytext=(text_x, text_y),
            fontsize=7.5,
            ha="center", va="center",
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.8, alpha=0.55),
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.8),
            zorder=6,
        )

    ax.invert_yaxis()
    ax.set_xlabel("YouTube 좋아요 수 (팬덤 화력)", fontsize=12)
    ax.set_ylabel("멜론 순위 (낮을수록 상위권)", fontsize=12)
    ax.set_title("팬덤 화력 vs 대중성\n(봄날 · Never Ending Story 제외)",
                 fontsize=14, fontweight="bold")
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v/1_000_000:.1f}M" if v >= 1_000_000
                          else f"{int(v/1_000)}K")
    )
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ 팬덤 차트 저장: {output_path}")


# ============================================================
# 13. 파일 저장
# ============================================================
def save_to_files(df_chart, df_weekly, df_youtube):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_chart.to_csv(f"music_chart_{ts}.csv", index=False, encoding="utf-8-sig")
    print(f"📁 CSV 저장 완료")

    with pd.ExcelWriter(f"music_chart_{ts}.xlsx", engine="openpyxl") as w:
        df_chart[df_chart["source"] == "melon"].to_excel(w, sheet_name="멜론", index=False)
        df_chart[df_chart["source"] == "bugs"].to_excel(w, sheet_name="벅스", index=False)
        if not df_weekly.empty:
            df_weekly.to_excel(w, sheet_name="주간순위", index=False)
        if not df_youtube.empty:
            df_youtube.to_excel(w, sheet_name="YouTube", index=False)
    print(f"📁 엑셀 저장 완료")


# ============================================================
# 13. MariaDB 저장
# ============================================================
def save_to_mariadb(df_chart, df_weekly, df_youtube, db_config):
    from sqlalchemy import create_engine, text

    db_name = "로컬" if db_config["host"] == "localhost" else "Aiven"
    print(f"\n🗄️  {db_name} DB 저장 시작...")

    connect_args = {}
    if db_config.get("ssl"):
        connect_args = {"ssl": {"ssl_mode": "REQUIRED"}}

    engine = create_engine(
        f"mysql+pymysql://{db_config['user']}:{db_config['password']}"
        f"@{db_config['host']}:{db_config['port']}/{db_config['database']}",
        connect_args=connect_args
    )

    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS chart_data (
                id INT AUTO_INCREMENT PRIMARY KEY,
                `rank` INT NOT NULL, title VARCHAR(500) NOT NULL,
                artist VARCHAR(300) NOT NULL, album VARCHAR(500) DEFAULT '',
                likes INT DEFAULT 0, song_id VARCHAR(50) DEFAULT '',
                source VARCHAR(20) NOT NULL, crawled_at DATETIME NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_source (source), INDEX idx_rank (source, `rank`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS weekly_rank (
                id INT AUTO_INCREMENT PRIMARY KEY,
                week_offset INT NOT NULL, week_label VARCHAR(100),
                `rank` INT NOT NULL, title VARCHAR(500) NOT NULL,
                artist VARCHAR(300) NOT NULL, likes INT DEFAULT 0,
                crawled_at DATETIME NOT NULL,
                INDEX idx_title (title(100)), INDEX idx_week (week_offset)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS youtube_stats (
                id INT AUTO_INCREMENT PRIMARY KEY,
                `rank` INT NOT NULL, title VARCHAR(500) NOT NULL,
                artist VARCHAR(300) NOT NULL, video_title VARCHAR(500),
                video_id VARCHAR(50), view_count BIGINT DEFAULT 0,
                like_count BIGINT DEFAULT 0, comment_count BIGINT DEFAULT 0,
                comment1 TEXT, comment2 TEXT, comment3 TEXT, comment4 TEXT,
                crawled_at DATETIME NOT NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS quiz_scores (
                id INT AUTO_INCREMENT PRIMARY KEY,
                nickname VARCHAR(100) NOT NULL,
                score INT DEFAULT 0,
                updated_at DATETIME NOT NULL,
                UNIQUE KEY uk_nickname (nickname),
                INDEX idx_score (score)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """))
        db_schema = db_config["database"]
        migrations = [
            ("chart_data",    "likes",    "INT DEFAULT 0"),
            ("chart_data",    "song_id",  "VARCHAR(50) DEFAULT ''"),
            ("weekly_rank",   "likes",    "INT DEFAULT 0"),
            ("youtube_stats", "comment1", "TEXT"),
            ("youtube_stats", "comment2", "TEXT"),
            ("youtube_stats", "comment3", "TEXT"),
            ("youtube_stats", "comment4", "TEXT"),
        ]
        for table, col, col_def in migrations:
            row = conn.execute(text(
                "SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS "
                f"WHERE TABLE_SCHEMA='{db_schema}' AND TABLE_NAME='{table}' AND COLUMN_NAME='{col}'"
            )).scalar()
            if row == 0:
                conn.execute(text(f"ALTER TABLE `{table}` ADD COLUMN `{col}` {col_def}"))
        conn.commit()

    df_chart.to_sql("chart_data", con=engine, if_exists="append", index=False, method="multi", chunksize=100)
    print(f"  ✅ chart_data: {len(df_chart)}건")
    if not df_weekly.empty:
        # 주간 데이터는 매 실행마다 최신 데이터로 교체 (누적 방지)
        with engine.connect() as conn:
            conn.execute(text("TRUNCATE TABLE weekly_rank"))
            conn.commit()
        df_weekly.to_sql("weekly_rank", con=engine, if_exists="append", index=False, method="multi", chunksize=100)
        print(f"  ✅ weekly_rank: {len(df_weekly)}건 (교체 저장)")
    if not df_youtube.empty:
        df_youtube.to_sql("youtube_stats", con=engine, if_exists="append", index=False, method="multi", chunksize=100)
        print(f"  ✅ youtube_stats: {len(df_youtube)}건")

    engine.dispose()


# ============================================================
# 14. 메인 실행
# ============================================================
if __name__ == "__main__":
    YOUTUBE_API_KEY = "AIzaSyARmDkqalzAT5wGW2ffg6WVPxxxuntscGo"

    # ---- 실시간 크롤링 (멜론 + 벅스 + YouTube) ----
    df_chart, _, df_youtube = crawl_all(youtube_api_key=YOUTUBE_API_KEY)

    # ---- 주간 순위: weekly_rank.csv 에서 로드 ----
    df_weekly = load_weekly_from_csv("weekly_rank.csv")

    save_to_files(df_chart, df_weekly, df_youtube)

    local_config = {
        "host": "localhost", "port": 3306,
        "user": "root", "password": "1234",
        "database": "music_chart", "ssl": False
    }
    save_to_mariadb(df_chart, df_weekly, df_youtube, local_config)

    aiven_config = {
        "host": "mysql-22039057-musicproject1.c.aivencloud.com",
        "port": 25918, "user": "avnadmin",
        "password": "your_aiven_password_here",
        "database": "music_chart", "ssl": True
    }
    save_to_mariadb(df_chart, df_weekly, df_youtube, aiven_config)

    print("\n📋 미리보기:")
    print(df_chart.head(10).to_string(index=False))
    print("\n🏆 사이트별 1위:")
    for src in ["melon", "bugs"]:
        top = df_chart[(df_chart["source"] == src) & (df_chart["rank"] == 1)]
        if not top.empty:
            r = top.iloc[0]
            print(f"  {src:>6}: {r['title']} - {r['artist']}")

    # ---- 주간 ML 예측 차트 ----
    if not df_weekly.empty:
        print("\n📈 주간 순위 ML 예측 시작...")
        analyze_and_predict_weekly(df_weekly, output_path="weekly_prediction.png")

    # ---- 팬덤 화력 vs 대중성 차트 ----
    if not df_youtube.empty:
        print("\n🔥 팬덤 vs 대중성 차트 생성...")
        plot_fandom_chart(df_chart, df_youtube, output_path="fandom_chart.png")

    # ---- 월간 장르 분석 ----
    DATA_FILE = "monthly_genre_data.csv"
    if os.path.exists(DATA_FILE):
        print(f"\n📂 기존 월간 데이터 로드: {DATA_FILE}")
        df_monthly = pd.read_csv(DATA_FILE, dtype={"song_id": str, "year_month": str})
        df_monthly["month"] = df_monthly["year_month"].map(lambda x: int(str(x)[4:6]))
        df_monthly["season"] = df_monthly["month"].map(_month_to_season)
    else:
        print("\n🕷️  월간 차트 크롤링 시작 (최근 12개월)...")
        df_monthly = crawl_all_months(months_back=24, start_offset=1)
        df_monthly.to_csv(DATA_FILE, index=False, encoding="utf-8-sig")
        print(f"💾 월간 데이터 저장: {DATA_FILE}")

    print(f"\n📊 월간 수집: {len(df_monthly)}건 / {df_monthly['year_month'].nunique()}개월")
    plot_genre_season(df_monthly, top_n=8)
