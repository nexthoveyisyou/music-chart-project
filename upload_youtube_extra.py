"""
upload_youtube_extra.py
youtube_music_stats_final.csv (11~100위) → Aiven youtube_stats 테이블 업로드
실행: python upload_youtube_extra.py
"""
import os
import re
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime

AIVEN_HOST     = "mysql-22039057-musicproject1.c.aivencloud.com"
AIVEN_PORT     = 25918
AIVEN_USER     = "avnadmin"
AIVEN_PASSWORD = os.environ.get("AIVEN_PASSWORD", "")
AIVEN_DB       = "music_chart"
CSV_PATH       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "youtube_music_stats_final.csv")

def extract_video_id(url):
    m = re.search(r"[?&]v=([A-Za-z0-9_-]{11})", str(url))
    return m.group(1) if m else ""

def main():
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    df["video_id"]      = df["Video URL"].apply(extract_video_id)
    df["video_title"]   = df["Title"]
    df["view_count"]    = df["Views"].fillna(0).astype(int)
    df["like_count"]    = df["Likes"].fillna(0).astype(int)
    df["comment_count"] = df["Comments"].fillna(0).astype(int)
    df["rank"]          = df["Rank"].astype(int)
    df["title"]         = df["Title"]
    df["artist"]        = df["Artist"]
    df["comment1"]      = ""
    df["comment2"]      = ""
    df["comment3"]      = ""
    df["comment4"]      = ""
    df["crawled_at"]    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    upload = df[["rank", "title", "artist", "video_title", "video_id",
                 "view_count", "like_count", "comment_count",
                 "comment1", "comment2", "comment3", "comment4", "crawled_at"]]

    engine = create_engine(
        f"mysql+pymysql://{AIVEN_USER}:{AIVEN_PASSWORD}@{AIVEN_HOST}:{AIVEN_PORT}/{AIVEN_DB}",
        connect_args={"ssl": {"ssl_mode": "REQUIRED"}}
    )

    with engine.connect() as conn:
        min_rank = int(upload["rank"].min())
        max_rank = int(upload["rank"].max())
        conn.execute(text(f"DELETE FROM youtube_stats WHERE `rank` BETWEEN {min_rank} AND {max_rank}"))
        conn.commit()
        print(f"  기존 {min_rank}~{max_rank}위 데이터 삭제")

    upload.to_sql("youtube_stats", con=engine, if_exists="append",
                  index=False, method="multi", chunksize=50)
    print(f"  OK {len(upload)}곡 업로드 완료 ({min_rank}~{max_rank}위)")
    engine.dispose()

if __name__ == "__main__":
    main()
