# 🎵 멜론·벅스 음원차트 분석 + 4주 예측 + 퀴즈 게임

## 📁 파일 구조
```
music_chart_project/
├── music_chart_crawler.py    # 크롤링 (멜론+벅스+주간8주+YouTube)
├── streamlit_app.py          # 대시보드 + 예측 + 산점도 + 퀴즈
├── schema.sql                # DB 테이블 생성
├── secrets_example.toml      # Streamlit DB 접속 설정 예시
└── README.md
```

## 🚀 실행 순서

### 1. 패키지 설치
```bash
pip install selenium beautifulsoup4 pandas pymysql sqlalchemy openpyxl streamlit plotly google-api-python-client
```

### 2. DB 테이블 생성 (로컬 MariaDB)
```bash
mysql.exe -u root -p1234 < schema.sql
```

### 3. 크롤링 실행
```bash
python music_chart_crawler.py
```

### 4. Streamlit 대시보드
```bash
mkdir .streamlit
copy secrets_example.toml .streamlit\secrets.toml
streamlit run streamlit_app.py
```

## 🎯 주요 기능

| 기능 | 설명 |
|------|------|
| 📊 차트 대시보드 | 멜론/벅스 TOP10, 공통곡 비교, 아티스트 빈도 |
| 📈 4주 순위 예측 | 8주 주간 데이터 기반 추세/모멘텀/좋아요 알고리즘 |
| 🔥 팬덤 vs 대중성 | YouTube 좋아요 vs 순위 산점도 |
| 🎮 가사 퀴즈 | 가사+가수 보고 곡 제목 맞추기 |

## 📈 예측 알고리즘 설명
1. **추세(Trend)**: 최근 3~4주 순위 변동의 가중 이동평균
2. **모멘텀(Momentum)**: 최근 vs 과거 평균 순위 차이
3. **좋아요 보정**: 좋아요 많은 곡 = 순위 유지력 강함
4. **패턴 규칙**: 급상승곡은 추가 상승, 정체곡은 점진 하락

## 👥 역할 분담
- A: 크롤러 (멜론+벅스+주간차트)
- B: DB 설계 + Aiven 연동 + YouTube API
- C: Streamlit 대시보드 + 예측 시각화 + 퀴즈 게임
