# src 폴더 안내

현재 `app.py`는 수행평가 제출과 실행 편의성을 위해 단일 파일 구조를 유지합니다.
향후 GitHub에서 프로젝트를 더 발전시킬 때 아래와 같이 모듈 분리를 할 수 있습니다.

- `models.py`: Paper 데이터 모델
- `connectors.py`: PubMed, arXiv, Semantic Scholar, Google Scholar 검색 연결
- `summary.py`: TF-IDF 기반 요약
- `translation.py`: 한국어/영어 번역 및 전문용어 보정
- `analysis.py`: 트렌드 분석, 추천, 키워드 네트워크
- `storage.py`: 사용자/북마크 JSON 저장
