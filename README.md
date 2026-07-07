# ResearchNavigator

**ResearchNavigator**는 화학공학·생명과학 분야의 연구 논문을 검색하고 분석하기 위한 Streamlit 기반 학술 정보 탐색 플랫폼입니다.

이 프로젝트는 고등학생 수행평가 및 포트폴리오용으로 제작되었으며, 단순 논문 검색을 넘어 연구 동향 분석, 유사 논문 추천, 키워드 네트워크, PDF 요약 기능까지 포함합니다.

## 주요 기능

- PubMed 논문 검색
- arXiv 논문 검색
- Semantic Scholar 논문 검색
- Google Scholar 검색 지원
- 한국어 검색어 자동 영어 번역
- TF-IDF 기반 자동 요약
- 한국어 의역 요약
- 바이오/제약 전문용어 자동 보정
- 논문 북마크 저장
- 폴더별 논문 관리
- CSV / Excel 내보내기
- DOI, PMID, arXiv ID, PDF 링크 확인
- 학교 Wi-Fi 및 보안망 대응 네트워크 진단
- 검색 결과 기반 **AI 유사 논문 추천**
- 연도별 **논문 트렌드 분석**
- 검색 결과 기반 **키워드 네트워크 시각화**
- PDF 논문 업로드 후 텍스트 추출 및 요약

## 화학공학 연계성

화학공학 분야에서는 촉매, 배터리, 수소 생산, 탄소 포집, 반도체 공정, 고분자 소재 등 빠르게 변화하는 연구 주제를 지속적으로 파악하는 능력이 중요합니다. 이 프로그램은 여러 학술 데이터베이스의 논문을 통합 검색하고, 연도별 연구량 변화와 핵심 키워드 관계를 시각화하여 연구 동향을 파악할 수 있도록 설계되었습니다.

## 설치 방법

```bash
pip install -r requirements.txt
```

## 실행 방법

```bash
streamlit run app.py
```

## 폴더 구조

```text
ResearchNavigator/
├─ app.py
├─ README.md
├─ requirements.txt
├─ LICENSE
├─ .gitignore
├─ gitignore_visible_copy.txt
├─ GITHUB_UPLOAD_GUIDE.md
├─ PROJECT_STRUCTURE.txt
├─ data/
│  ├─ users.json
│  ├─ bookmarks.json
│  └─ README_DATA_FILES.txt
├─ logs/
│  └─ KEEP_LOGS_FOLDER.txt
├─ assets/
│  └─ KEEP_ASSETS_FOLDER.txt
└─ src/
   ├─ __init__.py
   └─ README_MODULES.md
```

## 주의사항

- `users.json`, `bookmarks.json`은 실행 중 개인 데이터가 저장될 수 있으므로 공개 저장소에 올릴 때는 빈 `{}` 상태를 권장합니다.
- Google Scholar는 공식 API가 아니므로 학교망 또는 보안망에서 차단될 수 있습니다.
- 일부 학술 API는 네트워크 환경에 따라 응답이 느리거나 제한될 수 있습니다.

## License

MIT License
