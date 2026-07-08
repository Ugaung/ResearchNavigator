import re
import os
import json
import time
import hashlib
import uuid
import requests
import feedparser
import streamlit as st
import xml.etree.ElementTree as ET
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import streamlit.components.v1 as components

from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Tuple
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from io import BytesIO
from deep_translator import GoogleTranslator
import plotly.express as px

# ResearchNavigator v2: chemical engineering analysis modules
try:
    from src.chemeng_dashboard import render_chemeng_dashboard
except Exception as _chemeng_import_error:
    render_chemeng_dashboard = None
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Google Scholar용
try:
    from scholarly import scholarly
    SCHOLARLY_AVAILABLE = True
except Exception:
    SCHOLARLY_AVAILABLE = False

# =========================================================
# 기본 설정
# =========================================================
st.set_page_config(
    page_title="바이오 논문 검색/요약 시스템",
    layout="wide"
)

USERS_FILE = "users.json"
BOOKMARKS_FILE = "bookmarks.json"


# =========================================================
# 학교 Wi-Fi / 보안망 대응 네트워크 유틸
# =========================================================
APP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json;q=0.8,*/*;q=0.7",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Connection": "close",
}

NETWORK_LOG_FILE = "network_errors.log"


def build_retry_session() -> requests.Session:
    """학교/기관 Wi-Fi에서 순간 차단·지연이 생겨도 몇 번 재시도하는 세션."""
    session = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(APP_HEADERS)
    return session


HTTP_SESSION = build_retry_session()


def record_network_error(source: str, error: Exception | str):
    """오류를 숨기지 않고 Streamlit 세션과 파일에 기록한다."""
    msg = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {source}: {error}"
    try:
        with open(NETWORK_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass
    try:
        if "network_errors" not in st.session_state:
            st.session_state.network_errors = []
        st.session_state.network_errors.append(msg)
        st.session_state.network_errors = st.session_state.network_errors[-50:]
    except Exception:
        pass
    print(msg)


def http_get(url, **kwargs):
    """requests.get 대체 함수: User-Agent, timeout, retry, SSL 오류 기록을 통합 적용."""
    timeout = kwargs.pop("timeout", 60)
    headers = kwargs.pop("headers", None) or {}
    merged_headers = dict(APP_HEADERS)
    merged_headers.update(headers)
    try:
        return HTTP_SESSION.get(url, timeout=timeout, headers=merged_headers, **kwargs)
    except Exception as e:
        record_network_error(str(url), e)
        raise


def run_network_diagnostics() -> Dict[str, Tuple[bool, str]]:
    """학교 Wi-Fi에서 어떤 학술 API가 막히는지 확인한다."""
    targets = {
        "PubMed": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=cancer&retmode=json&retmax=1",
        "arXiv": "http://export.arxiv.org/api/query?search_query=all:cancer&start=0&max_results=1",
        "Semantic Scholar": "https://api.semanticscholar.org/graph/v1/paper/search?query=cancer&limit=1&fields=title",
        "Google Translate": "https://translate.google.com",
        "Google Scholar": "https://scholar.google.com",
    }
    results = {}
    for name, url in targets.items():
        try:
            r = http_get(url, timeout=12)
            ok = 200 <= r.status_code < 400
            results[name] = (ok, f"HTTP {r.status_code}")
        except Exception as e:
            results[name] = (False, str(e)[:180])
    return results

# =========================================================
# JSON 파일 유틸
# =========================================================
def ensure_json_file(path, default_data):
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default_data, f, ensure_ascii=False, indent=2)

def load_json(path, default_data):
    ensure_json_file(path, default_data)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return default_data

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

ensure_json_file(USERS_FILE, {})
ensure_json_file(BOOKMARKS_FILE, {})

# =========================================================
# 세션 초기화
# =========================================================
def init_session():
    defaults = {
        "logged_in": False,
        "current_user": None,
        "papers": [],
        "last_query": "",
        "translated_query": "",
        "generated_summaries": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# =========================================================
# 비밀번호 해시
# =========================================================
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

# =========================================================
# 데이터 모델
# =========================================================
@dataclass
class Paper:
    source: str
    source_id: Optional[str] = None
    title: str = ""
    authors: List[str] = field(default_factory=list)
    abstract: str = ""
    summary_en: str = ""
    summary_ko: str = ""
    published_date: Optional[str] = None
    year: Optional[int] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    arxiv_id: Optional[str] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    image_urls: List[str] = field(default_factory=list)
    citation_count: int = 0
    relevance_score: float = 0.0

    disease_tags: List[str] = field(default_factory=list)
    tech_tags: List[str] = field(default_factory=list)
    folder_tags: List[str] = field(default_factory=list)
    notes: str = ""
    reading_status: str = "읽기 전"
    importance: int = 3

    def unique_key(self):
        if self.doi:
            return f"doi::{self.doi.lower()}"
        if self.pmid:
            return f"pmid::{self.pmid}"
        if self.arxiv_id:
            return f"arxiv::{self.arxiv_id}"
        if self.source_id:
            return f"{self.source.lower()}::{self.source_id}"
        return f"title::{normalize_title(self.title)}"

# =========================================================
# 텍스트 유틸
# =========================================================
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def normalize_title(title: str) -> str:
    if not title:
        return ""
    title = title.lower().strip()
    title = re.sub(r"[^a-z0-9가-힣\s]", "", title)
    title = re.sub(r"\s+", " ", title)
    return title

def contains_korean(text: str) -> bool:
    return bool(re.search(r"[가-힣]", text or ""))

def extract_year_from_text(text: str):
    if not text:
        return None
    m = re.search(r"(19|20)\d{2}", str(text))
    if m:
        return int(m.group())
    return None

# =========================================================
# 바이오/제약 전문용어 사전
# =========================================================
BIO_TERM_MAP = {
    "single-cell rna sequencing": "단일세포 RNA 시퀀싱",
    "single cell rna sequencing": "단일세포 RNA 시퀀싱",
    "scrna-seq": "단일세포 RNA 시퀀싱",
    "rna sequencing": "RNA 시퀀싱",
    "rna-seq": "RNA 시퀀싱",
    "bulk rna sequencing": "벌크 RNA 시퀀싱",
    "transcriptome": "전사체",
    "transcriptomics": "전사체 분석",
    "proteomics": "단백질체 분석",
    "metabolomics": "대사체 분석",
    "genomics": "유전체 분석",
    "epigenomics": "후성유전체 분석",
    "multiomics": "멀티오믹스 분석",
    "multi-omics": "멀티오믹스 분석",
    "spatial transcriptomics": "공간 전사체 분석",
    "whole exome sequencing": "전장 엑솜 시퀀싱",
    "whole genome sequencing": "전장 유전체 시퀀싱",
    "variant calling": "변이 검출",
    "copy number variation": "복제수 변이",
    "cnv": "복제수 변이",
    "gene expression": "유전자 발현",
    "differential expression": "차등 발현",
    "pathway analysis": "경로 분석",
    "gene set enrichment analysis": "유전자 집합 풍부도 분석",
    "gsea": "유전자 집합 풍부도 분석",
    "machine learning": "기계학습",
    "deep learning": "딥러닝",
    "foundation model": "파운데이션 모델",
    "large language model": "대규모 언어모델",

    "tumor microenvironment": "종양 미세환경",
    "tumour microenvironment": "종양 미세환경",
    "immune microenvironment": "면역 미세환경",
    "immune checkpoint inhibitor": "면역관문 억제제",
    "immune checkpoint inhibitors": "면역관문 억제제",
    "checkpoint inhibitor": "면역관문 억제제",
    "checkpoint inhibitors": "면역관문 억제제",
    "pd-1 inhibitor": "PD-1 억제제",
    "pd-l1 inhibitor": "PD-L1 억제제",
    "ctla-4 inhibitor": "CTLA-4 억제제",
    "car-t": "CAR-T 세포치료제",
    "car t": "CAR-T 세포치료제",
    "t cell receptor": "T세포 수용체",
    "cytokine release syndrome": "사이토카인 방출 증후군",
    "tumor infiltrating lymphocyte": "종양침윤림프구",
    "tumor-infiltrating lymphocyte": "종양침윤림프구",
    "til": "종양침윤림프구",
    "neoantigen": "신생항원",
    "biomarker": "바이오마커",
    "predictive biomarker": "예측 바이오마커",
    "prognostic biomarker": "예후 바이오마커",

    "antibody-drug conjugate": "항체-약물 접합체",
    "adc": "항체-약물 접합체",
    "bispecific antibody": "이중특이항체",
    "monoclonal antibody": "단일클론 항체",
    "small molecule": "저분자 화합물",
    "small-molecule": "저분자 화합물",
    "cell therapy": "세포치료",
    "gene therapy": "유전자치료",
    "mrna vaccine": "mRNA 백신",
    "messenger rna vaccine": "mRNA 백신",
    "lipid nanoparticle": "지질나노입자",
    "pharmacokinetics": "약동학",
    "pk": "약동학",
    "pharmacodynamics": "약력학",
    "pd": "약력학",
    "drug resistance": "약물 내성",
    "combination therapy": "병용요법",
    "monotherapy": "단독요법",
    "dose escalation": "용량 증량",
    "dose expansion": "확장 코호트 평가",

    "alzheimer's disease": "알츠하이머병",
    "parkinson's disease": "파킨슨병",
    "non-small cell lung cancer": "비소세포폐암",
    "small cell lung cancer": "소세포폐암",
    "triple-negative breast cancer": "삼중음성유방암",
    "hepatocellular carcinoma": "간세포암",
    "colorectal cancer": "대장암",
    "glioblastoma": "교모세포종",
    "organoid": "오가노이드",
    "patient-derived organoid": "환자유래 오가노이드",
    "patient-derived xenograft": "환자유래 이종이식 모델",
    "xenograft": "이종이식 모델",
    "knockout": "유전자 결손",
    "knockdown": "유전자 발현 억제",
    "wild type": "야생형",
    "stem cell": "줄기세포",
    "induced pluripotent stem cell": "유도만능줄기세포",
    "ipsc": "유도만능줄기세포",

    "overall survival": "전체 생존기간",
    "progression-free survival": "무진행 생존기간",
    "objective response rate": "객관적 반응률",
    "duration of response": "반응 지속기간",
    "disease control rate": "질병 조절률",
    "hazard ratio": "위험비",
    "confidence interval": "신뢰구간",
    "adverse event": "이상반응",
    "serious adverse event": "중대한 이상반응",
    "treatment-emergent adverse event": "치료 중 발생 이상반응",
    "randomized controlled trial": "무작위 대조시험",
    "double-blind": "이중눈가림",
    "open-label": "공개표지",
    "phase i": "1상",
    "phase ii": "2상",
    "phase iii": "3상",
    "cohort": "코호트",
    "real-world data": "실사용 데이터",
    "real-world evidence": "실사용 근거",
}

BIO_TERM_ITEMS = sorted(BIO_TERM_MAP.items(), key=lambda x: len(x[0]), reverse=True)

# =========================================================
# 검색어 한국어 -> 영어 번역
# =========================================================
def translate_query_if_needed(query: str) -> str:
    if not query or not query.strip():
        return query
    query = query.strip()
    if not contains_korean(query):
        return query
    try:
        translated = GoogleTranslator(source="auto", target="en").translate(query)
        if translated and translated.strip():
            return translated.strip()
        return query
    except:
        return query

# =========================================================
# 번역 전 영어 문장 내 전문용어 치환
# =========================================================
def replace_bio_terms_in_english_text(text: str) -> str:
    if not text:
        return text
    result = text
    for eng, kor in BIO_TERM_ITEMS:
        pattern = re.compile(rf"(?i)\b{re.escape(eng)}\b")
        result = pattern.sub(kor, result)
    return result

# =========================================================
# 한국어 후처리
# =========================================================
def polish_korean_text(text: str) -> str:
    if not text:
        return text

    replacements = {
        "종양 미세 환경": "종양 미세환경",
        "면역 체크포인트": "면역관문",
        "체크포인트 억제제": "면역관문 억제제",
        "세포 치료": "세포치료",
        "유전자 치료": "유전자치료",
        "저 분자": "저분자",
        "단일 세포": "단일세포",
        "전 사체": "전사체",
        "유전 자": "유전자",
        "생존 전체": "전체 생존",
        "진행 없는 생존": "무진행 생존",
        "부작용 사건": "이상반응",
        "신뢰 간격": "신뢰구간",
        "위험 비율": "위험비",
        "약동력학": "약력학",
        "약동학적": "약동학",
        "임상 시험": "임상시험",
        "무작위화된 대조 시험": "무작위 대조시험",
        "이중 맹검": "이중눈가림",
        "개방 라벨": "공개표지",
        "실세계 데이터": "실사용 데이터",
        "실세계 근거": "실사용 근거",
        "오가노이드들": "오가노이드",
        "이 연구는 보여준다": "이 연구는 시사한다",
        "이 연구는 입증한다": "이 연구는 보여준다",
        "유의하게 향상되었다": "유의하게 개선되었다",
        "효율": "효과",
    }

    result = text
    for a, b in replacements.items():
        result = result.replace(a, b)

    result = re.sub(r"\s+", " ", result)
    result = result.replace(" .", ".")
    result = result.replace(" ,", ",")
    result = result.replace(" :", ":")
    result = result.replace(" ;", ";")
    return result.strip()

# =========================================================
# 요약문 영어 -> 한국어 번역
# =========================================================
def translate_summary_to_korean(summary_text: str) -> str:
    if not summary_text or not summary_text.strip():
        return "요약문이 없어 번역할 수 없습니다."

    lines = [x.strip() for x in summary_text.split("\n") if x.strip()]
    if not lines:
        return "요약문이 없어 번역할 수 없습니다."

    translated_lines = []

    for line in lines:
        try:
            preprocessed = replace_bio_terms_in_english_text(line)
            ko = GoogleTranslator(source="auto", target="ko").translate(preprocessed)
            if ko and ko.strip():
                translated_lines.append(polish_korean_text(ko.strip()))
            else:
                translated_lines.append(polish_korean_text(preprocessed))
        except:
            translated_lines.append(polish_korean_text(line))

    return "\n".join(translated_lines)

# =========================================================
# 회원 관리
# =========================================================
class UserManager:
    def __init__(self, users_file=USERS_FILE):
        self.users_file = users_file

    def load_users(self):
        return load_json(self.users_file, {})

    def save_users(self, users):
        save_json(self.users_file, users)

    def register(self, username, password, name, email, major=""):
        users = self.load_users()

        username = username.strip()
        password = password.strip()
        name = name.strip()
        email = email.strip()
        major = major.strip()

        if not username or not password or not name or not email:
            return False, "아이디, 비밀번호, 이름, 이메일은 필수입니다."

        if username in users:
            return False, "이미 존재하는 아이디입니다."

        users[username] = {
            "username": username,
            "password_hash": hash_password(password),
            "name": name,
            "email": email,
            "major": major,
        }
        self.save_users(users)
        return True, "회원가입이 완료되었습니다."

    def login(self, username, password):
        users = self.load_users()
        if username not in users:
            return False, "존재하지 않는 아이디입니다."

        if users[username]["password_hash"] != hash_password(password):
            return False, "비밀번호가 올바르지 않습니다."

        return True, users[username]

    def get_user(self, username):
        users = self.load_users()
        return users.get(username)

    def update_profile(self, username, name, email, major):
        users = self.load_users()
        if username not in users:
            return False, "회원 정보가 존재하지 않습니다."

        users[username]["name"] = name.strip()
        users[username]["email"] = email.strip()
        users[username]["major"] = major.strip()
        self.save_users(users)
        return True, "회원 정보가 수정되었습니다."

    def change_password(self, username, old_password, new_password):
        users = self.load_users()
        if username not in users:
            return False, "회원 정보가 존재하지 않습니다."

        if users[username]["password_hash"] != hash_password(old_password):
            return False, "현재 비밀번호가 올바르지 않습니다."

        if not new_password.strip():
            return False, "새 비밀번호를 입력해주세요."

        users[username]["password_hash"] = hash_password(new_password)
        self.save_users(users)
        return True, "비밀번호가 변경되었습니다."

# =========================================================
# 자동 태그 규칙
# =========================================================
DISEASE_TAG_RULES = {
    "유방암": [
        "breast cancer", "breast carcinoma", "mammary carcinoma", "tnbc",
        "triple-negative breast cancer", "삼중음성유방암", "유방암"
    ],
    "폐암": [
        "lung cancer", "nsclc", "non-small cell lung cancer",
        "small cell lung cancer", "sclc", "폐암", "비소세포폐암", "소세포폐암"
    ],
    "대장암": [
        "colorectal cancer", "colon cancer", "rectal cancer", "crc", "대장암", "결장암", "직장암"
    ],
    "간암": [
        "hepatocellular carcinoma", "hcc", "liver cancer", "간암", "간세포암"
    ],
    "췌장암": [
        "pancreatic cancer", "pancreatic adenocarcinoma", "pdac", "췌장암"
    ],
    "난소암": [
        "ovarian cancer", "ovarian carcinoma", "난소암"
    ],
    "전립선암": [
        "prostate cancer", "전립선암"
    ],
    "백혈병": [
        "leukemia", "aml", "all", "cml", "cll", "백혈병"
    ],
    "림프종": [
        "lymphoma", "hodgkin lymphoma", "non-hodgkin lymphoma", "림프종"
    ],
    "교모세포종": [
        "glioblastoma", "gbm", "교모세포종"
    ],
    "알츠하이머병": [
        "alzheimer", "alzheimer's disease", "알츠하이머", "알츠하이머병"
    ],
    "파킨슨병": [
        "parkinson", "parkinson's disease", "파킨슨", "파킨슨병"
    ],
    "자가면역질환": [
        "autoimmune disease", "autoimmunity", "자가면역", "자가면역질환"
    ],
    "염증성 장질환": [
        "inflammatory bowel disease", "ibd", "crohn", "ulcerative colitis",
        "염증성 장질환", "크론병", "궤양성 대장염"
    ],
    "류마티스관절염": [
        "rheumatoid arthritis", "ra", "류마티스관절염"
    ],
}

TECH_TAG_RULES = {
    "단일세포": [
        "single-cell", "single cell", "scrna-seq", "단일세포"
    ],
    "공간전사체": [
        "spatial transcriptomics", "spatial transcriptome", "공간 전사체", "공간전사체"
    ],
    "RNA시퀀싱": [
        "rna-seq", "rna sequencing", "bulk rna", "전사체", "rna 시퀀싱"
    ],
    "멀티오믹스": [
        "multiomics", "multi-omics", "omics", "proteomics", "metabolomics", "genomics", "멀티오믹스"
    ],
    "오가노이드": [
        "organoid", "pdo", "patient-derived organoid", "오가노이드"
    ],
    "ADC": [
        "adc", "antibody-drug conjugate", "항체-약물 접합체"
    ],
    "이중특이항체": [
        "bispecific antibody", "이중특이항체"
    ],
    "면역항암": [
        "immune checkpoint", "pd-1", "pd-l1", "ctla-4", "immunotherapy",
        "면역관문", "면역항암", "면역치료"
    ],
    "CAR-T": [
        "car-t", "car t", "chimeric antigen receptor", "CAR-T"
    ],
    "바이오마커": [
        "biomarker", "predictive biomarker", "prognostic biomarker", "바이오마커"
    ],
    "AI/머신러닝": [
        "artificial intelligence", "ai", "machine learning", "deep learning",
        "neural network", "transformer", "large language model", "머신러닝", "딥러닝"
    ],
    "유전자치료": [
        "gene therapy", "crispr", "base editing", "prime editing", "유전자치료"
    ],
    "세포치료": [
        "cell therapy", "stem cell", "mesenchymal stem cell", "세포치료"
    ],
}

def detect_tags_from_text(title: str, abstract: str, summary_en: str = "", summary_ko: str = "") -> Tuple[List[str], List[str]]:
    text = " ".join([
        title or "",
        abstract or "",
        summary_en or "",
        summary_ko or ""
    ]).lower()

    disease_tags = []
    tech_tags = []

    for tag, keywords in DISEASE_TAG_RULES.items():
        for kw in keywords:
            if kw.lower() in text:
                disease_tags.append(tag)
                break

    for tag, keywords in TECH_TAG_RULES.items():
        for kw in keywords:
            if kw.lower() in text:
                tech_tags.append(tag)
                break

    return sorted(set(disease_tags)), sorted(set(tech_tags))

# =========================================================
# 북마크 관리
# =========================================================
class BookmarkManager:
    def __init__(self, bookmark_file=BOOKMARKS_FILE):
        self.bookmark_file = bookmark_file

    def load_all(self):
        return load_json(self.bookmark_file, {})

    def save_all(self, data):
        save_json(self.bookmark_file, data)

    def _normalize_user_record(self, data, username):
        record = data.get(username)
        changed = False

        if record is None:
            record = {"bookmarks": [], "folders": []}
            data[username] = record
            changed = True
        elif isinstance(record, list):
            record = {"bookmarks": record, "folders": []}
            data[username] = record
            changed = True
        elif isinstance(record, dict):
            if "bookmarks" not in record or not isinstance(record.get("bookmarks"), list):
                record["bookmarks"] = record.get("bookmarks", []) if isinstance(record.get("bookmarks"), list) else []
                changed = True
            if "folders" not in record or not isinstance(record.get("folders"), list):
                record["folders"] = record.get("folders", []) if isinstance(record.get("folders"), list) else []
                changed = True
        else:
            record = {"bookmarks": [], "folders": []}
            data[username] = record
            changed = True

        # 각 북마크에 기본값 보정
        for item in record["bookmarks"]:
            if "folder_tags" not in item or not isinstance(item.get("folder_tags"), list):
                item["folder_tags"] = []
                changed = True
            if "notes" not in item:
                item["notes"] = ""
                changed = True
            if "reading_status" not in item:
                item["reading_status"] = "읽기 전"
                changed = True
            if "importance" not in item:
                item["importance"] = 3
                changed = True

        return record, changed

    def _make_bookmark_id(self, item: dict) -> str:
        seed = "|".join([
            str(item.get("doi", "") or ""),
            str(item.get("pmid", "") or ""),
            str(item.get("arxiv_id", "") or ""),
            str(item.get("source", "") or ""),
            str(item.get("source_id", "") or ""),
            str(item.get("title", "") or ""),
            str(item.get("url", "") or ""),
        ]).strip("|")
        return hashlib.md5(seed.encode("utf-8")).hexdigest() if seed else str(uuid.uuid4())

    def ensure_bookmark_ids(self, username=None):
        data = self.load_all()
        changed = False

        usernames = [username] if username else list(data.keys())
        for user in usernames:
            record, record_changed = self._normalize_user_record(data, user)
            changed = changed or record_changed
            for item in record["bookmarks"]:
                if not item.get("bookmark_id"):
                    item["bookmark_id"] = self._make_bookmark_id(item)
                    changed = True
                if "folder_tags" not in item or not isinstance(item.get("folder_tags"), list):
                    item["folder_tags"] = []
                    changed = True
                if "notes" not in item:
                    item["notes"] = ""
                    changed = True
                if "reading_status" not in item:
                    item["reading_status"] = "읽기 전"
                    changed = True
                if "importance" not in item:
                    item["importance"] = 3
                    changed = True

        if changed:
            self.save_all(data)

    def get_user_bookmarks(self, username):
        self.ensure_bookmark_ids(username)
        data = self.load_all()
        record, changed = self._normalize_user_record(data, username)
        if changed:
            self.save_all(data)
        return record.get("bookmarks", [])

    def get_user_folders(self, username):
        self.ensure_bookmark_ids(username)
        data = self.load_all()
        record, changed = self._normalize_user_record(data, username)

        # 북마크에 붙은 폴더 태그도 폴더 목록에 반영
        folder_set = set(record.get("folders", []))
        for item in record.get("bookmarks", []):
            for f in item.get("folder_tags", []):
                if f:
                    folder_set.add(f)

        folders = sorted(folder_set)
        if folders != sorted(record.get("folders", [])):
            record["folders"] = folders
            changed = True

        if changed:
            self.save_all(data)
        return folders

    def create_folder(self, username, folder_name):
        folder_name = clean_text(folder_name).strip()
        if not folder_name:
            return False, "폴더 이름을 입력해주세요."

        data = self.load_all()
        record, _ = self._normalize_user_record(data, username)
        folders = record.get("folders", [])
        if folder_name in folders:
            return False, "이미 존재하는 폴더입니다."

        folders.append(folder_name)
        record["folders"] = sorted(set(folders))
        self.save_all(data)
        return True, f"'{folder_name}' 폴더를 만들었습니다."

    def rename_folder(self, username, old_name, new_name):
        new_name = clean_text(new_name).strip()
        if not old_name or not new_name:
            return False, "폴더 이름을 확인해주세요."

        data = self.load_all()
        record, _ = self._normalize_user_record(data, username)
        folders = record.get("folders", [])
        if old_name not in folders:
            return False, "기존 폴더를 찾지 못했습니다."
        if new_name != old_name and new_name in folders:
            return False, "같은 이름의 폴더가 이미 있습니다."

        record["folders"] = [new_name if f == old_name else f for f in folders]
        for item in record.get("bookmarks", []):
            tags = item.get("folder_tags", [])
            item["folder_tags"] = [new_name if f == old_name else f for f in tags]

        record["folders"] = sorted(set(record["folders"]))
        self.save_all(data)
        return True, "폴더 이름을 변경했습니다."

    def delete_folder(self, username, folder_name):
        data = self.load_all()
        record, _ = self._normalize_user_record(data, username)
        folders = record.get("folders", [])
        if folder_name not in folders:
            return False, "삭제할 폴더를 찾지 못했습니다."

        record["folders"] = [f for f in folders if f != folder_name]
        for item in record.get("bookmarks", []):
            item["folder_tags"] = [f for f in item.get("folder_tags", []) if f != folder_name]

        self.save_all(data)
        return True, f"'{folder_name}' 폴더를 삭제했습니다."

    def update_bookmark_folders(self, username, bookmark_id, folder_names):
        data = self.load_all()
        record, _ = self._normalize_user_record(data, username)
        folders = sorted(set([clean_text(f).strip() for f in folder_names if clean_text(f).strip()]))

        # 폴더 목록 자동 확장
        record["folders"] = sorted(set(record.get("folders", []) + folders))

        found = False
        for item in record.get("bookmarks", []):
            if item.get("bookmark_id") == bookmark_id:
                item["folder_tags"] = folders
                found = True
                break

        if not found:
            return False, "북마크를 찾지 못했습니다."

        self.save_all(data)
        return True, "북마크 폴더를 저장했습니다."

    def bulk_update_folders(self, username, bookmark_ids, mode, folder_names):
        data = self.load_all()
        record, _ = self._normalize_user_record(data, username)
        folder_names = sorted(set([clean_text(f).strip() for f in folder_names if clean_text(f).strip()]))
        record["folders"] = sorted(set(record.get("folders", []) + folder_names))
        changed = 0
        for item in record.get("bookmarks", []):
            if item.get("bookmark_id") in bookmark_ids:
                current = set(item.get("folder_tags", []))
                if mode == "replace":
                    new_tags = folder_names
                elif mode == "add":
                    new_tags = sorted(current | set(folder_names))
                elif mode == "remove":
                    new_tags = sorted(current - set(folder_names))
                else:
                    new_tags = sorted(current)
                item["folder_tags"] = new_tags
                changed += 1
        self.save_all(data)
        return True, f"{changed}개 북마크의 폴더를 업데이트했습니다."

    def update_bookmark_metadata(self, username, bookmark_id, notes=None, reading_status=None, importance=None):
        data = self.load_all()
        record, _ = self._normalize_user_record(data, username)
        found = False
        for item in record.get("bookmarks", []):
            if item.get("bookmark_id") == bookmark_id:
                if notes is not None:
                    item["notes"] = notes
                if reading_status is not None:
                    item["reading_status"] = reading_status
                if importance is not None:
                    item["importance"] = int(importance)
                found = True
                break
        if not found:
            return False, "북마크를 찾지 못했습니다."
        self.save_all(data)
        return True, "북마크 메모/상태를 저장했습니다."

    def add_bookmark(self, username, paper: Paper):
        data = self.load_all()
        record, _ = self._normalize_user_record(data, username)

        disease_tags, tech_tags = detect_tags_from_text(
            paper.title, paper.abstract, paper.summary_en, paper.summary_ko
        )
        paper.disease_tags = disease_tags
        paper.tech_tags = tech_tags
        if not getattr(paper, "folder_tags", None):
            paper.folder_tags = []

        key = paper.unique_key()

        new_title_norm = normalize_title(paper.title)
        for item in record["bookmarks"]:
            existing = Paper(**{
                "source": item.get("source", ""),
                "source_id": item.get("source_id"),
                "title": item.get("title", ""),
                "authors": item.get("authors", []),
                "abstract": item.get("abstract", ""),
                "summary_en": item.get("summary_en", ""),
                "summary_ko": item.get("summary_ko", ""),
                "published_date": item.get("published_date"),
                "year": item.get("year"),
                "journal": item.get("journal"),
                "doi": item.get("doi"),
                "pmid": item.get("pmid"),
                "arxiv_id": item.get("arxiv_id"),
                "url": item.get("url"),
                "pdf_url": item.get("pdf_url"),
                "citation_count": item.get("citation_count", 0),
                "relevance_score": item.get("relevance_score", 0.0),
                "disease_tags": item.get("disease_tags", []),
                "tech_tags": item.get("tech_tags", []),
                "folder_tags": item.get("folder_tags", []),
                "notes": item.get("notes", ""),
                "reading_status": item.get("reading_status", "읽기 전"),
                "importance": item.get("importance", 3),
            })
            existing_norm = normalize_title(existing.title)
            if existing.unique_key() == key:
                return False, "이미 북마크한 논문입니다."
            if new_title_norm and existing_norm and new_title_norm == existing_norm:
                return False, "제목이 동일한 북마크가 이미 있습니다."
            if new_title_norm and existing_norm and title_similarity_score(new_title_norm, existing_norm) >= 0.96:
                return False, "제목이 매우 유사한 북마크가 이미 있습니다."

        paper_dict = asdict(paper)
        paper_dict.setdefault("folder_tags", [])
        paper_dict.setdefault("notes", "")
        paper_dict.setdefault("reading_status", "읽기 전")
        paper_dict.setdefault("importance", 3)
        paper_dict["bookmark_id"] = self._make_bookmark_id(paper_dict)
        record["bookmarks"].append(paper_dict)
        self.save_all(data)
        return True, "북마크에 저장되었습니다."

    def remove_bookmark(self, username, bookmark_id):
        data = self.load_all()
        record, _ = self._normalize_user_record(data, username)

        before = len(record["bookmarks"])
        record["bookmarks"] = [item for item in record["bookmarks"] if item.get("bookmark_id") != bookmark_id]

        if len(record["bookmarks"]) == before:
            return False, "삭제할 북마크를 찾지 못했습니다."

        self.save_all(data)
        return True, "북마크가 삭제되었습니다."
# =========================================================
# 문장 분리 / 요약 유틸
# =========================================================
def split_sentences(text: str) -> List[str]:
    text = clean_text(text)
    if not text:
        return []

    text = text.replace("e.g.", "eg")
    text = text.replace("i.e.", "ie")
    text = text.replace("et al.", "et al")

    parts = re.split(r'(?<=[\.\!\?])\s+', text)
    sentences = []

    for p in parts:
        p = p.strip()
        if len(p) >= 20:
            sentences.append(p)

    if len(sentences) <= 1 and ";" in text:
        parts = [x.strip() for x in text.split(";") if len(x.strip()) >= 20]
        sentences = parts if parts else [text]

    if not sentences and text:
        sentences = [text]

    return sentences

def tokenize_for_keywords(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9가-힣\s\-]", " ", text)
    tokens = text.split()

    stopwords = {
        "the", "and", "or", "of", "to", "in", "for", "on", "with", "by", "a", "an",
        "is", "are", "was", "were", "be", "this", "that", "these", "those",
        "we", "our", "they", "their", "it", "as", "at", "from", "can", "may",
        "using", "used", "use", "show", "shows", "shown", "study", "results",
        "result", "methods", "method", "background", "objective", "objectives",
        "conclusion", "conclusions", "patients", "patient", "analysis",
        "그리고", "또한", "대한", "위한", "에서", "으로", "했다", "한다", "있는", "있다"
    }

    filtered = []
    for t in tokens:
        if len(t) <= 2:
            continue
        if t in stopwords:
            continue
        filtered.append(t)
    return filtered

def build_query_keywords(query: str, title: str = "", abstract: str = "") -> List[str]:
    base = " ".join([query or "", title or "", abstract or ""])
    tokens = tokenize_for_keywords(base)
    freq = Counter(tokens)
    return [w for w, _ in freq.most_common(20)]

def sentence_score(sentence: str, keywords: List[str]) -> float:
    s = sentence.lower()
    score = 0.0

    for kw in keywords:
        if kw in s:
            score += 1.2

    bonus_patterns = [
        "conclude", "conclusion", "suggest", "demonstrate", "show", "revealed",
        "associated", "significant", "improved", "improvement", "effective",
        "results", "found", "indicate", "therefore"
    ]
    for p in bonus_patterns:
        if p in s:
            score += 0.8

    if re.search(r"\d", sentence):
        score += 0.5

    length = len(sentence)
    if length < 40:
        score -= 0.7
    elif length > 500:
        score -= 0.5

    return score

def tfidf_sentence_scores(sentences: List[str]) -> List[float]:
    if not sentences:
        return []
    if len(sentences) == 1:
        return [1.0]

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        X = vectorizer.fit_transform(sentences)
        scores = np.asarray(X.sum(axis=1)).ravel().tolist()
        return scores
    except:
        return [1.0 for _ in sentences]

def extractive_summarize(title: str, abstract: str, line_count: int = 3, query: str = "") -> str:
    abstract = clean_text(abstract)
    if not abstract:
        return "초록이 없어 요약할 수 없습니다."

    line_count = max(1, min(line_count, 5))
    sentences = split_sentences(abstract)

    if not sentences:
        return "초록이 너무 짧거나 형식이 맞지 않아 요약할 수 없습니다."

    if len(sentences) <= line_count:
        return "\n".join(sentences)

    keywords = build_query_keywords(query=query, title=title, abstract=abstract)
    tfidf_scores = tfidf_sentence_scores(sentences)

    scored = []
    for idx, sent in enumerate(sentences):
        score = tfidf_scores[idx] + sentence_score(sent, keywords)

        if idx == 0:
            score += 0.4
        if idx == len(sentences) - 1:
            score += 0.4

        scored.append((idx, sent, score))

    top = sorted(scored, key=lambda x: x[2], reverse=True)[:line_count]
    top = sorted(top, key=lambda x: x[0])

    return "\n".join([x[1] for x in top])

# =========================================================
# 논문 검색 커넥터
# =========================================================
class PubMedConnector:
    BASE_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    BASE_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    def search(self, query: str, max_results: int = 10) -> List[Paper]:
        papers = []

        search_params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": max_results,
            "sort": "relevance"
        }

        try:
            search_resp = http_get(self.BASE_SEARCH_URL, params=search_params, timeout=30)
            search_resp.raise_for_status()
            search_data = search_resp.json()
            id_list = search_data.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                return []

            fetch_params = {
                "db": "pubmed",
                "id": ",".join(id_list),
                "retmode": "xml"
            }

            fetch_resp = http_get(self.BASE_FETCH_URL, params=fetch_params, timeout=30)
            fetch_resp.raise_for_status()
            root = ET.fromstring(fetch_resp.text)

            for article in root.findall(".//PubmedArticle"):
                try:
                    pmid_elem = article.find(".//PMID")
                    pmid = pmid_elem.text if pmid_elem is not None else None

                    title_elem = article.find(".//ArticleTitle")
                    title = "".join(title_elem.itertext()).strip() if title_elem is not None else "제목 없음"

                    abstract_texts = []
                    for abstract_elem in article.findall(".//Abstract/AbstractText"):
                        txt = "".join(abstract_elem.itertext()).strip()
                        label = abstract_elem.attrib.get("Label")
                        if txt:
                            if label:
                                txt = f"{label}: {txt}"
                            abstract_texts.append(txt)
                    abstract = "\n".join(abstract_texts)

                    author_list = []
                    for author in article.findall(".//Author"):
                        lastname = author.findtext("LastName", default="")
                        forename = author.findtext("ForeName", default="")
                        collective = author.findtext("CollectiveName", default="")
                        if collective:
                            author_list.append(collective)
                        else:
                            full = f"{forename} {lastname}".strip()
                            if full:
                                author_list.append(full)

                    journal = article.findtext(".//Journal/Title", default="")

                    year = None
                    pub_year = article.findtext(".//PubDate/Year")
                    medline_date = article.findtext(".//PubDate/MedlineDate")
                    if pub_year and pub_year.isdigit():
                        year = int(pub_year)
                    elif medline_date:
                        year = extract_year_from_text(medline_date)

                    doi = None
                    for aid in article.findall(".//ArticleId"):
                        if aid.attrib.get("IdType") == "doi":
                            doi = aid.text
                            break

                    paper = Paper(
                        source="PubMed",
                        source_id=pmid,
                        pmid=pmid,
                        title=clean_text(title),
                        authors=author_list,
                        abstract=clean_text(abstract),
                        published_date=str(year) if year else "",
                        year=year,
                        journal=journal,
                        doi=doi,
                        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                        citation_count=0
                    )
                    papers.append(paper)
                except:
                    continue

        except Exception as e:
            print(f"[PubMed Error] {e}")

        return papers

class ArxivConnector:
    BASE_URL = "http://export.arxiv.org/api/query"

    def search(self, query: str, max_results: int = 10) -> List[Paper]:
        papers = []
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }

        try:
            response = http_get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            feed = feedparser.parse(response.text)

            for entry in feed.entries:
                authors = [author.name for author in entry.authors] if hasattr(entry, "authors") else []
                published = getattr(entry, "published", "")
                title = clean_text(getattr(entry, "title", ""))
                abstract = clean_text(getattr(entry, "summary", ""))
                link = getattr(entry, "link", "")

                pdf_url = None
                if hasattr(entry, "links"):
                    for l in entry.links:
                        if getattr(l, "type", "") == "application/pdf":
                            pdf_url = l.href

                arxiv_id = entry.id.split("/")[-1] if hasattr(entry, "id") else None
                year = extract_year_from_text(published)

                paper = Paper(
                    source="arXiv",
                    source_id=arxiv_id,
                    arxiv_id=arxiv_id,
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    published_date=published,
                    year=year,
                    url=link,
                    pdf_url=pdf_url,
                    citation_count=0
                )
                papers.append(paper)

        except Exception as e:
            print(f"[arXiv Error] {e}")

        return papers

class SemanticScholarConnector:
    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

    def search(self, query: str, max_results: int = 10) -> List[Paper]:
        papers = []

        params = {
            "query": query,
            "limit": max_results,
            "fields": "paperId,title,abstract,authors,year,citationCount,url,publicationDate,venue,externalIds"
        }

        try:
            response = http_get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            for item in data.get("data", []):
                authors = [a.get("name", "") for a in item.get("authors", [])]
                external_ids = item.get("externalIds", {}) or {}
                source_id = item.get("paperId") or external_ids.get("CorpusId") or external_ids.get("DOI")

                paper = Paper(
                    source="Semantic Scholar",
                    source_id=str(source_id) if source_id else None,
                    title=clean_text(item.get("title", "")),
                    authors=authors,
                    abstract=clean_text(item.get("abstract", "") or ""),
                    year=item.get("year"),
                    published_date=item.get("publicationDate"),
                    journal=item.get("venue"),
                    doi=external_ids.get("DOI"),
                    url=item.get("url"),
                    citation_count=item.get("citationCount", 0) or 0
                )
                papers.append(paper)

        except Exception as e:
            print(f"[Semantic Scholar Error] {e}")

        return papers

    # -----------------------------------------------------
    # DOI 또는 제목으로 citation count 보강
    # -----------------------------------------------------
    def get_citation_count_by_title_or_doi(self, title: str, doi: str = None) -> int:
        # 1순위: DOI 직접 조회
        if doi:
            try:
                url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
                params = {"fields": "title,citationCount"}
                r = http_get(url, params=params, timeout=20)
                if r.status_code == 200:
                    data = r.json()
                    return data.get("citationCount", 0) or 0
            except Exception as e:
                print("[SemanticScholar DOI citation enrich error]", e)

        # 2순위: 제목 검색
        if title and title.strip():
            try:
                params = {
                    "query": title,
                    "limit": 5,
                    "fields": "title,citationCount,year,externalIds"
                }
                r = http_get(self.BASE_URL, params=params, timeout=20)
                if r.status_code == 200:
                    data = r.json().get("data", [])

                    if not data:
                        return 0

                    # 가장 비슷한 제목 선택
                    norm_target = normalize_title(title)
                    best_count = 0
                    best_score = -1

                    for item in data:
                        candidate_title = clean_text(item.get("title", ""))
                        norm_candidate = normalize_title(candidate_title)

                        score = title_similarity_score(norm_target, norm_candidate)

                        if score > best_score:
                            best_score = score
                            best_count = item.get("citationCount", 0) or 0

                    # 제목이 어느 정도 유사할 때만 채택
                    if best_score >= 0.55:
                        return best_count

            except Exception as e:
                print("[SemanticScholar Title citation enrich error]", e)

        return 0

class GoogleScholarConnector:
    """
    Google Scholar는 공식 무료 API가 없어서 scholarly 라이브러리를 사용.
    """
    def search(self, query: str, max_results: int = 10) -> List[Paper]:
        papers = []

        if not SCHOLARLY_AVAILABLE:
            return papers

        try:
            search_query = scholarly.search_pubs(query)
            count = 0

            for item in search_query:
                if count >= max_results:
                    break

                try:
                    bib = item.get("bib", {}) if isinstance(item, dict) else {}
                    title = clean_text(bib.get("title", ""))
                    abstract = clean_text(bib.get("abstract", "") or "")
                    authors_raw = bib.get("author", "")
                    authors = []

                    if isinstance(authors_raw, str) and authors_raw.strip():
                        authors = [a.strip() for a in authors_raw.split(" and ") if a.strip()]
                    elif isinstance(authors_raw, list):
                        authors = authors_raw

                    year = None
                    if bib.get("pub_year"):
                        try:
                            year = int(str(bib.get("pub_year")))
                        except:
                            year = extract_year_from_text(str(bib.get("pub_year")))

                    venue = bib.get("venue", "") or bib.get("journal", "")
                    url = item.get("pub_url") or item.get("eprint_url") or None
                    citedby = item.get("num_citations", 0) or 0
                    paper_id = item.get("author_id") or item.get("container_type") or title

                    paper = Paper(
                        source="Google Scholar",
                        source_id=str(paper_id),
                        title=title if title else "제목 없음",
                        authors=authors,
                        abstract=abstract,
                        year=year,
                        published_date=str(year) if year else "",
                        journal=venue,
                        url=url,
                        citation_count=citedby
                    )
                    papers.append(paper)
                    count += 1
                    time.sleep(0.2)

                except:
                    continue

        except Exception as e:
            print(f"[Google Scholar Error] {e}")

        return papers

# =========================================================
# 제목 유사도 계산
# =========================================================
def title_similarity_score(a: str, b: str) -> float:
    """
    아주 단순한 제목 유사도:
    - 완전 일치면 1.0
    - 토큰 겹침 기반 점수
    """
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    a_tokens = set(a.split())
    b_tokens = set(b.split())

    if not a_tokens or not b_tokens:
        return 0.0

    inter = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    return inter / union if union > 0 else 0.0

# =========================================================
# 논문 서비스
# =========================================================
def deduplicate_papers(papers: List[Paper]) -> List[Paper]:
    """
    같은 논문이 여러 소스에서 들어왔을 때:
    - DOI가 같으면 같은 논문
    - 아니면 정규화 제목이 매우 비슷하면 같은 논문으로 취급
    - citation_count는 더 큰 값으로 갱신
    """
    result = []

    for paper in papers:
        matched = None

        for existing in result:
            same = False

            # 1) DOI 동일
            if paper.doi and existing.doi and paper.doi.lower() == existing.doi.lower():
                same = True

            # 2) PMID 동일
            elif paper.pmid and existing.pmid and paper.pmid == existing.pmid:
                same = True

            # 3) arXiv ID 동일
            elif paper.arxiv_id and existing.arxiv_id and paper.arxiv_id == existing.arxiv_id:
                same = True

            # 4) 제목 유사도
            else:
                sim = title_similarity_score(
                    normalize_title(paper.title),
                    normalize_title(existing.title)
                )
                if sim >= 0.88:
                    same = True

            if same:
                matched = existing
                break

        if matched is None:
            result.append(paper)
        else:
            # 더 좋은 정보로 보강
            if (not matched.abstract or len(matched.abstract) < 20) and paper.abstract:
                matched.abstract = paper.abstract

            if paper.citation_count > matched.citation_count:
                matched.citation_count = paper.citation_count

            if not matched.url and paper.url:
                matched.url = paper.url
            if not matched.pdf_url and paper.pdf_url:
                matched.pdf_url = paper.pdf_url
            if not matched.published_date and paper.published_date:
                matched.published_date = paper.published_date
            if not matched.year and paper.year:
                matched.year = paper.year
            if not matched.journal and paper.journal:
                matched.journal = paper.journal
            if not matched.authors and paper.authors:
                matched.authors = paper.authors
            if not matched.doi and paper.doi:
                matched.doi = paper.doi
            if not matched.pmid and paper.pmid:
                matched.pmid = paper.pmid
            if not matched.arxiv_id and paper.arxiv_id:
                matched.arxiv_id = paper.arxiv_id

            # source 표시를 합치고 싶으면 이렇게
            if matched.source != paper.source:
                source_set = set([s.strip() for s in matched.source.split(",") if s.strip()])
                source_set.add(paper.source)
                matched.source = ", ".join(sorted(source_set))

    return result

def sort_papers(papers: List[Paper], sort_by: str) -> List[Paper]:
    if sort_by == "latest":
        return sorted(papers, key=lambda x: (x.year or 0, x.published_date or ""), reverse=True)
    elif sort_by == "citation":
        return sorted(papers, key=lambda x: x.citation_count or 0, reverse=True)
    return papers

class SearchService:
    def __init__(self):
        self.pubmed = PubMedConnector()
        self.arxiv = ArxivConnector()
        self.semantic = SemanticScholarConnector()
        self.scholar = GoogleScholarConnector()

    def enrich_citation_counts(self, papers: List[Paper]) -> List[Paper]:
        """
        모든 논문에 대해 citation_count 보강
        - citation_count가 0인 논문만 보강
        - DOI 우선, 없으면 제목 검색
        - 학교 Wi-Fi에서 Semantic Scholar가 막히면 조용히 건너뛰고 검색 결과는 유지
        """
        total = len(papers)
        for i, paper in enumerate(papers):
            if not paper.citation_count or paper.citation_count == 0:
                try:
                    c = self.semantic.get_citation_count_by_title_or_doi(
                        title=paper.title,
                        doi=paper.doi
                    )
                    paper.citation_count = c
                except Exception as e:
                    record_network_error("Semantic Scholar citation enrich", e)

            if total > 0:
                time.sleep(0.05)

        return papers

    def _safe_source_search(self, source_name: str, connector, query: str, max_results: int) -> List[Paper]:
        """한 검색원이 실패해도 전체 검색이 멈추지 않게 보호한다."""
        try:
            return connector.search(query, max_results=max_results)
        except Exception as e:
            record_network_error(source_name, e)
            return []

    def search_all(self, query: str,
                   use_pubmed=True, use_arxiv=True, use_semantic=True, use_scholar=True,
                   max_results_per_source=10):
        papers = []
        jobs = []

        if use_pubmed:
            jobs.append(("PubMed", self.pubmed))
        if use_arxiv:
            jobs.append(("arXiv", self.arxiv))
        if use_semantic:
            jobs.append(("Semantic Scholar", self.semantic))
        if use_scholar:
            jobs.append(("Google Scholar", self.scholar))

        # 학교망에서 특정 API가 오래 멈춰도 다른 API 결과를 먼저 확보하기 위해 병렬 검색
        if jobs:
            with ThreadPoolExecutor(max_workers=min(len(jobs), 4)) as executor:
                future_map = {
                    executor.submit(self._safe_source_search, name, connector, query, max_results_per_source): name
                    for name, connector in jobs
                }
                for future in as_completed(future_map):
                    name = future_map[future]
                    try:
                        papers.extend(future.result())
                    except Exception as e:
                        record_network_error(name, e)

        papers = deduplicate_papers(papers)

        # 인용수 보강은 실패해도 검색 결과 자체는 유지
        try:
            papers = self.enrich_citation_counts(papers)
        except Exception as e:
            record_network_error("citation enrichment", e)

        papers = deduplicate_papers(papers)
        return papers

class LocalSummaryService:
    def summarize(self, title: str, abstract: str, line_count: int = 3, query: str = ""):
        summary_en = extractive_summarize(
            title=title,
            abstract=abstract,
            line_count=line_count,
            query=query
        )
        summary_ko = translate_summary_to_korean(summary_en)

        return {
            "en": summary_en,
            "ko": summary_ko
        }

# =========================================================
# 서비스 인스턴스
# =========================================================
user_manager = UserManager()
bookmark_manager = BookmarkManager()
search_service = SearchService()
summary_service = LocalSummaryService()

# =========================================================
# 인증 UI
# =========================================================
def render_auth_page():
    st.title("📚 회원 기능 포함 무료 논문 검색 및 요약 시스템")
    st.caption("회원가입 / 로그인 / 마이페이지 / 북마크 / Google Scholar / 자동 태그 / 인용수 보강 기능 포함")

    tab1, tab2 = st.tabs(["로그인", "회원가입"])

    with tab1:
        st.subheader("로그인")
        login_username = st.text_input("아이디", key="login_username")
        login_password = st.text_input("비밀번호", type="password", key="login_password")

        if st.button("로그인", key="login_btn"):
            ok, result = user_manager.login(login_username, login_password)
            if ok:
                st.session_state.logged_in = True
                st.session_state.current_user = result["username"]
                st.success(f"{result['name']}님, 로그인되었습니다.")
                st.rerun()
            else:
                st.error(result)

    with tab2:
        st.subheader("회원가입")
        reg_username = st.text_input("아이디", key="reg_username")
        reg_password = st.text_input("비밀번호", type="password", key="reg_password")
        reg_password2 = st.text_input("비밀번호 확인", type="password", key="reg_password2")
        reg_name = st.text_input("이름", key="reg_name")
        reg_email = st.text_input("이메일", key="reg_email")
        reg_major = st.text_input("전공/소속(선택)", key="reg_major")

        if st.button("회원가입", key="register_btn"):
            if reg_password != reg_password2:
                st.error("비밀번호 확인이 일치하지 않습니다.")
            else:
                ok, msg = user_manager.register(
                    reg_username, reg_password, reg_name, reg_email, reg_major
                )
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

# =========================================================
# 북마크 필터링/분류용 유틸
# =========================================================
def build_tag_index(bookmarks: List[dict], key_name: str) -> Dict[str, List[dict]]:
    grouped = defaultdict(list)
    for item in bookmarks:
        tags = item.get(key_name, []) or []
        if not tags:
            grouped["태그 없음"].append(item)
        else:
            for t in tags:
                grouped[t].append(item)
    return dict(grouped)


def paper_to_export_row(item: dict) -> dict:
    return {
        "title": item.get("title", ""),
        "source": item.get("source", ""),
        "year": item.get("year", ""),
        "journal": item.get("journal", ""),
        "authors": ", ".join(item.get("authors", [])) if isinstance(item.get("authors"), list) else item.get("authors", ""),
        "citation_count": item.get("citation_count", 0),
        "disease_tags": ", ".join(item.get("disease_tags", [])) if item.get("disease_tags") else "",
        "tech_tags": ", ".join(item.get("tech_tags", [])) if item.get("tech_tags") else "",
        "folder_tags": ", ".join(item.get("folder_tags", [])) if item.get("folder_tags") else "",
        "reading_status": item.get("reading_status", "읽기 전"),
        "importance": item.get("importance", 3),
        "notes": item.get("notes", ""),
        "doi": item.get("doi", ""),
        "pmid": item.get("pmid", ""),
        "arxiv_id": item.get("arxiv_id", ""),
        "url": item.get("url", ""),
        "pdf_url": item.get("pdf_url", ""),
        "image_urls": "\n".join(item.get("image_urls", [])) if item.get("image_urls") else "",
        "abstract": item.get("abstract", ""),
        "summary_en": item.get("summary_en", ""),
        "summary_ko": item.get("summary_ko", ""),
    }

def make_bookmark_export_bytes(bookmarks: List[dict], export_format: str = "csv"):
    rows = [paper_to_export_row(item) for item in bookmarks]
    df = pd.DataFrame(rows)
    if export_format == "xlsx":
        bio = BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="bookmarks")
        bio.seek(0)
        return bio.getvalue()
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def highlight_text(text: str, keywords: List[str]) -> str:
    if not text:
        return ""
    highlighted = text
    for kw in sorted([k.strip() for k in keywords if k and k.strip()], key=len, reverse=True):
        try:
            highlighted = re.sub(f"({re.escape(kw)})", r"<mark>\1</mark>", highlighted, flags=re.I)
        except re.error:
            pass
    return highlighted

def bookmark_sort_key(item: dict, mode: str):
    if mode == "최신 저장순":
        return item.get("bookmark_id", "")
    if mode == "연도순":
        return item.get("year") or 0
    if mode == "인용수순":
        return item.get("citation_count", 0)
    if mode == "제목순":
        return normalize_title(item.get("title", ""))
    if mode == "중요도순":
        return item.get("importance", 0)
    return item.get("bookmark_id", "")


def normalize_external_paper_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url

def extract_paper_images_from_html(soup: BeautifulSoup, base_url: str, max_images: int = 8) -> List[str]:
    image_urls = []
    seen = set()

    def add_url(raw):
        if not raw:
            return
        full = urljoin(base_url, raw)
        low = full.lower()
        if not low.startswith(("http://", "https://")):
            return
        if any(bad in low for bad in ["logo", "icon", "sprite", "avatar"]):
            return
        if full not in seen:
            seen.add(full)
            image_urls.append(full)

    for prop in ["og:image", "twitter:image"]:
        tag = soup.find("meta", attrs={"property": prop}) or soup.find("meta", attrs={"name": prop})
        if tag and tag.get("content"):
            add_url(tag.get("content"))

    selectors = [
        "figure img", "img[alt*='fig' i]", "img[alt*='figure' i]",
        "img[class*='fig' i]", "img[src*='figure' i]", "img[src*='fig' i]"
    ]
    for sel in selectors:
        for img in soup.select(sel):
            add_url(img.get("src") or img.get("data-src") or img.get("data-original"))
            if len(image_urls) >= max_images:
                return image_urls[:max_images]

    for img in soup.find_all("img"):
        add_url(img.get("src") or img.get("data-src") or img.get("data-original"))
        if len(image_urls) >= max_images:
            break
    return image_urls[:max_images]

def fetch_paper_from_url(url: str) -> Tuple[Optional[Paper], str]:
    url = normalize_external_paper_url(url)
    if not url:
        return None, "링크를 입력해주세요."
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = http_get(url, headers=headers, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        return None, f"링크를 불러오지 못했습니다: {e}"

    final_url = resp.url
    content_type = resp.headers.get("Content-Type", "").lower()
    if "pdf" in content_type or final_url.lower().endswith('.pdf'):
        title = os.path.basename(urlparse(final_url).path) or "직접 추가한 PDF"
        paper = Paper(source="Custom Link", title=title, url=final_url, pdf_url=final_url, abstract="", source_id=final_url)
        return paper, "PDF 링크에서 논문을 추가했습니다."

    html = resp.text
    soup = BeautifulSoup(html, 'html.parser')

    def meta(*names):
        for name in names:
            tag = soup.find('meta', attrs={'name': name}) or soup.find('meta', attrs={'property': name})
            if tag and tag.get('content'):
                return clean_text(tag.get('content'))
        return ""

    title = meta('citation_title', 'dc.title', 'og:title') or clean_text(soup.title.get_text()) if soup.title else ''
    abstract = meta('citation_abstract', 'description', 'og:description', 'dc.description')
    authors = [clean_text(t.get('content')) for t in soup.find_all('meta', attrs={'name': 'citation_author'}) if t.get('content')]
    journal = meta('citation_journal_title', 'citation_conference_title', 'dc.source')
    doi = meta('citation_doi', 'dc.identifier')
    pub_date = meta('citation_publication_date', 'dc.date')
    year = None
    m = re.search(r'(19|20)\d{2}', pub_date or '')
    if m:
        year = int(m.group(0))
    pdf_url = meta('citation_pdf_url')
    if pdf_url:
        pdf_url = urljoin(final_url, pdf_url)

    source_name = urlparse(final_url).netloc.replace('www.', '') or 'Custom Link'
    image_urls = extract_paper_images_from_html(soup, final_url)

    paper = Paper(
        source=source_name,
        source_id=final_url,
        title=title or final_url,
        authors=authors,
        abstract=abstract,
        published_date=pub_date or None,
        year=year,
        journal=journal or None,
        doi=doi or None,
        url=final_url,
        pdf_url=pdf_url or None,
        image_urls=image_urls,
    )
    return paper, "링크에서 논문 정보를 불러왔습니다."

def render_pdf_embed(pdf_url: str, height: int = 700):
    if not pdf_url:
        return
    safe_url = pdf_url.replace('"', '%22')
    components.html(f'<iframe src="{safe_url}" width="100%" height="{height}" style="border:none;"></iframe>', height=height+20)

def render_paper_detail_block(item: dict, detail_key: str):
    user_keywords_raw = st.session_state.get("highlight_keywords", "")
    auto_keywords = []
    auto_keywords += item.get("disease_tags", [])[:5]
    auto_keywords += item.get("tech_tags", [])[:5]
    user_keywords = [x.strip() for x in user_keywords_raw.split(",") if x.strip()]
    highlight_keywords = list(dict.fromkeys(auto_keywords + user_keywords))

    tab1, tab2, tab3, tab4 = st.tabs(["기본 정보", "초록/요약", "링크/식별자", "시각자료/PDF"])

    with tab1:
        st.write(f"**제목:** {item.get('title', '제목 없음')}")
        st.write(f"**출처:** {item.get('source', '')}")
        st.write(f"**저자:** {', '.join(item.get('authors', [])) if item.get('authors') else '정보 없음'}")
        st.write(f"**연도:** {item.get('year', '정보 없음')}")
        st.write(f"**저널/학회:** {item.get('journal', '정보 없음')}")
        st.write(f"**인용수:** {item.get('citation_count', 0)}")
        st.write(f"**질환 태그:** {', '.join(item.get('disease_tags', [])) if item.get('disease_tags') else '없음'}")
        st.write(f"**기술 태그:** {', '.join(item.get('tech_tags', [])) if item.get('tech_tags') else '없음'}")
        st.write(f"**읽기 상태:** {item.get('reading_status', '읽기 전')} | **중요도:** {'★'*int(item.get('importance',3))}")

    with tab2:
        st.write("**초록**")
        st.markdown(highlight_text(item.get("abstract", "초록 없음"), highlight_keywords), unsafe_allow_html=True)
        if item.get("summary_en"):
            st.write("**영문 요약**")
            st.info(item.get("summary_en"))
        if item.get("summary_ko"):
            st.write("**한국어 의역 요약**")
            st.success(item.get("summary_ko"))
        if item.get("notes"):
            st.write("**내 메모**")
            st.write(item.get("notes"))

    with tab3:
        if item.get("doi"):
            st.write("**DOI**")
            st.code(item.get("doi"), language=None)
            st.markdown(f"[DOI 링크 열기](https://doi.org/{item.get('doi')})")
        if item.get("pmid"):
            st.write("**PMID**")
            st.code(str(item.get("pmid")), language=None)
            st.markdown(f"[PubMed에서 보기](https://pubmed.ncbi.nlm.nih.gov/{item.get('pmid')}/)")
        if item.get("arxiv_id"):
            st.write("**arXiv ID**")
            st.code(str(item.get("arxiv_id")), language=None)
            st.markdown(f"[arXiv 페이지 열기](https://arxiv.org/abs/{item.get('arxiv_id')})")
        if item.get("url"):
            st.markdown(f"[논문 원문/초록 페이지 열기]({item.get('url')})")
        if item.get("pdf_url"):
            st.markdown(f"[PDF 바로가기]({item.get('pdf_url')})")
        elif item.get("arxiv_id"):
            st.markdown(f"[arXiv PDF 바로가기](https://arxiv.org/pdf/{item.get('arxiv_id')}.pdf)")
        elif item.get("pmid") and item.get("doi"):
            st.caption("PubMed 논문은 DOI 링크에서 원문/PDF를 확인할 수 있습니다.")

    with tab4:
        image_urls = item.get("image_urls", []) or []
        if image_urls:
            st.write(f"**논문 페이지에서 수집한 시각자료 미리보기 ({len(image_urls)}개)**")
            st.image(image_urls, use_container_width=True)
        else:
            st.caption("현재 저장된 시각자료가 없습니다. 직접 링크 추가 기능으로 가져온 논문이나 그림이 포함된 원문 페이지에서 수집된 경우 여기에 표시됩니다.")

        pdf_candidate = item.get("pdf_url") or (f"https://arxiv.org/pdf/{item.get('arxiv_id')}.pdf" if item.get("arxiv_id") else None)
        if pdf_candidate:
            st.write("**브라우저 내 PDF 보기**")
            render_pdf_embed(pdf_candidate, height=720)


def render_bookmark_card(item: dict, idx: int, username: str, section_name: str = "all"):
    bookmark_id = item.get("bookmark_id")
    if not bookmark_id:
        bookmark_id = bookmark_manager._make_bookmark_id(item)
        item["bookmark_id"] = bookmark_id

    title = item.get("title", "제목 없음")
    expander_title = f"{idx+1}. {title}" if isinstance(idx, int) and idx >= 0 else title

    with st.expander(expander_title):
        st.write(f"**출처:** {item.get('source', '')}")
        st.write(f"**저자:** {', '.join(item.get('authors', [])) if item.get('authors') else '정보 없음'}")
        st.write(f"**연도:** {item.get('year', '정보 없음')} | **저널/학회:** {item.get('journal', '정보 없음')} | **인용수:** {item.get('citation_count', 0)}")
        st.write(f"**읽기 상태:** {item.get('reading_status', '읽기 전')} | **중요도:** {'★'*int(item.get('importance',3))}")

        folder_tags = item.get("folder_tags", []) or []
        st.write(f"**북마크 폴더:** {', '.join(folder_tags) if folder_tags else '미지정'}")

        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            if st.button("상세 보기", key=f"detail_toggle_{username}_{section_name}_{bookmark_id}"):
                state_key = f"show_detail_{username}_{section_name}_{bookmark_id}"
                st.session_state[state_key] = not st.session_state.get(state_key, False)
        with c2:
            delete_key = f"delete_bookmark_{username}_{section_name}_{bookmark_id}"
            if st.button("이 북마크 삭제", key=delete_key):
                ok, msg = bookmark_manager.remove_bookmark(username, bookmark_id)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
        with c3:
            if item.get("pdf_url"):
                st.markdown(f"[PDF]({item.get('pdf_url')})")
            elif item.get("url"):
                st.markdown(f"[원문 링크]({item.get('url')})")

        folder_options = bookmark_manager.get_user_folders(username)
        selected_folders = st.multiselect(
            "이 북마크를 넣을 폴더",
            options=folder_options,
            default=[f for f in folder_tags if f in folder_options],
            key=f"folder_select_{username}_{section_name}_{bookmark_id}"
        )
        if st.button("폴더 저장", key=f"folder_save_{username}_{section_name}_{bookmark_id}"):
            ok, msg = bookmark_manager.update_bookmark_folders(username, bookmark_id, selected_folders)
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

        st.text_area("개인 메모", value=item.get("notes",""), key=f"note_{username}_{section_name}_{bookmark_id}", height=100)
        m1, m2 = st.columns([1,1])
        with m1:
            status = st.selectbox("읽기 상태", ["읽기 전","읽는 중","읽음","핵심 논문","재확인 필요"],
                                  index=["읽기 전","읽는 중","읽음","핵심 논문","재확인 필요"].index(item.get("reading_status","읽기 전")) if item.get("reading_status","읽기 전") in ["읽기 전","읽는 중","읽음","핵심 논문","재확인 필요"] else 0,
                                  key=f"status_{username}_{section_name}_{bookmark_id}")
        with m2:
            importance = st.slider("중요도",1,5,int(item.get("importance",3)), key=f"importance_{username}_{section_name}_{bookmark_id}")
        if st.button("메모/상태 저장", key=f"meta_save_{username}_{section_name}_{bookmark_id}"):
            ok, msg = bookmark_manager.update_bookmark_metadata(
                username,
                bookmark_id,
                notes=st.session_state.get(f"note_{username}_{section_name}_{bookmark_id}", ""),
                reading_status=status,
                importance=importance
            )
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

        state_key = f"show_detail_{username}_{section_name}_{bookmark_id}"
        if st.session_state.get(state_key, False):
            render_paper_detail_block(item, state_key)

# =========================================================
# 마이페이지

# =========================================================


def render_mypage(username):
    user = user_manager.get_user(username)
    if not user:
        st.error("회원 정보를 불러올 수 없습니다.")
        return

    st.subheader("👤 마이페이지")
    tabs = st.tabs(["내 정보", "회원정보 수정", "비밀번호 변경", "내 북마크", "통계 대시보드", "논문 비교"])

    with tabs[0]:
        st.write(f"**아이디:** {user['username']}")
        st.write(f"**이름:** {user.get('name', '')}")
        st.write(f"**이메일:** {user.get('email', '')}")
        st.write(f"**전공/소속:** {user.get('major', '')}")

    with tabs[1]:
        new_name = st.text_input("이름 수정", value=user.get("name", ""), key="edit_name")
        new_email = st.text_input("이메일 수정", value=user.get("email", ""), key="edit_email")
        new_major = st.text_input("전공/소속 수정", value=user.get("major", ""), key="edit_major")
        if st.button("회원정보 저장", key="save_profile_btn"):
            ok, msg = user_manager.update_profile(username, new_name, new_email, new_major)
            st.success(msg) if ok else st.error(msg)
            if ok: st.rerun()

    with tabs[2]:
        old_pw = st.text_input("현재 비밀번호", type="password", key="old_pw")
        new_pw = st.text_input("새 비밀번호", type="password", key="new_pw")
        new_pw2 = st.text_input("새 비밀번호 확인", type="password", key="new_pw2")
        if st.button("비밀번호 변경", key="change_pw_btn"):
            if new_pw != new_pw2:
                st.error("새 비밀번호 확인이 일치하지 않습니다.")
            else:
                ok, msg = user_manager.change_password(username, old_pw, new_pw)
                st.success(msg) if ok else st.error(msg)

    bookmark_manager.ensure_bookmark_ids(username)
    bookmarks = bookmark_manager.get_user_bookmarks(username)
    folders = bookmark_manager.get_user_folders(username)

    with tabs[3]:
        st.markdown("### 📁 북마크 폴더 관리")
        f1, f2 = st.columns([2, 1.2])
        with f1:
            new_folder_name = st.text_input("새 폴더 이름", placeholder="예: 암 연구 / 면역치료 / scRNA-seq", key="new_bookmark_folder_name")
        with f2:
            if st.button("폴더 생성", key="create_bookmark_folder_btn"):
                ok, msg = bookmark_manager.create_folder(username, new_folder_name)
                st.success(msg) if ok else st.warning(msg)
                if ok: st.rerun()

        if folders:
            m1, m2, m3 = st.columns([1.2, 1.5, 1])
            with m1:
                folder_to_manage = st.selectbox("관리할 폴더", folders, key="folder_manage_select")
            with m2:
                rename_folder_name = st.text_input("폴더 이름 변경", value=folder_to_manage if folders else "", key="rename_bookmark_folder_name")
            with m3:
                st.write("")
                if st.button("폴더 이름 변경", key="rename_bookmark_folder_btn"):
                    ok, msg = bookmark_manager.rename_folder(username, folder_to_manage, rename_folder_name)
                    st.success(msg) if ok else st.warning(msg)
                    if ok: st.rerun()
                if st.button("폴더 삭제", key="delete_bookmark_folder_btn"):
                    ok, msg = bookmark_manager.delete_folder(username, folder_to_manage)
                    st.success(msg) if ok else st.warning(msg)
                    if ok: st.rerun()

        if not bookmarks:
            st.info("저장된 북마크가 없습니다.")
        else:
            st.write(f"총 **{len(bookmarks)}개**의 북마크가 있습니다.")
            q_col, d_col, t_col, f_col, s_col, r_col = st.columns([2.1, 1, 1, 1, 1, 1])
            with q_col:
                bookmark_query = st.text_input("북마크 검색", placeholder="제목/초록/요약/저자/메모 검색", key="bookmark_search_query")
            disease_index = build_tag_index(bookmarks, "disease_tags")
            tech_index = build_tag_index(bookmarks, "tech_tags")
            folder_index = build_tag_index(bookmarks, "folder_tags")
            disease_options = ["전체"] + sorted(disease_index.keys())
            tech_options = ["전체"] + sorted(tech_index.keys())
            folder_options = ["전체"] + sorted(folder_index.keys())
            with d_col:
                disease_filter = st.selectbox("질환 태그", disease_options, key="bookmark_disease_filter")
            with t_col:
                tech_filter = st.selectbox("기술 태그", tech_options, key="bookmark_tech_filter")
            with f_col:
                folder_filter = st.selectbox("폴더", folder_options, key="bookmark_folder_filter")
            with s_col:
                sort_mode = st.selectbox("정렬", ["최신 저장순","연도순","인용수순","제목순","중요도순"], key="bookmark_sort_mode")
            with r_col:
                status_filter = st.selectbox("읽기 상태", ["전체","읽기 전","읽는 중","읽음","핵심 논문","재확인 필요"], key="bookmark_status_filter")

            st.session_state["highlight_keywords"] = st.text_input("강조 키워드(쉼표로 구분)", value=st.session_state.get("highlight_keywords",""), key="highlight_keywords_input")

            def bookmark_matches(item):
                blob = " ".join([
                    item.get("title", ""), item.get("abstract", ""), item.get("summary_en", ""), item.get("summary_ko", ""),
                    ", ".join(item.get("authors", [])) if item.get("authors") else "", item.get("notes","")
                ]).lower()
                if bookmark_query and bookmark_query.lower() not in blob:
                    return False
                if disease_filter != "전체" and disease_filter not in (item.get("disease_tags", []) or []):
                    return False
                if tech_filter != "전체" and tech_filter not in (item.get("tech_tags", []) or []):
                    return False
                if folder_filter != "전체" and folder_filter not in (item.get("folder_tags", []) or []):
                    return False
                if status_filter != "전체" and item.get("reading_status","읽기 전") != status_filter:
                    return False
                return True

            filtered = [b for b in bookmarks if bookmark_matches(b)]
            filtered = sorted(filtered, key=lambda x: bookmark_sort_key(x, sort_mode), reverse=(sort_mode in ["연도순","인용수순","중요도순","최신 저장순"]))

            st.markdown("### 📦 북마크 내보내기")
            e1, e2 = st.columns(2)
            with e1:
                csv_bytes = make_bookmark_export_bytes(filtered, "csv")
                st.download_button("CSV 다운로드", data=csv_bytes, file_name="bookmarks.csv", mime="text/csv")
            with e2:
                xlsx_bytes = make_bookmark_export_bytes(filtered, "xlsx")
                st.download_button("Excel 다운로드", data=xlsx_bytes, file_name="bookmarks.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            st.markdown("### 🧰 북마크 일괄 폴더 이동")
            selected_ids = []
            bulk_cols = st.columns([2,2,1])
            with bulk_cols[0]:
                bulk_folder_targets = st.multiselect("적용할 폴더", options=folders, key="bulk_folder_targets")
            with bulk_cols[1]:
                bulk_mode = st.selectbox("적용 방식", ["replace","add","remove"], format_func=lambda x: {"replace":"선택 폴더로 교체","add":"선택 폴더 추가","remove":"선택 폴더 제거"}[x], key="bulk_folder_mode")
            with bulk_cols[2]:
                st.caption("아래 북마크 목록에서 체크한 항목에 적용됩니다.")

            list_tabs = st.tabs(["전체 목록", "질환 태그별", "기술 태그별", "폴더별 보기"])
            with list_tabs[0]:
                for idx, item in enumerate(filtered):
                    bid = item.get("bookmark_id", f"b{idx}")
                    if st.checkbox(f"일괄 선택", key=f"bulk_select_all_{bid}"):
                        selected_ids.append(bid)
                    render_bookmark_card(item, idx, username, "all")
            with list_tabs[1]:
                disease_groups = build_tag_index(filtered, "disease_tags")
                for tag_name, items in sorted(disease_groups.items()):
                    st.markdown(f"#### {tag_name}")
                    for idx, item in enumerate(items):
                        bid = item.get("bookmark_id", f"d{idx}")
                        if st.checkbox(f"일괄 선택", key=f"bulk_select_d_{tag_name}_{bid}"):
                            selected_ids.append(bid)
                        render_bookmark_card(item, idx, username, f"disease_{tag_name}")
            with list_tabs[2]:
                tech_groups = build_tag_index(filtered, "tech_tags")
                for tag_name, items in sorted(tech_groups.items()):
                    st.markdown(f"#### {tag_name}")
                    for idx, item in enumerate(items):
                        bid = item.get("bookmark_id", f"t{idx}")
                        if st.checkbox(f"일괄 선택", key=f"bulk_select_t_{tag_name}_{bid}"):
                            selected_ids.append(bid)
                        render_bookmark_card(item, idx, username, f"tech_{tag_name}")
            with list_tabs[3]:
                folder_groups = build_tag_index(filtered, "folder_tags")
                for tag_name, items in sorted(folder_groups.items()):
                    st.markdown(f"#### {tag_name}")
                    for idx, item in enumerate(items):
                        bid = item.get("bookmark_id", f"f{idx}")
                        if st.checkbox(f"일괄 선택", key=f"bulk_select_f_{tag_name}_{bid}"):
                            selected_ids.append(bid)
                        render_bookmark_card(item, idx, username, f"folder_{tag_name}")

            selected_ids = list(dict.fromkeys(selected_ids))
            if st.button("선택 북마크에 폴더 일괄 적용", key="bulk_apply_folder_btn"):
                if not selected_ids:
                    st.warning("먼저 체크박스로 북마크를 선택해주세요.")
                else:
                    ok, msg = bookmark_manager.bulk_update_folders(username, selected_ids, bulk_mode, bulk_folder_targets)
                    st.success(msg) if ok else st.error(msg)
                    if ok: st.rerun()

    with tabs[4]:
        st.markdown("### 📊 북마크 통계 대시보드")
        if not bookmarks:
            st.info("북마크가 없어 통계를 표시할 수 없습니다.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("총 북마크", len(bookmarks))
            c2.metric("핵심 논문", sum(1 for b in bookmarks if b.get("reading_status") == "핵심 논문"))
            c3.metric("평균 인용수", round(sum(b.get("citation_count",0) for b in bookmarks)/max(len(bookmarks),1),1))
            c4.metric("폴더 수", len(folders))

            def count_tags(key):
                cnt = Counter()
                for b in bookmarks:
                    vals = b.get(key, []) or ["태그 없음"]
                    for v in vals:
                        cnt[v] += 1
                return cnt

            def normalize_year(y):
                y = str(y).strip()
                return y if y.isdigit() else "미상"

            source_df = pd.DataFrame(Counter([b.get("source", "기타") for b in bookmarks]).items(), columns=["source", "count"]).sort_values("count", ascending=False)
            disease_df = pd.DataFrame(count_tags("disease_tags").items(), columns=["tag", "count"]).sort_values("count", ascending=False).head(15)
            tech_df = pd.DataFrame(count_tags("tech_tags").items(), columns=["tag", "count"]).sort_values("count", ascending=False).head(15)
            year_counter = Counter([normalize_year(b.get("year", "미상")) for b in bookmarks])
            year_df = pd.DataFrame(year_counter.items(), columns=["year", "count"])
            if not year_df.empty:
                year_df["year_sort"] = year_df["year"].apply(lambda x: int(x) if str(x).isdigit() else 999999)
                year_df = year_df.sort_values(["year_sort", "year"]).drop(columns=["year_sort"])
            top_cited = pd.DataFrame(sorted(bookmarks, key=lambda x: x.get("citation_count", 0), reverse=True)[:10])[["title", "source", "year", "citation_count"]]

            # 읽기 상태 / 폴더 통계
            status_df = pd.DataFrame(Counter([b.get("reading_status", "읽기 전") for b in bookmarks]).items(), columns=["status", "count"]).sort_values("count", ascending=False)
            folder_counter = Counter()
            for b in bookmarks:
                folder_vals = b.get("folder_tags", []) or ["폴더 없음"]
                for f in folder_vals:
                    folder_counter[f] += 1
            folder_df = pd.DataFrame(folder_counter.items(), columns=["folder", "count"]).sort_values("count", ascending=False).head(15)

            st.markdown("#### 📈 인터랙티브 Plotly 그래프")
            st.caption("마우스를 올리면 값이 보이고, 범례 클릭/확대/축소도 가능합니다.")

            row1 = st.columns(2)
            with row1[0]:
                st.markdown("**출처별 북마크 수**")
                if not source_df.empty:
                    fig = px.bar(source_df, x="source", y="count", text="count", title="출처별 북마크 수")
                    fig.update_layout(xaxis_title="출처", yaxis_title="북마크 수")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.caption("표시할 데이터가 없습니다.")
            with row1[1]:
                st.markdown("**연도별 북마크 수**")
                if not year_df.empty:
                    year_plot_df = year_df.copy()
                    year_plot_df["year_label"] = year_plot_df["year"].astype(str)
                    fig = px.line(year_plot_df, x="year_label", y="count", markers=True, title="연도별 북마크 수")
                    fig.update_layout(xaxis_title="연도", yaxis_title="북마크 수")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.caption("표시할 데이터가 없습니다.")

            row2 = st.columns(2)
            with row2[0]:
                st.markdown("**질환 태그 TOP 15**")
                if not disease_df.empty:
                    fig = px.bar(disease_df.sort_values("count", ascending=True), x="count", y="tag", orientation="h", text="count", title="질환 태그 TOP 15")
                    fig.update_layout(xaxis_title="북마크 수", yaxis_title="질환 태그")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.caption("표시할 데이터가 없습니다.")
            with row2[1]:
                st.markdown("**기술 태그 TOP 15**")
                if not tech_df.empty:
                    fig = px.bar(tech_df.sort_values("count", ascending=True), x="count", y="tag", orientation="h", text="count", title="기술 태그 TOP 15")
                    fig.update_layout(xaxis_title="북마크 수", yaxis_title="기술 태그")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.caption("표시할 데이터가 없습니다.")

            row3 = st.columns(2)
            with row3[0]:
                st.markdown("**읽기 상태 분포**")
                if not status_df.empty:
                    fig = px.pie(status_df, names="status", values="count", hole=0.35, title="읽기 상태 분포")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.caption("표시할 데이터가 없습니다.")
            with row3[1]:
                st.markdown("**폴더별 북마크 수 TOP 15**")
                if not folder_df.empty:
                    fig = px.bar(folder_df.sort_values("count", ascending=True), x="count", y="folder", orientation="h", text="count", title="폴더별 북마크 수 TOP 15")
                    fig.update_layout(xaxis_title="북마크 수", yaxis_title="폴더")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.caption("표시할 데이터가 없습니다.")

            st.markdown("#### 📋 원본 통계 표")
            stat_tabs = st.tabs(["출처", "질환 태그", "기술 태그", "연도", "읽기 상태", "폴더", "인용수 TOP10"])
            with stat_tabs[0]:
                st.dataframe(source_df, use_container_width=True)
            with stat_tabs[1]:
                st.dataframe(disease_df, use_container_width=True)
            with stat_tabs[2]:
                st.dataframe(tech_df, use_container_width=True)
            with stat_tabs[3]:
                st.dataframe(year_df, use_container_width=True)
            with stat_tabs[4]:
                st.dataframe(status_df, use_container_width=True)
            with stat_tabs[5]:
                st.dataframe(folder_df, use_container_width=True)
            with stat_tabs[6]:
                st.markdown("#### 인용수 상위 논문 TOP 10")
                st.dataframe(top_cited, use_container_width=True)

    with tabs[5]:
        st.markdown("### 🆚 논문 비교")
        if not bookmarks:
            st.info("비교할 북마크가 없습니다.")
        else:
            options = {f"{b.get('title','제목 없음')} ({b.get('source','')}, {b.get('year','')})": b.get("bookmark_id") for b in bookmarks}
            selected_labels = st.multiselect("비교할 논문 2~3개 선택", list(options.keys()), max_selections=3, key="compare_papers_select")
            if len(selected_labels) < 2:
                st.caption("2개 이상 선택하면 비교표가 표시됩니다.")
            else:
                selected = [next(b for b in bookmarks if b.get("bookmark_id")==options[label]) for label in selected_labels]
                rows=[]
                fields=[("제목","title"),("출처","source"),("연도","year"),("저널/학회","journal"),("인용수","citation_count"),("질환 태그","disease_tags"),("기술 태그","tech_tags"),("읽기 상태","reading_status"),("중요도","importance"),("DOI","doi"),("PMID","pmid"),("arXiv ID","arxiv_id"),("URL","url"),("PDF","pdf_url"),("영문 요약","summary_en"),("한국어 요약","summary_ko"),("메모","notes")]
                for label,key in fields:
                    row={"항목":label}
                    for paper in selected:
                        val=paper.get(key,"")
                        if isinstance(val,list): val=", ".join(val)
                        row[paper.get("title","제목 없음")[:50]]=val
                    rows.append(row)
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

def render_main_app():
    username = st.session_state.current_user
    user = user_manager.get_user(username)

    st.title("📚 무료 논문 검색 및 요약 시스템")
    st.caption("PubMed / arXiv / Semantic Scholar / Google Scholar 통합 검색 + 회원 기능 + 한국어 검색 자동번역 + 바이오 전문용어 번역 보정 + 자동 태그 + 인용수 보강")

    top1, top2 = st.columns([4, 1])

    with top1:
        st.write(f"**현재 로그인:** {user.get('name', username)} ({username})")

    with top2:
        if st.button("로그아웃"):
            st.session_state.logged_in = False
            st.session_state.current_user = None
            st.session_state.papers = []
            st.session_state.last_query = ""
            st.session_state.translated_query = ""
            st.session_state.generated_summaries = {}
            st.rerun()

    st.sidebar.header("회원 메뉴")
    page = st.sidebar.radio("이동", ["논문 검색", "마이페이지", "화학공학 분석"])

    if page == "마이페이지":
        render_mypage(username)
        return

    if page == "화학공학 분석":
        if render_chemeng_dashboard is None:
            st.error("화학공학 분석 모듈을 불러오지 못했습니다. src 폴더와 requirements.txt를 확인해주세요.")
        else:
            user_bookmarks = bookmark_manager.get_user_bookmarks(username)
            render_chemeng_dashboard(
                papers=st.session_state.get("papers", []),
                bookmarks=user_bookmarks,
            )
        return

    # -------------------------
    # 검색 옵션
    # -------------------------
    st.sidebar.header("검색 옵션")

    use_pubmed = st.sidebar.checkbox("PubMed", value=True)
    use_arxiv = st.sidebar.checkbox("arXiv", value=True)
    use_semantic = st.sidebar.checkbox("Semantic Scholar", value=True)
    use_scholar = st.sidebar.checkbox("Google Scholar", value=True)

    if use_scholar and not SCHOLARLY_AVAILABLE:
        st.sidebar.warning("Google Scholar용 scholarly 라이브러리가 설치되지 않았습니다. `pip install scholarly` 후 다시 실행하세요.")

    with st.sidebar.expander("학교 Wi-Fi 진단", expanded=False):
        st.caption("학교망에서 PubMed, arXiv, Semantic Scholar, Google 서비스 접속 가능 여부를 확인합니다.")
        if st.button("네트워크 진단 실행", key="network_diag_btn"):
            diag = run_network_diagnostics()
            for service, (ok, msg) in diag.items():
                if ok:
                    st.success(f"{service}: 정상 ({msg})")
                else:
                    st.error(f"{service}: 실패 ({msg})")
            if st.session_state.get("network_errors"):
                with st.expander("최근 네트워크 오류 로그"):
                    for line in st.session_state.network_errors[-10:]:
                        st.code(line, language=None)

    sort_option_label = st.sidebar.selectbox(
        "정렬 기준",
        ["관련도순", "최신순", "인용수 높은 순"]
    )

    summary_line_option = st.sidebar.selectbox(
        "요약 줄 수",
        ["1줄", "3줄", "5줄"]
    )

    max_results = st.sidebar.slider("소스별 최대 검색 개수", 3, 20, 10)

    sort_map = {
        "관련도순": "relevance",
        "최신순": "latest",
        "인용수 높은 순": "citation"
    }

    line_map = {
        "1줄": 1,
        "3줄": 3,
        "5줄": 5
    }

    query = st.text_input(
        "논문 주제를 입력하세요",
        placeholder="예: 유방암 면역치료 / single-cell RNA sequencing / Alzheimer's disease"
    )

    st.markdown("### 🔗 논문 링크 직접 추가")
    st.caption("논문/초록 페이지 URL 또는 PDF 링크를 붙여 넣으면 검색 결과에 추가하고, 가능하면 논문 페이지의 그림/도표 이미지도 함께 가져옵니다.")
    direct_url = st.text_input("논문 링크(URL)", key="direct_paper_url", placeholder="https://...")
    dc1, dc2 = st.columns(2)
    with dc1:
        if st.button("링크 논문 불러오기", key="load_direct_paper_btn"):
            if not direct_url.strip():
                st.warning("먼저 논문 링크를 입력해주세요.")
            else:
                with st.spinner("링크에서 논문 정보를 읽는 중입니다..."):
                    paper, msg = fetch_paper_from_url(direct_url)
                if paper is None:
                    st.error(msg)
                else:
                    current = st.session_state.get("papers", [])
                    current.append(paper)
                    current = deduplicate_papers(current)
                    st.session_state.papers = current
                    st.success(msg + " 검색 결과 목록에 추가했습니다.")
                    st.rerun()
    with dc2:
        if st.button("링크 논문 바로 북마크", key="bookmark_direct_paper_btn"):
            if not direct_url.strip():
                st.warning("먼저 논문 링크를 입력해주세요.")
            else:
                with st.spinner("링크에서 논문 정보를 읽는 중입니다..."):
                    paper, msg = fetch_paper_from_url(direct_url)
                if paper is None:
                    st.error(msg)
                else:
                    disease_tags, tech_tags = detect_tags_from_text(paper.title, paper.abstract, paper.summary_en, paper.summary_ko)
                    paper.disease_tags = disease_tags
                    paper.tech_tags = tech_tags
                    ok, save_msg = bookmark_manager.add_bookmark(username, paper)
                    st.success(save_msg) if ok else st.warning(save_msg)

    if st.button("검색", key="search_btn"):
        if not query.strip():
            st.warning("검색어를 입력해주세요.")
        else:
            translated_query = translate_query_if_needed(query)

            with st.spinner("논문 검색 및 인용수 보강 중입니다..."):
                papers = search_service.search_all(
                    query=translated_query,
                    use_pubmed=use_pubmed,
                    use_arxiv=use_arxiv,
                    use_semantic=use_semantic,
                    use_scholar=use_scholar,
                    max_results_per_source=max_results
                )

            papers = sort_papers(papers, sort_map[sort_option_label])

            st.session_state.papers = papers
            st.session_state.last_query = query
            st.session_state.translated_query = translated_query
            st.session_state.generated_summaries = {}

    papers = st.session_state.papers
    original_query = st.session_state.last_query
    translated_query = st.session_state.translated_query

    if original_query:
        st.markdown("### 검색어 정보")
        st.write(f"**입력 검색어:** {original_query}")
        st.write(f"**실제 검색에 사용된 검색어:** {translated_query if translated_query else original_query}")

    if papers:
        papers = sort_papers(papers, sort_map[sort_option_label])
        st.success(f"총 {len(papers)}개의 논문을 찾았습니다.")

        for idx, paper in enumerate(papers, start=1):
            with st.container():
                st.markdown("---")
                st.subheader(f"{idx}. {paper.title}")

                col1, col2, col3 = st.columns([1.6, 1, 1])

                with col1:
                    st.write(f"**출처:** {paper.source}")
                    if paper.authors:
                        author_text = ", ".join(paper.authors[:6])
                        if len(paper.authors) > 6:
                            author_text += " 외"
                    else:
                        author_text = "정보 없음"
                    st.write(f"**저자:** {author_text}")

                with col2:
                    st.write(f"**연도:** {paper.year if paper.year else '정보 없음'}")
                    st.write(f"**저널/학회:** {paper.journal if paper.journal else '정보 없음'}")

                with col3:
                    st.write(f"**인용수:** {paper.citation_count}")
                    if paper.url:
                        st.markdown(f"[원문 링크 바로가기]({paper.url})")

                with st.expander("초록 / 상세 정보 보기"):
                    if paper.abstract and paper.abstract.strip():
                        st.write("**초록**")
                        st.write(paper.abstract)
                    else:
                        st.write("초록 정보가 없습니다.")
                    if paper.doi:
                        st.write(f"**DOI:** {paper.doi}")
                    if paper.pmid:
                        st.write(f"**PMID:** {paper.pmid}")
                    if paper.arxiv_id:
                        st.write(f"**arXiv ID:** {paper.arxiv_id}")
                    if paper.pdf_url:
                        st.markdown(f"[PDF 바로가기]({paper.pdf_url})")
                    if paper.image_urls:
                        st.write("**논문 페이지 시각자료 미리보기**")
                        st.image(paper.image_urls[:6], use_container_width=True)

                b1, b2 = st.columns([1, 1])
                summary_key = f"summary_{paper.unique_key()}"

                with b1:
                    if st.button(f"요약 생성 + 한국어 번역 ({summary_line_option})", key=f"summary_btn_{paper.unique_key()}"):
                        with st.spinner("요약 및 번역 생성 중입니다..."):
                            summary_dict = summary_service.summarize(
                                title=paper.title,
                                abstract=paper.abstract,
                                line_count=line_map[summary_line_option],
                                query=translated_query if translated_query else original_query
                            )
                            st.session_state.generated_summaries[summary_key] = summary_dict

                with b2:
                    if st.button("북마크 저장", key=f"bookmark_btn_{paper.unique_key()}"):
                        if summary_key in st.session_state.generated_summaries:
                            paper.summary_en = st.session_state.generated_summaries[summary_key].get("en", "")
                            paper.summary_ko = st.session_state.generated_summaries[summary_key].get("ko", "")

                        disease_tags, tech_tags = detect_tags_from_text(
                            paper.title, paper.abstract, paper.summary_en, paper.summary_ko
                        )
                        paper.disease_tags = disease_tags
                        paper.tech_tags = tech_tags

                        ok, msg = bookmark_manager.add_bookmark(username, paper)
                        if ok:
                            st.success(msg)
                        else:
                            st.warning(msg)

                if summary_key in st.session_state.generated_summaries:
                    summary_data = st.session_state.generated_summaries[summary_key]

                    preview_disease_tags, preview_tech_tags = detect_tags_from_text(
                        paper.title, paper.abstract,
                        summary_data.get("en", ""),
                        summary_data.get("ko", "")
                    )

                    st.write("**영문 요약**")
                    st.info(summary_data.get("en", ""))

                    st.write("**한국어 의역 요약**")
                    st.success(summary_data.get("ko", ""))

                    st.write(f"**예상 질환 태그:** {', '.join(preview_disease_tags) if preview_disease_tags else '없음'}")
                    st.write(f"**예상 기술 태그:** {', '.join(preview_tech_tags) if preview_tech_tags else '없음'}")

    else:
        st.info("검색어를 입력하고 '검색' 버튼을 누르면 결과가 여기에 표시됩니다.")

    st.markdown("---")
    st.markdown("### 현재 포함된 기능")
    st.markdown(
        """
- 회원가입 / 로그인 / 로그아웃
- 회원 정보 저장(JSON)
- 회원별 마이페이지
- 회원정보 수정 / 비밀번호 변경
- 회원별 논문 북마크 저장
- PubMed / arXiv / Semantic Scholar / Google Scholar 통합 검색
- 한국어 검색어 자동 영어 번역 검색
- 최신순 / 인용수순 정렬
- 1줄 / 3줄 / 5줄 요약
- 영문 요약 + 한국어 의역 요약 표시
- 바이오/제약 전문용어 자동 보정
- 북마크 저장 시 질환 태그 / 기술 태그 자동 분류
- 마이페이지에서 질환별 / 기술별 북마크 조회
- 북마크 제목/초록/저자 검색 + 질환/기술 태그 필터
- 북마크 CSV / Excel 내보내기
- 검색 결과/북마크 상세보기(초록, DOI, PMID, arXiv, PDF 링크)
- 논문 링크(URL/PDF) 직접 추가 및 바로 북마크 저장
- 논문 페이지에서 그림/도표 등 시각자료 수집 후 브라우저 내 미리보기
- 북마크 상세에서 PDF 브라우저 내 열람
- Semantic Scholar 기반 인용수 자동 보강
"""
    )

# =========================================================
# 앱 실행
# =========================================================
if not st.session_state.logged_in:
    render_auth_page()
else:
    render_main_app()