from __future__ import annotations
from collections import Counter
import re
from typing import Dict, List
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

STOPWORDS = set("""
the and or of to in for on with by a an is are was were be this that these those we our they their it as at from can may using used use show shows shown study results result methods method background objective objectives conclusion conclusions patients patient analysis
그리고 또한 대한 위한 에서 으로 했다 한다 있는 있다 연구 결과 방법 논문 분석 목적 결론 통해 관련
""".split())

def extract_pdf_text(uploaded_file, max_pages: int = 8) -> str:
    if fitz is None:
        return ""
    data = uploaded_file.read()
    doc = fitz.open(stream=data, filetype="pdf")
    texts = []
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        texts.append(page.get_text("text"))
    return "\n".join(texts).strip()

def extract_keywords(text: str, top_n: int = 20) -> List[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}|[가-힣]{2,}", text.lower())
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) >= 3]
    return [w for w, _ in Counter(tokens).most_common(top_n)]

def simple_summary(text: str, max_sentences: int = 5) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return "PDF에서 텍스트를 추출하지 못했습니다. 스캔 이미지 PDF일 수 있습니다."
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 50]
    if not sentences:
        return text[:1000]
    keywords = set(extract_keywords(text, top_n=20))
    scored = []
    for i, s in enumerate(sentences[:80]):
        score = sum(1 for kw in keywords if kw in s.lower())
        if re.search(r"\d", s): score += 0.5
        if i == 0: score += 0.5
        scored.append((score, i, s))
    top = sorted(scored, reverse=True)[:max_sentences]
    top = sorted(top, key=lambda x: x[1])
    return "\n".join(s for _, _, s in top)

def analyze_pdf(uploaded_file) -> Dict[str, object]:
    if fitz is None:
        return {"error": "PyMuPDF가 설치되어 있지 않습니다. pip install PyMuPDF 후 다시 실행하세요."}
    text = extract_pdf_text(uploaded_file)
    return {"text_length": len(text), "keywords": extract_keywords(text), "summary": simple_summary(text), "preview": text[:2000]}
