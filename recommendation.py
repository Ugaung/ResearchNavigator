from __future__ import annotations
from typing import Any, Iterable, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def _get(item: Any, key: str, default=""):
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)

def paper_label(item: Any) -> str:
    return str(_get(item, "title", "제목 없음") or "제목 없음")

def paper_text(item: Any) -> str:
    return " ".join(str(_get(item, k, "") or "") for k in ["title", "abstract", "summary_en", "summary_ko"])

def recommend_similar_papers(items: Iterable[Any], selected_index: int = 0, top_n: int = 5) -> List[Tuple[int, Any, float]]:
    papers = list(items)
    if len(papers) < 2:
        return []
    selected_index = max(0, min(selected_index, len(papers) - 1))
    docs = [paper_text(p) for p in papers]
    try:
        vec = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
        X = vec.fit_transform(docs)
        sims = cosine_similarity(X[selected_index], X).ravel()
    except Exception:
        return []
    ranked = []
    for i, score in enumerate(sims):
        if i != selected_index:
            ranked.append((i, papers[i], float(score)))
    ranked.sort(key=lambda x: x[2], reverse=True)
    return ranked[:top_n]
