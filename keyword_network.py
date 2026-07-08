from __future__ import annotations
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Iterable, List, Tuple
import re
import pandas as pd
import plotly.graph_objects as go
try:
    import networkx as nx
except Exception:
    nx = None

STOPWORDS = set("""
the and or of to in for on with by a an is are was were be this that these those we our they their it as at from can may using used use show shows shown study results result methods method background objective objectives conclusion conclusions patients patient analysis into within between among via based effect effects data paper research
그리고 또한 대한 위한 에서 으로 했다 한다 있는 있다 연구 결과 방법 논문 분석 목적 결론 통해 관련 대한
""".split())

def _get(item: Any, key: str, default=""):
    if isinstance(item, dict): return item.get(key, default)
    return getattr(item, key, default)

def _text(item: Any) -> str:
    return " ".join(str(_get(item, k, "") or "") for k in ["title", "abstract", "summary_en", "summary_ko"])

def extract_keywords(text: str, max_keywords: int = 12) -> List[str]:
    text = text.lower()
    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}|[가-힣]{2,}", text)
    filtered = [t for t in tokens if t not in STOPWORDS and len(t) >= 3]
    counts = Counter(filtered)
    return [w for w, _ in counts.most_common(max_keywords)]

def keyword_frequency(items: Iterable[Any], top_n: int = 30) -> pd.DataFrame:
    c = Counter()
    for item in items:
        c.update(extract_keywords(_text(item), max_keywords=20))
    return pd.DataFrame(c.most_common(top_n), columns=["keyword", "count"])

def build_network_edges(items: Iterable[Any], per_paper_keywords: int = 8) -> pd.DataFrame:
    edges = Counter()
    for item in items:
        kws = extract_keywords(_text(item), max_keywords=per_paper_keywords)
        for a, b in combinations(sorted(set(kws)), 2):
            edges[(a, b)] += 1
    return pd.DataFrame([(a, b, w) for (a, b), w in edges.items()], columns=["source", "target", "weight"])

def make_keyword_network_figure(items: Iterable[Any], max_edges: int = 60):
    if nx is None:
        return None, "networkx가 설치되어 있지 않습니다. requirements.txt 설치를 확인해주세요."
    edges_df = build_network_edges(items)
    if edges_df.empty:
        return None, "키워드 네트워크를 만들 충분한 텍스트가 없습니다."
    edges_df = edges_df.sort_values("weight", ascending=False).head(max_edges)
    G = nx.Graph()
    for _, row in edges_df.iterrows():
        G.add_edge(row["source"], row["target"], weight=float(row["weight"]))
    if len(G.nodes) == 0:
        return None, "네트워크 노드가 없습니다."
    pos = nx.spring_layout(G, seed=42, k=0.9)
    edge_x, edge_y = [], []
    for a, b in G.edges():
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.6), hoverinfo='none', mode='lines')
    node_x, node_y, text, sizes = [], [], [], []
    degrees = dict(G.degree())
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x); node_y.append(y); text.append(node); sizes.append(10 + degrees.get(node, 1) * 3)
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=text, textposition="top center", marker=dict(size=sizes), hoverinfo='text')
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(title="키워드 동시출현 네트워크", showlegend=False, margin=dict(l=20, r=20, t=50, b=20), xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig, ""
