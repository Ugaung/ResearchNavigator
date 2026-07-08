from __future__ import annotations
from typing import Any, Iterable, List
import pandas as pd
import plotly.express as px
import streamlit as st
from .chemeng_tags import tag_counts, top_tagged_papers, detect_chemeng_tags
from .trend_analysis import make_year_trend_chart
from .keyword_network import keyword_frequency, make_keyword_network_figure
from .recommendation import recommend_similar_papers, paper_label
from .pdf_analyzer import analyze_pdf

def _get(item: Any, key: str, default=""):
    if isinstance(item, dict): return item.get(key, default)
    return getattr(item, key, default)

def _merge_items(papers: Iterable[Any], bookmarks: Iterable[Any]) -> List[Any]:
    combined = []
    seen = set()
    for item in list(papers or []) + list(bookmarks or []):
        title = str(_get(item, "title", "") or "").strip().lower()
        doi = str(_get(item, "doi", "") or "").strip().lower()
        key = doi or title
        if key and key not in seen:
            combined.append(item); seen.add(key)
    return combined

def render_chemeng_dashboard(papers=None, bookmarks=None):
    st.title("⚗️ 화학공학 연구 분석 대시보드")
    st.caption("검색 결과와 북마크를 활용해 화학공학 태그, 연구 트렌드, 키워드, 유사 논문, PDF 분석을 수행합니다.")
    items = _merge_items(papers or [], bookmarks or [])
    st.info(f"분석 대상 논문 수: {len(items)}개  ·  검색 결과와 북마크를 합쳐 중복 제목/DOI를 제거했습니다.")
    tabs = st.tabs(["화학공학 태그", "연도별 트렌드", "키워드 분석", "유사 논문 추천", "PDF 분석"])

    with tabs[0]:
        st.subheader("화학공학 분야 자동 태그")
        if not items:
            st.warning("먼저 논문 검색을 실행하거나 북마크를 저장해주세요.")
        else:
            counts = tag_counts(items)
            if counts:
                df = pd.DataFrame(counts.most_common(), columns=["tag", "count"])
                st.plotly_chart(px.bar(df, x="tag", y="count", title="화학공학 태그 분포"), use_container_width=True)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("화학공학 관련 태그가 감지되지 않았습니다. 배터리, 촉매, 수소, CCUS 등의 키워드로 검색해보세요.")
            st.markdown("#### 태그가 감지된 논문")
            for item, tags in top_tagged_papers(items, limit=20):
                st.write(f"**{_get(item, 'title', '제목 없음')}**")
                st.caption("태그: " + ", ".join(tags))

    with tabs[1]:
        st.subheader("연도별 논문 트렌드")
        fig, df = make_year_trend_chart(items)
        if fig is None:
            st.warning("연도 정보가 있는 논문이 부족합니다.")
        else:
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df, use_container_width=True)

    with tabs[2]:
        st.subheader("키워드 빈도 및 네트워크")
        if not items:
            st.warning("분석할 논문이 없습니다.")
        else:
            freq = keyword_frequency(items, top_n=30)
            if not freq.empty:
                st.plotly_chart(px.bar(freq.head(20), x="keyword", y="count", title="상위 키워드"), use_container_width=True)
                st.dataframe(freq, use_container_width=True)
            fig, msg = make_keyword_network_figure(items)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            elif msg:
                st.info(msg)

    with tabs[3]:
        st.subheader("TF-IDF 기반 유사 논문 추천")
        if len(items) < 2:
            st.warning("유사 논문 추천에는 최소 2개 이상의 논문이 필요합니다.")
        else:
            labels = [paper_label(x) for x in items]
            selected = st.selectbox("기준 논문 선택", list(range(len(labels))), format_func=lambda i: labels[i][:120])
            recs = recommend_similar_papers(items, selected_index=selected, top_n=5)
            if not recs:
                st.info("추천 결과를 만들지 못했습니다. 초록이 포함된 논문이 더 많으면 정확도가 올라갑니다.")
            for idx, item, score in recs:
                st.write(f"**유사도 {score:.3f} · {paper_label(item)}**")
                abstract = str(_get(item, "abstract", "") or "")
                if abstract:
                    st.caption(abstract[:350] + ("..." if len(abstract) > 350 else ""))

    with tabs[4]:
        st.subheader("PDF 업로드 분석")
        uploaded = st.file_uploader("논문 PDF 업로드", type=["pdf"])
        if uploaded is not None:
            with st.spinner("PDF 텍스트 추출 및 요약 중..."):
                result = analyze_pdf(uploaded)
            if result.get("error"):
                st.error(result["error"])
            else:
                st.write(f"추출된 텍스트 길이: {result.get('text_length', 0)}자")
                st.markdown("#### 핵심 키워드")
                st.write(", ".join(result.get("keywords", [])))
                st.markdown("#### 자동 요약")
                st.info(result.get("summary", ""))
                with st.expander("추출 텍스트 미리보기"):
                    st.text(result.get("preview", ""))
