from __future__ import annotations
from typing import Any, Iterable
import pandas as pd
import plotly.express as px

def _get(item: Any, key: str, default=None):
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)

def year_dataframe(items: Iterable[Any]) -> pd.DataFrame:
    years = []
    for item in items:
        y = _get(item, "year")
        try:
            y = int(y) if y else None
        except Exception:
            y = None
        if y and 1900 <= y <= 2100:
            years.append(y)
    if not years:
        return pd.DataFrame(columns=["year", "count"])
    df = pd.Series(years).value_counts().sort_index().reset_index()
    df.columns = ["year", "count"]
    return df

def make_year_trend_chart(items: Iterable[Any]):
    df = year_dataframe(items)
    if df.empty:
        return None, df
    fig = px.line(df, x="year", y="count", markers=True, title="연도별 논문 수 트렌드")
    fig.update_layout(xaxis_title="연도", yaxis_title="논문 수")
    return fig, df
