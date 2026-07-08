from __future__ import annotations
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple

CHEMENG_TAG_RULES: Dict[str, List[str]] = {
    "배터리/전지": ["battery", "batteries", "lithium", "li-ion", "sodium-ion", "solid-state battery", "electrode", "cathode", "anode", "electrolyte", "배터리", "전지", "리튬", "전극", "전해질"],
    "촉매/반응공학": ["catalyst", "catalysis", "photocatalyst", "electrocatalyst", "reaction engineering", "kinetics", "촉매", "반응공학", "반응속도"],
    "수소/연료전지": ["hydrogen", "water splitting", "fuel cell", "pemfc", "electrolysis", "수소", "연료전지", "물분해", "전기분해"],
    "탄소포집/CCUS": ["carbon capture", "ccus", "ccs", "co2 capture", "carbon dioxide capture", "direct air capture", "탄소포집", "이산화탄소 포집"],
    "분리공정/막": ["membrane", "separation", "adsorption", "distillation", "filtration", "desalination", "분리공정", "막", "흡착", "증류", "여과"],
    "고분자/소재": ["polymer", "composite", "nanomaterial", "hydrogel", "porous material", "mof", "cof", "고분자", "복합재", "나노소재", "하이드로젤"],
    "반도체/공정": ["semiconductor", "etching", "deposition", "thin film", "photoresist", "process control", "반도체", "식각", "증착", "박막", "포토레지스트"],
    "바이오공정": ["bioprocess", "fermentation", "bioreactor", "enzyme", "metabolic engineering", "발효", "생물반응기", "효소", "대사공학"],
    "환경/수처리": ["wastewater", "water treatment", "pollutant", "remediation", "환경", "수처리", "폐수", "오염물질"],
    "공정최적화/AI": ["process optimization", "machine learning", "deep learning", "digital twin", "process control", "머신러닝", "딥러닝", "공정최적화", "디지털 트윈"],
}

def _get(item: Any, key: str, default=""):
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)

def paper_text(item: Any) -> str:
    parts = [_get(item, "title", ""), _get(item, "abstract", ""), _get(item, "summary_en", ""), _get(item, "summary_ko", "")]
    return " ".join(str(p or "") for p in parts).lower()

def detect_chemeng_tags(item: Any) -> List[str]:
    text = paper_text(item)
    tags = []
    for tag, keywords in CHEMENG_TAG_RULES.items():
        if any(kw.lower() in text for kw in keywords):
            tags.append(tag)
    return sorted(set(tags))

def tag_counts(items: Iterable[Any]) -> Counter:
    c = Counter()
    for item in items:
        tags = detect_chemeng_tags(item)
        if tags:
            c.update(tags)
    return c

def top_tagged_papers(items: Iterable[Any], limit: int = 20) -> List[Tuple[Any, List[str]]]:
    rows = []
    for item in items:
        tags = detect_chemeng_tags(item)
        if tags:
            rows.append((item, tags))
    return rows[:limit]
