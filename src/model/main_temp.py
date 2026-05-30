from config.paths import *
"""
급식 식단 교환 추천 분석
────────────────────────
--base   : 기반 데이터 (빈도 baseline 구축용, 여러 달)
--target : 분석할 데이터 (교환 추천 대상, 1달치)

swap_score 공식
───────────────
  swap_score = (α × menu_diversity + β × nutrient_sim)
               × log(1 + rep_gap)
               × date_penalty

  date_penalty = 1 / log(2 + day_gap)
    → 날짜가 가까울수록 인접 반복 메뉴 교환 우선 추천

변경 이력
─────────
- baseline=0 (신규 메뉴)는 반복 메뉴에서 제외
- 날짜 근접도 패널티(date_penalty) 추가
- 결과에 day_gap, date_penalty 컬럼 추가

실행 예시
─────────
# 기본 (기반 1-11월 / 대상 12월)
python meal_analysis.py --base base_data.csv --target target_data.csv

# 특정 학교만
python meal_analysis.py --base base.csv --target target.csv --schools "가경초,봉명초"

# 파라미터 조정
python meal_analysis.py --base base.csv --target target.csv --repeat 4 --alpha 0.6 --beta 0.4
"""

import argparse
import ast
import warnings
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════════
# 임베딩 백엔드
#   1순위: SentenceTransformer (ko-sroberta)
#   2순위: SentenceTransformer (multilingual)
#   3순위: TF-IDF char n-gram (오프라인 fallback)
# ════════════════════════════════════════════════════════════════

def _load_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
        for name in ["jhgan/ko-sroberta-multitask",
                     "paraphrase-multilingual-MiniLM-L12-v2"]:
            try:
                m = SentenceTransformer(name)
                print(f"[임베딩] SentenceTransformer 로드: {name}")
                return m
            except Exception:
                continue
    except ImportError:
        pass
    return None


class EmbeddingBackend:
    def __init__(self):
        self._st = _load_sentence_transformer()
        if self._st is None:
            print("[임베딩] TF-IDF char n-gram fallback 사용")
            self._tfidf   = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
            self._fitted  = False
        else:
            self._tfidf  = None
            self._fitted = True

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        if self._st:
            return self._st.encode(texts, show_progress_bar=False,
                                   batch_size=64, normalize_embeddings=True)
        mat = self._tfidf.fit_transform(texts).toarray()
        self._fitted = True
        return mat

    def transform(self, texts: list[str]) -> np.ndarray:
        if self._st:
            return self._st.encode(texts, show_progress_bar=False,
                                   batch_size=64, normalize_embeddings=True)
        if not self._fitted:
            raise RuntimeError("fit_transform 먼저 호출 필요")
        return self._tfidf.transform(texts).toarray()


# ════════════════════════════════════════════════════════════════
# 0. 데이터 로드 & 전처리
# ════════════════════════════════════════════════════════════════

def load_data(csv_path: str, label: str = "") -> pd.DataFrame:
    """
    CSV 로드 후 공통 전처리.
    label: 로그 출력용 ("기반" / "대상")
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["급식일자"] = df["급식일자"].astype(str).str.strip()
    df["month"]   = df["급식일자"].str[4:6].astype(int)

    # 숫자 추출 (주석 문자열 제거)
    def clean_num(v):
        import re
        if pd.isna(v): return None
        m = re.search(r"[\d.]+", str(v))
        return float(m.group()) if m else None

    for col in ["칼로리정보", "탄수화물(g)", "단백질(g)", "지방(g)",
                "칼슘(mg)", "철분(mg)"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_num)

    if "칼로리정보" in df.columns:
        df.rename(columns={"칼로리정보": "칼로리_num"}, inplace=True)

    def parse_menu(raw):
        try:
            return ast.literal_eval(str(raw))
        except Exception:
            return []

    df["menus"]     = df["menu_list"].apply(parse_menu)
    df["menus"]     = df["menus"].apply(lambda l: [m.strip() for m in l if m.strip()])
    df["menu_text"] = df["menus"].apply(lambda l: ", ".join(l))

    tag = f"[{label}]" if label else "[데이터]"
    months = sorted(df["month"].unique())
    print(f"{tag} {len(df):,}행  |  학교 {df['학교명'].nunique()}개  "
          f"|  포함 월: {months}")
    return df


# ════════════════════════════════════════════════════════════════
# 1. Baseline: 기반 데이터의 메뉴 월평균 등장 횟수
# ════════════════════════════════════════════════════════════════

def build_baseline(df_base: pd.DataFrame) -> dict:
    """
    기반 데이터 전체를 사용해 메뉴별 월평균 등장 횟수 계산.
    월 수는 데이터에 포함된 고유 월 수로 자동 산출.
    """
    n_months = df_base["month"].nunique() or 1
    counter  = Counter()
    for menus in df_base["menus"]:
        counter.update(menus)

    baseline = {m: cnt / n_months for m, cnt in counter.items()}

    print(f"\n[Baseline] 기반 데이터 {n_months}개월 · 메뉴 종류 {len(baseline):,}개")
    top = sorted(baseline.items(), key=lambda x: -x[1])[:10]
    print("  월평균 등장 Top 10:")
    for i, (m, avg) in enumerate(top, 1):
        print(f"    {i:2d}. {m:<22s}  {avg:6.1f}회/월")
    return baseline


# ════════════════════════════════════════════════════════════════
# 2. 반복 메뉴 탐지 (대상 데이터 기준)
# ════════════════════════════════════════════════════════════════

def detect_repeated_menus(
    df_target: pd.DataFrame,
    baseline: dict,
    repeat_threshold: int   = 5,
    baseline_multiplier: float = 1.5,
) -> tuple[set, pd.DataFrame]:
    """
    대상 데이터 내 메뉴 등장 횟수가
      - repeat_threshold 이상  OR
      - baseline 대비 baseline_multiplier 배 이상
    이면 반복 메뉴로 분류.
    """
    counter = Counter()
    for menus in df_target["menus"]:
        counter.update(menus)

    rows = []
    for menu, cnt in counter.items():
        avg = baseline.get(menu, 0)

        # baseline=0 → 기반 데이터에 없던 신규 메뉴 → 반복 아님
        if avg == 0:
            rows.append({
                "menu"        : menu,
                "target_count": cnt,
                "monthly_avg" : 0.0,
                "ratio"       : None,
                "is_repeat"   : False,
            })
            continue

        ratio = cnt / avg
        rows.append({
            "menu"        : menu,
            "target_count": cnt,
            "monthly_avg" : round(avg, 2),
            "ratio"       : round(ratio, 2),
            "is_repeat"   : cnt >= repeat_threshold or ratio >= baseline_multiplier,
        })

    stats    = (pd.DataFrame(rows)
                .sort_values("target_count", ascending=False)
                .reset_index(drop=True))
    repeated = set(stats[stats["is_repeat"]]["menu"])

    print(f"\n[반복 탐지] {repeat_threshold}회↑ OR baseline×{baseline_multiplier}↑"
          f"  →  반복 메뉴 {len(repeated)}개")
    show = stats[stats["is_repeat"]].head(15)
    if not show.empty:
        print(show[["menu", "target_count", "monthly_avg", "ratio"]].to_string(index=False))
    return repeated, stats


# ════════════════════════════════════════════════════════════════
# 3. 피처 빌드 (임베딩 + 영양소)
# ════════════════════════════════════════════════════════════════

NUTRIENT_COLS = ["칼로리_num", "탄수화물(g)", "단백질(g)",
                 "지방(g)", "칼슘(mg)", "철분(mg)"]


def build_features(df_target: pd.DataFrame,
                   backend: EmbeddingBackend) -> pd.DataFrame:
    """
    대상 데이터 전체에 대해 임베딩 + 영양소 벡터를 한번에 계산.
    월 필터 없음 — df_target 자체가 이미 분석 대상만 담고 있음.
    """
    df = df_target.copy().reset_index(drop=True)
    if df.empty:
        return df

    avail_nut = [c for c in NUTRIENT_COLS if c in df.columns]
    nut_raw   = df[avail_nut].fillna(0).values
    df["nutrient_vec"] = list(StandardScaler().fit_transform(nut_raw))

    embs = backend.fit_transform(df["menu_text"].tolist())
    df["embedding"] = list(embs)
    return df


# ════════════════════════════════════════════════════════════════
# 4. 날짜 교환 추천
# ════════════════════════════════════════════════════════════════

def _day_gap(d1: str, d2: str) -> int:
    """'20251203' 형식 두 날짜의 일수 차이"""
    from datetime import date
    def to_date(s):
        s = str(s)
        return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    return abs((to_date(d1) - to_date(d2)).days)


def compute_swap_candidates(
    df_sch: pd.DataFrame,
    repeated_menus: set,
    alpha: float = 0.5,
    beta:  float = 0.5,
    top_k: int   = 5,
) -> list[dict]:
    """
    swap_score = (α × menu_diversity + β × nutrient_sim)
                 × log(1 + rep_gap)
                 × date_penalty

    date_penalty = 1 / log(2 + day_gap)
        날짜가 가까울수록 패널티가 커져 인접 날짜 교환 우선 추천.
        day_gap=1  → 0.91
        day_gap=5  → 0.51
        day_gap=10 → 0.40
    """
    n = len(df_sch)
    if n < 2:
        return []

    df_sch = df_sch.copy()
    df_sch["rep_score"] = df_sch["menus"].apply(
        lambda l: sum(1 for m in l if m in repeated_menus)
    )

    menu_sim = cosine_similarity(np.vstack(df_sch["embedding"].values))
    nutr_sim = cosine_similarity(np.vstack(df_sch["nutrient_vec"].values))

    results = []
    for i, j in combinations(range(n), 2):
        ri, rj  = int(df_sch.iloc[i]["rep_score"]), int(df_sch.iloc[j]["rep_score"])
        rep_gap = abs(ri - rj)
        if rep_gap == 0:
            continue

        diversity    = 1.0 - float(menu_sim[i, j])
        n_sim        = (float(nutr_sim[i, j]) + 1) / 2
        d_gap        = _day_gap(df_sch.iloc[i]["급식일자"], df_sch.iloc[j]["급식일자"])
        date_penalty = 1.0 / np.log(2 + d_gap)
        score        = (alpha * diversity + beta * n_sim) * np.log1p(rep_gap) * date_penalty

        hi, lo = (i, j) if ri >= rj else (j, i)

        results.append({
            "date_hi"       : str(df_sch.iloc[hi]["급식일자"]),
            "date_lo"       : str(df_sch.iloc[lo]["급식일자"]),
            "rep_hi"        : int(df_sch.iloc[hi]["rep_score"]),
            "rep_lo"        : int(df_sch.iloc[lo]["rep_score"]),
            "rep_gap"       : rep_gap,
            "day_gap"       : d_gap,
            "date_penalty"  : round(date_penalty, 4),
            "menu_diversity": round(diversity, 4),
            "nutrient_sim"  : round(n_sim, 4),
            "swap_score"    : round(score, 4),
            "menus_hi"      : df_sch.iloc[hi]["menus"],
            "menus_lo"      : df_sch.iloc[lo]["menus"],
        })

    results.sort(key=lambda x: -x["swap_score"])
    return results[:top_k]


# ════════════════════════════════════════════════════════════════
# 5. 출력 & 저장
# ════════════════════════════════════════════════════════════════

def print_results(all_results: list[dict], max_schools: int = 3):
    seen = set()
    for row in all_results:
        s = row["학교명"]
        if s in seen:
            continue
        seen.add(s)
        school_rows = [r for r in all_results if r["학교명"] == s]

        print(f"\n{'═'*70}")
        print(f"  📍 {s}  —  교환 추천 Top {len(school_rows)}")
        print(f"{'═'*70}")
        for rank, c in enumerate(school_rows, 1):
            print(f"\n  [{rank}] {c['date_hi']}  ↔  {c['date_lo']}"
                  f"   (swap_score={c['swap_score']:.4f})")
            print(f"       반복 메뉴: {c['rep_hi']}개  ↔  {c['rep_lo']}개"
                  f"   (격차 {c['rep_gap']}개)  |  날짜 간격 {c['day_gap']}일")
            print(f"       다양성 {c['menu_diversity']:.3f}  "
                  f"|  영양 유사도 {c['nutrient_sim']:.3f}  "
                  f"|  날짜 패널티 {c['date_penalty']:.3f}")
            print(f"       {c['date_hi']}: {', '.join(c['menus_hi'])}")
            print(f"       {c['date_lo']}: {', '.join(c['menus_lo'])}")

        if len(seen) >= max_schools:
            rem = len({r["학교명"] for r in all_results}) - max_schools
            if rem > 0:
                print(f"\n  ... 외 {rem}개 학교 결과는 CSV 참조")
            break


def save_results(all_results: list[dict], out_path: str):
    if not all_results:
        print("[저장] 저장할 결과 없음")
        return
    rows = [{
        **{k: v for k, v in r.items() if k not in ("menus_hi", "menus_lo")},
        "menus_hi": ", ".join(r["menus_hi"]),
        "menus_lo": ", ".join(r["menus_lo"]),
    } for r in all_results]
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[저장] {out_path}  ({len(rows)}행)")


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main(
    base_path: str,
    target_path: str,
    target_schools: list[str] | None = None,
    repeat_threshold: int    = 5,
    baseline_multiplier: float = 1.5,
    alpha: float = 0.5,
    beta:  float = 0.5,
    top_k: int   = 5,
    out_csv: str = "swap_recommendations.csv",
):
    print("=" * 60)
    print(f"  기반 데이터 : {base_path}")
    print(f"  대상 데이터 : {target_path}")
    print("=" * 60)

    # 0. 로드
    df_base   = load_data(base_path,   label="기반")
    df_target = load_data(target_path, label="대상")

    # 두 파일의 학교 목록이 다를 수 있으므로 교집합만 분석
    schools_base   = set(df_base["학교명"].unique())
    schools_target = set(df_target["학교명"].unique())
    common         = schools_base & schools_target
    if not common:
        print("\n[경고] 두 파일에 공통 학교명이 없습니다. 학교명을 확인하세요.")
        return [], pd.DataFrame()

    if target_schools:
        common = common & set(target_schools)

    print(f"\n[공통 학교] {len(common)}개")

    # 1. baseline (기반 데이터 전체)
    baseline = build_baseline(df_base)

    # 2. 반복 메뉴 탐지 (대상 데이터 전체)
    repeated_menus, stats = detect_repeated_menus(
        df_target, baseline,
        repeat_threshold=repeat_threshold,
        baseline_multiplier=baseline_multiplier,
    )

    # 3. 임베딩 (대상 데이터 전체를 한번에 fit)
    print("\n[임베딩] 대상 데이터 벡터화 중...")
    backend    = EmbeddingBackend()
    df_feat    = build_features(df_target, backend)
    print(f"[임베딩] 완료  ({len(df_feat)}행)")

    # 4. 학교별 교환 추천
    schools = sorted(common)
    print(f"\n[분석] {len(schools)}개 학교 처리 중...\n")

    all_results = []
    for idx, school in enumerate(schools, 1):
        df_sch = df_feat[df_feat["학교명"] == school].copy().reset_index(drop=True)
        cands  = compute_swap_candidates(df_sch, repeated_menus, alpha, beta, top_k)
        status = f"{len(cands)}쌍 추천" if cands else "후보 없음"
        print(f"  ({idx:3d}/{len(schools)}) {school:<25s}  {status}")
        for c in cands:
            all_results.append({"학교명": school, **c})

    # 5. 출력 & 저장
    print_results(all_results, max_schools=3)
    save_results(all_results, out_path=out_csv)

    return all_results, stats


# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="급식 식단 교환 추천",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python meal_analysis.py --base data/base.csv --target data/target.csv
  python meal_analysis.py --base data/base.csv --target data/target.csv --schools "가경초등학교,봉명초등학교"
  python meal_analysis.py --base data/base.csv --target data/target.csv --repeat 4 --alpha 0.6 --beta 0.4
        """
    )
    parser.add_argument("--base",    type=str,
                        default=str(DATAPATH / "menu_basis.csv"),
                        help="기반 데이터 CSV (baseline 구축용, 여러 달 OK)")
    parser.add_argument("--target",  type=str,
                        default=str(DATAPATH / "menu_target.csv"),
                        help="대상 데이터 CSV (교환 추천 대상, 1달치)")
    parser.add_argument("--schools", type=str,   default=None,
                        help="분석할 학교 (콤마 구분), 생략시 전체")
    parser.add_argument("--repeat",  type=int,   default=5,
                        help="반복 기준 횟수 (default:5)")
    parser.add_argument("--mult",    type=float, default=1.5,
                        help="baseline 배율 기준 (default:1.5)")
    parser.add_argument("--alpha",   type=float, default=0.5,
                        help="메뉴 다양성 가중치 (default:0.5)")
    parser.add_argument("--beta",    type=float, default=0.5,
                        help="영양 유사도 가중치 (default:0.5)")
    parser.add_argument("--top",     type=int,   default=5,
                        help="학교당 추천 수 (default:5)")
    parser.add_argument("--out",     type=str,
                        default="swap_recommendations.csv",
                        help="결과 CSV 저장 경로")
    args = parser.parse_args()

    schools = [s.strip() for s in args.schools.split(",")] if args.schools else None

    main(
        base_path           = args.base,
        target_path         = args.target,
        target_schools      = schools,
        repeat_threshold    = args.repeat,
        baseline_multiplier = args.mult,
        alpha               = args.alpha,
        beta                = args.beta,
        top_k               = args.top,
        out_csv             = args.out,
    )