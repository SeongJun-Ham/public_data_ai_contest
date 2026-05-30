from config.paths import *
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


# ════════════════════════════════════════════════════════════════════════════
# 임베딩 백엔드 선택
#   1순위: SentenceTransformer (ko-sroberta)  → 로컬에 모델 있을 때
#   2순위: SentenceTransformer (multilingual) → 다국어 경량 모델
#   3순위: TF-IDF char n-gram                 → 완전 오프라인 fallback
# ════════════════════════════════════════════════════════════════════════════

def _load_sentence_transformer():
    """SentenceTransformer 로드 시도, 실패하면 None 반환"""
    try:
        from sentence_transformers import SentenceTransformer
        candidates = [
            "jhgan/ko-sroberta-multitask",
            "paraphrase-multilingual-MiniLM-L12-v2",
        ]
        for name in candidates:
            try:
                model = SentenceTransformer(name)
                print(f"[임베딩] SentenceTransformer 로드 성공: {name}")
                return model
            except Exception:
                continue
    except ImportError:
        pass
    return None


class EmbeddingBackend:
    """
    texts(list[str]) → numpy array (n, dim) 를 반환하는 공통 인터페이스
    """
    def __init__(self):
        self._st_model = _load_sentence_transformer()
        if self._st_model is None:
            print("[임베딩] SentenceTransformer 사용 불가 → TF-IDF char n-gram fallback")
            # TF-IDF는 fit 후 재사용; corpus 전체를 모아서 한번에 fit
            self._tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
            self._fitted = False
        else:
            self._tfidf = None
            self._fitted = True

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        if self._st_model is not None:
            return self._st_model.encode(texts, show_progress_bar=False,
                                         batch_size=64, normalize_embeddings=True)
        # TF-IDF: fit on all texts
        mat = self._tfidf.fit_transform(texts).toarray()
        self._fitted = True
        return mat

    def transform(self, texts: list[str]) -> np.ndarray:
        """fit 이후 소규모 벡터 변환용"""
        if self._st_model is not None:
            return self._st_model.encode(texts, show_progress_bar=False,
                                         batch_size=64, normalize_embeddings=True)
        if not self._fitted:
            raise RuntimeError("fit_transform 먼저 호출 필요")
        return self._tfidf.transform(texts).toarray()


# ════════════════════════════════════════════════════════════════════════════
# 0.  데이터 로드 & 전처리
# ════════════════════════════════════════════════════════════════════════════

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    df["급식일자"] = df["급식일자"].astype(str).str.strip()
    df["month"]   = df["급식일자"].str[4:6].astype(int)

    # 칼로리 숫자만 추출
    df["칼로리_num"] = (
        df["칼로리정보"].astype(str)
        .str.extract(r"([\d.]+)")[0]
        .astype(float)
    )

    # menu_list 파싱
    def parse_menu(raw):
        try:
            return ast.literal_eval(str(raw))
        except Exception:
            return []

    df["menus"] = df["menu_list"].apply(parse_menu)
    df["menus"] = df["menus"].apply(lambda lst: [m.strip() for m in lst if m.strip()])
    df["menu_text"] = df["menus"].apply(lambda lst: ", ".join(lst))

    print(f"[데이터] {len(df):,}행 로드  |  학교: {df['학교명'].nunique()}개  "
          f"|  월 범위: {df['month'].min()}~{df['month'].max()}월")
    return df


# ════════════════════════════════════════════════════════════════════════════
# 1.  1~11월 Baseline: 메뉴별 월평균 등장 횟수
# ════════════════════════════════════════════════════════════════════════════

def build_baseline(df: pd.DataFrame) -> dict:
    df_train = df[df["month"] <= 11]
    counter  = Counter()
    for menus in df_train["menus"]:
        counter.update(menus)

    baseline = {m: cnt / 11 for m, cnt in counter.items()}

    print(f"\n[Baseline] 메뉴 종류: {len(baseline):,}개  (1~11월 기준)")
    top = sorted(baseline.items(), key=lambda x: -x[1])[:10]
    print("  월평균 등장 Top 10:")
    for i, (m, avg) in enumerate(top, 1):
        print(f"    {i:2d}. {m:<22s}  {avg:6.1f}회/월")
    return baseline


# ════════════════════════════════════════════════════════════════════════════
# 2.  12월 반복 메뉴 탐지
# ════════════════════════════════════════════════════════════════════════════

def detect_repeated_menus(
    df: pd.DataFrame,
    baseline: dict,
    repeat_threshold: int   = 5,
    baseline_multiplier: float = 1.5,
) -> tuple[set, pd.DataFrame]:
    """
    repeat_threshold    : 12월 내 절대 등장 횟수 기준
    baseline_multiplier : baseline 대비 배율 기준
    둘 중 하나라도 해당하면 '반복 메뉴' 분류
    """
    df_dec   = df[df["month"] == 12]
    counter  = Counter()
    for menus in df_dec["menus"]:
        counter.update(menus)

    rows = []
    for menu, cnt in counter.items():
        avg   = baseline.get(menu, 0)
        ratio = (cnt / avg) if avg > 0 else float("inf")
        rows.append({
            "menu"       : menu,
            "dec_count"  : cnt,
            "monthly_avg": round(avg, 2),
            "ratio"      : round(ratio, 2),
            "is_repeat"  : cnt >= repeat_threshold or ratio >= baseline_multiplier,
        })

    stats    = pd.DataFrame(rows).sort_values("dec_count", ascending=False).reset_index(drop=True)
    repeated = set(stats[stats["is_repeat"]]["menu"])

    print(f"\n[반복 탐지] 기준: 12월 {repeat_threshold}회↑ OR baseline×{baseline_multiplier}↑")
    print(f"  → 반복 메뉴 {len(repeated)}개 탐지")

    show = stats[stats["is_repeat"]].head(15)
    if not show.empty:
        print(show[["menu","dec_count","monthly_avg","ratio"]].to_string(index=False))

    return repeated, stats


# ════════════════════════════════════════════════════════════════════════════
# 3.  날짜별 피처 빌드 (임베딩 + 영양소)
# ════════════════════════════════════════════════════════════════════════════

NUTRIENT_COLS = ["칼로리_num", "탄수화물(g)", "단백질(g)", "지방(g)", "칼슘(mg)", "철분(mg)"]


def build_features(
    df: pd.DataFrame,
    backend: EmbeddingBackend,
    month: int = 12,
) -> pd.DataFrame:
    """
    지정 월의 전체 데이터에 대해 임베딩 + 영양소 벡터를 한번에 계산
    (학교별로 반복 호출하지 않고 한번에 fit → transform)
    """
    df_m = df[df["month"] == month].copy().reset_index(drop=True)
    if df_m.empty:
        return df_m

    # ── 영양소 정규화
    nut_raw = df_m[NUTRIENT_COLS].fillna(0).values
    scaler  = StandardScaler()
    nut_scaled = scaler.fit_transform(nut_raw)
    df_m["nutrient_vec"] = list(nut_scaled)

    # ── 메뉴 임베딩 (전체 텍스트 한번에 fit)
    texts = df_m["menu_text"].tolist()
    embs  = backend.fit_transform(texts)
    df_m["embedding"] = list(embs)

    return df_m


# ════════════════════════════════════════════════════════════════════════════
# 4.  날짜 교환 추천
# ════════════════════════════════════════════════════════════════════════════

def compute_swap_candidates(
    df_sch: pd.DataFrame,
    repeated_menus: set,
    alpha: float = 0.5,
    beta:  float = 0.5,
    top_k: int   = 5,
) -> list[dict]:
    """
    swap_score = α × menu_diversity + β × nutrient_similarity

    menu_diversity  = 1 - cosine_sim(embed_i, embed_j)
        → 두 날의 메뉴가 다를수록 ↑  (교환 시 다양성 증가)
    nutrient_similarity = (cosine_sim(nutr_i, nutr_j) + 1) / 2
        → 영양 구성이 비슷할수록 ↑   (교환 후 영양 밸런스 유지)

    반복 점수 차이(rep_gap)가 클수록 교환 효과가 크므로
    최종 점수에 log(1 + rep_gap) 가중도 적용
    """
    n = len(df_sch)
    if n < 2:
        return []

    # 날짜별 반복 점수
    df_sch = df_sch.copy()
    df_sch["rep_score"] = df_sch["menus"].apply(
        lambda lst: sum(1 for m in lst if m in repeated_menus)
    )

    emb_mat  = np.vstack(df_sch["embedding"].values)
    nut_mat  = np.vstack(df_sch["nutrient_vec"].values)

    menu_sim = cosine_similarity(emb_mat)            # (n, n)
    nutr_sim = cosine_similarity(nut_mat)            # (n, n)

    results = []
    for i, j in combinations(range(n), 2):
        ri = int(df_sch.iloc[i]["rep_score"])
        rj = int(df_sch.iloc[j]["rep_score"])
        rep_gap = abs(ri - rj)
        if rep_gap == 0:
            continue                                  # 교환 효과 없음

        diversity = 1.0 - float(menu_sim[i, j])
        n_sim     = (float(nutr_sim[i, j]) + 1) / 2  # [0, 1]
        score     = (alpha * diversity + beta * n_sim) * np.log1p(rep_gap)

        hi, lo = (i, j) if ri >= rj else (j, i)

        results.append({
            "date_hi"      : str(df_sch.iloc[hi]["급식일자"]),
            "date_lo"      : str(df_sch.iloc[lo]["급식일자"]),
            "rep_hi"       : int(df_sch.iloc[hi]["rep_score"]),
            "rep_lo"       : int(df_sch.iloc[lo]["rep_score"]),
            "rep_gap"      : rep_gap,
            "menu_diversity": round(diversity, 4),
            "nutrient_sim" : round(n_sim, 4),
            "swap_score"   : round(score, 4),
            "menus_hi"     : df_sch.iloc[hi]["menus"],
            "menus_lo"     : df_sch.iloc[lo]["menus"],
        })

    results.sort(key=lambda x: -x["swap_score"])
    return results[:top_k]


# ════════════════════════════════════════════════════════════════════════════
# 5.  출력 & 저장
# ════════════════════════════════════════════════════════════════════════════

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
            print(f"       반복 메뉴: {c['date_hi']} {c['rep_hi']}개  ↔  "
                  f"{c['date_lo']} {c['rep_lo']}개   (격차 {c['rep_gap']}개)")
            print(f"       메뉴 다양성 {c['menu_diversity']:.3f}  |  "
                  f"영양 유사도 {c['nutrient_sim']:.3f}")
            hi_str = ", ".join(c["menus_hi"])
            lo_str = ", ".join(c["menus_lo"])
            print(f"       {c['date_hi']}: {hi_str}")
            print(f"       {c['date_lo']}: {lo_str}")

        if len(seen) >= max_schools:
            remaining = len({r["학교명"] for r in all_results}) - max_schools
            if remaining > 0:
                print(f"\n  ... 외 {remaining}개 학교 결과는 CSV 참조")
            break


def save_results(all_results: list[dict], out_path: str):
    if not all_results:
        print("[저장] 저장할 결과 없음")
        return
    rows = []
    for r in all_results:
        rows.append({
            **{k: v for k, v in r.items() if k not in ("menus_hi", "menus_lo")},
            "menus_hi": ", ".join(r["menus_hi"]),
            "menus_lo": ", ".join(r["menus_lo"]),
        })
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[저장] {out_path}  ({len(rows)}행)")


def main(
    csv_path: str,
    target_schools: list[str] | None = None,
    repeat_threshold: int    = 5,
    baseline_multiplier: float = 1.5,
    alpha: float  = 0.5,
    beta:  float  = 0.5,
    top_k: int    = 5,
    out_csv: str  = "swap_recommendations.csv",
):
    # 0. 로드
    df = load_data(csv_path)

    # 1. baseline (1~11월)
    baseline = build_baseline(df)

    # 2. 반복 메뉴 탐지 (12월)
    repeated_menus, dec_stats = detect_repeated_menus(
        df, baseline,
        repeat_threshold=repeat_threshold,
        baseline_multiplier=baseline_multiplier,
    )

    # 3. 임베딩 백엔드 초기화 + 12월 전체 피처 한번에 계산
    print("\n[임베딩] 12월 메뉴 텍스트 벡터화 중...")
    backend    = EmbeddingBackend()
    df_dec_all = build_features(df, backend, month=12)
    print(f"[임베딩] 완료  ({len(df_dec_all)}행)")

    # 4. 학교별 교환 추천
    schools = target_schools or df["학교명"].unique().tolist()
    print(f"\n[분석] {len(schools)}개 학교 처리 시작...\n")

    all_results = []
    for idx, school in enumerate(schools, 1):
        df_sch = df_dec_all[df_dec_all["학교명"] == school].copy().reset_index(drop=True)
        cands  = compute_swap_candidates(df_sch, repeated_menus, alpha, beta, top_k)

        status = f"{len(cands)}쌍 추천" if cands else "후보 없음"
        print(f"  ({idx:3d}/{len(schools)}) {school:<25s}  {status}")

        for c in cands:
            all_results.append({"학교명": school, **c})

    # 5. 출력 (상위 3개 학교 상세)
    print_results(all_results, max_schools=3)

    # 6. CSV 저장
    save_results(all_results, out_path=out_csv)

    return all_results, dec_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="급식 식단 교환 추천")
    parser.add_argument("--csv",      type=str,   default=DATAPATH / "menu_preproc.csv", help="급식 데이터 CSV 경로")
    parser.add_argument("--schools",help="분석할 학교 (콤마 구분), 생략시 전체")
    parser.add_argument("--repeat", type=int,   default=5,   help="12월 반복 기준 횟수 (default:5)")
    parser.add_argument("--mult",   type=float, default=1.5, help="baseline 배율 기준 (default:1.5)")
    parser.add_argument("--alpha",  type=float, default=0.5, help="메뉴 다양성 가중치 (default:0.5)")
    parser.add_argument("--beta",   type=float, default=0.5, help="영양 유사도 가중치 (default:0.5)")
    parser.add_argument("--top",    type=int,   default=5,   help="학교당 추천 수 (default:5)")
    parser.add_argument("--out",    default="swap_recommendations.csv", help="결과 CSV 저장 경로")
    args = parser.parse_args()

    schools = [s.strip() for s in args.schools.split(",")] if args.schools else None

    main(
        csv_path            = args.csv,
        target_schools      = schools,
        repeat_threshold    = args.repeat,
        baseline_multiplier = args.mult,
        alpha               = args.alpha,
        beta                = args.beta,
        top_k               = args.top,
        out_csv             = args.out,
    )