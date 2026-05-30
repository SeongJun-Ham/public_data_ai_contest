import re
from config.paths import *

import pandas as pd


def clean_number(value):
    """숫자 앞부분만 추출, 변환 실패하면 None"""
    if pd.isna(value):
        return None
    match = re.search(r"[\d.]+", str(value))
    return float(match.group()) if match else None


def parse_nutrition(text):
    result = {}

    if pd.isna(text):
        return result

    items = text.split("<br/>")

    for item in items:
        if ":" not in item:
            continue

        key, value = item.split(":", 1)
        key   = key.strip()
        value = value.strip()

        # 숫자 추출 (주석 문자열 제거)
        cleaned = clean_number(value)
        result[key] = cleaned if cleaned is not None else value

    return result


def preprocess_menu(text):
    if pd.isna(text):
        return []

    menus = text.split("<br/>")
    result = []

    for menu in menus:
        menu = re.sub(r"\([^)]*\)", "", menu)
        menu = re.sub(r"\[[^]]*\]", "", menu)
        menu = menu.strip()
        if menu:
            result.append(menu)

    return result


def main(file_name):
    data = pd.read_csv(DATAPATH / f"{file_name}.csv")
    save_dir = DATAPATH

    data = data[[
        "학교명",
        "급식일자",
        "급식인원수",
        "요리명",
        "칼로리정보",
        "영양정보"
    ]]

    pattern = "양념구매"
    data = data[~data["요리명"].str.contains(pattern, na=False)]

    # 칼로리도 숫자만 추출
    data["칼로리정보"] = data["칼로리정보"].apply(clean_number)

    nutrition_df = (
        data["영양정보"]
        .apply(parse_nutrition)
        .apply(pd.Series)
    )

    data = pd.concat([data, nutrition_df], axis=1)
    data = data.drop(columns=["영양정보"])

    data["menu_list"] = data["요리명"].apply(preprocess_menu)

    data = data[[
        '학교명', '급식일자', '급식인원수', '칼로리정보', '탄수화물(g)', '단백질(g)', '지방(g)',
        '비타민A(R.E)', '티아민(mg)', '리보플라빈(mg)', '비타민C(mg)', '칼슘(mg)', '철분(mg)',
        'menu_list'
    ]]

    data.to_csv(
        save_dir / "menu_preproc.csv",
        index=False,
        encoding="utf-8-sig"
    )


if __name__ == "__main__":
    main("충북_2025_급식식단정보")