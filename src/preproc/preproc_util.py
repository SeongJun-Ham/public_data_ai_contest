from config.paths import *
import pandas as pd




def main():
    data = pd.read_csv(DATAPATH / "menu_preproc.csv")
    save_dir = DATAPATH

    data_school = data[
        data["학교명"] == "가경초등학교"
    ]

    data_basis = data_school[
        data_school["급식일자"] < 20251200
    ]

    data_target = data_school[
        data_school["급식일자"] >= 20251200
    ]

    data_basis.to_csv(
        save_dir / "menu_basis.csv",
        index=False,
        encoding="utf-8-sig"
    )

    data_target.to_csv(
        save_dir / "menu_target.csv",
        index=False,
        encoding="utf-8-sig"
    )



if __name__ == "__main__":
    main()
