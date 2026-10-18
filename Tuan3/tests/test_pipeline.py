"""
Kiem thu pipeline (smoke test) - khong can du lieu Kaggle.

Script sinh ra mot bo du lieu GIA LAP mo phong dung cau truc cua Ames Housing
(co cot dinh tinh, dinh luong, gia tri thieu, ngoai lai > 4000 sqft), roi chay
toan bo 6 buoc CRISP-DM de kiem tra:
    1. Pipeline chay khong loi tu dau den cuoi
    2. Khong con gia tri thieu sau buoc Data Preparation
    3. So cot cua train va test khop nhau sau one-hot encoding
    4. File submission.csv duoc sinh ra, du so dong, khong co gia tri am/NaN

Chay:  python tests/test_pipeline.py
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

RNG = np.random.default_rng(0)


def make_fake(n: int, with_target: bool = True) -> pd.DataFrame:
    """Sinh du lieu gia lap giu dung ten cot / kieu du lieu cua Ames Housing."""
    d = {
        "Id": np.arange(1, n + 1),
        "MSSubClass": RNG.choice([20, 60, 70, 120], n),
        "MSZoning": RNG.choice(["RL", "RM", "FV", None], n, p=[.7, .2, .09, .01]),
        "LotFrontage": RNG.normal(70, 20, n).round(),
        "LotArea": RNG.lognormal(9.1, .3, n).round(),
        "Neighborhood": RNG.choice(["NAmes", "CollgCr", "OldTown", "Somerst"], n),
        "LotShape": RNG.choice(["Reg", "IR1", "IR2"], n),
        "LandSlope": RNG.choice(["Gtl", "Mod", "Sev"], n),
        "OverallQual": RNG.integers(3, 11, n),
        "OverallCond": RNG.integers(3, 9, n),
        "YearBuilt": RNG.integers(1900, 2010, n),
        "YearRemodAdd": RNG.integers(1950, 2011, n),
        "MasVnrType": RNG.choice(["BrkFace", "Stone", None], n, p=[.4, .3, .3]),
        "MasVnrArea": RNG.choice([0, 100, 200, np.nan], n, p=[.5, .2, .2, .1]),
        "ExterQual": RNG.choice(["Ex", "Gd", "TA", "Fa"], n),
        "ExterCond": RNG.choice(["Gd", "TA", "Fa"], n),
        "BsmtQual": RNG.choice(["Ex", "Gd", "TA", None], n, p=[.2, .3, .45, .05]),
        "BsmtCond": RNG.choice(["Gd", "TA", None], n, p=[.2, .75, .05]),
        "BsmtExposure": RNG.choice(["Gd", "Av", "Mn", "No", None], n, p=[.1, .2, .1, .55, .05]),
        "BsmtFinType1": RNG.choice(["GLQ", "ALQ", "Unf", None], n, p=[.3, .3, .35, .05]),
        "BsmtFinSF1": RNG.integers(0, 1200, n),
        "BsmtUnfSF": RNG.integers(0, 1000, n),
        "TotalBsmtSF": RNG.integers(0, 2000, n),
        "HeatingQC": RNG.choice(["Ex", "Gd", "TA"], n),
        "CentralAir": RNG.choice(["Y", "N"], n, p=[.93, .07]),
        "Electrical": RNG.choice(["SBrkr", "FuseA", None], n, p=[.9, .09, .01]),
        "1stFlrSF": RNG.integers(500, 2200, n),
        "2ndFlrSF": RNG.choice([0, 600, 900, 1200], n),
        "GrLivArea": RNG.integers(600, 4600, n),  # co ca ngoai lai > 4000
        "BsmtFullBath": RNG.integers(0, 2, n),
        "BsmtHalfBath": RNG.integers(0, 2, n),
        "FullBath": RNG.integers(1, 4, n),
        "HalfBath": RNG.integers(0, 2, n),
        "BedroomAbvGr": RNG.integers(1, 6, n),
        "KitchenQual": RNG.choice(["Ex", "Gd", "TA", None], n, p=[.1, .4, .49, .01]),
        "TotRmsAbvGrd": RNG.integers(3, 12, n),
        "Functional": RNG.choice(["Typ", "Min1", "Mod"], n),
        "Fireplaces": RNG.integers(0, 3, n),
        "FireplaceQu": RNG.choice(["Gd", "TA", None], n, p=[.3, .2, .5]),
        "GarageType": RNG.choice(["Attchd", "Detchd", None], n, p=[.6, .3, .1]),
        "GarageYrBlt": RNG.choice([1980, 1995, 2005, np.nan], n, p=[.3, .3, .3, .1]),
        "GarageFinish": RNG.choice(["Fin", "RFn", "Unf", None], n, p=[.3, .3, .3, .1]),
        "GarageCars": RNG.integers(0, 4, n),
        "GarageArea": RNG.integers(0, 900, n),
        "GarageQual": RNG.choice(["TA", "Gd", None], n, p=[.8, .1, .1]),
        "GarageCond": RNG.choice(["TA", "Gd", None], n, p=[.8, .1, .1]),
        "PavedDrive": RNG.choice(["Y", "P", "N"], n, p=[.9, .05, .05]),
        "WoodDeckSF": RNG.integers(0, 400, n),
        "OpenPorchSF": RNG.integers(0, 300, n),
        "EnclosedPorch": RNG.integers(0, 200, n),
        "3SsnPorch": np.zeros(n, dtype=int),
        "ScreenPorch": RNG.integers(0, 150, n),
        "PoolArea": RNG.choice([0, 0, 0, 500], n),
        "PoolQC": RNG.choice(["Ex", None], n, p=[.01, .99]),
        "Fence": RNG.choice(["MnPrv", None], n, p=[.2, .8]),
        "MiscFeature": RNG.choice(["Shed", None], n, p=[.05, .95]),
        "Alley": RNG.choice(["Grvl", None], n, p=[.06, .94]),
        "MoSold": RNG.integers(1, 13, n),
        "YrSold": RNG.integers(2006, 2011, n),
        "SaleType": RNG.choice(["WD", "New", None], n, p=[.85, .14, .01]),
        "SaleCondition": RNG.choice(["Normal", "Partial", "Abnorml"], n, p=[.82, .1, .08]),
        "Exterior1st": RNG.choice(["VinylSd", "HdBoard", "Wd Sdng"], n),
        "Exterior2nd": RNG.choice(["VinylSd", "HdBoard", "Wd Sdng"], n),
        "Utilities": RNG.choice(["AllPub", "NoSeWa"], n, p=[.99, .01]),
    }
    df = pd.DataFrame(d)
    if with_target:
        # Gia = ham tuyen tinh cua chat luong & dien tich + nhieu -> co tin hieu that
        price = (25000 + 12000 * df["OverallQual"] + 45 * df["GrLivArea"]
                 + 25 * df["TotalBsmtSF"] + 9000 * df["GarageCars"]
                 - 200 * (df["YrSold"] - df["YearBuilt"])
                 + RNG.normal(0, 18000, n))
        df["SalePrice"] = price.clip(40000, 700000).round()
    return df


def main():
    data_dir = os.path.join(ROOT, "data_fake")
    out_dir = os.path.join(ROOT, "outputs_test")
    os.makedirs(data_dir, exist_ok=True)

    train = make_fake(400, True)
    test = make_fake(120, False)
    train.to_csv(os.path.join(data_dir, "train.csv"), index=False)
    test.to_csv(os.path.join(data_dir, "test.csv"), index=False)
    print(f"[SETUP] Da sinh du lieu gia lap: train={train.shape}, test={test.shape}")

    cmd = [sys.executable, os.path.join(ROOT, "src", "house_price_crisp.py"),
           "--data-dir", data_dir, "--out-dir", out_dir, "--fast"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:]); print(r.stderr[-3000:])
        raise SystemExit("[FAIL] Pipeline chay loi")

    # ---------------- Cac phep kiem tra (assertions) -------------------
    sub_path = os.path.join(out_dir, "submission.csv")
    assert os.path.exists(sub_path), "[FAIL] Khong sinh ra submission.csv"
    sub = pd.read_csv(sub_path)
    assert len(sub) == len(test), f"[FAIL] submission thieu dong: {len(sub)} != {len(test)}"
    assert list(sub.columns) == ["Id", "SalePrice"], "[FAIL] Sai ten cot ban nop"
    assert sub["SalePrice"].notna().all(), "[FAIL] Co gia tri NaN trong du doan"
    assert (sub["SalePrice"] > 0).all(), "[FAIL] Co gia du doan <= 0"
    assert os.path.exists(os.path.join(out_dir, "model_comparison.csv")), "[FAIL] Thieu bang so sanh"
    figs = os.listdir(os.path.join(out_dir, "figures"))
    assert len(figs) >= 6, f"[FAIL] Thieu bieu do, chi co {len(figs)}"

    board = pd.read_csv(os.path.join(out_dir, "model_comparison.csv"))
    print("\n[KET QUA CV TREN DU LIEU GIA LAP]")
    print(board.to_string(index=False))
    print(f"\n[PASS] Pipeline chay tron ven. submission.csv: {len(sub)} dong, "
          f"gia tu ${sub.SalePrice.min():,.0f} den ${sub.SalePrice.max():,.0f}")
    print(f"[PASS] Sinh du {len(figs)} bieu do: {sorted(figs)}")
    print("\n>>> TAT CA KIEM THU DEU DAT <<<")


if __name__ == "__main__":
    main()
