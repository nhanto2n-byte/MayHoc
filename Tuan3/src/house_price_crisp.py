"""
=============================================================================
 LAB 02 - HOUSE PRICES: ADVANCED REGRESSION TECHNIQUES
 Quy trinh xay dung mo hinh theo 6 buoc CRISP-DM
 Du lieu: Ames Housing (De Cock, 2011) - Kaggle House Prices competition
=============================================================================

 Cach chay:
     python src/house_price_crisp.py --data-dir data --out-dir outputs

 Thu muc data/ can co: train.csv, test.csv  (tai tu Kaggle)
 Ket qua sinh ra trong outputs/: figures/*.png, model_comparison.csv,
                                 submission.csv, run_log.txt
=============================================================================
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # khong can man hinh -> luu thang ra file anh
import matplotlib.pyplot as plt

from scipy.stats import skew
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_FOLDS = 5


# ---------------------------------------------------------------------------
# Tien ich ghi log: in ra man hinh VA ghi vao outputs/run_log.txt
# ---------------------------------------------------------------------------
class Logger:
    def __init__(self, path: str):
        self.file = open(path, "w", encoding="utf-8")

    def __call__(self, *args):
        msg = " ".join(str(a) for a in args)
        print(msg)
        self.file.write(msg + "\n")
        self.file.flush()

    def section(self, title: str):
        self("\n" + "=" * 78)
        self(title)
        self("=" * 78)

    def close(self):
        self.file.close()


# ===========================================================================
# BUOC 1 - BUSINESS UNDERSTANDING (Hieu nghiep vu)
# ===========================================================================
BUSINESS_UNDERSTANDING = """
Muc tieu nghiep vu : Uoc luong gia ban (SalePrice) cua mot can nha o Ames, Iowa
                     dua tren 79 dac trung mo ta chat luong va so luong cac hang
                     muc vat ly cua bat dong san.
Muc tieu ky thuat  : Bai toan hoc co giam sat - HOI QUY (supervised regression).
Do do thanh cong   : RMSE tren log(SalePrice) - dung dung tieu chi cham cua
                     Kaggle. Lay log vi sai so 10% tren can nha 500k$ va tren
                     can nha 100k$ phai duoc coi la nghiem trong nhu nhau.
Rang buoc du lieu  : Theo De Cock (2011) - loai cac can > 4000 sqft (5 quan sat
                     bat thuong, trong do 3 la "Partial sale" khong phan anh gia
                     thi truong).
"""


# ===========================================================================
# BUOC 2 - DATA UNDERSTANDING (Hieu du lieu)
# ===========================================================================
def data_understanding(train: pd.DataFrame, test: pd.DataFrame, log: Logger, fig_dir: str):
    log.section("BUOC 2 | DATA UNDERSTANDING - KHAM PHA DU LIEU (EDA)")
    log(f"Kich thuoc train : {train.shape[0]} dong x {train.shape[1]} cot")
    log(f"Kich thuoc test  : {test.shape[0]} dong x {test.shape[1]} cot")

    num_cols = train.select_dtypes(include=[np.number]).columns
    cat_cols = train.select_dtypes(exclude=[np.number]).columns
    log(f"So cot dinh luong: {len(num_cols)} | So cot dinh tinh: {len(cat_cols)}")

    # --- 2.1 Thong ke bien muc tieu -------------------------------------
    y = train["SalePrice"]
    log("\n[2.1] Bien muc tieu SalePrice:")
    log(f"      min={y.min():,.0f}  max={y.max():,.0f}  mean={y.mean():,.0f}  median={y.median():,.0f}")
    nhan_xet = "lech phai manh -> CAN log-transform" if y.skew() > 0.75 else "tuong doi can doi"
    log(f"      do lech (skewness) = {y.skew():.3f}  ->  {nhan_xet}")
    log(f"      skewness sau log1p = {np.log1p(y).skew():.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(y, bins=50, color="#4C72B0", edgecolor="white")
    ax[0].set_title("SalePrice goc (lech phai)")
    ax[1].hist(np.log1p(y), bins=50, color="#55A868", edgecolor="white")
    ax[1].set_title("log1p(SalePrice) (gan chuan)")
    for a in ax:
        a.set_xlabel("Gia ban")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "01_target_distribution.png"), dpi=110)
    plt.close()

    # --- 2.2 Du lieu thieu ----------------------------------------------
    miss = train.isnull().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    miss_pct = (miss / len(train) * 100).round(2)
    miss_tbl = pd.DataFrame({"so_dong_thieu": miss, "ty_le_%": miss_pct})
    log("\n[2.2] Top 15 cot thieu du lieu nhieu nhat:")
    log(miss_tbl.head(15).to_string())
    miss_tbl.to_csv(os.path.join(fig_dir, "..", "missing_report.csv"))

    if len(miss_tbl) > 0:
        plt.figure(figsize=(9, 5))
        top = miss_tbl.head(20).iloc[::-1]
        plt.barh(top.index, top["ty_le_%"], color="#C44E52")
        plt.xlabel("% du lieu thieu")
        plt.title("20 cot thieu du lieu nhieu nhat (tap train)")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "02_missing_values.png"), dpi=110)
        plt.close()

    # --- 2.3 Tuong quan voi SalePrice -----------------------------------
    corr = train[num_cols].corr()["SalePrice"].drop("SalePrice").sort_values(ascending=False)
    log("\n[2.3] 10 bien so tuong quan MANH NHAT voi SalePrice:")
    log(corr.head(10).to_string())
    log("\n      10 bien so tuong quan YEU/AM nhat:")
    log(corr.tail(10).to_string())

    plt.figure(figsize=(8, 6))
    top_corr = corr.head(12).iloc[::-1]
    plt.barh(top_corr.index, top_corr.values, color="#4C72B0")
    plt.xlabel("He so tuong quan Pearson voi SalePrice")
    plt.title("12 dac trung so tuong quan manh nhat")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "03_top_correlations.png"), dpi=110)
    plt.close()

    # --- 2.4 Scatter GrLivArea: tim outlier theo goi y cua De Cock -------
    if "GrLivArea" in train.columns:
        out = train[(train["GrLivArea"] > 4000)]
        log(f"\n[2.4] So quan sat co GrLivArea > 4000 sqft (De Cock de nghi loai): {len(out)}")
        plt.figure(figsize=(6.5, 5))
        plt.scatter(train["GrLivArea"], train["SalePrice"], s=14, alpha=0.5, color="#4C72B0")
        if len(out):
            plt.scatter(out["GrLivArea"], out["SalePrice"], s=45, color="red", label="Outlier > 4000 sqft")
            plt.legend()
        plt.axvline(4000, ls="--", c="grey")
        plt.xlabel("GrLivArea (sqft)")
        plt.ylabel("SalePrice ($)")
        plt.title("SalePrice vs GrLivArea - phat hien ngoai lai")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "04_outliers_grlivarea.png"), dpi=110)
        plt.close()

    return corr


# ===========================================================================
# BUOC 3 - DATA PREPARATION (Chuan bi du lieu)
# ===========================================================================

# NA trong cac cot nay KHONG phai du lieu thieu, ma co nghia la "khong co"
NONE_COLS = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageType",
    "GarageFinish", "GarageQual", "GarageCond", "BsmtQual", "BsmtCond",
    "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "MasVnrType",
]
ZERO_COLS = [
    "GarageYrBlt", "GarageArea", "GarageCars", "BsmtFinSF1", "BsmtFinSF2",
    "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath", "MasVnrArea",
]
MODE_COLS = [
    "MSZoning", "Electrical", "KitchenQual", "Exterior1st",
    "Exterior2nd", "SaleType", "Functional", "Utilities",
]

# Thang do chat luong dang thu tu (ordinal) -> ma hoa thanh so co thu tu
QUAL_MAP = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
QUAL_COLS = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
    "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
]
ORDINAL_MAPS = {
    "BsmtExposure": {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "GarageFinish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "Functional": {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4, "Min2": 5, "Min1": 6, "Typ": 7},
    "CentralAir": {"N": 0, "Y": 1},
    "PavedDrive": {"N": 0, "P": 1, "Y": 2},
    "LotShape": {"IR3": 0, "IR2": 1, "IR1": 2, "Reg": 3},
    "LandSlope": {"Sev": 0, "Mod": 1, "Gtl": 2},
}


def data_preparation(train: pd.DataFrame, test: pd.DataFrame, log: Logger):
    log.section("BUOC 3 | DATA PREPARATION - LAM SACH & TAO DAC TRUNG")

    # --- 3.1 Loai ngoai lai theo khuyen nghi cua tac gia bo du lieu -----
    n0 = len(train)
    if "GrLivArea" in train.columns:
        train = train[train["GrLivArea"] <= 4000].reset_index(drop=True)
    log(f"[3.1] Loai ngoai lai GrLivArea > 4000: {n0} -> {len(train)} dong (-{n0 - len(train)})")

    # --- 3.2 Bien muc tieu: log1p ---------------------------------------
    y = np.log1p(train["SalePrice"].values)
    log("[3.2] Bien muc tieu chuyen sang log1p(SalePrice) (dung tieu chi cham Kaggle)")

    test_id = test["Id"].values if "Id" in test.columns else np.arange(len(test))
    n_train = len(train)

    X_train_raw = train.drop(columns=[c for c in ["Id", "SalePrice"] if c in train.columns])
    X_test_raw = test.drop(columns=[c for c in ["Id"] if c in test.columns])
    X_test_raw = X_test_raw[[c for c in X_train_raw.columns if c in X_test_raw.columns]]

    # Gop train + test de xu ly dong nhat (khong dung SalePrice -> khong ro ri)
    full = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)
    log(f"[3.3] Gop train+test de xu ly dong nhat: {full.shape}")

    # --- 3.4 Xu ly gia tri thieu theo NGHIA nghiep vu --------------------
    for c in NONE_COLS:
        if c in full.columns:
            full[c] = full[c].fillna("None")
    for c in ZERO_COLS:
        if c in full.columns:
            full[c] = full[c].fillna(0)
    if "LotFrontage" in full.columns and "Neighborhood" in full.columns:
        # Nha cung khu pho thuong co be ngang lo dat tuong duong nhau
        full["LotFrontage"] = full.groupby("Neighborhood")["LotFrontage"].transform(
            lambda s: s.fillna(s.median())
        )
        full["LotFrontage"] = full["LotFrontage"].fillna(full["LotFrontage"].median())
    for c in MODE_COLS:
        if c in full.columns and full[c].isnull().any():
            full[c] = full[c].fillna(full[c].mode()[0])
    # Quet phan con lai
    for c in full.columns:
        if full[c].isnull().any():
            if full[c].dtype == object:
                full[c] = full[c].fillna(full[c].mode()[0])
            else:
                full[c] = full[c].fillna(full[c].median())
    log(f"[3.4] Sau xu ly: tong so o con thieu = {int(full.isnull().sum().sum())}")

    # --- 3.5 Ma hoa bien thu tu (ordinal) --------------------------------
    n_ord = 0
    for c in QUAL_COLS:
        if c in full.columns and full[c].dtype == object:
            full[c] = full[c].map(QUAL_MAP).fillna(0).astype(int)
            n_ord += 1
    for c, m in ORDINAL_MAPS.items():
        if c in full.columns and full[c].dtype == object:
            full[c] = full[c].map(m).fillna(0).astype(int)
            n_ord += 1
    log(f"[3.5] Ma hoa {n_ord} bien thu tu (Ex>Gd>TA>Fa>Po) thanh so co thu bac")

    # --- 3.6 Tao dac trung moi (feature engineering) ---------------------
    def has(*cols):
        return all(c in full.columns for c in cols)

    created = []
    if has("TotalBsmtSF", "1stFlrSF", "2ndFlrSF"):
        # De Cock: rieng tong dien tich giai thich ~80% bien thien gia
        full["TotalSF"] = full["TotalBsmtSF"] + full["1stFlrSF"] + full["2ndFlrSF"]
        created.append("TotalSF")
    if has("FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"):
        full["TotalBath"] = (full["FullBath"] + 0.5 * full["HalfBath"]
                             + full["BsmtFullBath"] + 0.5 * full["BsmtHalfBath"])
        created.append("TotalBath")
    if has("YrSold", "YearBuilt"):
        full["HouseAge"] = full["YrSold"] - full["YearBuilt"]
        created.append("HouseAge")
    if has("YrSold", "YearRemodAdd"):
        full["RemodAge"] = full["YrSold"] - full["YearRemodAdd"]
        full["IsRemodeled"] = (full["YearRemodAdd"] != full["YearBuilt"]).astype(int)
        created += ["RemodAge", "IsRemodeled"]
    if has("OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "WoodDeckSF"):
        full["TotalPorchSF"] = (full["OpenPorchSF"] + full["EnclosedPorch"]
                                + full["3SsnPorch"] + full["ScreenPorch"] + full["WoodDeckSF"])
        created.append("TotalPorchSF")
    for src, name in [("PoolArea", "HasPool"), ("2ndFlrSF", "Has2ndFloor"),
                      ("GarageArea", "HasGarage"), ("TotalBsmtSF", "HasBsmt"),
                      ("Fireplaces", "HasFireplace")]:
        if src in full.columns:
            full[name] = (full[src] > 0).astype(int)
            created.append(name)
    if has("OverallQual", "TotalSF"):
        full["Qual_x_SF"] = full["OverallQual"] * full["TotalSF"]
        created.append("Qual_x_SF")
    log(f"[3.6] Tao moi {len(created)} dac trung: {', '.join(created)}")

    # --- 3.7 MSSubClass thuc chat la ma phan loai, khong phai so ---------
    if "MSSubClass" in full.columns:
        full["MSSubClass"] = full["MSSubClass"].astype(str)

    # --- 3.8 Giam do lech cua bien so bang log1p -------------------------
    num_feats = full.select_dtypes(include=[np.number]).columns
    sk = full[num_feats].apply(lambda s: skew(s.dropna()))
    skewed = sk[abs(sk) > 0.75].index
    full[skewed] = np.log1p(full[skewed].clip(lower=0))
    log(f"[3.7] log1p cho {len(skewed)} bien so co |skew| > 0.75")

    # --- 3.9 One-hot encoding cho bien danh muc --------------------------
    full = pd.get_dummies(full, drop_first=True).astype(float)
    log(f"[3.8] Sau one-hot encoding: {full.shape[1]} dac trung")

    X = full.iloc[:n_train].values
    X_test = full.iloc[n_train:].values
    return X, y, X_test, test_id, full.columns.tolist(), train


# ===========================================================================
# BUOC 4 - MODELING (Xay dung mo hinh)
# ===========================================================================
def rmse_cv(model, X, y, n_folds=N_FOLDS):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, scoring="neg_root_mean_squared_error", cv=kf)
    return -scores


def build_models():
    """Tra ve dict {ten: mo hinh}. Mo hinh tuyen tinh duoc dat trong Pipeline
    co RobustScaler vi chuan hoa la bat buoc voi Ridge/Lasso/ElasticNet."""
    models = {
        "Ridge": Pipeline([("sc", RobustScaler()), ("m", Ridge(alpha=10.0, random_state=RANDOM_STATE))]),
        "Lasso": Pipeline([("sc", RobustScaler()), ("m", Lasso(alpha=0.0005, max_iter=10000, random_state=RANDOM_STATE))]),
        "ElasticNet": Pipeline([("sc", RobustScaler()), ("m", ElasticNet(alpha=0.0005, l1_ratio=0.9, max_iter=10000, random_state=RANDOM_STATE))]),
        "RandomForest": RandomForestRegressor(n_estimators=400, max_features="sqrt", min_samples_leaf=1, n_jobs=-1, random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=800, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=RANDOM_STATE),
    }
    try:  # XGBoost neu may co cai dat
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(n_estimators=900, learning_rate=0.05, max_depth=3,
                                         subsample=0.8, colsample_bytree=0.8,
                                         reg_lambda=1.0, random_state=RANDOM_STATE, n_jobs=-1)
    except ImportError:
        pass
    return models


def modeling(X, y, log: Logger, out_dir: str, fast: bool = False):
    log.section("BUOC 4 | MODELING - HUAN LUYEN & TINH CHINH SIEU THAM SO")

    # --- 4.1 Tinh chinh alpha cho Ridge bang GridSearchCV ---------------
    log("[4.1] GridSearchCV tim alpha toi uu cho Ridge:")
    grid = GridSearchCV(
        Pipeline([("sc", RobustScaler()), ("m", Ridge(random_state=RANDOM_STATE))]),
        {"m__alpha": [0.1, 1, 5, 10, 20, 30, 50, 100]},
        scoring="neg_root_mean_squared_error",
        cv=KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
    )
    grid.fit(X, y)
    best_alpha = grid.best_params_["m__alpha"]
    log(f"      alpha tot nhat = {best_alpha} | RMSE(CV) = {-grid.best_score_:.5f}")

    models = build_models()
    models["Ridge"] = Pipeline([("sc", RobustScaler()), ("m", Ridge(alpha=best_alpha, random_state=RANDOM_STATE))])
    if fast:
        models["RandomForest"].set_params(n_estimators=80)
        models["GradientBoosting"].set_params(n_estimators=120)

    # --- 4.2 So sanh cac mo hinh bang 5-fold CV -------------------------
    log(f"\n[4.2] Danh gia {len(models)} mo hinh bang {N_FOLDS}-fold Cross-Validation:")
    rows = []
    for name, model in models.items():
        t0 = time.time()
        s = rmse_cv(model, X, y)
        rows.append({"Mo hinh": name, "RMSE_CV": s.mean(), "Do lech chuan": s.std(),
                     "Thoi gian (s)": round(time.time() - t0, 1)})
        log(f"      {name:<18} RMSE = {s.mean():.5f} (+/- {s.std():.5f})   [{rows[-1]['Thoi gian (s)']}s]")

    board = pd.DataFrame(rows).sort_values("RMSE_CV").reset_index(drop=True)
    board.to_csv(os.path.join(out_dir, "model_comparison.csv"), index=False)
    log("\n[4.3] Bang xep hang (RMSE cang nho cang tot):")
    log(board.to_string(index=False))

    return models, board


# ===========================================================================
# BUOC 5 - EVALUATION (Danh gia)
# ===========================================================================
def evaluation(models, board, X, y, log: Logger, fig_dir: str, feat_names):
    log.section("BUOC 5 | EVALUATION - DANH GIA MO HINH")

    best_name = board.iloc[0]["Mo hinh"]
    log(f"[5.1] Mo hinh tot nhat theo CV: {best_name} (RMSE={board.iloc[0]['RMSE_CV']:.5f})")

    # --- 5.2 Mo hinh ket hop (blending) ---------------------------------
    top2 = board.head(2)["Mo hinh"].tolist()
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = {n: np.zeros(len(y)) for n in top2}
    for tr, va in kf.split(X):
        for n in top2:
            import sklearn.base as skb
            m = skb.clone(models[n])
            m.fit(X[tr], y[tr])
            oof[n][va] = m.predict(X[va])
    blend = 0.5 * oof[top2[0]] + 0.5 * oof[top2[1]]
    rmse_blend = float(np.sqrt(mean_squared_error(y, blend)))
    log(f"[5.2] Blend 50/50 ({top2[0]} + {top2[1]}): RMSE = {rmse_blend:.5f}")
    use_blend = rmse_blend < board.iloc[0]["RMSE_CV"]
    log(f"      -> {'DUNG BLEND' if use_blend else 'GIU MO HINH DON'} cho ban nop cuoi cung")

    # --- 5.3 Danh gia tren tap holdout, quy doi ve don vi DOLA ----------
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    import sklearn.base as skb
    m = skb.clone(models[best_name]).fit(X_tr, y_tr)
    pred_log = m.predict(X_va)
    pred_usd, true_usd = np.expm1(pred_log), np.expm1(y_va)

    log("\n[5.3] Danh gia tren tap holdout 20% (quy ve dong dola de de dien giai):")
    log(f"      RMSE (log)          = {np.sqrt(mean_squared_error(y_va, pred_log)):.5f}")
    log(f"      R^2  (log)          = {r2_score(y_va, pred_log):.4f}")
    log(f"      MAE  ($)            = {mean_absolute_error(true_usd, pred_usd):,.0f}")
    log(f"      RMSE ($)            = {np.sqrt(mean_squared_error(true_usd, pred_usd)):,.0f}")
    log(f"      Bias trung binh ($) = {(pred_usd - true_usd).mean():,.0f}  (duong = du doan cao hon thuc te)")
    log(f"      Sai so lon nhat ($) = {np.max(np.abs(pred_usd - true_usd)):,.0f}")
    log(f"      Sai so % trung binh = {np.mean(np.abs(pred_usd - true_usd) / true_usd) * 100:.2f}%")

    # --- 5.4 Kiem tra thu cong 1 quan sat (theo yeu cau cua De Cock) ----
    i = 0
    log("\n[5.4] Kiem tra thu cong 1 quan sat (Model Check - De Cock 2011):")
    log(f"      Gia du doan = ${pred_usd[i]:,.0f} | Gia thuc te = ${true_usd[i]:,.0f} "
        f"| Sai so = ${abs(pred_usd[i] - true_usd[i]):,.0f}")

    # --- 5.5 Bieu do phan du + thuc te vs du doan ------------------------
    resid = y_va - pred_log
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].scatter(pred_log, resid, s=14, alpha=0.6, color="#4C72B0")
    ax[0].axhline(0, c="red", ls="--")
    ax[0].set_xlabel("Gia tri du doan (log)"); ax[0].set_ylabel("Phan du")
    ax[0].set_title("Phan du vs Du doan")
    ax[1].hist(resid, bins=40, color="#55A868", edgecolor="white")
    ax[1].set_title("Phan bo phan du (ky vong: chuong doi xung quanh 0)")
    ax[2].scatter(true_usd, pred_usd, s=14, alpha=0.6, color="#8172B2")
    lim = [min(true_usd.min(), pred_usd.min()), max(true_usd.max(), pred_usd.max())]
    ax[2].plot(lim, lim, "r--")
    ax[2].set_xlabel("Gia thuc te ($)"); ax[2].set_ylabel("Gia du doan ($)")
    ax[2].set_title("Thuc te vs Du doan")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "05_residuals.png"), dpi=110)
    plt.close()

    # --- 5.6 So sanh RMSE cac mo hinh ------------------------------------
    plt.figure(figsize=(8, 4.5))
    b = board.iloc[::-1]
    plt.barh(b["Mo hinh"], b["RMSE_CV"], xerr=b["Do lech chuan"], color="#4C72B0")
    plt.xlabel("RMSE (5-fold CV, thang log)")
    plt.title("So sanh cac mo hinh")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "06_model_comparison.png"), dpi=110)
    plt.close()

    # --- 5.7 Do quan trong cua dac trung ---------------------------------
    full_fit = skb.clone(models[best_name]).fit(X, y)
    est = full_fit.named_steps["m"] if isinstance(full_fit, Pipeline) else full_fit
    imp = getattr(est, "feature_importances_", None)
    if imp is None:
        imp = np.abs(getattr(est, "coef_", np.zeros(X.shape[1])))
    top = pd.Series(imp, index=feat_names).sort_values(ascending=False).head(15)
    log("\n[5.5] 15 dac trung anh huong manh nhat:")
    log(top.to_string())
    plt.figure(figsize=(8, 5.5))
    plt.barh(top.index[::-1], top.values[::-1], color="#DD8452")
    plt.title(f"Top 15 dac trung quan trong ({best_name})")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "07_feature_importance.png"), dpi=110)
    plt.close()

    return best_name, top2, use_blend


# ===========================================================================
# BUOC 6 - DEPLOYMENT (Trien khai)
# ===========================================================================
def deployment(models, best_name, top2, use_blend, X, y, X_test, test_id, log: Logger, out_dir: str):
    log.section("BUOC 6 | DEPLOYMENT - SINH FILE NOP KAGGLE")
    import sklearn.base as skb

    if use_blend:
        preds = np.zeros(len(X_test))
        for n in top2:
            preds += 0.5 * skb.clone(models[n]).fit(X, y).predict(X_test)
        log(f"[6.1] Huan luyen lai tren TOAN BO train, blend {top2[0]} + {top2[1]}")
    else:
        preds = skb.clone(models[best_name]).fit(X, y).predict(X_test)
        log(f"[6.1] Huan luyen lai {best_name} tren TOAN BO train")

    final = np.expm1(preds)  # doi nguoc log1p -> ve dong dola
    sub = pd.DataFrame({"Id": test_id, "SalePrice": final})
    path = os.path.join(out_dir, "submission.csv")
    sub.to_csv(path, index=False)
    log(f"[6.2] Da ghi {path} ({len(sub)} dong)")
    log(f"[6.3] Kiem tra nhanh ban nop: min=${final.min():,.0f} max=${final.max():,.0f} "
        f"mean=${final.mean():,.0f} | so gia tri am hoac NaN = "
        f"{int((final <= 0).sum() + np.isnan(final).sum())}")
    log(sub.head().to_string(index=False))
    return sub


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="House Prices - pipeline CRISP-DM")
    ap.add_argument("--data-dir", default="data", help="Thu muc chua train.csv va test.csv")
    ap.add_argument("--out-dir", default="outputs", help="Thu muc ghi ket qua")
    ap.add_argument("--fast", action="store_true", help="Che do chay nhanh (it cay hon) de kiem thu")
    args = ap.parse_args()

    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    log = Logger(os.path.join(args.out_dir, "run_log.txt"))

    tr_path = os.path.join(args.data_dir, "train.csv")
    te_path = os.path.join(args.data_dir, "test.csv")
    if not (os.path.exists(tr_path) and os.path.exists(te_path)):
        print(f"[LOI] Khong tim thay {tr_path} / {te_path}.\n"
              f"      Tai du lieu tai: https://www.kaggle.com/competitions/"
              f"house-prices-advanced-regression-techniques/data\n"
              f"      roi dat train.csv va test.csv vao thu muc '{args.data_dir}'.")
        sys.exit(1)

    log.section("BUOC 1 | BUSINESS UNDERSTANDING - HIEU BAI TOAN")
    log(BUSINESS_UNDERSTANDING)

    train = pd.read_csv(tr_path)
    test = pd.read_csv(te_path)

    data_understanding(train, test, log, fig_dir)
    X, y, X_test, test_id, feat_names, train_clean = data_preparation(train, test, log)
    models, board = modeling(X, y, log, args.out_dir, fast=args.fast)
    best_name, top2, use_blend = evaluation(models, board, X, y, log, fig_dir, feat_names)
    deployment(models, best_name, top2, use_blend, X, y, X_test, test_id, log, args.out_dir)

    log.section("HOAN TAT - TAT CA KET QUA NAM TRONG THU MUC: " + args.out_dir)
    log.close()


if __name__ == "__main__":
    main()
