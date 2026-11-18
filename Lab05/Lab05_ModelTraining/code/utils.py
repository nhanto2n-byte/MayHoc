"""
utils.py - Hàm dùng chung cho Lab05 (EDA + Phân lớp).
Các notebook chỉ cần:  from utils import *
"""
import re
import fnmatch
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from IPython.display import display
except ImportError:                      # chạy ngoài Jupyter
    display = print

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Đường dẫn & cấu hình chung
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"          # đặt các file CSV tải từ Kaggle vào đây
FIG_DIR = ROOT / "figures"        # ảnh biểu đồ xuất ra (dùng chèn vào báo cáo)
OUT_DIR = ROOT / "outputs"        # bảng kết quả, báo cáo HTML của công cụ EDA
for _d in (DATA_DIR, FIG_DIR, OUT_DIR):
    _d.mkdir(exist_ok=True)

RANDOM_STATE = 42
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.unicode_minus"] = False


# ----------------------------------------------------------------------------
# Đọc dữ liệu
# ----------------------------------------------------------------------------
def find_data(*patterns, required=True):
    """Tìm file trong thư mục data/ (không phân biệt hoa thường, hỗ trợ wildcard)."""
    files = [p for p in DATA_DIR.rglob("*") if p.is_file()]
    for pat in patterns:
        for p in files:
            if fnmatch.fnmatch(p.name.lower(), pat.lower()):
                return p
    if required:
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu khớp {patterns} trong {DATA_DIR}.\n"
            f"-> Tải file từ Kaggle (xem data/README.md) rồi chép vào thư mục data/."
        )
    return None


def load_table(*patterns, **kw):
    """Đọc CSV/Excel, tự nhận dấu phân cách (',', ';', tab) và encoding."""
    path = find_data(*patterns)
    print(f"[Đọc dữ liệu] {path.relative_to(ROOT)}")
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path, **kw)
    if "sep" not in kw:
        head = open(path, "rb").read(4096).decode("latin-1")
        first = head.splitlines()[0] if head else ""
        counts = {",": first.count(","), "\t": first.count("\t"), ";": first.count(";")}
        kw["sep"] = max(counts, key=counts.get)
    encs = [kw.pop("encoding")] if "encoding" in kw else ["utf-8", "latin-1"]
    for enc in encs:
        try:
            return pd.read_csv(path, encoding=enc, **kw)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Không đọc được {path}")


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def guess_column(df, candidates, required=True):
    """Tìm tên cột (bỏ qua hoa/thường, dấu cách, dấu chấm...) trong danh sách ứng viên."""
    lookup = {_norm(c): c for c in df.columns}
    for cand in candidates:
        if _norm(cand) in lookup:
            return lookup[_norm(cand)]
    if required:
        raise KeyError(f"Không tìm thấy cột nào trong {candidates}. Các cột hiện có: {list(df.columns)}")
    return None


def load_diabetes(clean_zeros=True):
    """Dữ liệu tiểu đường (Pima). clean_zeros=True: coi 0 ở các cột y khoa là thiếu (NaN)."""
    df = load_table("diabetes.csv", "diabetes*.csv", "pima*.csv")
    target = guess_column(df, ["Outcome", "diabetes", "Diabetes_binary", "class", "target"])
    if clean_zeros:
        for c in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
            if c in df.columns:
                df[c] = df[c].replace(0, np.nan)
    return df, target


# ----------------------------------------------------------------------------
# Tổng quan & thống kê mô tả
# ----------------------------------------------------------------------------
def overview(df, name="dataset", n=5):
    """In thông tin tổng quan: kích thước, kiểu dữ liệu, thiếu, trùng lặp."""
    print(f"===== {name} =====")
    print(f"Kích thước: {df.shape[0]} dòng x {df.shape[1]} cột")
    print(f"Số dòng trùng lặp: {df.duplicated().sum()}")
    info = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_%": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(),
    })
    display(info)
    display(df.head(n))


def describe_full(df):
    """Bảng thống kê mô tả mở rộng cho các cột số."""
    num = df.select_dtypes(include="number")
    t = pd.DataFrame({
        "count": num.count(),
        "missing": num.isna().sum(),
        "mean": num.mean(),
        "median": num.median(),
        "mode": num.apply(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan),
        "std": num.std(),
        "var": num.var(),
        "min": num.min(),
        "max": num.max(),
        "range": num.max() - num.min(),
        "Q1": num.quantile(0.25),
        "Q3": num.quantile(0.75),
    })
    t["IQR"] = t["Q3"] - t["Q1"]
    t["skew"] = num.skew()
    t["kurtosis"] = num.kurt()
    return t.round(3)


def outlier_table(df, k=1.5):
    """Đếm ngoại lai theo quy tắc IQR: ngoài [Q1 - k*IQR, Q3 + k*IQR]."""
    rows = []
    for c in df.select_dtypes(include="number").columns:
        s = df[c].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        n_out = int(((s < lo) | (s > hi)).sum())
        rows.append([c, lo, hi, n_out, round(100 * n_out / max(len(s), 1), 2)])
    out = pd.DataFrame(rows, columns=["column", "lower_fence", "upper_fence", "n_outliers", "outlier_%"])
    return out.sort_values("n_outliers", ascending=False).reset_index(drop=True)


def clip_outliers_iqr(df, cols, k=1.5):
    """Winsorize: kẹp giá trị ngoại lai về hai rào IQR (trả về bản sao)."""
    out = df.copy()
    for c in cols:
        q1, q3 = out[c].quantile(0.25), out[c].quantile(0.75)
        iqr = q3 - q1
        out[c] = out[c].clip(q1 - k * iqr, q3 + k * iqr)
    return out


# ----------------------------------------------------------------------------
# Vẽ & lưu hình
# ----------------------------------------------------------------------------
def slug(s):
    """Chuẩn hóa chuỗi thành tên file ASCII (bỏ dấu tiếng Việt)."""
    import unicodedata
    s = str(s).replace("đ", "d").replace("Đ", "D")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def save_fig(name, show=True, dpi=150):
    """Lưu hình vào figures/ (PNG) rồi hiển thị."""
    path = FIG_DIR / (name if name.lower().endswith((".png", ".pdf", ".svg")) else name + ".png")
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    print(f"[Đã lưu] {path.relative_to(ROOT)}")


def grid_plot(df, cols, kind="hist", ncols=3, hue=None, figsize_per=(4.2, 3.2), name=None):
    """Vẽ lưới histogram / boxplot cho nhiều cột."""
    cols = list(cols)
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, cols):
        if kind == "hist":
            sns.histplot(data=df, x=c, hue=hue, kde=True, ax=ax, element="step")
        elif kind == "box":
            if hue is None:
                sns.boxplot(data=df, x=c, ax=ax)
            else:
                sns.boxplot(data=df, x=hue, y=c, ax=ax)
        ax.set_title(c)
    for ax in axes[len(cols):]:
        ax.axis("off")
    if name:
        save_fig(name)
    else:
        plt.tight_layout()
        plt.show()


# ----------------------------------------------------------------------------
# Tiền xử lý & đánh giá mô hình
# ----------------------------------------------------------------------------
def make_preprocessor(X, scale=True):
    """ColumnTransformer: số -> impute median (+ chuẩn hóa), phân loại -> impute mode + One-Hot."""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline(num_steps), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols))
    return ColumnTransformer(transformers)


def evaluate_model(model, X_test, y_test, name="model", class_names=None, plot=True):
    """In classification report, vẽ confusion matrix, trả về dict các chỉ số."""
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, confusion_matrix,
                                 classification_report)
    y_true = np.asarray(y_test)
    y_pred = np.asarray(model.predict(X_test))
    classes = np.unique(np.concatenate([y_true, y_pred]))
    if len(classes) == 2:
        kw = dict(average="binary", pos_label=classes[1], zero_division=0)
    else:
        kw = dict(average="weighted", zero_division=0)
    res = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, **kw),
        "recall": recall_score(y_true, y_pred, **kw),
        "f1": f1_score(y_true, y_pred, **kw),
        "roc_auc": np.nan,
    }
    try:
        if len(classes) == 2:
            if hasattr(model, "predict_proba"):
                score = model.predict_proba(X_test)[:, 1]
            else:
                score = model.decision_function(X_test)
            res["roc_auc"] = roc_auc_score(y_true, score)
    except Exception:
        pass
    labels = ([str(c) for c in class_names]
              if (class_names is not None and len(class_names) == len(classes))
              else [str(c) for c in classes])
    print(f"\n===== {name} =====")
    print(classification_report(y_true, y_pred, labels=classes, target_names=labels, zero_division=0))
    if plot:
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        big = len(classes) >= 5
        plt.figure(figsize=(7, 5.5) if big else (4.8, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Dự đoán"); plt.ylabel("Thực tế"); plt.title(f"Confusion matrix - {name}")
        save_fig("cm_" + slug(name))
    return res


def results_table(results, sort_by="f1"):
    df = pd.DataFrame(results).set_index("model")
    return df.sort_values(sort_by, ascending=False).round(4)


def plot_model_comparison(results_df, name="model_comparison"):
    ax = results_df[["accuracy", "precision", "recall", "f1"]].plot.bar(figsize=(9, 4), rot=20)
    ax.set_ylim(0, 1.05); ax.set_title("So sánh các mô hình"); ax.legend(loc="lower right")
    save_fig(name)


def plot_roc(models, X_test, y_test, name="roc_curves"):
    """models: dict {tên: mô hình đã fit}. Chỉ dùng cho bài toán nhị phân."""
    from sklearn.metrics import roc_curve, roc_auc_score
    plt.figure(figsize=(5.5, 5))
    for n, m in models.items():
        s = m.predict_proba(X_test)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, s)
        plt.plot(fpr, tpr, label=f"{n} (AUC={roc_auc_score(y_test, s):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC curves"); plt.legend()
    save_fig(name)


def plot_cv_curve(cv_results_df, param, title, name, n_folds, ylabel="ROC AUC"):
    """Vẽ train/test score (mean ± 1 SE) theo một tham số - dùng chẩn đoán overfitting."""
    col = f"param_{param}"
    x = cv_results_df[col].astype(float)
    plt.figure(figsize=(6, 4))
    plt.errorbar(x, cv_results_df["mean_train_score"],
                 yerr=cv_results_df["std_train_score"] / np.sqrt(n_folds), label="Train (mean ± 1 SE)")
    plt.errorbar(x, cv_results_df["mean_test_score"],
                 yerr=cv_results_df["std_test_score"] / np.sqrt(n_folds), label="CV test (mean ± 1 SE)")
    plt.xlabel(param); plt.ylabel(ylabel); plt.title(title); plt.legend()
    save_fig(name)


def tree_forest_experiment(X_train, X_test, y_train, y_test, feature_names, class_names, tag, cv=5):
    """
    Quy trình chuẩn cho Cây quyết định + Rừng cây (bài toán nhị phân):
      1) Cây không giới hạn  -> thấy overfitting
      2) GridSearchCV theo max_depth + biểu đồ train/test  (như lab)
      3) GridSearchCV nhiều tham số cho cây
      4) Rừng cây: biểu đồ theo số cây, GridSearchCV nhiều tham số
      5) Vẽ cây, feature importance, so sánh mô hình
    Trả về (bảng kết quả, dict mô hình tốt nhất).
    """
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV

    rs = RANDOM_STATE
    results, models = [], {}

    # 1) Baseline không giới hạn độ sâu
    dt0 = DecisionTreeClassifier(random_state=rs).fit(X_train, y_train)
    print(f"[DT không giới hạn] depth={dt0.get_depth()} | acc train={dt0.score(X_train, y_train):.3f} "
          f"| acc test={dt0.score(X_test, y_test):.3f}  -> chênh lệch lớn = overfitting")
    results.append(evaluate_model(dt0, X_test, y_test, f"{tag} DT (không giới hạn)", class_names, plot=False))

    # 2) Quét max_depth
    gs_depth = GridSearchCV(DecisionTreeClassifier(random_state=rs),
                            {"max_depth": [1, 2, 3, 4, 5, 6, 8, 10, 12]},
                            scoring="roc_auc", cv=cv, refit=True, return_train_score=True)
    gs_depth.fit(X_train, y_train)
    plot_cv_curve(pd.DataFrame(gs_depth.cv_results_), "max_depth",
                  f"{tag}: ROC AUC theo max_depth", f"{slug(tag)}_dt_depth_curve", cv)
    print("max_depth tốt nhất:", gs_depth.best_params_, "| CV AUC =", round(gs_depth.best_score_, 4))

    # 3) Tinh chỉnh cây nhiều tham số
    gs_dt = GridSearchCV(DecisionTreeClassifier(random_state=rs),
                         {"max_depth": [3, 4, 5, 6, 8], "min_samples_leaf": [1, 5, 10, 20],
                          "criterion": ["gini", "entropy"]},
                         scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
    gs_dt.fit(X_train, y_train)
    best_dt = gs_dt.best_estimator_
    print("DT tốt nhất:", gs_dt.best_params_, "| CV AUC =", round(gs_dt.best_score_, 4))
    results.append(evaluate_model(best_dt, X_test, y_test, f"{tag} DT (tối ưu)", class_names))
    models["Decision Tree"] = best_dt

    # Vẽ cây (cắt ở độ sâu 3 cho dễ đọc)
    plt.figure(figsize=(18, 8))
    plot_tree(best_dt, feature_names=list(feature_names), class_names=[str(c) for c in class_names],
              filled=True, rounded=True, proportion=True, max_depth=3, fontsize=8)
    plt.title(f"{tag}: Cây quyết định tối ưu (hiển thị tối đa 3 tầng)")
    save_fig(f"{slug(tag)}_decision_tree")

    # 4) Rừng cây
    rf = RandomForestClassifier(random_state=rs, n_jobs=-1)
    gs_n = GridSearchCV(rf, {"n_estimators": list(range(10, 210, 20))},
                        scoring="roc_auc", cv=cv, refit=True, return_train_score=True)
    gs_n.fit(X_train, y_train)
    r = pd.DataFrame(gs_n.cv_results_)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    axs[0].plot(r["param_n_estimators"].astype(float), r["mean_fit_time"], "-o")
    axs[0].set_xlabel("Số cây"); axs[0].set_ylabel("Mean fit time (s)")
    axs[1].errorbar(r["param_n_estimators"].astype(float), r["mean_test_score"],
                    yerr=r["std_test_score"] / np.sqrt(cv))
    axs[1].set_xlabel("Số cây"); axs[1].set_ylabel("Mean CV ROC AUC ± 1 SE")
    save_fig(f"{slug(tag)}_rf_ntrees")

    gs_rf = GridSearchCV(RandomForestClassifier(random_state=rs, n_jobs=-1),
                         {"n_estimators": [100, 200], "max_depth": [3, 5, 8, None],
                          "min_samples_leaf": [1, 3, 5], "max_features": ["sqrt", "log2"]},
                         scoring="roc_auc", cv=cv, refit=True)
    gs_rf.fit(X_train, y_train)
    best_rf = gs_rf.best_estimator_
    print("RF tốt nhất:", gs_rf.best_params_, "| CV AUC =", round(gs_rf.best_score_, 4))
    print(f"[RF tối ưu] acc train={best_rf.score(X_train, y_train):.3f} | acc test={best_rf.score(X_test, y_test):.3f}")
    results.append(evaluate_model(best_rf, X_test, y_test, f"{tag} RF (tối ưu)", class_names))
    models["Random Forest"] = best_rf

    # 5) Feature importance
    imp = pd.Series(best_rf.feature_importances_, index=feature_names).sort_values()
    plt.figure(figsize=(6, max(3, 0.32 * len(imp))))
    imp.plot.barh(); plt.title(f"{tag}: Feature importance (Random Forest)"); plt.xlabel("Importance")
    save_fig(f"{slug(tag)}_feature_importance")
    display(imp.sort_values(ascending=False).round(4).to_frame("importance"))

    table = results_table(results)
    display(table)
    plot_model_comparison(table, f"{slug(tag)}_compare")
    plot_roc(models, X_test, y_test, f"{slug(tag)}_roc")
    return table, models
