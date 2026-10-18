# Lab 02 - House Prices: Advanced Regression Techniques

Bai lab xay dung mo hinh du doan gia nha tren bo du lieu **Ames Housing**
(De Cock, 2011) theo quy trinh **CRISP-DM**.

## 1. Cau truc thu muc

```
lab02_house_price/
├── README.md                 # file nay
├── requirements.txt          # thu vien can cai
├── data/                     # ĐẶT train.csv, test.csv CUA KAGGLE VAO DAY
├── src/
│   └── house_price_crisp.py  # pipeline chinh - 6 buoc CRISP-DM
├── tests/
│   └── test_pipeline.py      # kiem thu tu dong (khong can du lieu Kaggle)
├── outputs/                  # ket qua sinh ra khi chay
│   ├── run_log.txt           # toan bo log qua trinh chay
│   ├── missing_report.csv    # bao cao du lieu thieu
│   ├── model_comparison.csv  # bang so sanh cac mo hinh
│   ├── submission.csv        # file nop len Kaggle
│   └── figures/*.png         # 7 bieu do EDA & danh gia
└── report/
    └── BaoCao_Lab02_House_Price.docx
```

## 2. Cai dat

```bash
pip install -r requirements.txt
```

## 3. Tai du lieu

Tai `train.csv` va `test.csv` tai:
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
roi dat vao thu muc `data/`.

## 4. Chay

```bash
# Chay day du 6 buoc CRISP-DM
python src/house_price_crisp.py --data-dir data --out-dir outputs

# Chay nhanh (it cay hon) de kiem tra moi truong
python src/house_price_crisp.py --data-dir data --out-dir outputs --fast
```

## 5. Kiem thu

`tests/test_pipeline.py` sinh du lieu gia lap dung schema Ames roi chay toan bo
pipeline va kiem tra: khong con gia tri thieu, so cot train/test khop nhau,
`submission.csv` du dong va khong co gia tri am/NaN.

```bash
python tests/test_pipeline.py
```

## 6. Ket qua tham khao (5-fold CV, RMSE tren log(SalePrice))

Diem so thuc te khi chay tren du lieu Kaggle se duoc ghi trong
`outputs/model_comparison.csv` va `outputs/run_log.txt`.
