# Lab05 - Thực hành bài toán phân lớp (nhóm)

## Cấu trúc
```
Lab05_ModelTraining/
├── code/
│   ├── utils.py                                  # hàm dùng chung (đọc dữ liệu, thống kê, đánh giá mô hình, vẽ hình)
│   ├── 11_EDA_1_1_Thong_ke_mo_ta.ipynb           # Lab 1.1  Thống kê mô tả
│   ├── 12_EDA_1_2_Truc_quan_hoa.ipynb            # Lab 1.2  Xử lý & trực quan hóa
│   ├── 13_EDA_1_3_Don_bien_hai_bien.ipynb        # Lab 1.3  Đơn biến, hai biến, SweetViz, AutoViz
│   ├── 21_CLS_2_1_Decision_Tree_Random_Forest.ipynb   # Lab 2.1
│   ├── 22_CLS_2_2_SVM.ipynb                      # Lab 2.2
│   └── 23_CLS_2_3_Naive_Bayes.ipynb              # Lab 2.3
├── data/        # đặt file CSV tải từ Kaggle vào đây (xem data/README.md)
├── figures/     # ảnh biểu đồ tự động lưu khi chạy (chèn vào báo cáo)
├── outputs/     # bảng kết quả (.csv), báo cáo HTML của ydata-profiling / SweetViz, AutoViz
└── requirements.txt
```

## Cách chạy
1. `pip install -r requirements.txt`
2. Tải dữ liệu theo `data/README.md`, chép vào `data/`.
3. `cd code` rồi `jupyter notebook`, mở lần lượt từng notebook và chạy **Run All**.
   (Notebook phải chạy từ thư mục `code/` để import được `utils.py`.)

## Quy ước
- `random_state` cố định (42; riêng bài mẫu Credit Card/Iris/Digits theo đề) để tái lập kết quả.
- Chia train/test có `stratify` cho bài toán mất cân bằng lớp.
- Tiền xử lý (impute, chuẩn hóa) đặt trong Pipeline hoặc fit trên tập train để tránh rò rỉ dữ liệu.
- Chọn siêu tham số bằng cross-validation trên tập train; tập test chỉ dùng để báo cáo kết quả cuối.
- Cài đặt thư viện tùy chọn (ydata-profiling, dtale, sweetviz, autoviz, plotly) chưa có thì cell tương ứng chỉ in hướng dẫn cài, các cell khác vẫn chạy bình thường.
