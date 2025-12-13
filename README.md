# Decision Tree Demo - Iris & Titanic

Demo trực quan về **Decision Tree Classifier** với 2 dataset: **Iris** (lý thuyết cơ bản) và **Titanic** (ứng dụng thực tế), xây dựng bằng **Python** và **Streamlit**.

## Đề tài

**Nghiên cứu về Cây Quyết Định và các Độ đo, các Phương pháp tính Độ chính xác Phân lớp**

---

## Mục tiêu

1. **Lý thuyết**: Hiểu thuật toán Decision Tree, so sánh Gini vs Entropy
2. **Thực hành**: Áp dụng vào bài toán thực tế với dữ liệu phức tạp
3. **Đánh giá**: Các phương pháp đo lường độ chính xác (Accuracy, Precision, Recall, F1, ROC)

---

## Nội dung dự án

```
iris-decision-tree-demo/
├── main_app.py              # App tổng hợp (trang chủ + hướng dẫn)
├── iris_demo.py             # Demo Iris - Lý thuyết cơ bản
├── titanic_demo.py          # Demo Titanic - Ứng dụng thực tế
├── data/
│   └── titanic.csv          # Dataset Titanic
├── utils/
│   ├── __init__.py
│   └── preprocessing.py     # Tiền xử lý dữ liệu Titanic
├── requirements.txt         # Dependencies
└── README.md
```

### **Demo 1: Iris - Lý thuyết cơ bản**
- Phân loại hoa Iris thành 3 loại (Setosa, Versicolor, Virginica)
- So sánh Gini vs Entropy
- Visualization cây quyết định
- Feature Importance
- Confusion Matrix & Metrics

### **Demo 2: Titanic - Ứng dụng thực tế**
- Dự đoán khả năng sống sót sau thảm họa Titanic 1912
- Tiền xử lý dữ liệu thực tế (missing values, encoding)
- Feature Engineering (FamilySize, Title, HasCabin)
- Train/Test Split
- ROC Curve & AUC
- Dự đoán tương tác với giải thích

---

## Hướng dẫn cài đặt

1. **Clone repository**

```bash
git clone https://github.com/Viethiep49/iris-decision-tree-demo.git
cd <ten_folder>
```

2. **Cài đặt Python 3.12+ và tạo môi trường ảo**

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows
# hoặc
source .venv/bin/activate       # macOS/Linux
```

3. **Cài đặt các thư viện cần thiết**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Cài đặt Graphviz hệ thống** (để hiển thị cây quyết định):

* Windows: tải từ [Graphviz Download](https://graphviz.org/download/) và thêm vào PATH.
* Linux: `sudo apt install graphviz`
* macOS: `brew install graphviz`

---

## Cách chạy ứng dụng

### **Option 1: App tổng hợp (khuyên dùng)**
```bash
streamlit run main_app.py
```
- Trang chủ với hướng dẫn đầy đủ
- Menu chọn demo Iris hoặc Titanic
- Lý thuyết về Gini, Entropy, Metrics

### **Option 2: Chạy từng demo riêng lẻ**

**Demo Iris (Lý thuyết cơ bản):**
```bash
streamlit run iris_demo.py
```

**Demo Titanic (Ứng dụng thực tế):**
```bash
streamlit run titanic_demo.py
```

* Streamlit sẽ mở trình duyệt mặc định với giao diện tương tác.
* Bạn có thể thay đổi tham số, so sánh Gini vs Entropy, và dự đoán mẫu mới.

---

## Kiến thức chính

### **1. Các độ đo (Impurity Measures)**

#### Gini Index
```
Gini = 1 - Σ(pi²)
```
- Xác suất chọn 2 mẫu khác lớp
- Giá trị: 0 (thuần khiết) → 0.5 (tạp)
- Tính nhanh, hiệu quả

#### Entropy
```
Entropy = -Σ(pi × log2(pi))
```
- Độ hỗn loạn của dữ liệu
- Giá trị: 0 (thuần khiết) → log2(n_classes)
- Chính xác về mặt lý thuyết

### **2. Phương pháp đánh giá độ chính xác**

| Metric | Công thức | Ý nghĩa |
|--------|-----------|---------|
| **Accuracy** | (TP + TN) / Total | Tỷ lệ dự đoán đúng tổng thể |
| **Precision** | TP / (TP + FP) | Độ chính xác khi dự đoán Positive |
| **Recall** | TP / (TP + FN) | Khả năng tìm ra Positive |
| **F1-Score** | 2 × (P × R) / (P + R) | Trung bình hài hòa P và R |
| **AUC-ROC** | Diện tích dưới ROC | Khả năng phân biệt 2 lớp |

---

## Hướng dẫn Thuyết trình

### **Kịch bản đề xuất (20 phút)**

1. **Giới thiệu (2 phút)**
   - Cây Quyết Định là gì?
   - Ứng dụng thực tế

2. **Lý thuyết - Demo Iris (6 phút)**
   - Giải thích Gini vs Entropy
   - So sánh 2 độ đo
   - Đọc cây quyết định
   - Feature Importance

3. **Thực hành - Demo Titanic (7 phút)**
   - Khám phá dữ liệu (EDA)
   - Tiền xử lý (missing values, encoding)
   - Feature Engineering
   - ROC Curve & AUC

4. **Demo tương tác (3 phút)**
   - Dự đoán hoa Iris
   - Dự đoán sống sót Titanic

5. **Q&A (2 phút)**

---

## So sánh Iris vs Titanic

| Tiêu chí | Iris | Titanic |
|----------|------|---------|
| Số mẫu | 150 | ~900 |
| Features | 4 (số) | 10+ (số + chữ) |
| Classes | 3 (cân bằng) | 2 (không cân bằng) |
| Missing values | Không | Có nhiều |
| Độ phức tạp | Đơn giản | Phức tạp |
| Mục đích | Học lý thuyết | Ứng dụng thực tế |

---

## Tham khảo

* [Scikit-learn Decision Tree](https://scikit-learn.org/stable/modules/tree.html)
* [Streamlit Documentation](https://docs.streamlit.io/)
* [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)
* [Titanic Dataset](https://www.kaggle.com/c/titanic)
* [ROC Curve & AUC](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html)

---

## License

MIT License - Tự do sử dụng cho mục đích học tập
