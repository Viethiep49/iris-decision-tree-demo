# Iris Decision Tree Demo

Demo trực quan về **Decision Tree Classifier** sử dụng **Iris Dataset**, xây dựng bằng **Python** và **Streamlit**. Mục tiêu là minh họa cách phân loại hoa Iris dựa trên 4 thuộc tính: Sepal Length, Sepal Width, Petal Length, Petal Width.

---

## Nội dung dự án

* `iris_demo.py`: File chính chạy ứng dụng Streamlit.
* Hiển thị các tính năng:

  * Nhập thông số mẫu hoa Iris để dự đoán.
  * Hiển thị metric: Accuracy, Precision, Recall, F1-score.
  * Confusion Matrix trực quan.
  * Feature Importance.
  * Visualization cây quyết định.

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
pip install streamlit scikit-learn matplotlib seaborn pandas graphviz plotly
```

4. **Cài đặt Graphviz hệ thống** (để hiển thị cây quyết định):

* Windows: tải từ [Graphviz Download](https://graphviz.org/download/) và thêm vào PATH.
* Linux: `sudo apt install graphviz`
* macOS: `brew install graphviz`

---

## Cách chạy ứng dụng

```bash
streamlit run iris_demo.py
```

* Streamlit sẽ mở trình duyệt mặc định với giao diện tương tác.
* Bạn có thể nhập thông số hoa Iris, chọn độ đo (`gini` hoặc `entropy`) và độ sâu tối đa của cây.

---

## Demo ảnh hưởng khi nhập thông số

* Nếu nhập giá trị ngoài khoảng thực tế của hoa Iris, mô hình vẫn dự đoán nhưng kết quả không đáng tin cậy.
* Giá trị hợp lệ:

  * Sepal Length, Sepal Width, Petal Length, Petal Width ~ 0–10 cm.

---


## Tham khảo

* [Scikit-learn Decision Tree](https://scikit-learn.org/stable/modules/tree.html)
* [Streamlit Documentation](https://docs.streamlit.io/)
* [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)
