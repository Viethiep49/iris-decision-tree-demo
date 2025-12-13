# -*- coding: utf-8 -*-
# -------------------------------
# App tổng hợp: Iris + Titanic Decision Tree Demo
# -------------------------------

import streamlit as st

# Cấu hình trang
st.set_page_config(
    page_title="Decision Tree Demo - Iris & Titanic",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar - Menu chọn demo
st.sidebar.title("Decision Tree Demo")
st.sidebar.markdown("---")

demo_choice = st.sidebar.radio(
    "Chọn Demo:",
    ["Trang chủ", "Iris - Lý thuyết cơ bản", "Titanic - Ứng dụng thực tế"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### Đề tài
**Nghiên cứu về Cây Quyết Định**
- Các độ đo: Gini, Entropy
- Phương pháp tính độ chính xác
- Ứng dụng thực tế

### Thông tin
Môn: Trí Tuệ Nhân Tạo
Framework: Streamlit + Scikit-learn
""")

# -------------------------------
# Trang chủ
# -------------------------------
if demo_choice == "Trang chủ":
    st.title("Demo Cây Quyết Định (Decision Tree)")
    st.markdown("### Nghiên cứu về Cây Quyết Định và các Độ đo")

    st.markdown("---")

    # Giới thiệu
    st.header("Giới thiệu")
    st.markdown("""
    Demo này minh họa thuật toán **Decision Tree Classifier** với 2 bài toán:

    1. **Iris Dataset** - Lý thuyết cơ bản
       - Phân loại hoa Iris thành 3 loại
       - Dataset đơn giản, lý tưởng để học thuật toán
       - Tập trung vào: Gini vs Entropy, cấu trúc cây, metrics

    2. **Titanic Dataset** - Ứng dụng thực tế
       - Dự đoán khả năng sống sót sau thảm họa Titanic
       - Dataset phức tạp, gần với bài toán thực tế
       - Tập trung vào: Tiền xử lý, Feature Engineering, Interpretation
    """)

    # So sánh 2 dataset
    st.header("So sánh 2 Dataset")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Iris Dataset")
        st.markdown("""
        **Đặc điểm:**
        - 150 mẫu, 4 features (số)
        - 3 classes cân bằng
        - Không có missing values
        - Dataset sạch, đơn giản

        **Học được gì:**
        - Hiểu thuật toán Decision Tree
        - So sánh Gini vs Entropy
        - Đọc và phân tích cây
        - Feature Importance
        - Confusion Matrix & Metrics
        """)

    with col2:
        st.subheader("Titanic Dataset")
        st.markdown("""
        **Đặc điểm:**
        - ~900 mẫu, 10+ features (số + chữ)
        - 2 classes không cân bằng
        - Có missing values
        - Dataset phức tạp, thực tế

        **Học được gì:**
        - Tiền xử lý dữ liệu thực tế
        - Feature Engineering
        - Train/Test Split
        - Overfitting vs Underfitting
        - ROC Curve & AUC
        - Model Interpretation
        """)

    # Kiến thức chính
    st.header("Kiến thức chính")

    tab1, tab2, tab3 = st.tabs(["Độ đo", "Thuật toán", "Đánh giá"])

    with tab1:
        st.markdown("""
        ### Các độ đo (Impurity Measures)

        #### 1. Gini Index
        ```
        Gini = 1 - Σ(pi²)
        ```
        - **Ý nghĩa**: Xác suất chọn 2 mẫu khác lớp
        - **Giá trị**: 0 (thuần khiết) đến 0.5 (tạp)
        - **Ưu điểm**: Tính nhanh, hiệu quả

        #### 2. Entropy (Information Gain)
        ```
        Entropy = -Σ(pi × log2(pi))
        ```
        - **Ý nghĩa**: Độ hỗn loạn của dữ liệu
        - **Giá trị**: 0 (thuần khiết) đến log2(n_classes)
        - **Ưu điểm**: Chính xác về mặt lý thuyết

        #### 3. So sánh
        | Tiêu chí | Gini | Entropy |
        |----------|------|---------|
        | Tốc độ | Nhanh hơn | Chậm hơn |
        | Độ chính xác | Tương đương | Tương đương |
        | Cây tạo ra | Cân bằng | Chi tiết hơn |
        """)

    with tab2:
        st.markdown("""
        ### Thuật toán xây dựng cây

        #### Các bước:
        1. **Tính Impurity** cho tất cả features
        2. **Chọn feature tốt nhất** (Information Gain cao nhất)
        3. **Phân chia dữ liệu** theo feature đó
        4. **Lặp lại** cho các node con
        5. **Dừng** khi đạt điều kiện (max_depth, min_samples, ...)

        #### Thuật toán phổ biến:
        - **ID3**: Sử dụng Entropy (chỉ categorical)
        - **C4.5**: Cải tiến ID3 (xử lý numerical + pruning)
        - **CART**: Sử dụng Gini (scikit-learn dùng CART)

        #### Tham số quan trọng:
        - `max_depth`: Độ sâu tối đa → tránh overfitting
        - `min_samples_split`: Số mẫu tối thiểu để phân chia
        - `min_samples_leaf`: Số mẫu tối thiểu ở node lá
        - `criterion`: 'gini' hoặc 'entropy'
        """)

    with tab3:
        st.markdown("""
        ### Phương pháp đánh giá

        #### 1. Confusion Matrix
        Bảng so sánh dự đoán vs thực tế:
        ```
                  Predicted
                  0      1
        Actual 0  TN     FP
               1  FN     TP
        ```

        #### 2. Metrics
        - **Accuracy**: (TP + TN) / Total
        - **Precision**: TP / (TP + FP) - Độ chính xác khi dự đoán Positive
        - **Recall**: TP / (TP + FN) - Khả năng tìm ra Positive
        - **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)

        #### 3. ROC Curve & AUC
        - **ROC Curve**: Đồ thị TPR vs FPR
        - **AUC**: Diện tích dưới ROC → 1.0 là tốt nhất
        - **Ý nghĩa**: Đánh giá khả năng phân biệt 2 lớp

        #### 4. Train/Test Split
        - Chia dữ liệu: 70-80% train, 20-30% test
        - **Tránh overfitting**: Model học thuộc lòng train set
        - **Kiểm tra**: Test accuracy gần train accuracy → tốt
        """)

    # Hướng dẫn sử dụng
    st.header("Hướng dẫn sử dụng")
    st.markdown("""
    1. **Chọn demo** từ sidebar:
       - **Iris** - Bắt đầu ở đây để hiểu cơ bản
       - **Titanic** - Sau đó chuyển sang ứng dụng thực tế

    2. **Thay đổi tham số** ở sidebar:
       - Thử cả `gini` và `entropy`
       - Tăng/giảm `max_depth` → quan sát overfitting

    3. **Phân tích kết quả**:
       - Xem Feature Importance → feature nào quan trọng?
       - Đọc cây quyết định → hiểu cách phân loại
       - Confusion Matrix → đánh giá độ chính xác

    4. **Dự đoán mẫu mới**:
       - Iris: Nhập 4 thông số hoa
       - Titanic: Nhập thông tin hành khách
    """)

    # Gợi ý thuyết trình
    st.header("Gợi ý Thuyết trình")
    st.markdown("""
    ### Kịch bản demo:

    **Phần 1: Giới thiệu (2 phút)**
    - Cây Quyết Định là gì?
    - Ứng dụng trong thực tế

    **Phần 2: Lý thuyết - Iris (5 phút)**
    - Giải thích Gini vs Entropy
    - Demo so sánh 2 độ đo
    - Đọc cây quyết định
    - Giải thích Confusion Matrix

    **Phần 3: Thực hành - Titanic (5 phút)**
    - Tiền xử lý dữ liệu (missing values, encoding)
    - Feature Engineering (FamilySize, Title)
    - Train/Test split
    - ROC Curve & AUC

    **Phần 4: Dự đoán tương tác (3 phút)**
    - Demo dự đoán hoa Iris
    - Demo dự đoán sống sót Titanic
    - Giải thích kết quả

    **Phần 5: Q&A (5 phút)**
    - Trả lời câu hỏi
    """)

    st.markdown("---")
    st.info("Chọn demo từ sidebar để bắt đầu!")

# -------------------------------
# Demo Iris
# -------------------------------
elif demo_choice == "Iris - Lý thuyết cơ bản":
    st.info("Đang chạy Iris Demo...")

    st.markdown("""
    ### Iris Decision Tree Demo

    Demo này tập trung vào **lý thuyết cơ bản** của Decision Tree:
    - So sánh Gini vs Entropy
    - Hiểu cấu trúc cây quyết định
    - Feature Importance
    - Confusion Matrix & Metrics

    ---
    """)

    # Hướng dẫn chạy riêng
    st.code("streamlit run iris_demo.py", language="bash")

    st.warning("""
    **Lưu ý**: Do giới hạn kỹ thuật, vui lòng chạy Iris demo bằng lệnh trên trong terminal riêng.

    Hoặc mở file `iris_demo.py` trực tiếp.
    """)

# -------------------------------
# Demo Titanic
# -------------------------------
elif demo_choice == "Titanic - Ứng dụng thực tế":
    st.info("Đang chạy Titanic Demo...")

    st.markdown("""
    ### Titanic Decision Tree Demo

    Demo này tập trung vào **ứng dụng thực tế**:
    - Tiền xử lý dữ liệu phức tạp
    - Feature Engineering
    - Train/Test Split & Overfitting
    - ROC Curve & AUC
    - Model Interpretation

    ---
    """)

    # Hướng dẫn chạy riêng
    st.code("streamlit run titanic_demo.py", language="bash")

    st.warning("""
    **Lưu ý**: Do giới hạn kỹ thuật, vui lòng chạy Titanic demo bằng lệnh trên trong terminal riêng.

    Hoặc mở file `titanic_demo.py` trực tiếp.
    """)

# -------------------------------
# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Made with Streamlit")
