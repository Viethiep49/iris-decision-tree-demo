# -*- coding: utf-8 -*-
# -------------------------------
# Demo Decision Tree - Titanic Dataset với Streamlit
# Ứng dụng thực tế: Dự đoán sống sót
# -------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
import plotly.express as px
import plotly.graph_objects as go
from utils.preprocessing import load_titanic_data, preprocess_titanic, decode_prediction_input, get_feature_names

# -------------------------------
# Cấu hình trang
st.set_page_config(page_title="Titanic Decision Tree Demo", layout="wide")

# -------------------------------
# 1. Tiêu đề
st.title("Demo Decision Tree - Titanic Dataset")
st.markdown("**Dự đoán khả năng sống sót sau thảm họa Titanic năm 1912**")

# -------------------------------
# 2. Load và tiền xử lý dữ liệu
@st.cache_data
def load_and_process_data():
    df = load_titanic_data('data/titanic.csv')
    X, y, processed_df = preprocess_titanic(df)
    return df, X, y, processed_df

df_raw, X, y, df_processed = load_and_process_data()

# -------------------------------
# 3. Sidebar - Cấu hình
st.sidebar.header("Cài đặt mô hình")
criterion = st.sidebar.selectbox("Chọn độ đo (criterion)", ["gini", "entropy"])
max_depth = st.sidebar.slider("Độ sâu tối đa (max_depth)", 1, 10, 5)
min_samples_split = st.sidebar.slider("Min samples split", 2, 20, 2)
test_size = st.sidebar.slider("Tỷ lệ test set (%)", 10, 40, 20) / 100

st.sidebar.markdown("---")
st.sidebar.markdown("### Thông tin Dataset")
st.sidebar.write(f"Tổng số mẫu: {len(df_raw)}")
st.sidebar.write(f"Sống sót: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
st.sidebar.write(f"Tử vong: {len(y)-y.sum()} ({(len(y)-y.sum())/len(y)*100:.1f}%)")

# -------------------------------
# 4. Khám phá dữ liệu (EDA)
st.header("1. Khám phá Dữ liệu (EDA)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Tỷ lệ sống sót")
    survival_counts = df_raw['Survived'].value_counts()
    fig1 = go.Figure(data=[go.Pie(
        labels=['Tử vong', 'Sống sót'],
        values=survival_counts.values,
        hole=0.3,
        marker_colors=['#FF6B6B', '#4ECDC4']
    )])
    fig1.update_layout(height=300)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("Sống sót theo Giới tính")
    survival_sex = pd.crosstab(df_raw['Sex'], df_raw['Survived'], normalize='index') * 100
    fig2 = px.bar(survival_sex, barmode='group',
                  labels={'value': 'Tỷ lệ (%)', 'Sex': 'Giới tính'},
                  color_discrete_map={0: '#FF6B6B', 1: '#4ECDC4'})
    fig2.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    st.subheader("Sống sót theo Hạng vé")
    survival_class = pd.crosstab(df_raw['Pclass'], df_raw['Survived'], normalize='index') * 100
    fig3 = px.bar(survival_class, barmode='group',
                  labels={'value': 'Tỷ lệ (%)', 'Pclass': 'Hạng vé'},
                  color_discrete_map={0: '#FF6B6B', 1: '#4ECDC4'})
    fig3.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.subheader("Phân bố Tuổi")
    fig4 = go.Figure()
    fig4.add_trace(go.Histogram(x=df_raw[df_raw['Survived']==1]['Age'],
                                name='Sống sót', marker_color='#4ECDC4', opacity=0.7))
    fig4.add_trace(go.Histogram(x=df_raw[df_raw['Survived']==0]['Age'],
                                name='Tử vong', marker_color='#FF6B6B', opacity=0.7))
    fig4.update_layout(barmode='overlay', height=300, xaxis_title='Tuổi', yaxis_title='Số lượng')
    st.plotly_chart(fig4, use_container_width=True)

# -------------------------------
# 5. Train/Test Split và Huấn luyện
st.header("2. Huấn luyện Mô hình Decision Tree")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

model = DecisionTreeClassifier(
    criterion=criterion,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    random_state=42
)
model.fit(X_train, y_train)

# Dự đoán
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Accuracy
train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

col1, col2, col3 = st.columns(3)
col1.metric("Train Accuracy", f"{train_acc:.2%}")
col2.metric("Test Accuracy", f"{test_acc:.2%}")
col3.metric("Overfitting", f"{(train_acc - test_acc):.2%}",
            delta=None, delta_color="inverse")

# -------------------------------
# 6. Đánh giá mô hình
st.header("3. Đánh giá Mô hình")

tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Classification Report", "ROC Curve"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Test Set")
        cm_test = confusion_matrix(y_test, y_pred_test)
        fig_cm = plt.figure(figsize=(6, 5))
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Tử vong', 'Sống sót'],
                    yticklabels=['Tử vong', 'Sống sót'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix (Test Set)')
        st.pyplot(fig_cm)

    with col2:
        st.subheader("Giải thích")
        st.write(f"**True Negative (TN):** {cm_test[0,0]} - Dự đoán đúng tử vong")
        st.write(f"**False Positive (FP):** {cm_test[0,1]} - Dự đoán sai sống sót")
        st.write(f"**False Negative (FN):** {cm_test[1,0]} - Dự đoán sai tử vong")
        st.write(f"**True Positive (TP):** {cm_test[1,1]} - Dự đoán đúng sống sót")

with tab2:
    report = classification_report(y_test, y_pred_test,
                                   target_names=['Tử vong', 'Sống sót'],
                                   output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    st.dataframe(df_report.style.highlight_max(axis=0, color='lightgreen'))

with tab3:
    # ROC Curve
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                 name=f'ROC Curve (AUC = {roc_auc:.2f})',
                                 line=dict(color='#4ECDC4', width=3)))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                 name='Random Classifier',
                                 line=dict(color='gray', width=2, dash='dash')))
    fig_roc.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=500
    )
    st.plotly_chart(fig_roc, use_container_width=True)

# -------------------------------
# 7. Feature Importance
st.header("4. Feature Importance")

feature_names = get_feature_names()
feat_imp = pd.DataFrame({
    'Feature': feature_names,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=True)

fig_imp = px.bar(feat_imp, x='Importance', y='Feature', orientation='h',
                 color='Importance', color_continuous_scale='Viridis',
                 title='Độ quan trọng của từng đặc trưng')
st.plotly_chart(fig_imp, use_container_width=True)

# Giải thích
st.markdown("""
**Giải thích:**
- **Sex**: Giới tính (0=Nam, 1=Nữ) - Phụ nữ được ưu tiên cứu hộ
- **Pclass**: Hạng vé (1=First, 2=Second, 3=Third) - Hạng cao gần xuồng cứu sinh
- **Fare**: Giá vé - Phản ánh vị trí cabin
- **Age**: Tuổi - Trẻ em được ưu tiên
- **FamilySize**: Số người trong gia đình
- **Title**: Danh xưng (Mr, Mrs, Miss, Master)
""")

# -------------------------------
# 8. Visualize Decision Tree
st.header("5. Cây Quyết Định")

dot_data = export_graphviz(
    model,
    out_file=None,
    feature_names=feature_names,
    class_names=['Tử vong', 'Sống sót'],
    filled=True,
    rounded=True,
    special_characters=True,
    max_depth=3  # Giới hạn độ sâu hiển thị
)
graph = graphviz.Source(dot_data)
st.graphviz_chart(graph)

st.info("Mỗi node hiển thị: điều kiện phân chia, Gini/Entropy, số mẫu, và lớp dự đoán")

# -------------------------------
# 9. Dự đoán tương tác
st.header("6. Dự đoán Khả năng Sống sót")

st.markdown("**Nhập thông tin hành khách để dự đoán khả năng sống sót:**")

col1, col2, col3 = st.columns(3)

with col1:
    user_name = st.text_input("Tên", "John Doe")
    user_pclass = st.selectbox("Hạng vé", [1, 2, 3], index=2)
    user_sex = st.selectbox("Giới tính", ["Nam", "Nữ"])
    user_age = st.number_input("Tuổi", 0, 100, 25)

with col2:
    user_sibsp = st.number_input("Số anh chị em/vợ chồng", 0, 8, 0)
    user_parch = st.number_input("Số bố mẹ/con cái", 0, 6, 0)
    user_fare = st.number_input("Giá vé ($)", 0.0, 600.0, 15.0)

with col3:
    user_embarked = st.selectbox("Cảng lên tàu", ["Cherbourg", "Queenstown", "Southampton"])
    user_title = st.selectbox("Danh xưng", ["Mr", "Miss", "Mrs", "Master", "Rare"])

if st.button("Dự đoán", type="primary"):
    # Chuẩn bị input
    sample = decode_prediction_input(
        user_pclass, user_sex, user_age, user_sibsp,
        user_parch, user_fare, user_embarked, user_title
    )

    # Dự đoán
    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0]

    # Hiển thị kết quả
    if prediction == 1:
        st.success(f"**{user_name} có khả năng SỐNG SÓT** ({probability[1]:.1%} confidence)")
    else:
        st.error(f"**{user_name} có khả năng TỬ VONG** ({probability[0]:.1%} confidence)")

    # Giải thích
    st.markdown("**Phân tích:**")
    reasons = []
    if user_sex == "Nữ":
        reasons.append("- Nữ giới được ưu tiên cứu hộ (Women and children first)")
    else:
        reasons.append("- Nam giới có tỷ lệ sống sót thấp hơn")

    if user_pclass == 1:
        reasons.append("- Hạng vé First Class - gần xuồng cứu sinh")
    elif user_pclass == 3:
        reasons.append("- Hạng vé Third Class - xa xuồng cứu sinh")

    if user_age < 16:
        reasons.append("- Trẻ em được ưu tiên cứu hộ")
    elif user_age > 60:
        reasons.append("- Người cao tuổi khó di chuyển nhanh")

    family_size = user_sibsp + user_parch + 1
    if family_size == 1:
        reasons.append("- Đi một mình - không có người giúp đỡ")
    elif family_size <= 4:
        reasons.append("- Có gia đình nhỏ - cùng nhau hỗ trợ")
    else:
        reasons.append("- Gia đình lớn - khó di chuyển cùng lúc")

    for reason in reasons:
        st.write(reason)

    # Hiển thị xác suất
    st.markdown("**Xác suất:**")
    fig_prob = go.Figure(go.Bar(
        x=[probability[0], probability[1]],
        y=['Tử vong', 'Sống sót'],
        orientation='h',
        marker_color=['#FF6B6B', '#4ECDC4']
    ))
    fig_prob.update_layout(height=200, xaxis_title='Xác suất', showlegend=False)
    st.plotly_chart(fig_prob, use_container_width=True)

# -------------------------------
# Footer
st.markdown("---")
st.markdown("""
**Về Demo này:**
- Dataset: Titanic - Machine Learning from Disaster (Kaggle)
- Thuật toán: Decision Tree Classifier
- Framework: Streamlit + Scikit-learn
""")
