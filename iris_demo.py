# -------------------------------
# Demo Decision Tree - Iris Dataset với Streamlit
# -------------------------------

import streamlit as st
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import graphviz
import plotly.express as px
import numpy as np

# -------------------------------

# 1. Tiêu đề

st.title("Demo Decision Tree - Iris Dataset (Enhanced Visualization)")

# -------------------------------

# 2. Load dữ liệu

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# -------------------------------

# 3. Sidebar - chọn tham số mô hình

st.sidebar.header("Cài đặt mô hình")
criterion = st.sidebar.selectbox("Chọn độ đo (criterion)", ["gini", "entropy"])
max_depth = st.sidebar.slider("Độ sâu tối đa của cây (max_depth)", 1, 10, 4)

# -------------------------------

# 4. Huấn luyện mô hình

model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=42)
model.fit(X, y)

# -------------------------------

# 5. Nhập thông số mẫu mới

st.subheader("Nhập thông số mẫu mới để dự đoán")
sl = st.number_input("Sepal Length (cm)", 0.0, 10.0, 5.1)
sw = st.number_input("Sepal Width (cm)", 0.0, 10.0, 3.5)
pl = st.number_input("Petal Length (cm)", 0.0, 10.0, 1.4)
pw = st.number_input("Petal Width (cm)", 0.0, 10.0, 0.2)

if st.button("Dự đoán"):
sample = [[sl, sw, pl, pw]]
pred = model.predict(sample)
st.success(f"Mẫu này thuộc lớp: {target_names[pred][0]}")

# -------------------------------

# 6. Hiển thị metric và confusion matrix

st.subheader("Đánh giá mô hình trên toàn bộ dữ liệu")
y_pred = model.predict(X)
report = classification_report(y, y_pred, target_names=target_names, output_dict=True)
df_report = pd.DataFrame(report).transpose()
st.dataframe(df_report)

# Confusion matrix nâng cao

cm = confusion_matrix(y, y_pred)
cm_percent = cm / cm.sum(axis=1)[:, None] * 100
fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(cm_percent, annot=cm, fmt='d', cmap='YlGnBu',
xticklabels=target_names, yticklabels=target_names, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix (count & %)")
st.pyplot(fig)

# -------------------------------

# 7. Feature Importance nâng cao

st.subheader("Feature Importance")
feat_imp = pd.DataFrame({
'feature': feature_names,
'importance': model.feature_importances_
}).sort_values(by='importance', ascending=True)

fig2, ax2 = plt.subplots(figsize=(6,4))
sns.barplot(x='importance', y='feature', data=feat_imp, palette='viridis', ax=ax2)
ax2.set_xlabel("Importance")
ax2.set_ylabel("Feature")
ax2.set_title("Feature Importance")
st.pyplot(fig2)

# Plotly interactive (tùy chọn)

fig_plotly = px.bar(feat_imp, x='importance', y='feature', orientation='h',
color='importance', color_continuous_scale='Viridis',
title='Feature Importance (Interactive)')
st.plotly_chart(fig_plotly)

# -------------------------------

# 8. Trực quan hóa cây quyết định

st.subheader("Cây quyết định")
dot_data = export_graphviz(
model,
out_file=None,
feature_names=feature_names,
class_names=target_names,
filled=True,
rounded=True,
special_characters=True
)
graph = graphviz.Source(dot_data)
st.graphviz_chart(graph)
