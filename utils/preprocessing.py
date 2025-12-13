# -------------------------------
# Tiền xử lý dữ liệu Titanic
# -------------------------------

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_titanic_data(file_path):
    """Load Titanic dataset"""
    df = pd.read_csv(file_path)
    return df

def preprocess_titanic(df):
    """
    Tiền xử lý dữ liệu Titanic:
    - Xử lý missing values
    - Feature engineering
    - Encoding categorical variables
    """
    # Copy để không ảnh hưởng dữ liệu gốc
    data = df.copy()

    # 1. Xử lý missing values
    # Age: Điền median
    data['Age'].fillna(data['Age'].median(), inplace=True)

    # Embarked: Điền mode (giá trị phổ biến nhất)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

    # Fare: Điền median
    data['Fare'].fillna(data['Fare'].median(), inplace=True)

    # Cabin: Tạo feature mới - có cabin hay không
    data['HasCabin'] = data['Cabin'].notna().astype(int)

    # 2. Feature Engineering
    # Family Size
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

    # Is Alone
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)

    # Title từ Name
    data['Title'] = data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

    # Gom nhóm Title
    title_mapping = {
        'Mr': 'Mr',
        'Miss': 'Miss',
        'Mrs': 'Mrs',
        'Master': 'Master',
        'Dr': 'Rare',
        'Rev': 'Rare',
        'Col': 'Rare',
        'Major': 'Rare',
        'Mlle': 'Miss',
        'Countess': 'Rare',
        'Ms': 'Miss',
        'Lady': 'Rare',
        'Jonkheer': 'Rare',
        'Don': 'Rare',
        'Dona': 'Rare',
        'Mme': 'Mrs',
        'Capt': 'Rare',
        'Sir': 'Rare'
    }
    data['Title'] = data['Title'].map(title_mapping)
    data['Title'].fillna('Rare', inplace=True)

    # 3. Encoding categorical variables
    # Sex: male=0, female=1
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

    # Embarked: C=0, Q=1, S=2
    embarked_mapping = {'C': 0, 'Q': 1, 'S': 2}
    data['Embarked'] = data['Embarked'].map(embarked_mapping)

    # Title: Label Encoding
    title_encoding = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}
    data['Title'] = data['Title'].map(title_encoding)

    # 4. Chọn features để training
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
                'Embarked', 'FamilySize', 'IsAlone', 'HasCabin', 'Title']

    X = data[features]
    y = data['Survived'] if 'Survived' in data.columns else None

    return X, y, data

def get_feature_names():
    """Trả về tên các features"""
    return ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
            'Embarked', 'FamilySize', 'IsAlone', 'HasCabin', 'Title']

def decode_prediction_input(pclass, sex, age, sibsp, parch, fare, embarked, title):
    """
    Chuyển đổi input từ user thành format phù hợp với model
    """
    # Encoding
    sex_encoded = 0 if sex == 'Nam' else 1
    embarked_encoded = {'Cherbourg': 0, 'Queenstown': 1, 'Southampton': 2}[embarked]
    title_encoded = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}[title]

    # Tính toán features
    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0
    has_cabin = 0  # Mặc định không có cabin

    # Tạo feature vector
    features = [pclass, sex_encoded, age, sibsp, parch, fare,
                embarked_encoded, family_size, is_alone, has_cabin, title_encoded]

    return [features]
