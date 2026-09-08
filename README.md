# Kế Hoạch Toàn Diện Dự Án Machine Learning: Titanic Survival Prediction

Tài liệu này là cẩm nang chi tiết kết hợp chuẩn mực **Quy trình 7 bước phát triển Machine Learning (End-to-End ML Lifecycle)** cùng toàn bộ **Code Python thực thi hoàn chỉnh** trong Jupyter Notebook và ứng dụng thực tế.

---

## Mục Lục Tổng Quan 7 Giai Đoạn

1. [1. Data (Dữ liệu)](#1-data-dữ-liệu)
2. [2. Features (Đặc trưng & Tiền xử lý)](#2-features-đặc-trưng--tiền-xử-lý)
3. [3. Model (Lựa chọn thuật toán mô hình)](#3-model-lựa-chọn-thuật-toán-mô-hình)
4. [4. Training (Huấn luyện & Tối ưu tham số)](#4-training-huấn-luyện--tối-ưu-tham-số)
5. [5. Evaluation (Đánh giá & Diễn giải mô hình)](#5-evaluation-đánh-giá--diễn-giải-mô-hình)
6. [6. Prediction (Dự đoán trên tập Test & Xuất kết quả)](#6-prediction-dự-đoán-trên-tập-test--xuất-kết-quả)
7. [7. Deployment (Triển khai Web App & Kaggle Submission)](#7-deployment-triển-khai-web-app--kaggle-submission)

---

# 1. Data (Dữ liệu)

### 1.1. Mục tiêu & Nhiệm vụ

- **Thu thập dữ liệu:** Tải các file dữ liệu từ cuộc thi Titanic của Kaggle gồm:

  - `train.csv`: Tập dữ liệu huấn luyện có sẵn kết quả nhãn `Survived` (0 = Mất, 1 = Sống sót).

  - `test.csv`: Tập dữ liệu kiểm thử thực tế cần dự đoán kết quả `Survived`.

  - `gender_submission.csv`: File mẫu nộp bài chuẩn của Kaggle.

- **Khám phá dữ liệu (EDA - Exploratory Data Analysis):** Tìm hiểu ý nghĩa từng cột đặc trưng (Feature):

  - `Pclass`: Hạng vé (1 = Hạng nhất, 2 = Hạng nhì, 3 = Hạng phổ thông).

  - `Sex`: Giới tính (male, female).

  - `Age`: Độ tuổi hành khách.

  - `SibSp`: Số anh chị em / vợ chồng cùng đi trên tàu.

  - `Parch`: Số bố mẹ / con cái cùng đi trên tàu.

  - `Ticket`: Mã vé.

  - `Fare`: Giá vé.

  - `Cabin`: Số hiệu buồng phòng.

  - `Embarked`: Cảng lên tàu (C = Cherbourg, Q = Queenstown, S = Southampton).

- **Phát hiện vấn đề (Data Quality Issues):**

  - Nhận diện các giá trị khuyết thiếu (`Missing Values / NaN`), đặc biệt là ở các cột `Age` (~20% thiếu), `Cabin` (>77% thiếu) và `Embarked` (2 mẫu thiếu).

### 1.2. Code Triển Khai Trong Notebook

#### **Code Cell 1: Khởi tạo môi trường & Tải dữ liệu**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings

# Cấu hình thẩm mỹ & tắt các cảnh báo không cần thiết
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_palette('Set2')

# 1. Tải dữ liệu
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Đặt PassengerId làm index để không gây nhiễu trong quá trình học máy
train_df.set_index('PassengerId', inplace=True)
test_df.set_index('PassengerId', inplace=True)

print(f"Kích thước tập Train: {train_df.shape}")
print(f"Kích thước tập Test:  {test_df.shape}")
display(train_df.head())
```

#### **Code Cell 2: Khám phá phân phối & Phát hiện dữ liệu khuyết (Missing Values)**

```python
# 1. Thống kê dữ liệu khuyết
print("--- Dữ liệu khuyết ở tập Train ---")
missing_train = train_df.isnull().sum()
print(missing_train[missing_train > 0])

print("\n--- Dữ liệu khuyết ở tập Test ---")
missing_test = test_df.isnull().sum()
print(missing_test[missing_test > 0])

# 2. Đổi kiểu dữ liệu biến mục tiêu
train_df["Survived"] = train_df["Survived"].astype('category')

# 3. Vẽ biểu đồ phân bố sống sót & mối quan hệ với các biến quan trọng
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Tỉ lệ sống theo Giới tính (Sex)
sns.countplot(data=train_df, x='Sex', hue='Survived', ax=axes[0, 0], palette='pastel')
axes[0, 0].set_title('Tỉ lệ sống sót theo Giới tính (Sex)')
axes[0, 0].legend(title='Survived', labels=['Mất', 'Sống'])

# Tỉ lệ sống theo Hạng vé (Pclass)
sns.countplot(data=train_df, x='Pclass', hue='Survived', ax=axes[0, 1], palette='pastel')
axes[0, 1].set_title('Tỉ lệ sống sót theo Hạng vé (Pclass)')
axes[0, 1].legend(title='Survived', labels=['Mất', 'Sống'])

# Tỉ lệ sống theo Cảng lên tàu (Embarked)
sns.countplot(data=train_df, x='Embarked', hue='Survived', ax=axes[1, 0], palette='pastel')
axes[1, 0].set_title('Tỉ lệ sống sót theo Cảng lên tàu (Embarked)')
axes[1, 0].legend(title='Survived', labels=['Mất', 'Sống'])

# Phân phối Tuổi theo Sống/Mất
sns.kdeplot(data=train_df, x='Age', hue='Survived', fill=True, common_norm=False, ax=axes[1, 1], palette='tab10')
axes[1, 1].set_title('Phân phối Độ tuổi (Age) theo Sống/Mất')

plt.tight_layout()
plt.show()
```

> **Nhận xét EDA:**
> - Nữ giới có tỉ lệ sống sót cao vượt trội so với nam giới (ưu tiên cứu phụ nữ và trẻ em).
> - Hành khách ở Hạng 1 (Pclass = 1) có cơ hội sống sót cao nhất; Hạng 3 có tỉ lệ tử vong lớn nhất.
> - Cột `Cabin` thiếu hơn 77% dữ liệu nên sẽ loại bỏ. Cột `Age` cần điền khuyết thông minh.

---

# 2. Features (Đặc trưng & Tiền xử lý)

### 2.1. Mục tiêu & Phương pháp

- **Feature Engineering (Tạo đặc trưng mới):**

  - **Danh xưng (Title):** Trích xuất từ cột `Name` (Mr, Mrs, Miss, Master...). Giúp phản ánh cả địa vị xã hội lẫn tuổi tác (ví dụ: "Master" là các bé trai).

  - **Kích thước gia đình (FamilySize):** Tính bằng `SibSp + Parch + 1` (tính cả bản thân hành khách).

  - **Phân nhóm gia đình (Family_category):** Chia thành các nhóm `Single`, `Small`, `Medium`, `Large` vì gia đình nhỏ thường dễ thoát hiểm cùng nhau hơn gia đình quá đông hoặc đi một mình.

- **Xử lý dữ liệu thiếu (Imputation):**

  - Điền khuyết cột `Age` theo trung vị (`median`) của từng nhóm kết hợp `[Sex, Pclass]` để đảm bảo tính thực tế.

  - Điền `Embarked` bằng giá trị xuất hiện nhiều nhất (`most_frequent`).

  - Điền `Fare` (ở tập Test có 1 hành khách bị thiếu) bằng trung vị (`median`).

- **Mã hóa (Encoding) & Chuẩn hóa (Scaling):**

  - Mã hóa biến phân loại bằng `OneHotEncoder`.

  - Chuẩn hóa các cột số (`Age`, `Fare`) bằng `StandardScaler`.

- **Loại bỏ đặc trưng thừa (Feature Selection):** Loại bỏ `Cabin` (quá nhiều missing), `Ticket` (quá nhiều giá trị rời rạc), và `Name` (sau khi đã trích xuất `Title`).

### 2.2. Code Triển Khai Trong Notebook

#### **Code Cell 3: Trích xuất đặc trưng mới & Điền khuyết nhóm**

```python
# 1. Hàm trích xuất Danh xưng (Title) từ Name
def extract_title(name):
    match = re.search(r",\s*([^\.]+)\.", name)
    return match.group(1).strip() if match else "Other"

def group_titles(title):
    if title in ['Mr', 'Mrs', 'Miss', 'Master']:
        return title
    elif title in ['Ms', 'Mlle']:
        return 'Miss'
    elif title in ['Mme']:
        return 'Mrs'
    else:
        return 'Other'

for df in [train_df, test_df]:
    # Trích xuất Title
    df["Title"] = df["Name"].apply(extract_title).apply(group_titles)

    # Tạo FamilySize và Family_category
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["Family_category"] = pd.cut(df["FamilySize"],
                                   bins=[0, 1, 4, 6, 20],
                                   labels=["Single", "Small", "Medium", "Large"])

# 2. Điền khuyết cột Age dựa trên trung vị của nhóm Sex và Pclass
for df in [train_df, test_df]:
    df["Age"] = df.groupby(['Sex', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))

print("Feature Engineering hoàn tất!")
display(train_df[['Name', 'Title', 'FamilySize', 'Family_category', 'Age']].head())
```

#### **Code Cell 4: Xây dựng Scikit-Learn Pipeline tự động hóa**

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Chỉ định rõ ràng danh sách đặc trưng
num_features = ['Age', 'Fare']
cat_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'Family_category']
all_features = num_features + cat_features

# Tách X (đặc trưng) và y (nhãn mục tiêu)
X = train_df[all_features]
y = train_df["Survived"].astype(int)
X_test = test_df[all_features]

# Pipeline xử lý biến số
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline xử lý biến chữ (phân loại)
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Ghép 2 pipeline lại qua ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

print("Preprocessor Pipeline đã sẵn sàng!")
```

---

# 3. Model (Lựa chọn thuật toán mô hình)

### 3.1. Bản chất bài toán

- Bài toán Titanic là bài toán **Phân loại nhị phân (Binary Classification)**: Đầu ra là xác suất hành khách thuộc lớp `1` (Sống sót) hay `0` (Tử nạn).

### 3.2. Tập hợp các thuật toán khảo sát

Chúng ta khảo sát phổ rộng từ các mô hình truyền thống tới các thuật toán Ensemble tân tiến:

- **Nhóm Cơ bản:**

  - `Logistic Regression`: Nhanh, trực quan, giải thích xác suất tốt.

  - `K-Nearest Neighbors (KNN)`: Dựa trên khoảng cách lân cận.

  - `Decision Tree`: Phân nhánh theo quy tắc if-else.

- **Nhóm Nâng cao (Ensemble & Boosting):**

  - `Random Forest`: Tập hợp nhiều cây quyết định độc lập, giảm phương sai (variance).

  - `Gradient Boosting Classifier`: Tối ưu phần dư theo từng vòng lặp.

  - `Extra Trees Classifier`: Cực tiểu hóa tương quan giữa các cây.

  - `AdaBoost Classifier`: Tập trung vào các mẫu dữ liệu dự đoán sai.

  - `XGBoost Classifier`: Thuật toán Gradient Boosting tối ưu hóa cao độ, dẫn đầu các cuộc thi Kaggle.

  - `Support Vector Machines (SVC)`: Tìm siêu phẳng phân tách tối ưu.

---

# 4. Training (Huấn luyện & Tối ưu tham số)

### 4.1. Chiến lược chống Overfitting & Rò rỉ dữ liệu (Data Leakage)

- Sử dụng **Stratified 5-Fold Cross Validation**: Đảm bảo tỉ lệ nhãn `0` và `1` đồng đều giữa các fold.

- **Đóng gói Preprocessor vào Pipeline**: Dữ liệu validation chỉ được chuyển đổi dựa trên thông số tính toán từ fold huấn luyện của chính nó.

### 4.2. Code Triển Khai Trong Notebook

#### **Code Cell 5: Huấn luyện so sánh các mô hình cơ sở (Baseline Comparison)**

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier
)
from xgboost import XGBClassifier

seed = 42

baseline_classifiers = {
    'Logistic Regression': LogisticRegression(solver='liblinear', random_state=seed),
    'Linear SVC': LinearSVC(random_state=seed),
    'SVC (RBF)': SVC(probability=True, random_state=seed),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=seed),
    'Random Forest': RandomForestClassifier(random_state=seed),
    'Gradient Boosting': GradientBoostingClassifier(random_state=seed),
    'Extra Trees': ExtraTreesClassifier(random_state=seed),
    'AdaBoost': AdaBoostClassifier(random_state=seed),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=seed)
}

# Đóng gói Preprocessor với từng Classifier thành từng Full Pipeline hoàn chỉnh
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
results = []

for name, clf in baseline_classifiers.items():
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', clf)
    ])
    scores = cross_val_score(model_pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    for fold_idx, score in enumerate(scores):
        results.append({'Model': name, 'Fold': fold_idx + 1, 'Accuracy': score})

df_results = pd.DataFrame(results)

# Trực quan hóa kết quả các Baseline Model
plt.figure(figsize=(13, 6))
order = df_results.groupby('Model')['Accuracy'].mean().sort_values(ascending=False).index
sns.boxplot(data=df_results, x='Model', y='Accuracy', order=order, palette='vlag')
plt.xticks(rotation=35, ha='right')
plt.title('So sánh độ chính xác của 10 mô hình Baseline (5-Fold CV)')
plt.tight_layout()
plt.show()

# Hiển thị bảng xếp hạng
summary = df_results.groupby('Model')['Accuracy'].agg(['mean', 'std']).loc[order]
display(summary)
```

#### **Code Cell 6: Tinh chỉnh siêu tham số (Hyperparameter Tuning với GridSearchCV)**

Sau khi khảo sát, ta chọn ra 2 thuật toán mạnh mẽ nhất là `Random Forest` và `XGBoost` để tinh chỉnh:
```python
from sklearn.model_selection import GridSearchCV

# 1. Tối ưu Random Forest
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(random_state=seed))
])

rf_params = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [4, 6, 8, 10],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
}

grid_rf = GridSearchCV(rf_pipeline, rf_params, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
grid_rf.fit(X, y)

print("Best Random Forest Params:", grid_rf.best_params_)
print(f"Best Random Forest CV Score: {grid_rf.best_score_:.4f}")

# 2. Tối ưu XGBoost
xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBClassifier(eval_metric='logloss', random_state=seed))
])

xgb_params = {
    'model__n_estimators': [100, 150, 200],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__max_depth': [3, 4, 5, 6],
    'model__subsample': [0.8, 1.0]
}

grid_xgb = GridSearchCV(xgb_pipeline, xgb_params, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
grid_xgb.fit(X, y)

print("Best XGBoost Params:", grid_xgb.best_params_)
print(f"Best XGBoost CV Score: {grid_xgb.best_score_:.4f}")
```

---

# 5. Evaluation (Đánh giá & Diễn giải mô hình)

### 5.1. Tiêu chí đánh giá toàn diện

- **Accuracy (Độ chính xác):** Tỉ lệ dự đoán đúng trên toàn bộ tập dữ liệu (thang đo chính của Kaggle Titanic).

- **Precision (Độ chuẩn xác), Recall (Độ bao phủ) & F1-Score:** Đảm bảo mô hình không dự đoán thiên lệch về một lớp khi mẫu sống sót chiếm tỉ lệ ít hơn.

- **Confusion Matrix:** Nhìn rõ số trường hợp False Positive và False Negative.

- **Feature Importance (Độ quan trọng của đặc trưng):** Giải thích quyết định của mô hình (Explainable AI).

### 5.2. Code Triển Khai Trong Notebook

#### **Code Cell 7: Đánh giá chi tiết qua Confusion Matrix & Classification Report**

```python
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix

best_rf = grid_rf.best_estimator_

# Dự đoán cross-validation out-of-fold
y_pred_cv = cross_val_predict(best_rf, X, y, cv=cv)

print("=== CLASSIFICATION REPORT ===")
print(classification_report(y, y_pred_cv, target_names=['Mất (0)', 'Sống (1)']))

# Vẽ ma trận nhầm lẫn (Confusion Matrix)
cm = confusion_matrix(y, y_pred_cv)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Dự đoán Mất', 'Dự đoán Sống'],
            yticklabels=['Thực tế Mất', 'Thực tế Sống'])
plt.title('Ma Trận Nhầm Lẫn (Confusion Matrix)')
plt.show()
```

#### **Code Cell 8: Phân tích độ quan trọng của đặc trưng (Feature Importance)**

```python
# Lấy mô hình classifier bên trong pipeline tốt nhất
trained_rf_model = best_rf.named_steps['model']

# Lấy danh sách tên các cột sau khi qua OneHotEncoder
onehot_cols = best_rf.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_features)
feature_names = num_features + list(onehot_cols)

importances = trained_rf_model.feature_importances_
feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_df.head(12), x='Importance', y='Feature', palette='crest')
plt.title('Top 12 Đặc Trưng Quan Trọng Nhất Quyết Định Sự Sống Còn')
plt.show()
```

#### **Code Cell 9: Xây dựng Mô hình kết hợp (Voting Classifier Ensemble)**

```python
from sklearn.ensemble import VotingClassifier

# Khởi tạo Logistic Regression với cấu hình chuẩn
log_clf = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(solver='liblinear', random_state=seed))
])

# Kết hợp 3 mô hình: Random Forest (tối ưu), XGBoost (tối ưu), Logistic Regression
ensemble_voting = VotingClassifier(
    estimators=[
        ('rf', grid_rf.best_estimator_),
        ('xgb', grid_xgb.best_estimator_),
        ('lr', log_clf)
    ],
    voting='soft'  # Bầu chọn theo xác suất dự đoán trung bình
)

ensemble_scores = cross_val_score(ensemble_voting, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
print(f"Ensemble Voting CV Accuracy: {ensemble_scores.mean():.4f} (+/- {ensemble_scores.std():.4f})")
```

---

# 6. Prediction (Dự đoán trên tập Test & Xuất kết quả)

### 6.1. Quy tắc cốt lõi

- Tập `test.csv` phải đi qua cùng một Pipeline tiền xử lý đã fit trên tập Train để tránh mọi sai lệch định dạng.

- Đầu ra phải là định dạng 2 cột: `PassengerId` và `Survived`.

### 6.2. Code Triển Khai Trong Notebook

#### **Code Cell 10: Huấn luyện toàn bộ dữ liệu & Xuất file nộp bài Kaggle**

```python
# Huấn luyện mô hình Voting cuối cùng trên 100% dữ liệu Train (X, y)
ensemble_voting.fit(X, y)

# Dự đoán trên tập kiểm thử test.csv
final_predictions = ensemble_voting.predict(X_test)

# Đóng gói DataFrame submission đúng chuẩn Kaggle
submission_df = pd.DataFrame({
    'PassengerId': test_df.index,
    'Survived': final_predictions.astype(int)
})

# Lưu ra file CSV
submission_df.to_csv('submission.csv', index=False)
print(" Đã xuất file submission.csv thành công!")
print(f"Số lượng bản ghi dự đoán: {len(submission_df)}")
print("Tỉ lệ dự đoán sống sót:", submission_df['Survived'].value_counts(normalize=True).to_dict())
display(submission_df.head(10))
```

---

# 7. Deployment (Triển khai Web App & Kaggle Submission)

### 7.1. Triển khai cấp độ 1: Nộp bài lên Kaggle

1. Đăng nhập vào trang cuộc thi [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic).
2. Nhấn nút **Submit Predictions**.
3. Tải file `submission.csv` vừa tạo lên.
4. Xem điểm số (Public Score) trên Bảng xếp hạng (Leaderboard). Điểm số thông thường nằm trong khoảng **78% - 82%**, một kết quả xuất sắc cho bài toán này.

---

### 7.2. Triển khai cấp độ 2: Lưu mô hình (Model Persistence)

Để phục vụ cho ứng dụng thực tế hoặc web app, ta đóng gói và lưu mô hình vào file `.pkl` bằng thư viện `joblib`.

#### **Code Cell 11: Lưu mô hình xuống ổ đĩa**

```python
import joblib

# Lưu toàn bộ Pipeline (bao gồm cả preprocessor và model)
joblib.dump(ensemble_voting, 'titanic_model_pipeline.pkl')
print(" Đã lưu mô hình thành công vào file: titanic_model_pipeline.pkl")
```

---

### 7.3. Triển khai cấp độ 3: Ứng dụng Web App dự đoán tương tác với Streamlit

Tạo một file có tên `app.py`. Người dùng có thể tự nhập thông tin cá nhân và xem hệ thống tính toán xác suất sống sót theo thời gian thực.

#### **Mã nguồn file `app.py`:**

```python
import streamlit as st
import pandas as pd
import joblib

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# Load mô hình đã lưu
@st.cache_resource
def load_model():
    return joblib.load('titanic_model_pipeline.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"Chưa tìm thấy file mô hình `titanic_model_pipeline.pkl`. Vui lòng chạy notebook để huấn luyện và lưu mô hình trước!")
    st.stop()

st.title("🚢 Dự Đoán Khả Năng Sống Sót - Tàu Titanic")
st.markdown("Nhập các thông số hành khách dưới đây để mô hình Machine Learning dự đoán xác suất sống sót:")

# Giao diện nhập dữ liệu người dùng
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox("Hạng vé (Pclass):", options=[1, 2, 3], format_func=lambda x: f"Hạng {x}")
        sex = st.selectbox("Giới tính (Sex):", options=["male", "female"], format_func=lambda x: "Nam" if x == "male" else "Nữ")
        age = st.slider("Tuổi (Age):", min_value=1, max_value=85, value=28)
        title = st.selectbox("Danh xưng (Title):", options=["Mr", "Miss", "Mrs", "Master", "Other"])

    with col2:
        fare = st.number_input("Giá vé (Fare, £):", min_value=0.0, max_value=600.0, value=32.0, step=1.0)
        embarked = st.selectbox("Cảng lên tàu (Embarked):", options=["S", "C", "Q"],
                                format_func=lambda x: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[x])
        sibsp = st.number_input("Số anh/chị/em/vợ/chồng (SibSp):", min_value=0, max_value=10, value=0)
        parch = st.number_input("Số cha/mẹ/con cái (Parch):", min_value=0, max_value=10, value=0)

    submitted = st.form_submit_button("🚀 Dự Đoán Ngay", use_container_width=True)

if submitted:
    # Tính toán đặc trưng dẫn xuất
    family_size = sibsp + parch + 1
    if family_size == 1:
        family_cat = "Single"
    elif family_size <= 4:
        family_cat = "Small"
    elif family_size <= 6:
        family_cat = "Medium"
    else:
        family_cat = "Large"

    # Tạo DataFrame đầu vào đúng định dạng các cột huấn luyện
    input_data = pd.DataFrame([{
        'Age': age,
        'Fare': fare,
        'Pclass': pclass,
        'Sex': sex,
        'Embarked': embarked,
        'Title': title,
        'Family_category': family_cat
    }])

    # Dự đoán kết quả & xác suất
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    survival_prob = probabilities[1] * 100

    st.divider()
    if prediction == 1:
        st.success(f"🎉 **Kết quả: CÓ KHẢ NĂNG SỐNG SÓT!**")
        st.metric("Xác suất sống sót", f"{survival_prob:.1f}%")
        st.balloons()
    else:
        st.error(f"☠️ **Kết quả: KHÔNG SỐNG SÓT!**")
        st.metric("Xác suất sống sót", f"{survival_prob:.1f}%")
```

#### **Cách chạy Web App:**

Trong terminal, bạn cài đặt Streamlit (nếu chưa có) và khởi chạy:
```bash
pip install streamlit
streamlit run app.py
```

---

## 📌 Bảng Tóm Tắt Quy Trình 7 Bước

| Bước | Tên Giai Đoạn | Công Cụ / Kỹ Thuật Chính | Sản Phẩm Đầu Ra |
| :--- | :--- | :--- | :--- |
| **1** | **Data** | Pandas, Seaborn, Matplotlib | Thấu hiểu dữ liệu, phân bố, phát hiện missing |
| **2** | **Features** | Regex, Imputer, OneHotEncoder, StandardScaler, Pipeline | Bộ đặc trưng sạch, chống rò rỉ dữ liệu |
| **3** | **Model** | Logistic Regression, RF, XGBoost, SVC, KNN, Tree | Danh sách ứng viên mô hình tiềm năng |
| **4** | **Training** | Stratified 5-Fold CV, GridSearchCV | Siêu tham số tối ưu cho RF và XGBoost |
| **5** | **Evaluation** | Confusion Matrix, Precision, Recall, Feature Importance | Đánh giá độ tin cậy và lý giải mô hình |
| **6** | **Prediction** | Ensemble Soft Voting Classifier | File `submission.csv` nộp Kaggle |
| **7** | **Deployment** | Kaggle Leaderboard, Joblib, Streamlit Web App | Ứng dụng web tương tác thực tế |

#   T i t a n i c - - - M a c h i n e - L e a r n i n g - f r o m - D i s a s t e r
=======
# Kế Hoạch Toàn Diện Dự Án Machine Learning: Titanic Survival Prediction

Tài liệu này là cẩm nang chi tiết kết hợp chuẩn mực **Quy trình 7 bước phát triển Machine Learning (End-to-End ML Lifecycle)** cùng toàn bộ **Code Python thực thi hoàn chỉnh** trong Jupyter Notebook và ứng dụng thực tế.

---

## Mục Lục Tổng Quan 7 Giai Đoạn

1. [1. Data (Dữ liệu)](#1-data-dữ-liệu)
2. [2. Features (Đặc trưng & Tiền xử lý)](#2-features-đặc-trưng--tiền-xử-lý)
3. [3. Model (Lựa chọn thuật toán mô hình)](#3-model-lựa-chọn-thuật-toán-mô-hình)
4. [4. Training (Huấn luyện & Tối ưu tham số)](#4-training-huấn-luyện--tối-ưu-tham-số)
5. [5. Evaluation (Đánh giá & Diễn giải mô hình)](#5-evaluation-đánh-giá--diễn-giải-mô-hình)
6. [6. Prediction (Dự đoán trên tập Test & Xuất kết quả)](#6-prediction-dự-đoán-trên-tập-test--xuất-kết-quả)
7. [7. Deployment (Triển khai Web App & Kaggle Submission)](#7-deployment-triển-khai-web-app--kaggle-submission)

---

# 1. Data (Dữ liệu)

### 1.1. Mục tiêu & Nhiệm vụ

- **Thu thập dữ liệu:** Tải các file dữ liệu từ cuộc thi Titanic của Kaggle gồm:

  - `train.csv`: Tập dữ liệu huấn luyện có sẵn kết quả nhãn `Survived` (0 = Mất, 1 = Sống sót).

  - `test.csv`: Tập dữ liệu kiểm thử thực tế cần dự đoán kết quả `Survived`.

  - `gender_submission.csv`: File mẫu nộp bài chuẩn của Kaggle.

- **Khám phá dữ liệu (EDA - Exploratory Data Analysis):** Tìm hiểu ý nghĩa từng cột đặc trưng (Feature):

  - `Pclass`: Hạng vé (1 = Hạng nhất, 2 = Hạng nhì, 3 = Hạng phổ thông).

  - `Sex`: Giới tính (male, female).

  - `Age`: Độ tuổi hành khách.

  - `SibSp`: Số anh chị em / vợ chồng cùng đi trên tàu.

  - `Parch`: Số bố mẹ / con cái cùng đi trên tàu.

  - `Ticket`: Mã vé.

  - `Fare`: Giá vé.

  - `Cabin`: Số hiệu buồng phòng.

  - `Embarked`: Cảng lên tàu (C = Cherbourg, Q = Queenstown, S = Southampton).

- **Phát hiện vấn đề (Data Quality Issues):**

  - Nhận diện các giá trị khuyết thiếu (`Missing Values / NaN`), đặc biệt là ở các cột `Age` (~20% thiếu), `Cabin` (>77% thiếu) và `Embarked` (2 mẫu thiếu).

### 1.2. Code Triển Khai Trong Notebook

#### **Code Cell 1: Khởi tạo môi trường & Tải dữ liệu**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings

# Cấu hình thẩm mỹ & tắt các cảnh báo không cần thiết
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_palette('Set2')

# 1. Tải dữ liệu
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Đặt PassengerId làm index để không gây nhiễu trong quá trình học máy
train_df.set_index('PassengerId', inplace=True)
test_df.set_index('PassengerId', inplace=True)

print(f"Kích thước tập Train: {train_df.shape}")
print(f"Kích thước tập Test:  {test_df.shape}")
display(train_df.head())
```

#### **Code Cell 2: Khám phá phân phối & Phát hiện dữ liệu khuyết (Missing Values)**

```python
# 1. Thống kê dữ liệu khuyết
print("--- Dữ liệu khuyết ở tập Train ---")
missing_train = train_df.isnull().sum()
print(missing_train[missing_train > 0])

print("\n--- Dữ liệu khuyết ở tập Test ---")
missing_test = test_df.isnull().sum()
print(missing_test[missing_test > 0])

# 2. Đổi kiểu dữ liệu biến mục tiêu
train_df["Survived"] = train_df["Survived"].astype('category')

# 3. Vẽ biểu đồ phân bố sống sót & mối quan hệ với các biến quan trọng
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Tỉ lệ sống theo Giới tính (Sex)
sns.countplot(data=train_df, x='Sex', hue='Survived', ax=axes[0, 0], palette='pastel')
axes[0, 0].set_title('Tỉ lệ sống sót theo Giới tính (Sex)')
axes[0, 0].legend(title='Survived', labels=['Mất', 'Sống'])

# Tỉ lệ sống theo Hạng vé (Pclass)
sns.countplot(data=train_df, x='Pclass', hue='Survived', ax=axes[0, 1], palette='pastel')
axes[0, 1].set_title('Tỉ lệ sống sót theo Hạng vé (Pclass)')
axes[0, 1].legend(title='Survived', labels=['Mất', 'Sống'])

# Tỉ lệ sống theo Cảng lên tàu (Embarked)
sns.countplot(data=train_df, x='Embarked', hue='Survived', ax=axes[1, 0], palette='pastel')
axes[1, 0].set_title('Tỉ lệ sống sót theo Cảng lên tàu (Embarked)')
axes[1, 0].legend(title='Survived', labels=['Mất', 'Sống'])

# Phân phối Tuổi theo Sống/Mất
sns.kdeplot(data=train_df, x='Age', hue='Survived', fill=True, common_norm=False, ax=axes[1, 1], palette='tab10')
axes[1, 1].set_title('Phân phối Độ tuổi (Age) theo Sống/Mất')

plt.tight_layout()
plt.show()
```

> **Nhận xét EDA:**
> - Nữ giới có tỉ lệ sống sót cao vượt trội so với nam giới (ưu tiên cứu phụ nữ và trẻ em).
> - Hành khách ở Hạng 1 (Pclass = 1) có cơ hội sống sót cao nhất; Hạng 3 có tỉ lệ tử vong lớn nhất.
> - Cột `Cabin` thiếu hơn 77% dữ liệu nên sẽ loại bỏ. Cột `Age` cần điền khuyết thông minh.

---

# 2. Features (Đặc trưng & Tiền xử lý)

### 2.1. Mục tiêu & Phương pháp

- **Feature Engineering (Tạo đặc trưng mới):**

  - **Danh xưng (Title):** Trích xuất từ cột `Name` (Mr, Mrs, Miss, Master...). Giúp phản ánh cả địa vị xã hội lẫn tuổi tác (ví dụ: "Master" là các bé trai).

  - **Kích thước gia đình (FamilySize):** Tính bằng `SibSp + Parch + 1` (tính cả bản thân hành khách).

  - **Phân nhóm gia đình (Family_category):** Chia thành các nhóm `Single`, `Small`, `Medium`, `Large` vì gia đình nhỏ thường dễ thoát hiểm cùng nhau hơn gia đình quá đông hoặc đi một mình.

- **Xử lý dữ liệu thiếu (Imputation):**

  - Điền khuyết cột `Age` theo trung vị (`median`) của từng nhóm kết hợp `[Sex, Pclass]` để đảm bảo tính thực tế.

  - Điền `Embarked` bằng giá trị xuất hiện nhiều nhất (`most_frequent`).

  - Điền `Fare` (ở tập Test có 1 hành khách bị thiếu) bằng trung vị (`median`).

- **Mã hóa (Encoding) & Chuẩn hóa (Scaling):**

  - Mã hóa biến phân loại bằng `OneHotEncoder`.

  - Chuẩn hóa các cột số (`Age`, `Fare`) bằng `StandardScaler`.

- **Loại bỏ đặc trưng thừa (Feature Selection):** Loại bỏ `Cabin` (quá nhiều missing), `Ticket` (quá nhiều giá trị rời rạc), và `Name` (sau khi đã trích xuất `Title`).

### 2.2. Code Triển Khai Trong Notebook

#### **Code Cell 3: Trích xuất đặc trưng mới & Điền khuyết nhóm**

```python
# 1. Hàm trích xuất Danh xưng (Title) từ Name
def extract_title(name):
    match = re.search(r",\s*([^\.]+)\.", name)
    return match.group(1).strip() if match else "Other"

def group_titles(title):
    if title in ['Mr', 'Mrs', 'Miss', 'Master']:
        return title
    elif title in ['Ms', 'Mlle']:
        return 'Miss'
    elif title in ['Mme']:
        return 'Mrs'
    else:
        return 'Other'

for df in [train_df, test_df]:
    # Trích xuất Title
    df["Title"] = df["Name"].apply(extract_title).apply(group_titles)

    # Tạo FamilySize và Family_category
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["Family_category"] = pd.cut(df["FamilySize"],
                                   bins=[0, 1, 4, 6, 20],
                                   labels=["Single", "Small", "Medium", "Large"])

# 2. Điền khuyết cột Age dựa trên trung vị của nhóm Sex và Pclass
for df in [train_df, test_df]:
    df["Age"] = df.groupby(['Sex', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))

print("Feature Engineering hoàn tất!")
display(train_df[['Name', 'Title', 'FamilySize', 'Family_category', 'Age']].head())
```

#### **Code Cell 4: Xây dựng Scikit-Learn Pipeline tự động hóa**

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Chỉ định rõ ràng danh sách đặc trưng
num_features = ['Age', 'Fare']
cat_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'Family_category']
all_features = num_features + cat_features

# Tách X (đặc trưng) và y (nhãn mục tiêu)
X = train_df[all_features]
y = train_df["Survived"].astype(int)
X_test = test_df[all_features]

# Pipeline xử lý biến số
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline xử lý biến chữ (phân loại)
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Ghép 2 pipeline lại qua ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

print("Preprocessor Pipeline đã sẵn sàng!")
```

---

# 3. Model (Lựa chọn thuật toán mô hình)

### 3.1. Bản chất bài toán

- Bài toán Titanic là bài toán **Phân loại nhị phân (Binary Classification)**: Đầu ra là xác suất hành khách thuộc lớp `1` (Sống sót) hay `0` (Tử nạn).

### 3.2. Tập hợp các thuật toán khảo sát

Chúng ta khảo sát phổ rộng từ các mô hình truyền thống tới các thuật toán Ensemble tân tiến:

- **Nhóm Cơ bản:**

  - `Logistic Regression`: Nhanh, trực quan, giải thích xác suất tốt.

  - `K-Nearest Neighbors (KNN)`: Dựa trên khoảng cách lân cận.

  - `Decision Tree`: Phân nhánh theo quy tắc if-else.

- **Nhóm Nâng cao (Ensemble & Boosting):**

  - `Random Forest`: Tập hợp nhiều cây quyết định độc lập, giảm phương sai (variance).

  - `Gradient Boosting Classifier`: Tối ưu phần dư theo từng vòng lặp.

  - `Extra Trees Classifier`: Cực tiểu hóa tương quan giữa các cây.

  - `AdaBoost Classifier`: Tập trung vào các mẫu dữ liệu dự đoán sai.

  - `XGBoost Classifier`: Thuật toán Gradient Boosting tối ưu hóa cao độ, dẫn đầu các cuộc thi Kaggle.

  - `Support Vector Machines (SVC)`: Tìm siêu phẳng phân tách tối ưu.

---

# 4. Training (Huấn luyện & Tối ưu tham số)

### 4.1. Chiến lược chống Overfitting & Rò rỉ dữ liệu (Data Leakage)

- Sử dụng **Stratified 5-Fold Cross Validation**: Đảm bảo tỉ lệ nhãn `0` và `1` đồng đều giữa các fold.

- **Đóng gói Preprocessor vào Pipeline**: Dữ liệu validation chỉ được chuyển đổi dựa trên thông số tính toán từ fold huấn luyện của chính nó.

### 4.2. Code Triển Khai Trong Notebook

#### **Code Cell 5: Huấn luyện so sánh các mô hình cơ sở (Baseline Comparison)**

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier
)
from xgboost import XGBClassifier

seed = 42

baseline_classifiers = {
    'Logistic Regression': LogisticRegression(solver='liblinear', random_state=seed),
    'Linear SVC': LinearSVC(random_state=seed),
    'SVC (RBF)': SVC(probability=True, random_state=seed),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=seed),
    'Random Forest': RandomForestClassifier(random_state=seed),
    'Gradient Boosting': GradientBoostingClassifier(random_state=seed),
    'Extra Trees': ExtraTreesClassifier(random_state=seed),
    'AdaBoost': AdaBoostClassifier(random_state=seed),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=seed)
}

# Đóng gói Preprocessor với từng Classifier thành từng Full Pipeline hoàn chỉnh
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
results = []

for name, clf in baseline_classifiers.items():
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', clf)
    ])
    scores = cross_val_score(model_pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    for fold_idx, score in enumerate(scores):
        results.append({'Model': name, 'Fold': fold_idx + 1, 'Accuracy': score})

df_results = pd.DataFrame(results)

# Trực quan hóa kết quả các Baseline Model
plt.figure(figsize=(13, 6))
order = df_results.groupby('Model')['Accuracy'].mean().sort_values(ascending=False).index
sns.boxplot(data=df_results, x='Model', y='Accuracy', order=order, palette='vlag')
plt.xticks(rotation=35, ha='right')
plt.title('So sánh độ chính xác của 10 mô hình Baseline (5-Fold CV)')
plt.tight_layout()
plt.show()

# Hiển thị bảng xếp hạng
summary = df_results.groupby('Model')['Accuracy'].agg(['mean', 'std']).loc[order]
display(summary)
```

#### **Code Cell 6: Tinh chỉnh siêu tham số (Hyperparameter Tuning với GridSearchCV)**

Sau khi khảo sát, ta chọn ra 2 thuật toán mạnh mẽ nhất là `Random Forest` và `XGBoost` để tinh chỉnh:
```python
from sklearn.model_selection import GridSearchCV

# 1. Tối ưu Random Forest
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(random_state=seed))
])

rf_params = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [4, 6, 8, 10],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
}

grid_rf = GridSearchCV(rf_pipeline, rf_params, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
grid_rf.fit(X, y)

print("Best Random Forest Params:", grid_rf.best_params_)
print(f"Best Random Forest CV Score: {grid_rf.best_score_:.4f}")

# 2. Tối ưu XGBoost
xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBClassifier(eval_metric='logloss', random_state=seed))
])

xgb_params = {
    'model__n_estimators': [100, 150, 200],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__max_depth': [3, 4, 5, 6],
    'model__subsample': [0.8, 1.0]
}

grid_xgb = GridSearchCV(xgb_pipeline, xgb_params, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
grid_xgb.fit(X, y)

print("Best XGBoost Params:", grid_xgb.best_params_)
print(f"Best XGBoost CV Score: {grid_xgb.best_score_:.4f}")
```

---

# 5. Evaluation (Đánh giá & Diễn giải mô hình)

### 5.1. Tiêu chí đánh giá toàn diện

- **Accuracy (Độ chính xác):** Tỉ lệ dự đoán đúng trên toàn bộ tập dữ liệu (thang đo chính của Kaggle Titanic).

- **Precision (Độ chuẩn xác), Recall (Độ bao phủ) & F1-Score:** Đảm bảo mô hình không dự đoán thiên lệch về một lớp khi mẫu sống sót chiếm tỉ lệ ít hơn.

- **Confusion Matrix:** Nhìn rõ số trường hợp False Positive và False Negative.

- **Feature Importance (Độ quan trọng của đặc trưng):** Giải thích quyết định của mô hình (Explainable AI).

### 5.2. Code Triển Khai Trong Notebook

#### **Code Cell 7: Đánh giá chi tiết qua Confusion Matrix & Classification Report**

```python
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix

best_rf = grid_rf.best_estimator_

# Dự đoán cross-validation out-of-fold
y_pred_cv = cross_val_predict(best_rf, X, y, cv=cv)

print("=== CLASSIFICATION REPORT ===")
print(classification_report(y, y_pred_cv, target_names=['Mất (0)', 'Sống (1)']))

# Vẽ ma trận nhầm lẫn (Confusion Matrix)
cm = confusion_matrix(y, y_pred_cv)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Dự đoán Mất', 'Dự đoán Sống'],
            yticklabels=['Thực tế Mất', 'Thực tế Sống'])
plt.title('Ma Trận Nhầm Lẫn (Confusion Matrix)')
plt.show()
```

#### **Code Cell 8: Phân tích độ quan trọng của đặc trưng (Feature Importance)**

```python
# Lấy mô hình classifier bên trong pipeline tốt nhất
trained_rf_model = best_rf.named_steps['model']

# Lấy danh sách tên các cột sau khi qua OneHotEncoder
onehot_cols = best_rf.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_features)
feature_names = num_features + list(onehot_cols)

importances = trained_rf_model.feature_importances_
feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_df.head(12), x='Importance', y='Feature', palette='crest')
plt.title('Top 12 Đặc Trưng Quan Trọng Nhất Quyết Định Sự Sống Còn')
plt.show()
```

#### **Code Cell 9: Xây dựng Mô hình kết hợp (Voting Classifier Ensemble)**

```python
from sklearn.ensemble import VotingClassifier

# Khởi tạo Logistic Regression với cấu hình chuẩn
log_clf = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(solver='liblinear', random_state=seed))
])

# Kết hợp 3 mô hình: Random Forest (tối ưu), XGBoost (tối ưu), Logistic Regression
ensemble_voting = VotingClassifier(
    estimators=[
        ('rf', grid_rf.best_estimator_),
        ('xgb', grid_xgb.best_estimator_),
        ('lr', log_clf)
    ],
    voting='soft'  # Bầu chọn theo xác suất dự đoán trung bình
)

ensemble_scores = cross_val_score(ensemble_voting, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
print(f"Ensemble Voting CV Accuracy: {ensemble_scores.mean():.4f} (+/- {ensemble_scores.std():.4f})")
```

---

# 6. Prediction (Dự đoán trên tập Test & Xuất kết quả)

### 6.1. Quy tắc cốt lõi

- Tập `test.csv` phải đi qua cùng một Pipeline tiền xử lý đã fit trên tập Train để tránh mọi sai lệch định dạng.

- Đầu ra phải là định dạng 2 cột: `PassengerId` và `Survived`.

### 6.2. Code Triển Khai Trong Notebook

#### **Code Cell 10: Huấn luyện toàn bộ dữ liệu & Xuất file nộp bài Kaggle**

```python
# Huấn luyện mô hình Voting cuối cùng trên 100% dữ liệu Train (X, y)
ensemble_voting.fit(X, y)

# Dự đoán trên tập kiểm thử test.csv
final_predictions = ensemble_voting.predict(X_test)

# Đóng gói DataFrame submission đúng chuẩn Kaggle
submission_df = pd.DataFrame({
    'PassengerId': test_df.index,
    'Survived': final_predictions.astype(int)
})

# Lưu ra file CSV
submission_df.to_csv('submission.csv', index=False)
print(" Đã xuất file submission.csv thành công!")
print(f"Số lượng bản ghi dự đoán: {len(submission_df)}")
print("Tỉ lệ dự đoán sống sót:", submission_df['Survived'].value_counts(normalize=True).to_dict())
display(submission_df.head(10))
```

---

# 7. Deployment (Triển khai Web App & Kaggle Submission)

### 7.1. Triển khai cấp độ 1: Nộp bài lên Kaggle

1. Đăng nhập vào trang cuộc thi [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic).
2. Nhấn nút **Submit Predictions**.
3. Tải file `submission.csv` vừa tạo lên.
4. Xem điểm số (Public Score) trên Bảng xếp hạng (Leaderboard). Điểm số thông thường nằm trong khoảng **78% - 82%**, một kết quả xuất sắc cho bài toán này.

---

### 7.2. Triển khai cấp độ 2: Lưu mô hình (Model Persistence)

Để phục vụ cho ứng dụng thực tế hoặc web app, ta đóng gói và lưu mô hình vào file `.pkl` bằng thư viện `joblib`.

#### **Code Cell 11: Lưu mô hình xuống ổ đĩa**

```python
import joblib

# Lưu toàn bộ Pipeline (bao gồm cả preprocessor và model)
joblib.dump(ensemble_voting, 'titanic_model_pipeline.pkl')
print(" Đã lưu mô hình thành công vào file: titanic_model_pipeline.pkl")
```

---

### 7.3. Triển khai cấp độ 3: Ứng dụng Web App dự đoán tương tác với Streamlit

Tạo một file có tên `app.py`. Người dùng có thể tự nhập thông tin cá nhân và xem hệ thống tính toán xác suất sống sót theo thời gian thực.

#### **Mã nguồn file `app.py`:**

```python
import streamlit as st
import pandas as pd
import joblib

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# Load mô hình đã lưu
@st.cache_resource
def load_model():
    return joblib.load('titanic_model_pipeline.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"Chưa tìm thấy file mô hình `titanic_model_pipeline.pkl`. Vui lòng chạy notebook để huấn luyện và lưu mô hình trước!")
    st.stop()

st.title("🚢 Dự Đoán Khả Năng Sống Sót - Tàu Titanic")
st.markdown("Nhập các thông số hành khách dưới đây để mô hình Machine Learning dự đoán xác suất sống sót:")

# Giao diện nhập dữ liệu người dùng
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox("Hạng vé (Pclass):", options=[1, 2, 3], format_func=lambda x: f"Hạng {x}")
        sex = st.selectbox("Giới tính (Sex):", options=["male", "female"], format_func=lambda x: "Nam" if x == "male" else "Nữ")
        age = st.slider("Tuổi (Age):", min_value=1, max_value=85, value=28)
        title = st.selectbox("Danh xưng (Title):", options=["Mr", "Miss", "Mrs", "Master", "Other"])

    with col2:
        fare = st.number_input("Giá vé (Fare, £):", min_value=0.0, max_value=600.0, value=32.0, step=1.0)
        embarked = st.selectbox("Cảng lên tàu (Embarked):", options=["S", "C", "Q"],
                                format_func=lambda x: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[x])
        sibsp = st.number_input("Số anh/chị/em/vợ/chồng (SibSp):", min_value=0, max_value=10, value=0)
        parch = st.number_input("Số cha/mẹ/con cái (Parch):", min_value=0, max_value=10, value=0)

    submitted = st.form_submit_button("🚀 Dự Đoán Ngay", use_container_width=True)

if submitted:
    # Tính toán đặc trưng dẫn xuất
    family_size = sibsp + parch + 1
    if family_size == 1:
        family_cat = "Single"
    elif family_size <= 4:
        family_cat = "Small"
    elif family_size <= 6:
        family_cat = "Medium"
    else:
        family_cat = "Large"

    # Tạo DataFrame đầu vào đúng định dạng các cột huấn luyện
    input_data = pd.DataFrame([{
        'Age': age,
        'Fare': fare,
        'Pclass': pclass,
        'Sex': sex,
        'Embarked': embarked,
        'Title': title,
        'Family_category': family_cat
    }])

    # Dự đoán kết quả & xác suất
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    survival_prob = probabilities[1] * 100

    st.divider()
    if prediction == 1:
        st.success(f"🎉 **Kết quả: CÓ KHẢ NĂNG SỐNG SÓT!**")
        st.metric("Xác suất sống sót", f"{survival_prob:.1f}%")
        st.balloons()
    else:
        st.error(f"☠️ **Kết quả: KHÔNG SỐNG SÓT!**")
        st.metric("Xác suất sống sót", f"{survival_prob:.1f}%")
```

#### **Cách chạy Web App:**

Trong terminal, bạn cài đặt Streamlit (nếu chưa có) và khởi chạy:
```bash
pip install streamlit
streamlit run app.py
```

---

## 📌 Bảng Tóm Tắt Quy Trình 7 Bước

| Bước | Tên Giai Đoạn | Công Cụ / Kỹ Thuật Chính | Sản Phẩm Đầu Ra |
| :--- | :--- | :--- | :--- |
| **1** | **Data** | Pandas, Seaborn, Matplotlib | Thấu hiểu dữ liệu, phân bố, phát hiện missing |
| **2** | **Features** | Regex, Imputer, OneHotEncoder, StandardScaler, Pipeline | Bộ đặc trưng sạch, chống rò rỉ dữ liệu |
| **3** | **Model** | Logistic Regression, RF, XGBoost, SVC, KNN, Tree | Danh sách ứng viên mô hình tiềm năng |
| **4** | **Training** | Stratified 5-Fold CV, GridSearchCV | Siêu tham số tối ưu cho RF và XGBoost |
| **5** | **Evaluation** | Confusion Matrix, Precision, Recall, Feature Importance | Đánh giá độ tin cậy và lý giải mô hình |
| **6** | **Prediction** | Ensemble Soft Voting Classifier | File `submission.csv` nộp Kaggle |
| **7** | **Deployment** | Kaggle Leaderboard, Joblib, Streamlit Web App | Ứng dụng web tương tác thực tế |

#   T i t a n i c - - - M a c h i n e - L e a r n i n g - f r o m - D i s a s t e r
>>>>>>> e1d0bd24fd5012d30042188ca1a208e8e43b773b
