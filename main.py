import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, f1_score, precision_score, recall_score,
    roc_curve, roc_auc_score, log_loss, auc
)
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import fbeta_score
import warnings
warnings.filterwarnings('ignore')

# Import data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
data = pd.read_csv(url, header=None, names=cols)
print("Dataset shape:", data.shape)

# 1. TIỀN XỬ LÝ DỮ LIỆU
print("\n Mô tả thống kê nâng cao:")
desc = data.describe().T
desc["missing"] = data.isnull().sum()
desc["zeros"] = (data == 0).sum()
desc["std/mean"] = desc["std"] / desc["mean"]
print(desc)
plt.figure(figsize=(15, 6))
data.boxplot()
plt.title("Boxplot toàn bộ thuộc tính (tìm outliers)")
plt.xticks(rotation=45)
plt.show()
plt.figure(figsize=(15, 6))
sns.violinplot(data=data)
plt.title("Violin plot các thuộc tính")
plt.xticks(rotation=45)
plt.show()

# Bước 1 : Xử lý Missing Values (Thay 0 bằng NaN và Impute)
# ---------------------------------------------------------
cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
data[cols_with_zero] = data[cols_with_zero].replace(0, np.nan)
# Chuẩn hóa tạm để KNN tính khoảng cách đúng
scaler_for_impute = StandardScaler()
scaled_cols = scaler_for_impute.fit_transform(data[cols_with_zero])
# Điền dữ liệu khuyết
imputer = KNNImputer(n_neighbors=7)
imputed_scaled = imputer.fit_transform(scaled_cols)
# Trả lại giá trị thực (Inverse)
data[cols_with_zero] = scaler_for_impute.inverse_transform(imputed_scaled)

# BƯỚC 2: Feature Engineering (Tạo feature mới trên dữ liệu ĐÃ SẠCH)
# Lúc này Glucose và BMI đã đầy đủ số liệu, phép chia sẽ luôn ra số thực
print("Creating new features...")
data['Glucose_BMI_Ratio'] = data['Glucose'] / (data['BMI'] + 1e-5)
data['BloodPressure_Age_Interaction'] = data['BloodPressure'] * data['Age'] / 100
data['Insulin_Glucose_Ratio'] = data['Insulin'] / (data['Glucose'] + 1e-5)
data['Metabolic_Age_Index'] = data['BMI'] * data['Age'] / 100
data['Pregnancy_Age_Risk'] = data['Pregnancies'] * data['Age'] / 100

# BƯỚC 3: Xử lý Outlier (Trên dữ liệu đã đầy đủ và có feature mới)
# ---------------------------------------------------------
def remove_outlier_robust(df, col):
    if col in ["Insulin", "SkinThickness", "DiabetesPedigreeFunction"]: # Có thể giữ lại hoặc xử lý tùy ý
        return df
    Q1 = df[col].quantile(0.10)
    Q3 = df[col].quantile(0.90)
    IQR = Q3 - Q1
    lower = Q1 - 3.0 * IQR
    upper = Q3 + 3.0 * IQR
    df[col] = df[col].clip(lower, upper)
    return df
# Áp dụng cho cả cột cũ và cột mới tạo
for c in data.columns.drop('Outcome'):
    data = remove_outlier_robust(data, c)

# BƯỚC 4: Tách, SMOTE và Chuẩn hóa cuối cùng
# ---------------------------------------------------------
X = data.drop('Outcome', axis=1)
y = data['Outcome']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
scaler_final = StandardScaler()
X_scaled = scaler_final.fit_transform(X_resampled)

# Chia tập dữ liệu (Train / Val / Test)
X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y_resampled, test_size=0.15, random_state=42, stratify=y_resampled)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)

print(f"Data ready. Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

# 2. HUẤN LUYỆN LOGISTIC REGRESSION & VẼ LEARNING CURVE
print("TRAINING BASE LOGISTIC REGRESSION (WITH HISTORY)...")

# Cấu hình model để hỗ trợ bước lặp thủ công (warm_start)
base_lr_model = LogisticRegression(
    random_state=42,
    solver='saga',      
    warm_start=True,    
    max_iter=1,         # Chỉ chạy 1 epoch mỗi lần gọi fit
    C=1.0               
)
cost_train_hist, cost_val_hist = [], []
acc_train_hist, acc_val_hist = [], []
n_iterations = 200

# Vòng lặp huấn luyện
for i in range(n_iterations):
    base_lr_model.fit(X_train, y_train)
    
    # Ghi lại Cost (Log Loss)
    train_proba = base_lr_model.predict_proba(X_train)
    val_proba = base_lr_model.predict_proba(X_val)
    cost_train_hist.append(log_loss(y_train, train_proba))
    cost_val_hist.append(log_loss(y_val, val_proba))
    
    # Ghi lại Accuracy
    acc_train_hist.append(base_lr_model.score(X_train, y_train))
    acc_val_hist.append(base_lr_model.score(X_val, y_val))

# 3. ĐÁNH GIÁ VÀ VẼ BIỂU ĐỒ

# Dự đoán cuối cùng
y_val_pred = base_lr_model.predict(X_val)
y_val_prob = base_lr_model.predict_proba(X_val)[:, 1]

# Tính toán metrics
lr_acc = accuracy_score(y_val, y_val_pred)
lr_f1 = f1_score(y_val, y_val_pred)
lr_pre = precision_score(y_val, y_val_pred)
lr_rec = recall_score(y_val, y_val_pred)
lr_auc = roc_auc_score(y_val, y_val_prob)

print(f"Final Results (Iteration {n_iterations}):")
print(f"Accuracy  : {lr_acc:.4f}")
print(f"F1-Score  : {lr_f1:.4f}")
print(f"ReCall    : {lr_rec:.4f}")
print(f"Precision : {lr_pre:.4f}")
print(f"AUC       : {lr_auc:.4f}")

# Vẽ Learning Curve
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(cost_train_hist, label="Train Cost", color='blue')
plt.plot(cost_val_hist, label="Validation Cost", color='orange')
plt.xlabel("Iterations")
plt.ylabel("Cost (Log Loss)")
plt.title("Learning Curve - Cost")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(acc_train_hist, label="Train Accuracy", color='blue')
plt.plot(acc_val_hist, label="Validation Accuracy", color='orange')
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.title("Learning Curve - Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Vẽ Confusion Matrix
cm = confusion_matrix(y_val, y_val_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Không mắc", "Mắc"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Validation set")
plt.show()

# 4. TỐI ƯU HÓA THRESHOLD (TĂNG RECALL)

print("\n" + "="*50)
print(" OPTIMIZING THRESHOLD FOR HIGH RECALL (F2-SCORE)")
print("="*50)

def find_best_threshold_f2(y_true, y_prob):
    """
    Tìm threshold tối ưu để tối đa hóa F2-Score 
    (Ưu tiên Recall cao hơn Precision)
    """
    thresholds = np.arange(0.3, 1.0, 0.01)
    f2_scores = []
    recalls = []
    precisions = []
    
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        # F2 score: Beta = 2 nghĩa là coi trọng Recall gấp 2 lần Precision
        score = fbeta_score(y_true, y_pred_t, beta=2, zero_division=0)
        f2_scores.append(score)
        recalls.append(recall_score(y_true, y_pred_t, zero_division=0))
        precisions.append(precision_score(y_true, y_pred_t, zero_division=0))
        
    best_idx = np.argmax(f2_scores)
    return thresholds[best_idx], f2_scores[best_idx], recalls[best_idx], precisions[best_idx]

# Tìm threshold tốt nhất
best_thresh, best_f2, best_rec, best_pre = find_best_threshold_f2(y_val, y_val_prob)

print(f"Optimal Threshold (F2) : {best_thresh:.4f}")
print(f"Best F2-Score          : {best_f2:.4f}")
print(f"Recall at new thresh   : {best_rec:.4f}")
print(f"Precision at new thresh: {best_pre:.4f}")

# Áp dụng Threshold mới để dự đoán lại
y_val_pred_new = (y_val_prob >= best_thresh).astype(int)

# So sánh kết quả
print("\n--- SO SÁNH HIỆU QUẢ ---")
print(f"Recall cũ (Thresh=0.5)    : {lr_rec:.4f}")
print(f"Recall mới (Thresh={best_thresh:.2f}): {best_rec:.4f} (Tăng khả năng phát hiện bệnh)")

# Vẽ Confusion Matrix mới
cm_new = confusion_matrix(y_val, y_val_pred_new)

plt.figure(figsize=(12, 5))  

# Ma trận cũ
plt.subplot(1, 2, 1)
disp_old = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Không mắc", "Mắc"])
disp_old.plot(cmap="Blues", ax=plt.gca(), colorbar=False)
plt.title(f"Threshold = 0.5\n(Missed: {cm[1,0]} cases)") # FN là dòng 2 cột 1

# Ma trận mới
plt.subplot(1, 2, 2)
disp_new = ConfusionMatrixDisplay(confusion_matrix=cm_new, display_labels=["Không mắc", "Mắc"])
disp_new.plot(cmap="Greens", ax=plt.gca(), colorbar=False)
plt.title(f"Threshold = {best_thresh:.4f}\n(Missed: {cm_new[1,0]} cases)")
plt.tight_layout()
plt.show()

"""# ENSEMBLE MODEL VỚI HYPERPARAMETER TUNING"""

# Định nghĩa các models
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

# Hyperparameter grids
param_grids = {
    
    'Random Forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    },
    'Gradient Boosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1, 0.15],
        'max_depth': [3, 4, 5]
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    }
}

print(" Training and tuning models...")

best_models = {}
best_scores = {}

for name, model in models.items():
    print(f"\n Tuning {name}...")
    grid_search = GridSearchCV(
        model, param_grids[name], 
        cv=5, scoring='accuracy', n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)
    best_models[name] = grid_search.best_estimator_
    best_scores[name] = grid_search.best_score_

    print(f"Best params: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")

# Hiển thị kết quả tuning
print("HYPERPARAMETER TUNING RESULTS")
for name, score in best_scores.items():
    print(f"{name:20}: {score:.4f}")

# Tạo Ensemble Model
print("\n Creating Ensemble Model...")

voting_clf = VotingClassifier(
    estimators=[
        ('rf', best_models['Random Forest']), 
        ('gb', best_models['Gradient Boosting']),
        ('svm', best_models['SVM'])
    ],
    voting='soft'
)
voting_clf.fit(X_train, y_train)

"""# 📈 ĐÁNH GIÁ MÔ HÌNH"""

def evaluate_model(model, X, y, model_name="Model"):
    """Đánh giá toàn diện mô hình"""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    pre = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    
    print(f"\n{model_name} Results:")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"Precision: {pre:.4f}")
    print(f"Recall   : {rec:.4f}")
    
    if y_prob is not None:
        auc_score = roc_auc_score(y, y_prob)
        print(f"AUC      : {auc_score:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Không mắc", "Mắc"])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()
    
    
    return acc, f1, pre, rec

# Đánh giá tất cả models
print("📈 MODEL EVALUATION ON VALIDATION SET")
print("="*60)

val_results = {}
for name, model in best_models.items():
    print(f"\n--- {name} ---")
    acc, f1, pre, rec = evaluate_model(model, X_val, y_val, name)
    val_results[name] = {'accuracy': acc, 'f1': f1, 'precision': pre, 'recall': rec}

# Đánh giá Ensemble
print("\n--- ENSEMBLE MODEL ---")
ensemble_acc, ensemble_f1, ensemble_pre, ensemble_rec = evaluate_model(
    voting_clf, X_val, y_val, "Ensemble"
)
val_results['Ensemble'] = {
    'accuracy': ensemble_acc, 
    'f1': ensemble_f1, 
    'precision': ensemble_pre, 
    'recall': ensemble_rec
}

# So sánh kết quả
print("\n" + "="*60)
print("MODEL COMPARISON - VALIDATION SET")
print("="*60)
results_df = pd.DataFrame(val_results).T
results_df = results_df.sort_values('accuracy', ascending=False)
print(results_df.round(4))

# Chọn model tốt nhất
best_model_name = results_df.index[0]
best_model = voting_clf if best_model_name == 'Ensemble' else best_models[best_model_name]

print(f"\n BEST MODEL: {best_model_name}")
print(f" Validation Accuracy: {results_df.loc[best_model_name, 'accuracy']:.4f}")

"""#  ĐÁNH GIÁ TRÊN TEST SET & TÌM THRESHOLD TỐI ƯU"""

# Đánh giá trên test set với model tốt nhất
print(" FINAL EVALUATION ON TEST SET")

# Dự đoán probabilities
y_test_prob = best_model.predict_proba(X_test)[:, 1]

# Tìm threshold tối ưu bằng Youden's Index
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
youden_j = tpr + (1 - fpr) - 1
best_idx = np.argmax(youden_j)
best_threshold = thresholds[best_idx]

print(f"Optimal threshold: {best_threshold:.4f}")

# Đánh giá với threshold tối ưu
y_test_pred_opt = (y_test_prob >= best_threshold).astype(int)

# Tính metrics
test_acc = accuracy_score(y_test, y_test_pred_opt)
test_f1 = f1_score(y_test, y_test_pred_opt)
test_pre = precision_score(y_test, y_test_pred_opt)
test_rec = recall_score(y_test, y_test_pred_opt)
test_auc = roc_auc_score(y_test, y_test_prob)

print(f"\nTEST SET RESULTS (Threshold = {best_threshold:.4f})")
print(f"Accuracy : {test_acc:.4f}")
print(f"F1-Score : {test_f1:.4f}")
print(f"Precision: {test_pre:.4f}")
print(f"Recall   : {test_rec:.4f}")
print(f"AUC      : {test_auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred_opt)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Không mắc", "Mắc"])
disp.plot(cmap="Blues")
plt.title(f"Final Test Set - {best_model_name}\nAccuracy: {test_acc:.4f}")
plt.show()

# Feature Importance (nếu có)
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': data.columns.drop('Outcome'),
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()
    
    print("\nTOP 10 FEATURES BY IMPORTANCE:")
    print(feature_importance.head(10))
# So sánh với baseline (threshold = 0.5)
y_test_pred_base = (y_test_prob >= 0.5).astype(int)
base_acc = accuracy_score(y_test, y_test_pred_base)
print(f"\nIMPROVEMENT COMPARISON:")
print(f"Baseline (threshold=0.5)  : {base_acc:.4f}")
print(f"Optimized (threshold={best_threshold:.4f}): {test_acc:.4f}")
print(f"Improvement               : +{(test_acc - base_acc):.4f}")
print("FINAL RESULTS SUMMARY")
print(f"\n Best Model: {best_model_name}")
print(f"Optimal Threshold: {best_threshold:.4f}")
print(f"AUC Score: {test_auc:.4f}")

# 5. TRIỂN KHAI DỰ ĐOÁN (SỬ DỤNG LOGISTIC REGRESSION

def prepare_patient_data(input_dict=None):
    """
    Trả về vector features (8 gốc + 5 feature mới).
    - Nếu input_dict = None → nhập từ bàn phím
    - Nếu input_dict != None → dùng dữ liệu truyền vào (fake)
    """

    ranges = {
        "Pregnancies": (0, 17),
        "Glucose": (44, 199),
        "BloodPressure": (24, 122),
        "SkinThickness": (7, 99),
        "Insulin": (14, 846),
        "BMI": (18.2, 67.1),
        "DiabetesPedigreeFunction": (0.078, 2.42),
        "Age": (21, 81)
    }

    features = []

    for feature in X.columns[:8]:  # 8 cột gốc
        low, high = ranges[feature]

        if input_dict is None:
            # nhập từ bàn phím
            while True:
                try:
                    value = float(input(f"{feature} ({low}-{high}): "))
                    if low <= value <= high:
                        features.append(value)
                        break
                    else:
                        print(f"Nằm ngoài khoảng {low}-{high}. Nhập lại!")
                except:
                    print("Vui lòng nhập số hợp lệ!")
        else:
            # dùng dữ liệu fake
            value = input_dict[feature]
            features.append(value)

    # Tự tạo feature mới
    Glucose = features[X.columns.get_loc("Glucose")]
    BMI = features[X.columns.get_loc("BMI")]
    BloodPressure = features[X.columns.get_loc("BloodPressure")]
    Age = features[X.columns.get_loc("Age")]
    Insulin = features[X.columns.get_loc("Insulin")]
    Preg = features[X.columns.get_loc("Pregnancies")]

    new_features = [
        Glucose / (BMI + 1e-5),
        BloodPressure * Age / 100,
        Insulin / (Glucose + 1e-5),
        BMI * Age / 100,
        Preg * Age / 100
    ]

    full_features = np.array(features + new_features).reshape(1, -1)
    return full_features
def predict_patient_logistic(model, scaler, full_features, threshold=0.32):
    """
    Dự đoán sử dụng mô hình Logistic Regression với ngưỡng tối ưu.
    """
    # 1. Chuẩn hóa dữ liệu (Sử dụng scaler đã fit từ tập train)
    scaled_features = scaler.transform(full_features)
    
    # 2. Dự đoán xác suất (Lấy xác suất của lớp 1 - Mắc bệnh)
    prob = model.predict_proba(scaled_features)[0][1]
    
    # 3. So sánh với ngưỡng (Threshold 0.32 tối ưu cho Recall)
    pred = 1 if prob >= threshold else 0

    # 4. In kết quả
    print("KẾT QUẢ CHẨN ĐOÁN (LOGISTIC REGRESSION)")
    print(f"Xác suất mắc bệnh dự tính: {prob:.4f} ({prob*100:.2f}%)")
    print(f"Ngưỡng quyết định (Recall): {threshold}")
    
    if pred == 1:
        print("KẾT LUẬN: CÓ NGUY CƠ MẮC TIỂU ĐƯỜNG")
        print("   (Khuyến nghị: Cần đi khám chuyên sâu ngay)")
    else:
        print("KẾT LUẬN: AN TOÀN (Nguy cơ thấp)")
    
    return pred, prob

# CHẠY THỬ NGHIỆM

# Dữ liệu mẫu 1 (Người trẻ, chỉ số bình thường)
sample1 = {
    "Pregnancies": 1, "Glucose": 85, "BloodPressure": 66,
    "SkinThickness": 29, "Insulin": 0, "BMI": 26.6,
    "DiabetesPedigreeFunction": 0.351, "Age": 31
}

# Dữ liệu mẫu 2 (Người trung niên, chỉ số cao - Nguy cơ cao)
sample2 = {
    "Pregnancies": 5, "Glucose": 166, "BloodPressure": 72,
    "SkinThickness": 19, "Insulin": 175, "BMI": 35.8, # BMI cao
    "DiabetesPedigreeFunction": 0.587, "Age": 51
}

# 1. Dự đoán Sample 1
print("Testing Sample 1...")
data1 = prepare_patient_data(sample1)
# Lưu ý: Truyền base_lr_model và threshold tối ưu của Logistic Regression
predict_patient_logistic(base_lr_model, scaler_final, data1, threshold=0.3)

# 2. Dự đoán Sample 2
print("Testing Sample 2...")
data2 = prepare_patient_data(sample2)
predict_patient_logistic(base_lr_model, scaler_final, data2, threshold=0.3)
# 3 . nhap vào
dattta = prepare_patient_data()
predict_patient_logistic(base_lr_model,scaler_final,dattta,  threshold=0.3)