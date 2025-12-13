Dưới đây là file `README.md` chi tiết và chuyên nghiệp dành cho dự án của bạn. File này được viết bằng Markdown, bạn có thể lưu lại dưới tên `README.md` để hiển thị đẹp mắt trên Github hoặc nộp kèm báo cáo.

-----

# 🏥 Dự Đoán Nguy Cơ Mắc Bệnh Tiểu Đường (Diabetes Prediction)

Dự án này xây dựng một hệ thống Machine Learning để dự đoán khả năng mắc bệnh tiểu đường ở bệnh nhân nữ (trên 21 tuổi) dựa trên các chỉ số y tế lâm sàng. Dự án sử dụng bộ dữ liệu chuẩn **Pima Indians Diabetes** và áp dụng các kỹ thuật xử lý dữ liệu nâng cao, Feature Engineering, và mô hình Ensemble Learning.

## 📋 Mục Lục

  - [Giới thiệu](https://www.google.com/search?q=%23gi%E1%BB%9Bi-thi%E1%BB%87u)
  - [Bộ Dữ Liệu](https://www.google.com/search?q=%23b%E1%BB%99-d%E1%BB%AF-li%E1%BB%87u)
  - [Cài Đặt & Yêu Cầu](https://www.google.com/search?q=%23c%C3%A0i-%C4%91%E1%BA%B7t--y%C3%AAu-c%E1%BA%A7u)
  - [Quy Trình Xử Lý (Pipeline)](https://www.google.com/search?q=%23quy-tr%C3%ACnh-x%E1%BB%AD-l%C3%BD-pipeline)
  - [Mô Hình & Thuật Toán](https://www.google.com/search?q=%23m%C3%B4-h%C3%ACnh--thu%E1%BA%ADt-to%C3%A1n)
  - [Kết Quả Đánh Giá](https://www.google.com/search?q=%23k%E1%BA%BFt-qu%E1%BA%A3-%C4%91%C3%A1nh-gi%C3%A1)
  - [Hướng Dẫn Sử Dụng](https://www.google.com/search?q=%23h%C6%B0%E1%BB%9Bng-d%E1%BA%ABn-s%E1%BB%AD-d%E1%BB%A5ng)

## Giới thiệu

Mục tiêu của dự án là hỗ trợ chẩn đoán sớm bệnh tiểu đường bằng cách phân tích các chỉ số như Glucose, BMI, Insulin, v.v. Hệ thống tập trung vào việc tối ưu hóa chỉ số **Recall** (Độ nhạy) để giảm thiểu tỷ lệ bỏ sót người bệnh (False Negative), đồng thời giải quyết vấn đề mất cân bằng dữ liệu (Imbalanced Data).

## Bộ Dữ Liệu

Dữ liệu được lấy từ nguồn [Pima Indians Diabetes Database](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv).

**Các đặc trưng gốc (Original Features):**

1.  `Pregnancies`: Số lần mang thai.
2.  `Glucose`: Nồng độ đường huyết (2 giờ sau khi uống dung dịch đường).
3.  `BloodPressure`: Huyết áp tâm trương (mm Hg).
4.  `SkinThickness`: Độ dày nếp gấp da cơ tam đầu (mm).
5.  `Insulin`: Nồng độ insulin huyết thanh (mu U/ml).
6.  `BMI`: Chỉ số khối cơ thể (cân nặng/chiều cao^2).
7.  `DiabetesPedigreeFunction`: Chỉ số di truyền bệnh tiểu đường.
8.  `Age`: Tuổi.
9.  `Outcome`: Nhãn (1: Mắc bệnh, 0: Không mắc).

**Các đặc trưng tạo mới (Engineered Features):**

  - `Glucose_BMI_Ratio`: Tương quan giữa đường huyết và cân nặng.
  - `BloodPressure_Age_Interaction`: Tương tác giữa huyết áp và tuổi.
  - `Insulin_Glucose_Ratio`: Đánh giá mức độ kháng Insulin.
  - `Metabolic_Age_Index`: Chỉ số trao đổi chất dựa trên tuổi và BMI.
  - `Pregnancy_Age_Risk`: Nguy cơ tích lũy từ thai kỳ và tuổi tác.

## Cài Đặt & Yêu Cầu

Dự án yêu cầu **Python 3.x** và các thư viện sau. Bạn có thể cài đặt bằng lệnh:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn
```

## Quy Trình Xử Lý (Pipeline)

1.  **Khám Phá Dữ Liệu (EDA):**

      - Thống kê mô tả (Descriptive Statistics).
      - Trực quan hóa phân phối và ngoại lai bằng Boxplot và Violin plot.

2.  **Xử Lý Dữ Liệu Khuyết (Missing Values):**

      - Phát hiện các giá trị `0` phi lý trong các cột sinh học (Glucose, BP, Skin, Insulin, BMI).
      - Thay thế `0` bằng `NaN`.
      - Sử dụng **KNN Imputer** (K-Nearest Neighbors) để điền dữ liệu khuyết dựa trên sự tương đồng giữa các mẫu.

3.  **Feature Engineering:**

      - Tạo ra 5 đặc trưng mới giúp mô hình nắm bắt tốt hơn các mối quan hệ phi tuyến tính giữa các chỉ số sức khỏe.

4.  **Xử Lý Ngoại Lai (Outliers):**

      - Sử dụng phương pháp IQR (Interquartile Range) để kẹp (clip) các giá trị ngoại lai, giúp mô hình bền vững hơn.

5.  **Chống Rò Rỉ Dữ Liệu (Prevent Data Leakage):**

      - **QUAN TRỌNG:** Thực hiện chia tập Train/Test/Val **TRƯỚC** khi áp dụng các kỹ thuật cân bằng dữ liệu.

6.  **Cân Bằng Dữ Liệu (Data Balancing):**

      - Sử dụng kỹ thuật **SMOTE** (Synthetic Minority Over-sampling Technique) chỉ trên tập **Train** để giải quyết vấn đề mất cân bằng nhãn (số lượng người khỏe mạnh nhiều hơn người bệnh).

7.  **Chuẩn Hóa (Scaling):**

      - Sử dụng `StandardScaler` để đưa dữ liệu về phân phối chuẩn (mean=0, std=1).

## 🤖 Mô Hình & Thuật Toán

Dự án triển khai và so sánh các phương pháp sau:

1.  **Logistic Regression (Base Model):**

      - Huấn luyện thủ công qua từng epoch để vẽ biểu đồ **Learning Curve** (Cost & Accuracy).
      - Tối ưu hóa ngưỡng quyết định (Threshold Tuning) dựa trên F2-Score để ưu tiên Recall.

2.  **Ensemble Learning (Advanced Model):**

      - Kết hợp sức mạnh của 3 thuật toán:
          - **Random Forest**
          - **Gradient Boosting**
          - **Support Vector Machine (SVM)**
      - Sử dụng **GridSearchCV** để tinh chỉnh siêu tham số (Hyperparameter Tuning).
      - **Voting Classifier** (Soft Voting) để tổng hợp kết quả dự đoán.

## Kết Quả Đánh Giá

Hệ thống sử dụng các chỉ số đánh giá toàn diện:

  - **Accuracy:** Độ chính xác tổng thể.
  - **Recall (Sensitivity):** Tỷ lệ phát hiện đúng người bệnh (Chỉ số quan trọng nhất).
  - **Precision:** Độ chính xác trong các dự đoán bệnh.
  - **F1-Score & F2-Score:** Trung bình điều hòa giữa Precision và Recall.
  - **ROC - AUC:** Diện tích dưới đường cong ROC.

*Biểu đồ Learning Curve và Confusion Matrix được tạo ra để trực quan hóa hiệu năng mô hình.*

## Hướng Dẫn Sử Dụng

1.  **Chạy toàn bộ quy trình:**
    Chạy file script chính (ví dụ `main.py` hoặc `diabetes_prediction.py`).

2.  **Dự đoán cho bệnh nhân mới:**
    Hàm `prepare_patient_data()` và `predict_patient_logistic()` đã được tích hợp sẵn. Bạn có thể nhập thông tin bệnh nhân từ bàn phím hoặc truyền vào dictionary dữ liệu:

    ```python
    sample_patient = {
        "Pregnancies": 5, "Glucose": 166, "BloodPressure": 72,
        "SkinThickness": 19, "Insulin": 175, "BMI": 35.8,
        "DiabetesPedigreeFunction": 0.587, "Age": 51
    }
    # Dự đoán
    data_processed = prepare_patient_data(sample_patient)
    predict_patient_logistic(base_lr_model, scaler_final, data_processed, threshold=0.3)
    ```

-----

**Tác giả:** Nhóm 12 - Học phần Trí Tuệ Nhân Tạo.
