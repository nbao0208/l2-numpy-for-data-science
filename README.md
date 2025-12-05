# NUMPY FOR DATA SCIENCE

> **Mô tả ngắn gọn:** Phân tích, khám phá cũng như tiền xử lý và xây dựng mô hình trên tập dữ liệu

---

## Mục lục 
1. [Giới thiệu](#giới-thiệu)
2. [Dataset](#dataset)
3. [Phương pháp & Thuật toán](#phương-pháp--thuật-toán)
4. [Cài đặt & Thiết lập](#cài-đặt--thiết-lập)
5. [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
6. [Kết quả](#kết-quả)
7. [Cấu trúc dự án](#cấu-trúc-dự-án)
8. [Thách thức & Giải pháp](#thách-thức--giải-pháp)
9. [Hướng phát triển](#hướng-phát-triển)
10. [Tác giả & Liên hệ](#tác-giả--liên-hệ)
---

## Giới thiệu

**Bài toán:**
Dự đoán một người dùng hay nhóm người dùng nào có xu hướng rời bỏ dịch vụ tín dụng của một ngân hàng hay tổ chức tín dụng.

**Động lực & Ứng dụng thực tế:**
Đối với ngân hàng hay một tổ chức tín dụng, việc dự đoán một người dùng nào đó có thể rời bỏ dịch vụ hay không nhằm đáp ứng dịch vụ tốt hơn để giữ chân họ là nhu cầu cần thiết và cấp bách hiện nay, thời đại lòng tin và uy tín được đánh giá rất cao.

**Mục tiêu cụ thể:**

* Mục tiêu số 1: Phân tích hành vi và thói quen giao dịch của các nhóm khách hàng.

* Mục tiêu số 2: Xây dựng được mô hình dự đoán xem một người dùng có ý định rời bỏ dịch vụ tín dụng hay không với độ chính xác cao.

## Dataset

**Nguồn dữ liệu:**
Lấy từ Kaggle

Link dataset:
https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers

**Đặc điểm dữ liệu:**

* **Kích thước:** 10127 dòng, 23 cột

* **Mô tả Features:** (Bảng dưới kiểu dữ liệu đã được chuyển đổi để phù hợp với ngữ cảnh bài toán)

| Tên cột | Ý nghĩa | Kiểu dữ liệu |
| :--- | :--- | :--- |
| `CLIENTNUM` | Mã định danh duy nhất của khách hàng. | Integer |
| `Attrition_Flag` | **Biến mục tiêu (Target)**. Trạng thái khách hàng (Existing Customer: Còn sử dụng, Attrited Customer: Đã rời bỏ). | Categorical |
| `Customer_Age` | Tuổi của khách hàng. | Integer |
| `Gender` | Giới tính (M=Male, F=Female). | Categorical |
| `Dependent_count` | Số lượng người phụ thuộc. | Integer |
| `Education_Level` | Trình độ học vấn. | Categorical |
| `Marital_Status` | Tình trạng hôn nhân. | Categorical |
| `Income_Category` | Phân khúc thu nhập hàng năm. | Categorical |
| `Card_Category` | Loại thẻ đang sử dụng (Blue, Silver, Gold, Platinum). | Categorical |
| `Months_on_book` | Thời gian quan hệ với ngân hàng (tính bằng tháng). | Integer |
| `Total_Relationship_Count` | Tổng số lượng sản phẩm khách hàng đang sử dụng. | Integer |
| `Months_Inactive_12_mon` | Số tháng không hoạt động trong 12 tháng qua. | Integer |
| `Contacts_Count_12_mon` | Số lần liên hệ với ngân hàng trong 12 tháng qua. | Integer |
| `Credit_Limit` | Hạn mức tín dụng. | Float |
| `Total_Revolving_Bal` | Tổng số dư nợ quay vòng (tiền nợ thẻ chưa trả). | Float |
| `Avg_Open_To_Buy` | Hạn mức tín dụng còn khả dụng trung bình (Credit Limit - Revolving Bal). | Float |
| `Total_Amt_Chng_Q4_Q1` | Tỷ lệ thay đổi số tiền giao dịch (Q4 so với Q1). | Float |
| `Total_Trans_Amt` | Tổng số tiền giao dịch (trong 12 tháng qua). | Integer |
| `Total_Trans_Ct` | Tổng số lần giao dịch (trong 12 tháng qua). | Integer |
| `Total_Ct_Chng_Q4_Q1` | Tỷ lệ thay đổi số lượng giao dịch (Q4 so với Q1). | Float |
| `Avg_Utilization_Ratio` | Tỷ lệ sử dụng thẻ trung bình. | Float |
| `Naive_Bayes_...` | *Các cột này sinh ra dựa trên xử lý của tác giả, được tác giả khuyến nghị loại bỏ trước khi thực hiện bất kỳ hành động nào.* | Ignore |

## Phương pháp & Thuật toán
**Quy trình xử lý dữ liệu (Pipeline):**
1.  Bước 1: Khám phá dữ liệu
2.  Bước 2: Chuyển đổi kiểu dữ liệu phù hợp
3.  Bước 3: Chuẩn hoá các dữ liệu numerical, One-Hot encoding các kiểu dữ liệu categorical bao gồm cả nhãn mục tiêu là `Attrition_Flag`


**Thuật toán sử dụng:**
Dự án sử dụng **Softmax Regression** (Multinomial Logistic Regression) để giải quyết bài toán phân loại đa lớp.

Công thức toán học áp dụng:
1. Tính điểm số (Logits):
   $$Z = X \cdot W + b$$
2. Tính xác suất dự đoán (Hàm kích hoạt Softmax):
   $$\hat{y}_k = \frac{e^{z_k}}{\sum_{j} e^{z_j}}$$
3. Hàm mất mát (Categorical Cross-Entropy):
   $$J = - \frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y^{(i)}_k \log(\hat{y}^{(i)}_k)$$

**Cách implement bằng NumPy:**
Sử dụng kỹ thuật **Vectorization** (tính toán trên ma trận) thay thế hoàn toàn cho các vòng lặp `for` để tối ưu tốc độ xử lý dữ liệu lớn:

* **Phép nhân ma trận:** Thay vì nhân từng phần tử, công thức $Z = X \cdot W + b$ được chuyển thành phép toán ma trận hiệu năng cao: `Z = X @ self.W + self.b`.
* **Numerical Stability (Ổn định số học):** Để tránh tràn số (overflow) khi tính hàm mũ $e^Z$, áp dụng kỹ thuật trừ đi giá trị lớn nhất trong hàng: `np.exp(Z - np.max(Z, axis=1, keepdims=True))`.
* **Tính Gradient:** Việc tính đạo hàm ngược (Backpropagation) để cập nhật trọng số được thực hiện cùng lúc cho toàn bộ batch dữ liệu: `dW = X.T @ dZ` với `dZ = (Y_predcit - Y_one_hot) / m`

## Cài đặt & Thiết lập
Yêu cầu hệ thống: Python 3.x, và các thư viện cần thiết.

```bash
# Clone repository này
git clone https://github.com/nbao0208/l2-numpy-for-data-science.git

# Di chuyển vào thư mục dự án
cd l2-numpy-for-data-science

# Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
```

## Hướng dẫn sử dụng

1. Khám phá dữ liệu: vào file 01_data_exploration.ipynb, chọn kernel phù hợp và sau đó `Run All`. File sẽ hiển thị các chart đánh giá và nhận xét trực quan về dữ liệu

2. Tiền xử lý dữ liệu: vào file 02_preprocessing.ipynb, chọn kernel phù hợp và sau đó `Run All`. File sẽ hiển thị quá trình nhận xét và tiền xử lý dữ liệu. Sau đó dữ liệu sau khi tiền xử lý sẽ được lưu vào thư mục processed trong thư mục data

3. Xây dựng mô hình: vào file 03_modeling.ipynb, chọn kernel phù hợp và sau đó `Run All`. File sẽ hiển thị quá trình huấn luyện dữ liệu và đánh giá sau khi huấn luyện xong

## Kết quả

Sau quá trình huấn luyện và đánh giá trên tập kiểm thử (Test Set), mô hình Softmax Regression đạt được các chỉ số khả quan:

**1. Tổng quan:**

  * **Accuracy (Độ chính xác toàn cục):** $91.02\%$ ($0.9102$)

**2. Confusion Matrix:**

| Actual \\ Predicted | Pred 0 (Existing) | Pred 1 (Churn) |
| :--- | :--- | :--- |
| **Actual 0** | **1648** (TN) | 55 (FP) |
| **Actual 1** | 127 (FN) | **196** (TP) |

**3. Chỉ số chi tiết từng lớp (Per-class Metrics):**

| Class | Precision | Recall | F1-score |
| :--- | :--- | :--- | :--- |
| **Class 0** (Existing) | 0.9285 | 0.9677 | 0.9477 |
| **Class 1** (Churn) | 0.7809 | 0.6068 | 0.6829 |

**4. Phân tích & Đánh giá:**

  * **Hiệu suất tốt trên nhóm đa số (Class 0):** Mô hình dự đoán rất chính xác khách hàng trung thành với F1-score lên tới $0.94$.
  * **Thách thức với nhóm rời bỏ (Class 1):** Mặc dù độ chính xác tổng thể cao ($91\%$), nhưng Recall của Class 1 chỉ đạt $\sim 60\%$. Điều này có nghĩa là mô hình bỏ sót khoảng $40\%$ số khách hàng thực sự rời bỏ (127 trường hợp False Negative).
  * **Nguyên nhân:** Do dữ liệu mất cân bằng (Imbalanced Dataset), số lượng mẫu Class 0 áp đảo Class 1, khiến mô hình có xu hướng thiên vị Class 0.

-----

## Cấu trúc dự án

Cấu trúc thư mục của dự án được tổ chức như sau:

```text
├── data/
│   ├── raw/BankChurners.csv       # Dữ liệu gốc
│   └── processed/                 # Dữ liệu đã qua xử lý 
├── notebooks/                     # Jupyter Notebooks 
├── src/
│   ├── data_processing.py           # Class xử lý dữ liệu & chuẩn hóa
│   ├── models.py                   # Class SoftmaxRegression
│   └── visualization.py             
├── requirements.txt               # Danh sách thư viện cần thiết
└── README.md                      # Tài liệu dự án
```

-----

## Thách thức và giải pháp

Trong quá trình xây dựng mô hình thủ công bằng NumPy đã gặp và giải quyết các vấn đề sau:

**1. Vấn đề tràn số (Numerical Overflow):**

  * **Khó khăn:** Hàm `np.exp(z)` trả về `inf` khi giá trị $z$ lớn, gây lỗi `NaN` khi tính xác suất.
  * **Giải pháp:** Áp dụng kỹ thuật **Shift Invariance**: Trừ đi giá trị lớn nhất trong hàng trước khi tính mũ ($e^{z - max(z)}$).

**2. Broadcasting & Vectorization:**

  * **Khó khăn:** Việc khớp kích thước ma trận khi tính toán Gradient (`dW`, `db`) và cập nhật trọng số mà không dùng vòng lặp rất dễ gây lỗi dimension mismatch.
  * **Giải pháp:** Kiểm soát chặt chẽ shape của ma trận, sử dụng `keepdims=True` 

-----

## Hướng phát triển

Để cải thiện hiệu suất mô hình, đặc biệt là khả năng phát hiện khách hàng rời bỏ (Class 1), các hướng phát triển tiếp theo bao gồm:

1.  **Xử lý mất cân bằng dữ liệu:** Áp dụng kỹ thuật **SMOTE** (Synthetic Minority Over-sampling Technique) hoặc gán trọng số (Class Weights) lớn hơn cho Class 1 trong hàm Loss.
2.  **Tối ưu hóa Hyperparameter:** Sử dụng Grid Search để tìm Learning Rate và số Epoch tối ưu nhất.
3.  **Feature Engineering:** Tạo thêm các đặc trưng mới hoặc sử dụng PCA để giảm chiều dữ liệu nhiễu.
4.  **Mở rộng mô hình:** Nâng cấp từ Softmax Regression (Linear) lên mạng Neural Network đơn giản (MLP) để nắm bắt các mối quan hệ phi tuyến tính.

-----

## Tác giả & liên hệ

Dự án được thực hiện bởi:

  * **Nguyễn Đình Quốc Bảo**
  * **Vai trò:** Sinh viên
  * **Liên hệ:** ndqbao23@clc.fitus.edu.vn

-----
