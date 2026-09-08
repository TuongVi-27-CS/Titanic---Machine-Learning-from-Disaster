# Tài liệu Web App - Dự đoán cơ hội sống sót Titanic 🚢

Đây là tài liệu mô tả chi tiết về ứng dụng web dự đoán khả năng sống sót của hành khách trên tàu Titanic, được xây dựng bằng Flask.

## 1. Tổng quan
Ứng dụng cho phép người dùng nhập các thông tin của một hành khách (hạng vé, giới tính, tuổi, giá vé, v.v.) và trả về dự đoán liệu người đó có khả năng sống sót hay không, kèm theo xác suất (%) dựa trên mô hình Machine Learning đã được huấn luyện.

## 2. Công nghệ sử dụng
- **Backend:** Flask (Python)
- **Frontend:** HTML, CSS (Vanilla), JavaScript (Fetch API) - được nhúng trực tiếp trong file `app.py`.
- **Machine Learning:** `scikit-learn`, `pandas`, `joblib` (dùng để load mô hình).

## 3. Cấu trúc ứng dụng
Toàn bộ logic của ứng dụng được đặt trong file [`app.py`](./app.py).

### File mô hình (`titanic_model_pipeline.pkl`)
- Ứng dụng sử dụng một mô hình đã được huấn luyện và lưu lại dưới dạng file `.pkl`.
- Khi server khởi động, ứng dụng sẽ cố gắng đọc file này thông qua `joblib.load()`.
- Nếu không tìm thấy hoặc có lỗi xảy ra, ứng dụng vẫn chạy nhưng API sẽ trả về lỗi khi người dùng bấm dự đoán.

### Giao diện (Frontend)
- Được định nghĩa trong biến `HTML_TEMPLATE`.
- Form nhập liệu gồm các trường: `Pclass`, `Sex`, `Age`, `Title`, `Fare`, `Embarked`, `SibSp`, `Parch`.
- Khi form được submit, JavaScript sẽ chặn hành vi mặc định, gom dữ liệu thành JSON và gửi một POST request đến `/api/predict`.

### API Endpoint (`/api/predict`)
- **Method:** `POST`
- **Body:** JSON chứa thông tin người dùng nhập.
- **Logic:**
  1. Kiểm tra xem mô hình đã được load thành công chưa.
  2. Lấy dữ liệu từ request.
  3. Tính toán thêm trường `Family_category` (Kích thước gia đình: Single, Small, Medium, Large) dựa vào `SibSp` và `Parch`.
  4. Tạo DataFrame (`pandas`) từ dữ liệu.
  5. Đưa dữ liệu vào mô hình để dự đoán (`model.predict`) và lấy xác suất (`model.predict_proba`).
- **Response:** JSON trả về kết quả dự đoán (1 là sống, 0 là chết) và xác suất sống sót.

## 4. Hướng dẫn chạy môi trường Local
1. Đảm bảo bạn đã cài đặt Python.
2. Cài đặt các thư viện cần thiết:
   ```bash
   pip install flask pandas joblib scikit-learn
   ```
3. Đảm bảo file `titanic_model_pipeline.pkl` nằm cùng thư mục với `app.py`.
4. Chạy ứng dụng:
   ```bash
   python app.py
   ```
5. Mở trình duyệt và truy cập: `http://127.0.0.1:5000`

## 5. Các lỗi thường gặp khi Deploy (Vercel, Render...)
**Lỗi: "Không tìm thấy dữ liệu" hoặc "Không tìm thấy file mô hình trên server."**
- **Nguyên nhân 1:** File `titanic_model_pipeline.pkl` không được tải lên server (có thể do bị chặn bởi `.gitignore` hoặc quên commit/push file này lên GitHub).
- **Nguyên nhân 2:** Thiếu các thư viện yêu cầu trong file `requirements.txt`. Khi server deploy, nó cần cài đặt đủ `scikit-learn`, `pandas`, `joblib` để đọc được file mô hình. Nếu thiếu, hàm `joblib.load()` sẽ bị lỗi (Exception) dẫn đến biến `model` bị gán bằng `None`.
- **Nguyên nhân 3:** Phiên bản `scikit-learn` trên môi trường deploy khác với phiên bản được dùng để train file `.pkl` lúc ở máy local. Hãy đảm bảo version giống nhau trong `requirements.txt`.
