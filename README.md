# Ứng dụng Phân loại Tin tức AG (AG News Classification App)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red.svg)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange.svg)

## 📖 Giới thiệu

Đây là một ứng dụng web đơn giản được xây dựng bằng **Streamlit** để phân loại các bản tin tiếng Anh vào 4 chủ đề khác nhau. Mô hình học máy được huấn luyện trên bộ dữ liệu **AG News Classification**, sử dụng các kỹ thuật xử lý ngôn ngữ tự nhiên (NLP) cơ bản và mô hình **Logistic Regression**.

Ứng dụng cho phép người dùng nhập vào tiêu đề và mô tả của một bản tin, sau đó trả về chủ đề dự đoán cùng với độ tin cậy của dự đoán đó.

## 📸 Demo ứng dụng

*Mẹo: Hãy chạy ứng dụng trên máy của bạn, chụp một bức ảnh màn hình đẹp và thay thế liên kết bên dưới để README của bạn trông chuyên nghiệp hơn.*


![Demo ứng dụng](https://github.com/user-attachments/assets/3593dd7d-add9-4bfe-9f1f-b5a00c69c9a7)

## ✨ Tính năng

-   Giao diện web trực quan, dễ sử dụng được xây dựng bằng Streamlit.
-   Phân loại tin tức thành 4 chủ đề: **Thế giới (World)**, **Thể thao (Sports)**, **Kinh doanh (Business)**, và **Khoa học/Công nghệ (Sci/Tech)**.
-   Áp dụng các bước tiền xử lý văn bản chuẩn (loại bỏ stop words, stemming, TF-IDF).
-   Hiển thị kết quả dự đoán cùng với phần trăm độ tin cậy.

## 🛠️ Công nghệ sử dụng

-   **Ngôn ngữ**: Python 3.9+
-   **Giao diện Web**: Streamlit
-   **Học máy & NLP**:
    -   Scikit-learn (cho mô hình Logistic Regression và TfidfVectorizer)
    -   NLTK (cho việc tiền xử lý văn bản)
    -   Pandas & NumPy (để xử lý dữ liệu)
-   **Lưu trữ mô hình**: Joblib

## 📂 Cấu trúc thư mục
![Screenshot 2025-06-21 080838](https://github.com/user-attachments/assets/df22c82b-b255-42e7-85c4-87fec5dcc220)


