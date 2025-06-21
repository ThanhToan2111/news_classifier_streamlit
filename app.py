import streamlit as st
st.set_page_config(page_title="Phân loại Tin tức AG", layout="wide")

import joblib 
import re 
import string 
import nltk
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer

nltk.download('stopwords')

@st.cache_resource 
def load_model_and_vectorizer():
    model = joblib.load('logistic_regression_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub('<.*?>', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    tokens = re.findall(r"[\w']+", text.lower())
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation and not word.isdigit()]
    extra_words = ['href', 'lt', 'gt', 'ii', 'iii', 'ie', 'quot', 'com']
    tokens = [word for word in tokens if word not in extra_words]
    stemmed_tokens = [PorterStemmer().stem(word) for word in tokens]
    return ' '.join(stemmed_tokens)

st.title("Ứng dụng Phân loại Tin tức AG")
st.write("Nhập vào tiêu đề và mô tả của một bản tin để phân loại nó vào một trong bốn chủ đề: Thế giới, Thể thao, Kinh doanh, hoặc Khoa học/Công nghệ.")

class_map = {
    1: 'Thế giới (World)',
    2: 'Thể thao (Sports)',
    3: 'Kinh doanh (Business)',
    4: 'Khoa học/Công nghệ (Sci/Tech)'
}

with st.form("news_form"):
    title = st.text_input("Tiêu đề (Title)")
    description = st.text_area("Mô tả (Description)")
    submitted = st.form_submit_button("Phân loại")

if submitted:
    if not title and not description:
        st.error("Vui lòng nhập tiêu đề hoặc mô tả.")
    else:
        combined_text = title + " " + description
        processed_text = preprocess_text(combined_text)
        text_vector = vectorizer.transform([processed_text])
        prediction = model.predict(text_vector)
        predicted_class_id = prediction[0]
        prediction_proba = model.predict_proba(text_vector)
        confidence = prediction_proba[0][predicted_class_id - 1] * 100
        predicted_class_name = class_map[predicted_class_id]
        st.success(f"**Chủ đề được dự đoán là:** {predicted_class_name}")
        st.info(f"**Độ tin cậy:** {confidence:.2f}%")
