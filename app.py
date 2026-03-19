import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

nltk.download('stopwords')

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Fake News Dashboard", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* Background Gradient */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Glass Cards */
.card {
    background: rgba(255, 255, 255, 0.1);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    transition: 0.3s;
}
.card:hover {
    transform: scale(1.05);
}

/* Titles */
h1, h2, h3 {
    color: #00FFD1;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #ff416c, #ff4b2b);
    color: white;
    border-radius: 10px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🧠 AI Fake News Detection Dashboard")
st.markdown("### 🚀 Intelligent NLP + Machine Learning System")

# ---------------- LOAD DATA ----------------
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true])
data = data[['text','label']]

# ---------------- CLEAN TEXT ----------------
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

data['text'] = data['text'].apply(clean_text)

# ---------------- MODEL ----------------
X = data['text']
y = data['label']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

vectorizer = TfidfVectorizer(max_df=0.7)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

model = SVC(kernel='linear')
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🎬 Dashboard Menu")
page = st.sidebar.radio("", ["🏠 Home","📊 Analytics","🧪 Prediction"], label_visibility="hidden")

# ================= HOME =================
if page == "🏠 Home":

    st.markdown("## 📌 Key Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"<div class='card'>📄 Total News<br><h2>{len(data)}</h2></div>", unsafe_allow_html=True)

    with col2:
        st.markdown(f"<div class='card'>⚠️ Fake News<br><h2>{sum(data['label']==0)}</h2></div>", unsafe_allow_html=True)

    with col3:
        st.markdown(f"<div class='card'>🎯 Accuracy<br><h2>{round(accuracy*100,2)}%</h2></div>", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("## 📊 Distribution")

    fig = px.pie(values=data['label'].value_counts(),
                 names=["Real","Fake"],
                 hole=0.5)

    st.plotly_chart(fig, use_container_width=True)

# ================= ANALYTICS =================
elif page == "📊 Analytics":

    st.markdown("## 📈 Deep Analytics")

    data['length'] = data['text'].apply(len)

    fig = px.histogram(data, x='length', nbins=60)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("## ☁️ Word Cloud (Fake News)")

    fake_words = " ".join(data[data['label']==0]['text'])

    wc = WordCloud(width=800,height=400,background_color='black').generate(fake_words)

    fig2, ax = plt.subplots()
    ax.imshow(wc)
    ax.axis("off")
    st.pyplot(fig2)

    st.markdown("## 📉 Confusion Matrix")

    cm = confusion_matrix(y_test,y_pred)

    fig3, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm')
    st.pyplot(fig3)

# ================= PREDICTION =================
elif page == "🧪 Prediction":

    st.markdown("## 🔎 Fake News Detector")

    user_input = st.text_area("Paste News Article Here...")

    if st.button("🚀 Analyze"):

        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])
        pred = model.predict(vector)

        if pred[0] == 0:
            st.error("⚠️ Fake News Detected")
        else:
            st.success("✅ Real News")
