import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

import streamlit as st
import pickle 
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tk = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))

#Streamlit APP
st.set_page_config(page_title="SMS Spam Detection",page_icon="📩",layout="centered")

st.markdown(
    """
    <style>
    body{
      font-family:Arial,sans-serif;
    }
    .main{
      backgound:white;
      padding:20px
      border-radius:10px
      box-shadow:0px 4px 6px rgba(0,0,0,0.1);
    }
    .stButton > button{
      backgound-color:#007bff
      color: white;
      border: none;
      border-radius: 5px;
      padding: 10px 20px;
      font-size: 16px;
      cursor: pointer;

    }
    .stButton > button:hover {
      background-color: #0056b3;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📩 SMS Spam Detection Model")
st.markdown("**Predict if a message is spam or not using AI model.**")
st.info("Enter your SMS message below to check its classification.")
    
#Input 
input_sms = st.text_area("📝 Enter the SMS:", height=150, placeholder="Type your message here...")


if st.button("🔍 Predict"):
    if input_sms.strip():
        # Preprocess
        transformed_sms = transform_text(input_sms)
        # Vectorize
        vector_input = tk.transform([transformed_sms])
        # Predict
        result = model.predict(vector_input)[0]
        # Display Result
        if result == 1:
            st.markdown("### 🚨 **Spam Message Detected!**", unsafe_allow_html=True)
            st.error("Be cautious! This message might be harmful or unwanted.")
        else:
            st.markdown("### ✅ **Not Spam**", unsafe_allow_html=True)
            st.success("This message seems safe.")
    else:
        st.warning("Please enter a valid SMS to analyze.")

st.markdown("---")
st.markdown("Made with ❤️ by **Anshil Maurya**")        
