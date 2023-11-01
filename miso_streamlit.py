import streamlit as st
import pickle
import tensorflow as tf
from keras.preprocessing import sequence
import re
import keras

# Load the tokenizer and pre-trained model
with open('tokenizer_final.pickle', 'rb') as handle:
    load_tokenizer = pickle.load(handle)
load_model = keras.models.load_model("misogyny3_model.h5")

# Function to preprocess the text input
def preprocess_text(text):
    TEXT_CLEANING_RE = "&.*?;|<.*?>\S+|@\S+|https?:\S+|https?://\S+|www\.\S+|http?:\S|[^A-Za-z0-9]+"
    stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']
    
    # Remove special characters, numbers, and stopwords
    text = re.sub(TEXT_CLEANING_RE, ' ', text.lower()).strip()
    tokens = [token for token in text.split() if token not in stopwords]
    return " ".join(tokens)

# # Streamlit UI
st.title('Misogyny Detector')
st.write('This app predicts whether a text contains misogynistic content or not.')

user_input = st.text_area('Enter a text:')
if st.button('Predict'):
    if not user_input:
        st.warning('Please enter a text for prediction.')
    else:
        preprocessed_input = [preprocess_text(user_input)]
        seq = load_tokenizer.texts_to_sequences(preprocessed_input)
        padded = sequence.pad_sequences(seq, maxlen=1000)
        pred = load_model.predict(padded)
        if pred < 0.5:
            st.error('Misogynistic content detected.')
        else:
            st.success('Non-misogynistic content detected.')

# (Optional) You can also provide some additional information about your model and preprocessing steps
st.write('**Model Information:**')
st.write('- Pretrained Model: LSTM Neural Network')
st.write('- Text Preprocessing: Text cleaning (removing special characters, numbers) and stopword removal')

st.write('**Disclaimer:** This model is for demonstration purposes and may not be 100% accurate. Use it with caution.')


