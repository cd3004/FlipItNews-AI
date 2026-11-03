
import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import re
import string

from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="FlipItNews AI",
    page_icon="📰",
    layout="wide"
)

# Load trained models
@st.cache_resource
def load_models():

    model = tf.keras.models.load_model(
        "/content/flipitnews_rnn.keras"
    )

    with open("/content/flipitnews_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("/content/flipitnews_label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    return model, tokenizer, encoder

model, tokenizer, encoder = load_models()

MAX_LEN = 500

# Dashboard design
st.title("📰 FlipItNews AI News Classifier")

st.write(
    "An NLP-based application that automatically "
    "classifies news articles into different categories."
)

st.divider()

st.subheader("Enter News Article")

title = st.text_input("News Headline")

content = st.text_area(
    "News Article Content",
    height=200,
    placeholder="Paste your news article here..."
)

if st.button("Predict News Category", type="primary"):

    if not title.strip() and not content.strip():

        st.warning("Please enter a headline or news article.")

    else:

        text = title + " " + content

        # Text preprocessing
        text = text.lower()
        text = text.translate(
            str.maketrans('', '', string.punctuation)
        )
        text = re.sub(r'\s+', ' ', text).strip()

        # Tokenization and padding
        sequence = tokenizer.texts_to_sequences([text])

        padded = pad_sequences(
            sequence,
            maxlen=MAX_LEN,
            padding='post',
            truncating='post'
        )

        # Prediction
        prediction = model.predict(padded, verbose=0)[0]

        predicted_index = np.argmax(prediction)

        category = encoder.inverse_transform(
            [predicted_index]
        )[0]

        confidence = prediction[predicted_index] * 100

        st.success("Classification Completed!")

        st.subheader("Prediction Result")

        st.metric(
            "Predicted News Category",
            category.upper()
        )

        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )

        st.subheader("Category Probability")

        for i, label in enumerate(encoder.classes_):

            st.write(f"**{label.title()}**")

            st.progress(float(prediction[i]))

            st.write(f"{prediction[i]*100:.2f}%")

st.divider()

st.caption(
    "FlipItNews AI | Powered by NLP, Word2Vec and LSTM"
)
