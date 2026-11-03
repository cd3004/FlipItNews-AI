
import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import re
import string
from pathlib import Path
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).parent
MAX_LEN = 500

st.set_page_config(
    page_title="FlipItNews AI",
    page_icon="📰",
    layout="wide"
)

# --------------------------------------------------
# 2. LOAD TRAINED MODEL AND SUPPORTING FILES
# --------------------------------------------------

@st.cache_resource
def load_models():

    # Load trained RNN model
    model = tf.keras.models.load_model(
        BASE_DIR / "flipitnews_rnn.keras"
    )

    # Load tokenizer
    with open(BASE_DIR / "flipitnews_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    # Load label encoder
    with open(BASE_DIR / "flipitnews_label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    # Load stopwords
    with open(BASE_DIR / "flipitnews_stopwords.pkl", "rb") as f:
        stop_words = pickle.load(f)

    return model, tokenizer, encoder, stop_words


model, tokenizer, encoder, stop_words = load_models()

# --------------------------------------------------
# 3. TEXT PREPROCESSING
# --------------------------------------------------

def preprocess_text(text):

    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove English stopwords
    tokens = [
        word for word in text.split()
        if word not in stop_words
    ]

    return " ".join(tokens)


# --------------------------------------------------
# 4. APPLICATION HEADER
# --------------------------------------------------

st.title("📰 FlipItNews AI News Classifier")

st.write(
    "An intelligent news classification system powered by "
    "Natural Language Processing and Recurrent Neural Networks."
)

st.divider()

# --------------------------------------------------
# 5. NEWS INPUT SECTION
# --------------------------------------------------

st.subheader("News Article Classification")

title = st.text_input(
    "Enter News Headline",
    placeholder="Enter the news headline..."
)

content = st.text_area(
    "Enter News Article Content",
    height=220,
    placeholder="Paste your news article here..."
)

# --------------------------------------------------
# 6. PREDICTION
# --------------------------------------------------

if st.button("Predict News Category", type="primary"):

    if not title.strip() and not content.strip():

        st.warning(
            "Please enter a news headline or article."
        )

    else:

        with st.spinner("Analyzing the news article..."):

            # Combine headline and article
            combined_text = title + " " + content

            # Preprocess input
            cleaned_text = preprocess_text(combined_text)

            # Convert text into sequences
            sequence = tokenizer.texts_to_sequences(
                [cleaned_text]
            )

            # Pad sequence
            padded = pad_sequences(
                sequence,
                maxlen=MAX_LEN,
                padding="post",
                truncating="post"
            )

            # Predict category
            prediction = model.predict(
                padded,
                verbose=0
            )[0]

            predicted_index = np.argmax(prediction)

            category = encoder.inverse_transform(
                [predicted_index]
            )[0]

            confidence = float(
                prediction[predicted_index]
            ) * 100

        # --------------------------------------------------
        # 7. DISPLAY RESULTS
        # --------------------------------------------------

        st.success("News classification completed!")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Predicted Category",
                value=category.upper()
            )

        with col2:
            st.metric(
                label="Prediction Confidence",
                value=f"{confidence:.2f}%"
            )

        # --------------------------------------------------
        # 8. CATEGORY-WISE PREDICTION SCORES
        # --------------------------------------------------

        st.subheader("Category-wise Prediction Scores")

        for i, label in enumerate(encoder.classes_):

            score = float(prediction[i])

            st.write(f"**{label.title()}**")

            st.progress(score)

            st.caption(
                f"{score * 100:.2f}%"
            )

# --------------------------------------------------
# 9. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "FlipItNews AI | NLP | Word2Vec | LSTM | RGIPT"
)
