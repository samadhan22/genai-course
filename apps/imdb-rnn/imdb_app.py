import json
import re
from pathlib import Path

import streamlit as st
from tensorflow import keras


app_dir = Path(__file__).parent
VOCAB_SIZE = 10_000
REVIEW_LENGTH = 200


@st.cache_resource
def load_artifacts():
    model = keras.models.load_model(
        app_dir / "imdb_rnn.keras", compile=False
    )
    word_index = json.loads(
        (app_dir / "imdb_word_index.json").read_text(encoding="utf-8")
    )
    return model, word_index


def encode_review(review, word_index):
    words = re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", review.lower())

    token_ids = [1]  # Start of review.
    for word in words:
        index = word_index.get(word)
        if index is not None and index + 3 < VOCAB_SIZE:
            token_ids.append(index + 3)
        else:
            token_ids.append(2)  # Unknown word.

    return keras.utils.pad_sequences(
        [token_ids],
        maxlen=REVIEW_LENGTH,
        padding="post",
        truncating="post",
    )


st.title("IMDb review sentiment")

with st.form("sentiment_form"):
    review = st.text_area("Movie review")
    submitted = st.form_submit_button("Predict sentiment")

if submitted:
    if review.strip():
        model, word_index = load_artifacts()
        features = encode_review(review, word_index)
        score = float(model.predict(features, verbose=0)[0, 0])
        label = "Positive" if score >= 0.5 else "Negative"

        st.metric("Positive sentiment score", f"{score:.1%}")
        st.write(f"Predicted sentiment: {label}")
    else:
        st.warning("Enter a movie review.")

        