import json
import re
from pathlib import Path

import numpy as np
import streamlit as st
from tensorflow import keras


APP_DIR = Path(__file__).parent


@st.cache_resource
def load_model_and_vocabulary():
    vocabulary = json.loads(
        (APP_DIR / "hamlet_vocab.json").read_text(encoding="utf-8")
    )
    id_to_word = vocabulary["id_to_word"]
    word_to_id = {word: index for index, word in enumerate(id_to_word)}
    model = keras.models.load_model(
        APP_DIR / "hamlet_lstm.keras", compile=False
    )
    return model, id_to_word, word_to_id, vocabulary["context_length"]


st.title("Hamlet next-word predictor")

with st.form("phrase_form"):
    prompt = st.text_input("Start of a phrase", value="to be or not to")
    submitted = st.form_submit_button("Predict next word")

if submitted:
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", prompt.lower())

    if not words:
        st.warning("Enter a phrase with at least one word.")
    else:
        model, id_to_word, word_to_id, context_length = (
            load_model_and_vocabulary()
        )

        token_ids = [
            word_to_id.get(word, 1) for word in words[-context_length:]
        ]
        model_input = keras.utils.pad_sequences(
            [token_ids], maxlen=context_length, padding="post", value=0
        )

        probabilities = model.predict(model_input, verbose=0)[0]
        # Skip [PAD] and [UNK], then map the slice positions back to IDs.
        top_ids = np.argsort(probabilities[2:])[-3:][::-1] + 2

        for rank, token_id in enumerate(top_ids, start=1):
            word = id_to_word[token_id]
            score = probabilities[token_id]
            st.write(f"{rank}. **{word}** — {score:.2%}")