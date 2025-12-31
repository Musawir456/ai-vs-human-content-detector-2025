import streamlit as st
import pickle


# 1) Load models
@st.cache_resource
def load_model():
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("logreg_model.pkl", "rb") as f:
        model = pickle.load(f)
    return tfidf, model


tfidf, model = load_model()


# 2) App UI
st.title("AI vs Human Content Detector (2025 Dataset)")

st.write(
    "In this app, you can paste any text, "
    "and the model will predict whether it is **AI-generated** or **human-written**."
)

user_text = st.text_area("Paste your text here:", height=200)

if st.button("Check"):
    if user_text.strip() == "":
        st.warning("Please enter or paste some text first.")
    else:
        X_input = tfidf.transform([user_text])
        pred = model.predict(X_input)[0]

        # NOTE: Check the Kaggle description:
        # If 1 = AI and 0 = Human, keep this mapping
        label_map = {0: "Human-written", 1: "AI-generated"}

        result = label_map.get(int(pred), str(pred))

        st.subheader("Result:")
        st.success(f"Prediction: **{result}**")
