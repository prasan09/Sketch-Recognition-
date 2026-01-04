import streamlit as st
from PIL import Image
from face_match import recognize_face
from face_match import artist_feedback


st.set_page_config(page_title="Sketch Recognition", layout="centered")

st.title("Sketch Recognition & Feedback")

uploaded = st.file_uploader(
    "Upload a sketch",
    type=["jpg", "png", "jpeg"]
)

if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Sketch")
    

    if st.button("Analyze"):
        with st.spinner("Analyzing..."):
            # 🔥 FIX HERE
            with open("temp.jpg", "wb") as f:
                f.write(uploaded.getvalue())

            name, confidence = recognize_face("temp.jpg")

        if name is None:
            st.error("Face not detected")
        else:
            st.success(f"Guess: {name}")
            st.write(f"Confidence: {confidence}")
        
        feedback = artist_feedback("temp.jpg", name)

        if feedback:
            st.subheader("Artist Feedback")
            for part, score in feedback.items():
                if score >= 85:
                    st.write(f"{part}: Very good ✔️ ({score}%)")
                elif score >= 65:
                    st.write(f"{part}: Good, can improve ({score}%)")
                else:
                    st.write(f"{part}: Needs improvement ❌ ({score}%)")

