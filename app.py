import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

st.set_page_config(page_title="Lung Cancer Detection", layout="centered")
st.title("Detecting Lung Cancer BY A.I")
st.write("Welcome To This App Please Load Your C.T")

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("Lung_cancer.keras")
model = load_my_model()

# set upload button
uploaded_file = st.file_uploader("Upload your file", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # showing pic in the page
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.success("The Image Has Been Uploaded Successfully")
    img_resixed = image.convert("RGB").resize((224, 224))
    img_array = np.asarray(img_resixed, dtype="float32")
    img_array_normal = img_array/255
    img_tensor = np.expand_dims(img_array_normal, axis=0)
    st.info("The Image Is Ready")
    st.markdown("---")

    if st.button("Predict"):
        with st.spinner("Predicting..."):
            prediction = model.predict(img_tensor)
            predicted_class_index = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class_index]*100
            class_names = ["خوش‌خیم (Benign)", "بدخیم (Malignant)", "نرمال و سالم (Normal)"]
            result = class_names[predicted_class_index]

            st.subheader("📋 نتیجه تحلیل مدل:")

            if predicted_class_index == 2:  # کلاس نرمال
                st.success(f"وضعیت: {result} | میزان اطمینان مدل: {confidence:.2f}%")
                st.balloons()  # یک افکت قشنگ برای خبر خوب سالم بودن ریه
            elif predicted_class_index == 0:  # کلاس خوش‌خیم
                st.warning(f"وضعیت: {result} | میزان اطمینان مدل: {confidence:.2f}%")
            else:  # کلاس بدخیم
                st.error(f"وضعیت: {result} | میزان اطمینان مدل: {confidence:.2f}%")

            st.write("📊 جزئیات دقیق احتمالات هر کلاس:")
            for idx, name in enumerate(class_names):
                st.write(f"- {name}: {prediction[0][idx] * 100:.2f}%")