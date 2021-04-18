import streamlit as st
from cassava.pretrained import get_model
import numpy as np
from PIL import Image
import json



st.set_option('deprecation.showfileUploaderEncoding',False)
@st.cache(allow_output_mutation=True)
def load_model():
    model = get_model(name="tf_efficientnet_b4")
    return model

model = load_model()

st.write("""
    # Cassava Leaf Disease Classification
    """)

file = st.file_uploader("Upload your Image Here",type=["jpg","png"])


def make_prediction(image,model):
    img = np.array(image)
    value = model.predit_as_json(img)
    return value


if file is None:
    st.text("Please Upload an image file")
else:
    image = Image.open(file)
    st.image(image,use_column_width=True)
    prediction = make_prediction(image=image,model=model)
    print(prediction)
    st.json(prediction)
    st.success("Prediction made sucessful")