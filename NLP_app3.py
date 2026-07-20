import pandas as pd
import numpy as np
import streamlit as st
import cv2
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model
model=load_model('digit_recognization_model.keras')

st.title("handwriten digit recognization")
canvas_result=st_canvas(
    fill_color="#00000000",
    stroke_width=10,
    stroke_color="#000000",
    background_color="#FFFFFF",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

if st.button("predict"):
    st.write("predicting...")
    # convert canvas image to numpy array
    img=canvas_result.image_data.astype(np.uint8)
    # convert image to greyscale
    grey_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #resize image to 28X28
    grey_img=cv2.resize(grey_img,(28,28))
    # normalize
    grey_img= grey_img/255.0
    # reshape the pixel value to be between 0 and 1
    grey_img=grey_img.reshape(-1,784)

    result=model.predict(grey_img)
    #get index of highest probability digit
    index=np.argmax(result)
    st.write(f"The predicted digit is: {index}") 
