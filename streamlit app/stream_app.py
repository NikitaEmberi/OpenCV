from numpy.lib.type_check import imag
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
try:
    import face_recognition
except:
    st.write("No module for face_recognition do not try.")
import random

mp_drawing = mp.solutions.drawing_utils
#face detection utility
mp_face_detection = mp.solutions.face_detection

#model for detecting the face
model_detection = mp_face_detection.FaceDetection()

#For selfie segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation

#Model for selfie segmentation
model = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
 
#Face Mesh
mp_face_mesh = mp.solutions.face_mesh
#Model for Face Mesh
model_face_mesh = mp_face_mesh.FaceMesh()

def get_image_download_link(img):
    img = Image.fromarray(img)
    img.save(f'img{random.randint(0,1000)}.png')
    st.success("Downloaded Successfully!")

st.title("Imageara")
st.subheader("Edit Images with OpenCV!")

add_selectbox = st.sidebar.selectbox(
    "What operations you would like to perform?",
    ("About","Face Recognition", "Face Detection","Selfie Segmentation","Grayscale","Meshing","BGR Image")
)


if add_selectbox == "About":
    st.write('''This application allows user to perform basic OpenCV operations on images
            and see the corresponding results. This application operates on libraries
            like mediapipe, dlib, face_recognition,cv2.''')

elif add_selectbox == "Grayscale":
    image_file_path = st.sidebar.file_uploader("Upload image",key="gray")
    if image_file_path is not None:
        image = np.array(Image.open(image_file_path))
        st.sidebar.image(image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        download=st.button('Download Image')
        st.image(gray_image)
        if download:
            get_image_download_link(gray_image)


elif add_selectbox == "BGR Image":
    choices = st.sidebar.radio("Choose one of these:",
                                ("Blue","Green","Red"))
    image_file_path = st.sidebar.file_uploader("Upload image",key="2")
    if image_file_path is not None:
        image = np.array(Image.open(image_file_path))
        st.sidebar.image(image)
        zeros = np.zeros(image.shape[:2], dtype="uint8")
        r, g, b = cv2.split(image)
        if choices == "Red":
            img = cv2.merge([r,zeros,zeros])
        elif choices == "Green":
            img = cv2.merge([zeros,g,zeros])
        elif choices == "Blue":
            img = cv2.merge([zeros, zeros, b])
        download=st.button('Download Image')
        st.image(img)
        if download:
            get_image_download_link(img)


elif add_selectbox == "Meshing":
    image_file_path = st.sidebar.file_uploader("Upload image")
    if image_file_path is not None:
        image = np.array(Image.open(image_file_path))
        st.sidebar.image(image)
        results = model_face_mesh.process(image)
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(image, face_landmarks)
        download=st.button('Download Image')
        st.image(image)
        if download:
            get_image_download_link(image)


elif add_selectbox == "Face Recognition":
    image_file_path = st.sidebar.file_uploader("Choose Image to train against",key="train")
    if image_file_path is not None:
        image = np.array(Image.open(image_file_path))
        st.sidebar.image(image)
        image_train = face_recognition.load_image_file(image_file_path)
        image_encodings_train = face_recognition.face_encodings(image_train)[0]
        image_location_train = face_recognition.face_locations(image_train)[0]

        actual_image_file_path =st.sidebar.file_uploader("Choose image to test with",key="actual_image")
        if actual_image_file_path is not None:
            image_test = face_recognition.load_image_file(actual_image_file_path)
            image_encodings_test = face_recognition.face_encodings(image_test)[0]
            image_location_test = face_recognition.face_locations(image_test)[0]
            results = face_recognition.compare_faces([image_encodings_test], image_encodings_train)[0]
            dst = face_recognition.face_distance([image_encodings_train],image_encodings_test)
            name = ""
            if results:
                image_train = cv2.cvtColor(image_train, cv2.COLOR_BGR2RGB)
                name = "Matched"
            else:
                name = "Not matched."
            cv2.rectangle(image_test, 
                    (image_location_test[3], image_location_test[0]),
                    (image_location_test[1], image_location_test[2]),
                    (0, 255, 0),
                    3)
            cv2.putText(image_test,f"{name}",
                    (60, 60),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1,
                    (0, 255,0),
                    3)
            download=st.button('Download Image')
            st.image(image_test)
            if download:
                get_image_download_link(image_test)


elif add_selectbox == "Face Detection":
    image_file_path = st.sidebar.file_uploader("Upload image",key="2")
    if image_file_path is not None:
        image = np.array(Image.open(image_file_path))
        st.sidebar.image(image)
        results = model_detection.process(image)

        for face_landmarks in results.detections:
            mp_drawing.draw_detection(image, face_landmarks)
        download=st.button('Download Image')
        st.image(image)
        if download:
            get_image_download_link(image)


elif add_selectbox == "Selfie Segmentation":
    image_file_path = st.sidebar.file_uploader("Upload image",key="3")
    if image_file_path is not None:
        image = np.array(Image.open(image_file_path))
        bg_color_zeros = np.zeros(image.shape, dtype="uint8")
        st.sidebar.image(image)
        results = model.process(image)
        condition = np.stack((results.segmentation_mask,)*3,axis = -1) > 0.8
        choices = st.sidebar.radio("Choose one of these:",
                                        ("Blue","Gray","White","Upload Custom Background"))
        if choices == "Blue":
            bg_color = (41, 113, 227)
            bg_color_zeros[:] = bg_color
            output_image = np.where(condition,image,bg_color_zeros)
        elif choices == "Gray":
            bg_color = (192, 192, 192)
            bg_color_zeros[:] = bg_color
            output_image = np.where(condition,image,bg_color_zeros)
        elif choices == "White":
            bg_color = (233, 238, 247)
            bg_color_zeros[:] = bg_color
            output_image = np.where(condition,image,bg_color_zeros)
        elif choices == "Upload Custom Background":
            bg_image_file_path = st.sidebar.file_uploader("Choose image",key="4")
            if bg_image_file_path is not None:
                bg_image = np.array(Image.open(bg_image_file_path))
                bg_image = cv2.resize(bg_image,(image.shape[1],image.shape[0]))
                output_image = np.where(condition,image,bg_image)
        st.image(output_image)
        download=st.button('Download Image')
        if download:
            get_image_download_link(output_image)

    

