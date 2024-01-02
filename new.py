import streamlit as st
import cv2
from mtcnn import MTCNN
from PIL import Image
import numpy as np
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import requests
import base64
import streamlit as st
import plotly.express as px

df = px.data.iris()

@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("pexels-photo-3473569.jpeg")
img1 = get_img_as_base64("pexell.jpg")

# Your existing Streamlit code here

# Define the CSS style
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:image/png;base64,{img}");
background-size: 180%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}
[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img1}");
background-size: 180%;
background-blur: 100%;
background-position: center;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stHeader"] {{
background: linear-gradient (...);
}} {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

with st.sidebar:
    st.sidebar.title("Created By")
    st.sidebar.text("Hindav")


def get_wikipedia_info(topic):
    response = requests.get(
        f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}",
        headers={"User-Agent": "Streamlit App"}
    )
    data = response.json()
    return data['extract'], data['thumbnail']['source']


detector = MTCNN()
model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')
feature_list = pickle.load(open('embedding.pkl','rb'))
filenames = pickle.load(open('filenames.pkl','rb'))

def save_uploaded_image(uploaded_image):
    try:
        with open(os.path.join('uploads',uploaded_image.name),'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except:
        return False

def extract_features(img_path,model,detector):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)

    x, y, width, height = results[0]['box']

    face = img[y:y + height, x:x + width]

    #  extract its features
    image = Image.fromarray(face)
    image = image.resize((224, 224))

    face_array = np.asarray(image)

    face_array = face_array.astype('float32')

    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result

def recommend(feature_list,features):
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos

def extract_features_from_video(video_path, model, detector, resize_factor=0.5, skip_frames=5):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % skip_frames != 0:
            continue

        # Resize frame
        frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)

        # Detect faces in each frame
        results = detector.detect_faces(frame)
        if results:
            x, y, width, height = results[0]['box']
            x, y, width, height = max(0, x), max(0, y), max(0, width), max(0, height)
            face = frame[y:y + height, x:x + width]

            # Resize and preprocess the face image
            image = Image.fromarray(face)
            image = image.resize((224, 224))
            face_array = np.asarray(image)
            face_array = face_array.astype('float32')
            expanded_img = np.expand_dims(face_array, axis=0)
            preprocessed_img = preprocess_input(expanded_img)
            features = model.predict(preprocessed_img).flatten()

            frames.append(features)

    cap.release()
    return frames
st.markdown("""
    <link rel="stylesheet" type="text/css" href="style.css">
""", unsafe_allow_html=True)

st.title('Celebrity Face Recognition')

uploaded_image = st.file_uploader('Upload an image')
uploaded_video = st.file_uploader('Upload a video')

if uploaded_image is not None:
    # save the image in a directory
    if save_uploaded_image(uploaded_image):
        # load the image
        display_image = Image.open(uploaded_image)

        # extract the features
        features = extract_features(os.path.join('uploads',uploaded_image.name),model,detector)
        # recommend
        index_pos = recommend(feature_list,features)
        predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))
        predicted_actor_readable = predicted_actor.replace('_', ' ').title()
        # display
        col1,col2 = st.columns(2)

        with col1:
            st.header('The image you uploaded')
            st.image(display_image, width=300)
        with col2:
            st.header("Seems like " + predicted_actor_readable)
            # Use predicted_actor as the topic for Wikipedia information
            note, image_url = get_wikipedia_info(predicted_actor_readable)
            st.image(image_url, width=300)

        st.write(note)

if uploaded_video is not None:
    recognize_video = st.button('Recognize Faces in Video')
    if recognize_video:
        # Save the uploaded video
        video_path = os.path.join('uploads', uploaded_video.name)
        with open(video_path, 'wb') as f:
            f.write(uploaded_video.getbuffer())

        # Extract features from the video with downsampling and skipping frames
        video_features = extract_features_from_video(video_path, model, detector)

        # Recommend based on the last frame's features
        index_pos = recommend(feature_list, video_features[-1])

        # Display result
        predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))
        predicted_actor_readable = predicted_actor.replace('_', ' ').title()
        st.header("Seems like " + predicted_actor_readable)
        # Use predicted_actor as the topic for Wikipedia information
        note, image_url = get_wikipedia_info(predicted_actor_readable)
        st.image(image_url, width=300)
        st.write(note)