from keras_vggface.vggface import VGGFace
import numpy as np
from keras_vggface.utils import preprocess_input
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from mtcnn import MTCNN
from PIL import Image

# Load precomputed features and filenames
feature_list = np.array(pickle.load(open('embedding.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load VGGFace model
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

# Initialize MTCNN detector
detector = MTCNN()

# Load sample image and perform face detection
sample_img = cv2.imread('sample/sample.jpg')
results = detector.detect_faces(sample_img)

# Extract face from the image
x, y, width, height = results[0]['box']
face = sample_img[y:y+height, x:x+width]

# Resize the face to match the input size of the VGGFace model
image = Image.fromarray(face)
image = image.resize((224, 224))
face_array = np.asarray(image)

# Preprocess the face for model prediction
face_array = face_array.astype('float32')
expanded_img = np.expand_dims(face_array, axis=0)
preprocessed_img = preprocess_input(expanded_img)

# Extract features for the input face
result = model.predict(preprocessed_img).flatten()

# Calculate cosine similarity with all the precomputed features
similarity = [cosine_similarity(result.reshape(1, -1), feature.reshape(1, -1))[0][0] for feature in feature_list]

# Find the index of the image with the highest similarity
index_pos = np.argmax(similarity)

# Load and display the recommended image
temp_img = cv2.imread(filenames[index_pos])
cv2.imshow('output', temp_img)
cv2.waitKey(0)