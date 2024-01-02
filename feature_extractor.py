# pip install mtcnn==0.1.0
# pip install tensorflow==2.3.1
# pip install keras==2.4.3
# pip install keras-vggface==0.6
# pip install keras_applications==1.0.8

# import os
# import pickle

# actors = os.listdir('dataset')

# filenames = []

# for actor in actors:
#     filenames.extend(
#         os.path.join('dataset', actor, file)
#         for file in os.listdir(os.path.join('dataset', actor))
#     )
# pickle.dump(filenames,open('filenames.pkl','wb'))


import tensorflow
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions
import numpy as np
import pickle
from tqdm import tqdm

filenames = pickle.load(open('filenames.pkl', 'rb'))

model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')


def feature_extractor(img_path, model):
    from keras.preprocessing import image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result


features = []

for file in tqdm(filenames):
    extracted_feature = feature_extractor(file, model)
    features.append(extracted_feature)

pickle.dump(features, open('embedding.pkl', 'wb'))
