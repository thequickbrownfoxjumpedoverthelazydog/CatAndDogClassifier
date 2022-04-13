import cv2
from keras.models import load_model

img = cv2.imread('WIN_20210823_22_58_58_Pro.jpg')

def prepare(filepath): 
    SIZE = 32 
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (SIZE, SIZE))
    return new_array.reshape(-1, SIZE, SIZE, 1)

classifier = load_model('CatAndDogClassifier.h5')

prediction = classifier.predict([prepare('CATTEST.jpg')])

print(prediction)