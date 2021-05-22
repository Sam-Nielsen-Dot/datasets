import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from numpy import asarray
import cv2
import time
from win10toast import ToastNotifier

asian = 2
white = 0
black = 4
brown = 1

races = {
    2:"asian",
    0:"white",
    1:"brown",
    4:"black",
    3:"south-east asian"
}

comments = {
    "white":[
        "pasty ass white boi",
        "go back to the wiggles concert",
        "u must be gay",
        "cracker boy",
        "go chocke on some spice"
    ],
    "asian":[
        "zipper head",
        "slit eyes"
    ],
    "brown":[
        "go eat some curry u currymuncher",
        "okey dokey"
    ],
    "black":[
        "THE N WORD",
        "eat sum kfc"
    ],
    "south-east asian":[
        "ur weird",
        "stp being weird",
    ]
}


def load_model(name):
    model = tf.keras.models.load_model(name)

    #model.summary()

    return model


def predict(model, image):
    # load the image and convert into
    # numpy array
    img = Image.open(image)
    
    # asarray() class is used to convert
    # PIL images into NumPy arrays
    numpydata = asarray(img)

    numpydata = np.resize(numpydata, (1, 48, 48, 1))

    prediction = model.predict(numpydata)

    print(prediction)

    return prediction

def main():
    model = load_model('model_ethnicity.h5')

    camera_port = 0
    camera = cv2.VideoCapture(camera_port)
    time.sleep(0.1)  # If you don't wait, the image will be dark
    return_value, image = camera.read()
    cv2.imwrite("test.png", image)
    del(camera)

    model = load_model('model_ethnicity.h5')
    prediction = predict(model, "test.png")[0]
    race = get_race(prediction)

    model = load_model('model_gender.h5')
    prediction = predict(model, "test.png")[0]
    gender = get_gender(prediction)
    

    toaster = ToastNotifier()
    #toaster.show_toast(race, comments[race][0]) 
    toaster.show_toast(race, gender) 

def get_race(prediction):
    max = 0
    for k in range(0, 5):
        if prediction[k] >= max:
            num = k
            max = prediction[k]
    print(max)
    print(num)
    return races[num]

def get_gender(prediction):
    if prediction[0] > prediction[1]:
        return "female"
    else:
        return "male"

main()

#print(get_gender(predict(load_model("model_ethnicity.h5"), "blackman.jpg")[0]))