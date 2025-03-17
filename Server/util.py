import joblib
import json
import numpy as np
import base64
import cv2
import pywt
import pandas as pd
from datetime import datetime
import shutil
import os

__class_name_to_number = {}
__class_number_to_name = {}

__model = None

def classify_image(image_base64_data):

    imgs = get_cropped_image_if_2_eyes(image_base64_data)

    # result = []
    result = []
    path_to_cropped = "./cropped"
    if os.path.exists(path_to_cropped):  # checks if path to code exists
        shutil.rmtree(path_to_cropped)  # recursively deletes the entire directory and its contents
    os.mkdir(path_to_cropped)  # new empty directory

    count = 1
    for img in imgs:
        cropped_file_name = "cropped" + str(count) + ".png"
        cropped_file_path = path_to_cropped + "/" + cropped_file_name
        cv2.imwrite(cropped_file_path, img)
        count += 1

        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))

        len_image_array = 32*32*3 + 32*32

        final = combined_img.reshape(1,len_image_array).astype(float)
        result.append({
            'class': class_number_to_name(__model.predict(final)[0]),
            'class_probability': np.around(__model.predict_proba(final)*100,2).tolist()[0],
            'class_dictionary': __class_name_to_number
        })

        name = class_number_to_name(__model.predict(final)[0]);
        if len(name) == 0:
            print("Error")
        else:
            result.append(name)
            file_path = "attendance.xlsx"
            sheet_name = "Sheet1"

            df = pd.read_excel(file_path, sheet_name=sheet_name)

            name_column = "Name"
            today_date = datetime.today().strftime('%Y-%m-%d')  # Get today's date as column name
            if today_date not in df.columns:
                df[today_date] = "Absent"

            df.loc[df[name_column] == name, today_date] = "Present"
            df.to_excel(file_path, index=False)

    return result

    # return {"status": "success", "message": "Attendance marked"}

def w2d(img, mode='haar', level=1):
    imArray = img
    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    #convert to float
    imArray =  np.float32(imArray)
    imArray /= 255;
    # compute coefficients
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)
    coeffs_H[0] *= 0;

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)

    return imArray_H

def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name

    with open("./artifacts/class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}

    global __model
    if __model is None:
        with open('./artifacts/saved_model.pkl', 'rb') as f:
            __model = joblib.load(f)
    print("loading saved artifacts...done")


def get_cv2_image_from_base64_string(b64str):
    '''
    credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
    :param uri:
    :return:
    '''
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_cropped_image_if_2_eyes(image_base64_data):
    face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')

    img = get_cv2_image_from_base64_string(image_base64_data)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) >= 2:
                cropped_faces.append(roi_color)

    return cropped_faces

# def get_b64():
#     with open("b64.txt") as f:
#         return f.read()

# if __name__ == '__main__':
#     load_saved_artifacts()
#
#     arr = classify_image(get_b64())
#     # arr = classify_image(None, r"Stranger_Things_Cast.jpg")
#     for elem in arr:
#         print(elem)



