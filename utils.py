from PIL import Image, ExifTags
import face_recognition
import numpy as np
import shutil
import cv2
import sys
import os


def PILtocv2(image_PIL):
    """
    Convert PIL Image Type to OpenCV Image Type
    """
    try:
        image_PIL = image_PIL.save('image.jpg')
    except:
        image_PIL = image_PIL.convert('RGB')
        image_PIL = image_PIL.save('image.jpg')
    return cv2.imread('image.jpg')


def encode_face(face_images):
    embedded_vetor = []
    for face_image in face_images:
        image = Image.open(face_image)

        # rotate when smartphone
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break

                exif = dict(image._getexif().items())

            if exif[orientation] == 3:
                image = image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                image = image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                image = image.rotate(90, expand=True)
        except (AttributeError, KeyError, IndexError):
            # cases: image don't have getexif
            pass

        # Convert to cv2 and get face's encode
        image = PILtocv2(image)
        # print(image.shape)
        location = [(0,223,223,0)]
        encode = face_recognition.face_encodings(image, location)
        # print('encode: ', len(encode))
        try: 
            for e in encode:
                embedded_vetor.append(e)
        except:
            print('Dont have any face in this picture')
            pass

    return embedded_vetor


def load_biometric_data():
    """
    biometrics: biometrics of all (n_images x 128)
    names: names of all (n_images)

    """
    # Load biometric data
    biometrics = []
    img_per_ppl = []
    names = []

    for data in os.listdir('./biometrics'):
        path = os.path.join('./biometrics',data)

        # append data and name of each image
        metric = np.load(path)
        biometrics.append(metric)
        
        num_img = metric.shape[0]
        names += [os.path.splitext(data)[0]] * num_img
        img_per_ppl.append(num_img)    
        
    return np.vstack(biometrics), np.array(names)


def image(frame, name, folder, biometrics, names, verify_confidence = 0.37):
    people_pred = []
    face_encodings = encode_face([os.path.join(folder, data) for data in sorted(os.listdir(folder), key = lambda x: int(x.lstrip('frame').rstrip('.jpg')))])
    print('{} face_encodings: '.format(name), len(face_encodings))

    # Face recognition
    face_distance_list = []
    for face_encoding in face_encodings:
        # List of boolean, shape: num_face x num biometric
        matches = face_recognition.compare_faces(biometrics, face_encoding, tolerance = verify_confidence)
        
        # Find the nearest name_faces
        prediction_one = "Unknown"
        face_distances = face_recognition.face_distance(biometrics, face_encoding)
        print('{} faces distance: '.format(name), min(face_distances))
        face_distance_list.append(round(min(face_distances), 2))
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            prediction_one = names[best_match_index]

        people_pred.append(prediction_one)
        print('{} prediction_one:'.format(name), prediction_one)

    return frame


# def biometric_predict():