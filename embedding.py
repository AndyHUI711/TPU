# coding: utf-8

# In[1]:
from PIL import Image
from facenet import Tpu_FaceRecognize
from pycoral.adapters.common import input_size
import numpy as np
import os
#from utils import *
import cv2
import h5py


def Create_embeddings(Embedding_book_path, face_engine):
    face_size = input_size(face_engine)
    print(face_size)

    img_arr, class_arr = align_face('Workers/', face_size)

    print("Test")
    embs = Tpu_FaceRecognize(face_engine, img_arr)

    f = h5py.File(Embedding_book_path, 'w')
    class_arr = [i.encode() for i in class_arr]
    f.create_dataset('class_name', data=class_arr)
    f.create_dataset('embeddings', data=embs)
    f.close()


def align_face(path='Workers/', face_size=(160, 160)):
    img_paths = os.listdir(path)
    class_names = [a.split('.')[0] for a in img_paths]
    img_paths = [os.path.join(path, p) for p in img_paths]
    scaled_arr = []
    class_names_arr = []

    for image_path, class_name in zip(img_paths, class_names):
        img = cv2.imread(image_path)
        scaled = cv2.resize(img, face_size, interpolation=cv2.INTER_LINEAR)

        scaled = Image.fromarray(cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB))
        scaled = np.asarray(img)

        scaled_arr.append(scaled)
        class_names_arr.append(class_name)

    scaled_arr = np.asarray(scaled_arr)
    class_names_arr = np.asarray(class_names_arr)
    print("scaled_arr", scaled_arr.shape)
    print('class_names_arr', class_names_arr)
    return scaled_arr, class_names_arr


# debug testing
if __name__ == '__main__':
    align_face((160, 160))