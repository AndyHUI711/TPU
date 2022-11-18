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
    #print(face_size) (160,160)


    img_arr, class_arr = align_face('Workers/', (160,160))



    embs = Tpu_FaceRecognize(face_engine, img_arr)
    #print(embs)

    f = h5py.File(Embedding_book_path, 'w')
    class_arr = [i.encode() for i in class_arr]
    #print(class_arr)
    f.create_dataset('class_name', data=class_arr)
    f.create_dataset('embeddings', data=embs)
    f.close()


    f = h5py.File(Embedding_book_path, 'r')
    #print("embedding test")
    class_arr = f['class_name'][:]
    class_arr = [k.decode() for k in class_arr]
    emb_arr = f['embeddings'][:]

    print('read_embedding: class_arr: {};'.format(class_arr))
    print('read_embedding: emb_arr: {};'.format(emb_arr))


def align_face(path, face_size):
    img_paths = os.listdir(path)
    class_names = [a.split('.')[0] for a in img_paths]
    img_paths = [os.path.join(path, p) for p in img_paths]
    scaled_arr = []
    class_names_arr = []

    for image_path, class_name in zip(img_paths, class_names):
        img = cv2.imread(image_path)
        scaled = cv2.resize(img, face_size, interpolation=cv2.INTER_LINEAR)
        print("1", scaled.shape)
        img = scaled.transpose((2, 0, 1))
        img = (img - 127.5) / 127.5
        print("2", img.shape)
        scaled = np.expand_dims(img, 0)
        print("3", scaled.shape)


        # scaled = Image.fromarray(cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB))
        # scaled = np.asarray(scaled)

        scaled_arr.append(scaled)
        class_names_arr.append(class_name)

    scaled_arr = np.asarray(scaled_arr)
    class_names_arr = np.asarray(class_names_arr)
    print("scaled_arr", scaled_arr.shape)
    print('class_names_arr', class_names_arr)
    return scaled_arr, class_names_arr


# debug testing
if __name__ == '__main__':
    align_face('Workers/', (160,160))