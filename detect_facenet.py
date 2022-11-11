import argparse
import cv2
import os

import h5py as h5py
import numpy as np
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference
from embedding import Create_embeddings
from facenet import Tpu_FaceRecognize

from pycoral.adapters import classify

class FaceNetRECOG:
    #def __init__(self):
        # self.name_list, self.known_embedding = self.loadFaceFeats()
        # self.name_list.append('UNKNOWN')
        # self.name_png_list = self.getNamePngs(self.name_list)

    ### need edit

    ### need edit
    def read_embedding(self, path, FaceNet_weight):
        # get known embedding
        try:
            f = h5py.File(path, 'r')
        except OSError:
            # Creates a new tf.lite.Interpreter instance using the given model.
            face_engine = make_interpreter(FaceNet_weight)
            face_engine.allocate_tensors()
            Create_embeddings(path,face_engine)
            f = h5py.File(path, 'r')

        class_arr = f['class_name'][:]
        class_arr = [k.decode() for k in class_arr]
        emb_arr = f['embeddings'][:]

        return class_arr, emb_arr   # name_list, known_embedding

    def crop_image(self, ans, frame, face_size):
        Images_cropped = []
        # objs is the bbox
        height, width, channels = frame.shape
        scale_x, scale_y = width / face_size[0], height / face_size[1]

        for obj in ans:
            img_crop = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #BBC = ans[i].bbox  # bounding_box_coordinate



            bbox = obj.bbox.scale(scale_x, scale_y)
            l, t, r, b = int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
            img_crop = img_crop[t:b, l:r]


            img_crop = cv2.resize(img_crop, (160, 160))

            Images_cropped.append(img_crop)

        return Images_cropped

    def main(self):
        default_model_dir = 'all_models'
        default_model = 'ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite'
        default_labels = 'label.txt'

        default_face_model = 'ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite'   # facenet weight

        default_embedding_dir = 'embedding_book'
        default_embedding_file = 'embeddings.h5'

        parser = argparse.ArgumentParser()
        parser.add_argument('--model', help='.tflite model path',
                            default=os.path.join(default_model_dir,default_model))
        parser.add_argument('--face_model', help='.tflite model path',
                            default=os.path.join(default_model_dir, default_face_model))
        parser.add_argument('--labels', help='label file path',
                            default=os.path.join(default_model_dir, default_labels))
        parser.add_argument('--top_k', type=int, default=3,
                            help='number of categories with highest score to display')
        parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 0)
        parser.add_argument('--threshold', type=float, default=0.1,
                            help='classifier score threshold')

        parser.add_argument("--threshold_face", type=float, default=1,
                            help="for facenet, higher threshold lower accuracy")

        parser.add_argument("--Embedding_book", default=os.path.join(default_embedding_dir,default_embedding_file),
                            help='saved embedding file path',)

        args = parser.parse_args()

        print('Loading {} with {} labels.'.format(args.model, args.labels))

        # read embedding
        class_arr, emb_arr = self.read_embedding(args.Embedding_book, args.face_model)

        interpreter = make_interpreter(args.model)
        interpreter.allocate_tensors()

        face_engine = make_interpreter(args.face_model)
        face_engine.allocate_tensors()

        labels = read_label_file(args.labels)
        inference_size = input_size(interpreter)
        face_size = input_size(face_engine)


        cap = cv2.VideoCapture(args.camera_idx)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            t1 = cv2.getTickCount()
            cv2_im = frame
            # frame to RGB and resize
            cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
            cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)

            # detect person face
            run_inference(interpreter, cv2_im_rgb.tobytes())
            objs = get_objects(interpreter, args.threshold)[:args.top_k]

            # objs is the bbox
            height, width, channels = cv2_im.shape
            scale_x, scale_y = width / inference_size[0], height / inference_size[1]
            if objs:
                print(objs) #[Object(id=0, score=0.16796875, bbox=BBox(xmin=126, ymin=134, xmax=221, ymax=247))]
                # bbox = obj.bbox.scale(scale_x, scale_y)
                # l, t, r, b = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
                # crop_face = frame[t:b, l:r]
                # crop the face part of the frame
                crop_face = self.crop_image(objs, frame, face_size)



                if cv2.waitKey(1) == ord('a'):
                    for k in range(0, len(crop_face)):
                        new_class_name = input('Please input name of worker:')
                        new_save = cv2.cvtColor(crop_face[k], cv2.COLOR_BGR2RGB)
                        cv2.imwrite('Workers/' + str(new_class_name) + '.jpg', new_save)
                    Create_embeddings(args.Embedding_book, face_engine)
                    class_arr, emb_arr = self.read_embedding(args.Embedding_book)

                # frame to RGB and resize
                face_im_rgb = cv2.cvtColor(crop_face, cv2.COLOR_BGR2RGB)
                face_im_rgb = cv2.resize(face_im_rgb, face_size)

                # h,w,c 2 c,h,w
                face_im_rgb = face_im_rgb.transpose((2, 0, 1))
                # [0,255] to [-1,1]
                face_im_rgb = (face_im_rgb - 127.5) / 127.5
                # dimensions expand
                face_input = np.expand_dims(face_im_rgb, 0)
                embs = Tpu_FaceRecognize(face_engine, face_input)

                face_num = len(objs)
                face_class = ['UNKNOWN'] * face_num

                for i in range(face_num):
                    diff = np.mean(np.square(embs[i] - emb_arr), axis=1)
                    min_diff = min(diff)

                    if min_diff < args.threshold_face:
                        index = np.argmin(diff)
                        face_class[i] = class_arr[index]
                print('Face_class:', face_class)
                print('Classes:', class_arr)

                for count, obj in enumerate(objs):
                    print('-----------------------------------------')
                    if labels:
                        print(labels[obj.label_id])
                    print('Score = ', obj.score)
                    box = obj.bounding_box.flatten().tolist()



                    # Draw a rectangle and label
                    cv2.rectangle(cv2_im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 2)
                    cv2.putText(cv2_im, '{}'.format(face_class[count]), (int(box[0]), int(box[1]) - 5),
                                cv2.FONT_HERSHEY_PLAIN,
                                1, (255, 0, 0), 1, cv2.LINE_AA)


            #cv2_im = self.append_objs_to_img(cv2_im, inference_size, objs, labels, name_overlay)

            # FPS display
            t2 = cv2.getTickCount()
            t = (t2 - t1) / cv2.getTickFrequency()
            fps = 1.0 / t
            cv2.putText(cv2_im, 'fps: {:.2f}'.format(fps), (5, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, cv2.LINE_AA)

            cv2.imshow('frame', cv2_im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def append_objs_to_img(self, cv2_im, inference_size, objs, labels, name_overlay):
        height, width, channels = cv2_im.shape
        scale_x, scale_y = width / inference_size[0], height / inference_size[1]
        for obj in objs:
            bbox = obj.bbox.scale(scale_x, scale_y)
            x0, y0 = int(bbox.xmin), int(bbox.ymin)
            x1, y1 = int(bbox.xmax), int(bbox.ymax)

            percent = int(100 * obj.score)
            label = '{}% {} {}'.format(percent, labels.get(obj.id, obj.id), name_overlay)

            cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        return cv2_im

if __name__ == '__main__':
    detFace = FaceNetRECOG()
    detFace.main()
