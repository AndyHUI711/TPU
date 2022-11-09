import argparse
import cv2
import os
import numpy as np
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

class FaceNetRECOG:
    def __init__(self):
        self.name_list, self.known_embedding = self.loadFaceFeats()
        self.name_list.append('UNKNOWN')
        self.name_png_list = self.getNamePngs(self.name_list)

    ### need edit
    def readPngFile(self, fileName):
        '''
        读取PNG图片
        '''
        # 解决中文路径问题
        png_img = cv2.imdecode(np.fromfile(fileName, dtype=np.uint8), -1)
        # 转为BGR，变成3通道
        png_img = cv2.cvtColor(png_img, cv2.COLOR_RGB2BGR)
        png_img = cv2.resize(png_img, (0, 0), fx=0.4, fy=0.4)
        return png_img

    def getNamePngs(self, name_list):
        '''
        生成每个人的名称PNG图片（以解决中文显示问题）
        '''

        real_name_list = []
        for name in name_list:
            real_name = name.split('_')[0]
            if real_name not in real_name_list:
                real_name_list.append(real_name)

        pngs_list = {}
        for name in tqdm.tqdm(real_name_list, desc='生成人脸标签PNG...'):

            filename = './images/name_png/' + name + '.png'
            # 如果存在，直接读取
            if os.path.exists(filename):
                png_img = self.readPngFile(filename)
                pngs_list[name] = png_img
                continue

            # 如果不存在，先生成
            # 背景
            bg = Image.new("RGBA", (400, 100), (0, 0, 0, 0))
            # 添加文字
            d = ImageDraw.Draw(bg)
            font = ImageFont.truetype('./fonts/MSYH.ttc', 80, encoding="utf-8")

            if name == '未知':
                color = (0, 0, 255, 255)
            else:
                color = (0, 255, 0, 255)

            d.text((0, 0), name, font=font, fill=color)
            # 保存
            bg.save(filename)
            # 再次检查
            if os.path.exists(filename):
                png_img = self.readPngFile(filename)
                pngs_list[name] = png_img

        return pngs_list

    def loadFaceFeats(self):
        '''
        加载目标人的特征
        '''
        # 记录名字
        name_list = []
        # 输入网络的所有人脸图片
        known_faces_input = []
        # 遍历
        known_face_list = glob.glob('./images/origin/*')
        for face in tqdm.tqdm(known_face_list, desc='处理目标人脸...'):
            name = face.split('\\')[-1].split('.')[0]
            name_list.append(name)
            # 裁剪人脸
            croped_face = self.getCropedFaceFromFile(face)
            if croped_face is None:
                print('图片：{} 未检测到人脸，跳过'.format(face))
                continue
            # 预处理
            img_input = self.imgPreprocess(croped_face)
            known_faces_input.append(img_input)
        # 转为Nummpy
        faces_input = np.array(known_faces_input)
        # 转tensor并放到GPU
        tensor_input = torch.from_numpy(faces_input).to(self.device)
        # 得到所有的embedding,转numpy
        known_embedding = self.facenet(tensor_input).detach().cpu().numpy()

        return name_list, known_embedding

    ### need edit


    def main(self):
        default_model_dir = 'all_models'
        default_model = 'ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite'
        default_labels = 'label.txt'
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', help='.tflite model path',
                            default=os.path.join(default_model_dir,default_model))
        parser.add_argument('--labels', help='label file path',
                            default=os.path.join(default_model_dir, default_labels))
        parser.add_argument('--top_k', type=int, default=3,
                            help='number of categories with highest score to display')
        parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 0)
        parser.add_argument('--threshold', type=float, default=0.1,
                            help='classifier score threshold')

        parser.add_argument("--threshold_face", type=float, default=1,
                            help="for facenet, higher threshold lower accuracy")

        args = parser.parse_args()

        print('Loading {} with {} labels.'.format(args.model, args.labels))
        interpreter = make_interpreter(args.model)
        interpreter.allocate_tensors()
        labels = read_label_file(args.labels)
        inference_size = input_size(interpreter)

        cap = cv2.VideoCapture(args.camera_idx)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
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
            for obj in objs:
                bbox = obj.bbox.scale(scale_x, scale_y)
                l, t, r, b = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
                # crop the face part of the frame
                crop_face = frame[t:b, l:r]


                ## detect crop_face to get embedding
                embedding = facenet

                dist_list = np.linalg.norm((embedding - self.known_embedding), axis=1)
                min_index = np.argmin(dist_list)

                pred_name = self.name_list[min_index]

                # mini distance
                min_dist = dist_list[min_index]
                if min_dist < args.threshold_face:
                    # detected
                    # name png
                    real_name = pred_name.split('_')[0]
                    name_overlay = self.name_png_list[real_name]
                else:
                    # un detected
                    name_overlay = self.name_png_list['UNKNOWN']

            # run_inference(interpreter, cv2_im_rgb.tobytes())
            # objs = get_objects(interpreter, args.threshold)[:args.top_k]
            # cv2_im = append_objs_to_img(cv2_im, inference_size, objs, labels)

            cv2_im = self.append_objs_to_img(cv2_im, inference_size, objs, labels, name_overlay)


            cv2.imshow('frame', cv2_im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def append_objs_to_img(self,cv2_im, inference_size, objs, labels, name_overlay):
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
