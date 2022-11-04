import os, argparse, cv2, sys, time, numpy
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate
from PIL import Image
from PIL import ImageDraw

# from model.SSD import FaceMaskDetection
# from model.FACENET import InceptionResnetV1

import re

'''
Requirements: 
1) Install the tflite_runtime package from here:
https://www.tensorflow.org/lite/guide/python
2) Camera to take inputs
3) Install libedgetpu
https://github.com/google-coral/edgetpu/tree/master/libedgetpu/direct
Download models:
$ wget https://github.com/google-coral/edgetpu/raw/master/test_data/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite
Run:
$ python3 edgetpu_face_detector.py --model mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite --edgetpu True 
'''

def get_cmd():
    default_model_dir = ''
    default_model = 'ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite'
    default_labels = 'labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path to tflite model.', required=False,
                        default=os.path.join(default_model_dir, default_model))
    parser.add_argument(
        '--threshold', help='Minimum confidence threshold.', default=1)
    parser.add_argument('--source', help='Video source.', default=0)
    parser.add_argument('--edgetpu', help='With EdgeTpu', default=False)
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))

    return parser.parse_args()

def load_labels(path):
    p = re.compile(r'\s*(\d+)(.+)')
    with open(path, 'r', encoding='utf-8') as f:
        lines = (p.match(line).groups() for line in f.readlines())
        return {int(num): text.strip() for num, text in lines}

def main():
    args = get_cmd()
    # Initialize the TF interpreter
    if args.edgetpu:
        interpreter = Interpreter(args.model, experimental_delegates=[
                                  load_delegate('libedgetpu.so.1.0')])
    else:
        interpreter = Interpreter(args.model)

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    width = input_details[0]['shape'][2]
    height = input_details[0]['shape'][1]
    cap = cv2.VideoCapture(args.source)
    image_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    image_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    while(True):
        ret, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        frame_resized = frame_resized.astype(numpy.float32)

        frame_resized /= 255

        input_data = numpy.expand_dims(frame_resized, axis=0)

        # #run inference
        # ssd_detector = FaceMaskDetection(face_mask_model_path, margin=0, GPU_ratio=0.1)
        # bboxes, re_confidence, re_classes, re_mask_id = ssd_detector.inference(input_data, image_height, image_width)

        # Run an inference
        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()


        # Results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        #classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]

        #print(boxes,classes,scores)

        for i in range(len(scores)):
            if ((scores[i] > args.threshold) and (scores[i] <= 1.0)):
                ymin = int(max(1, (boxes[i][0] * image_height)))
                xmin = int(max(1, (boxes[i][1] * image_width)))
                ymax = int(min(image_height, (boxes[i][2] * image_height)))
                xmax = int(min(image_width, (boxes[i][3] * image_width)))
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 4)
                object_name = 'face'
                label = '%s: %d%%' % (object_name, int(scores[i]*100))
                labelSize, baseLine = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.putText(frame, label, (xmin, label_ymin-7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.imshow('Object detector', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()