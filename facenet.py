import numpy as np
from pycoral.adapters import classify
from pycoral.utils.edgetpu import run_inference
def takeSecond(elem):
  return elem[0]

def Tpu_FaceRecognize(engine, face_img):

  faces = []
  for face in face_img:
    img = np.asarray(face).flatten()

    run_inference(engine, img.tobytes())

    result = classify.get_classes(engine, top_k=200, score_threshold=- 0.5)
    # result = engine.ClassifyWithInputTensor(img, top_k=3, threshold=-0.5)
    result.sort(key=takeSecond)

    np_result = []
    for i in range(0, len(result)):
      np_result.append(result[i][1])

    faces.append(np_result)
  np_face = np.array(faces)

  return np_face