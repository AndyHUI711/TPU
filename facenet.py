import numpy as np
from pycoral.adapters import classify
from pycoral.utils.edgetpu import run_inference
def takeSecond(elem):
  return elem[0]

def Tpu_FaceRecognize(engine, face_img):

  faces = []
  for face in face_img:
    img = np.asarray(face).flatten()

    run_inference(engine, img)

    """
    @Parameters: interpreter – The tf.lite.Interpreter to query for output.
    @Returns: The output tensor (flattened and dequantized) as numpy.array.
    """
    result = classify.get_scores(engine)


    # result = engine.ClassifyWithInputTensor(img, top_k=3, threshold=-0.5)
    print(result)
    #result.sort(key=takeSecond)

    # np_result = []
    # for i in range(0, len(result)):
    #   np_result.append(result[i][1])

    faces.append(result)

  np_face = np.array(faces)

  return np_face