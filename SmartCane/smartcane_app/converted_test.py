import cv2
import tensorflow as tf
from data_loader.display import create_mask, show_predictions
import numpy as np
from PIL import Image
import pandas as pd

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=500)])
    except RuntimeError as e:
        print(e)

IMG_WIDTH = 480
IMG_HEIGHT = 272
n_classes = 7

#Put in your local file path
tensorflow_lite_model_file = ""

interpreter = tf.lite.Interpreter(tensorflow_lite_model_file)
# Load TFLite model and allocate tensors.

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# input details
img = cv2.imread('./surface_img/MP_SEL_SUR_009569.jpg')
img = cv2.resize(img, (IMG_WIDTH,IMG_HEIGHT))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = tf.expand_dims(img, 0)

input_data = np.array(img, dtype=np.float32)
#print(input_data.shape)
'''
Get indexes of input and output layers
input_details[0]['index']를 출력하면 0, 딕셔너리 안에 index라는 key가 있고 그 Index값이 0임.
[1,IMG_HEIGHT, IMG_WIDTH,3] -> input에 들어가는 image의 shape 형태 [갯수, height, width, 채널]
'''
interpreter.resize_tensor_input(input_details[0]['index'],[1, IMG_HEIGHT, IMG_WIDTH, 3])
# allocate_tensor
interpreter.allocate_tensors()
'''
Transform input data (tensor_index, value)
tensor_index: Tensor index of tensor to set. This value can be gotten from the 'index' field in get_input_details.
value:	Value of tensor to set.
'''
interpreter.set_tensor(input_details[0]['index'], input_data)
# run the inference
interpreter.invoke()
# output_details[0]['index'] = the index which provides the input
output_data = interpreter.get_tensor(output_details[0]['index'])

pre = create_mask(output_data).numpy()

result1 = np.array(pre).T[0][0:80]
result2 = np.array(pre).T[0][80:160]
result3 = np.array(pre).T[0][160:240]
result4 = np.array(pre).T[0][240:320]
result5 = np.array(pre).T[0][320:400]
result6 = np.array(pre).T[0][400:480]

caution = {'clist1':0, 'clist2':0, 'clist3':0, 'clist4':0, 'clist5':0, 'clist6':0}

def caution_zone(result, clist):
    for j in result:
        for x in j:
            if x == 2:
                clist += 1
    return clist

caution['clist1'] = caution_zone(result1, caution['clist1'])
caution['clist2'] = caution_zone(result2, caution['clist2'])
caution['clist3'] = caution_zone(result3, caution['clist3'])
caution['clist4'] = caution_zone(result4, caution['clist4'])
caution['clist5'] = caution_zone(result5, caution['clist5'])
caution['clist6'] = caution_zone(result6, caution['clist6'])


def direction_guidance():
    key_min = min(caution.keys(), key=lambda k: caution[k])
    if key_min == 'clist1':
        print("맨 왼쪽")
    elif key_min == 'clist2':
        print("살짝 왼쪽")
    elif key_min == 'clist3':
        print("직진")
    elif key_min == 'clist4':
        print("직진")
    elif key_min == 'clist5':
        print("살짝 오른쪽")
    elif key_min == 'clist6':
        print("맨 오른쪽")


print("The number of label 2 in result1 : ", caution['clist1'])
print("The number of label 2 in result2 : ", caution['clist2'])
print("The number of label 2 in result3 : ", caution['clist3'])
print("The number of label 2 in result4 : ", caution['clist4'])
print("The number of label 2 in result5 : ", caution['clist5'])
print("The number of label 2 in result6 : ", caution['clist6'])

direction_guidance()
