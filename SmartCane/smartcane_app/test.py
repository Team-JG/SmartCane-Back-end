from django.shortcuts import render
from rest_framework import views
from rest_framework.response import Response
from .serializers import DirectionSerializer, ImageSerializer
from rest_framework.decorators import api_view
from rest_framework.views import APIView
from django.http.response import JsonResponse
from rest_framework.parsers import JSONParser, FileUploadParser
from rest_framework import status, viewsets
from .display import create_mask, show_predictions
import cv2
import tensorflow as tf
import os
import io
import numpy as np
from PIL import Image
import time
import asyncio

result = "default"


def predict():
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
    tensorflow_lite_model_file = "/Users/kim-yulhee/SmartCane-Back-end/SmartCane/smartcane_app/converted_model.tflite"

    interpreter = tf.lite.Interpreter(tensorflow_lite_model_file)
    # Load TFLite model and allocate tensors.

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()


    # input details
    img = cv2.imread('/Users/kim-yulhee/SmartCane-Back-end/SmartCane/smartcane_app/surface_img/test.png')
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

    # caution = {'clist1':0, 'clist2':0, 'clist3':0, 'clist4':0, 'clist5':0, 'clist6':0}
    # crosswalk = {'wlist1':0, 'wlist2':0, 'wlist3':0, 'wlist4':0, 'wlist5':0, 'wlist6':0}
    # roadway = {'rlist1':0, 'rlist2':0, 'rlist3':0, 'rlist4':0, 'rlist5':0, 'rlist6':0}

    caution = []
    crosswalk = []
    roadway = []

    async def caution_zone(result):
        cnt_2 = 0
        cnt_3 = 0
        cnt_5 = 0
        for j in result:
            for x in j:
                if x == 2:
                    cnt_2 += 1
                elif x == 3:
                    cnt_3 += 1
                elif x == 5:
                    cnt_5 += 1
        caution.append(cnt_2)
        crosswalk.append(cnt_3)
        roadway.append(cnt_5)


    async def find_section(list):
        #global result
        key_min = list.index(min(list))
        print(f'{list} 중 가장 적게 있는 구간 {key_min}입니다.')


    async def caution_async_process():
        start = time.time()
        await asyncio.wait([
            caution_zone(result1),
            caution_zone(result2),
            caution_zone(result3),
            caution_zone(result4),
            caution_zone(result5),
            caution_zone(result6),
        ])
        end = time.time()
        print(f'>>> caution_zone 비동기 처리 총 소요 시간: {end - start}')

        start = time.time()
        await asyncio.wait([
            find_section(caution),
            find_section(crosswalk),
            find_section(roadway),
        ])
        end = time.time()
        print(f'>>> direction_guidance 비동기 처리 총 소요 시간: {end - start}')


    asyncio.run(caution_async_process())



#""bike_lane_normal", "sidewalk_asphalt", "sidewalk_urethane""
# "caution_zone_stairs", "caution_zone_manhole", "caution_zone_tree_zone", "caution_zone_grating", "caution_zone_repair_zone"]
#"alley_crosswalk","roadway_crosswalk"
#"braille_guide_blocks_normal", "braille_guide_blocks_damaged"
#"roadway_normal","alley_normal","alley_speed_bump", "alley_damaged""
#"sidewalk_blocks","sidewalk_cement" , "sidewalk_soil_stone", "sidewalk_damaged","sidewalk_other"
# 2, 3, 5

@api_view(['GET','POST'])
def direction(request):
    if request.method == "POST":
        bytes = request.FILES['file'].file.getvalue()
        image = Image.open(io.BytesIO(bytes)).convert("RGB")
        image.save('/Users/kim-yulhee/SmartCane-Back-end/SmartCane/smartcane_app/surface_img/test.png', format='PNG')
        return Response("OK")
    else:
        predict()
        mydata = [{"result": result}]
        direction_serializers = DirectionSerializer(mydata, many=True)
        return Response(direction_serializers.data)
