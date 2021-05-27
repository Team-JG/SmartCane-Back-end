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
import json
from django.http import JsonResponse
from collections import ChainMap

#result = {}
result_list = []

def predict(image):
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
    #img = cv2.imread('/Users/kim-yulhee/SmartCane-Back-end/SmartCane/smartcane_app/surface_img/test.png')
    img = np.array(image)
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

    section1 = np.array(pre).T[0][0:80]
    section2 = np.array(pre).T[0][80:160]
    section3 = np.array(pre).T[0][160:320]
    section4 = np.array(pre).T[0][320:400]
    section5 = np.array(pre).T[0][400:480]

    
    result = {
        "left-end" : {
            "caution": 0,
            "cross-walk": 0,
            "road-way": 0
            },
        "left" : {
            "caution": 0, 
            "cross-walk": 0, 
            "road-way": 0
            },
        "front" : {
            "caution": 0, 
            "cross-walk": 0, 
            "road-way": 0
            },
        "right" : {
            "caution": 0, 
            "cross-walk": 0, 
            "road-way": 0
            },
        "right-end" : {
            "caution": 0, 
            "cross-walk": 0, 
            "road-way": 0
            }
    }

    async def caution_zone(result, dic):
        cnt_2 = 0
        cnt_3 = 0
        cnt_5 = 0
        k_list = []
        for j in result:
            for x in j:
                if x == 2:
                    cnt_2 += 1
                elif x == 3:
                    cnt_3 += 1
                elif x == 5:
                    cnt_5 += 1

        for key in dic:
            k_list.append(key)
        dic[k_list[0]] = cnt_2
        dic[k_list[1]] = cnt_3
        dic[k_list[2]] = cnt_5


    async def find_section(dic, s):
        context ={}
        for key in dic:
            if dic[key] > 1000:
                context["label"] = key
                context["direction"] = s
                result_list.append(context)


    async def caution_async_process():
        start = time.time()
        await asyncio.wait([
            caution_zone(section1, result["left-end"]),
            caution_zone(section2, result["left"]),
            caution_zone(section3, result["front"]),
            caution_zone(section4, result["right"]),
            caution_zone(section5, result["right-end"]),
        ])
        end = time.time()
        print(f'>>> caution_zone 비동기 처리 총 소요 시간: {end - start}')

        start = time.time()
        await asyncio.wait([
            find_section(result["left-end"], "left_end"),
            find_section(result["left"], "left"),
            find_section(result["front"], "front"),
            find_section(result["right"], "right"),
            find_section(result["right-end"], "right_end"),
        ])
        end = time.time()
        print(f'>>> direction_guidance 비동기 처리 총 소요 시간: {end - start}')


    asyncio.run(caution_async_process())
    print("=============")
    print(result_list)

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
        result_list.clear()
        predict(image)
        newdict = {}
        newdict["result"]=result_list
        return JsonResponse(newdict)
        #image.save('/Users/kim-yulhee/SmartCane-Back-end/SmartCane/smartcane_app/surface_img/test.png', format='PNG')
        #return Response("OK")
    else:
        #predict()
        #newdict = {}

        #newdict["result"]=test_lis
        #data = newdict
        return JsonResponse(newdict)
