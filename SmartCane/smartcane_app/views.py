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
from tensorflow import keras 
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

file_name = os.path.dirname(__file__) + '/pspunet_weight.h5'
model = keras.models.load_model(file_name)


#result = {}
result_list = []

def predict(image):
    img = np.array(image)
    img = cv2.resize(img, (IMG_WIDTH,IMG_HEIGHT))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = tf.expand_dims(img, 0)
    img = img / 255

    pre = model.predict(img)
    pre = create_mask(pre).numpy()

    section1 = np.array(pre).T[0][0:160]
    section2 = np.array(pre).T[0][160:320]
    section3 = np.array(pre).T[0][320:480]

    

    left = {
        "caution": 0, 
        "cross-walk": 0, 
        "road-way": 0
    }
    front = {
        "caution": 0, 
        "cross-walk": 0, 
        "road-way": 0
    }
    right = {
        "caution": 0, 
        "cross-walk": 0, 
        "road-way": 0
    }

    async def caution_zone(section, dic):
        dic["caution"] = (section == 2).sum()
        dic["cross-walk"] = (section == 3).sum()
        dic["road-way"] = (section == 5).sum()


    async def find_section(dic, s):
        context ={}
        for key in dic:
            if key == "road-way":
                if dic[key] > 4000:
                    context["label"] = key
                    context["direction"] = s
                    if not context in result_list:
                        result_list.append(context)
            elif key == "caution":
                if dic[key] > 1000:
                    context["label"] = key
                    context["direction"] = s
                    if not context in result_list:
                        result_list.append(context)
            elif key == "cross-walk":
                if dic[key] > 1000:
                    context["label"] = key
                    context["direction"] = s
                    if not context in result_list:
                        result_list.append(context)
       
            

    async def caution_async_process():
        start = time.time()
        await asyncio.wait([
            caution_zone(section1, left),
            caution_zone(section2, front),
            caution_zone(section3, right),
        ])
        end = time.time()
        print(f'>>> caution_zone 비동기 처리 총 소요 시간: {end - start}')

        start = time.time()
        await asyncio.wait([
            find_section(left, "left"),
            find_section(front, "front"),
            find_section(right, "right"),
        )]
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
        result_list.clear()
        predict(image)
        newdict = {}
        newdict["result"]=result_list
        return JsonResponse(newdict)
    else:
        return JsonResponse("GET Method")
