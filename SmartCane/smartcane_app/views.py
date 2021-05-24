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
from keras.models import load_model
import cv2
import tensorflow as tf
import os
import io
import PIL.Image as Image
import numpy as np
from PIL import Image
import pandas as pd

result = "default"


def DL():
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
    model = load_model(file_name)
    
    # input details
    img = cv2.imread('/Users/kim-yulhee/SmartCane-Back-end/SmartCane/smartcane_app/surface_img/test.png')
    img = cv2.resize(img, (IMG_WIDTH,IMG_HEIGHT))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img / 255

    img = tf.expand_dims(img, 0)

    pre = model.predict(img)
    pre = create_mask(pre).numpy()

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
        global result
        key_min = min(caution.keys(), key=lambda k: caution[k])
        if key_min == 'clist1':
            result="맨 왼쪽"
            print("맨 왼쪽")
        elif key_min == 'clist2':
            result="살짝 왼쪽"
            print("살짝 왼쪽")
        elif key_min == 'clist3':
            result="직진"
            print("직진")
        elif key_min == 'clist4':
            result="직진"
            print("직진")
        elif key_min == 'clist5':
            result="살짝 오른쪽"
            print("살짝 오른쪽")
        elif key_min == 'clist6':
            result="맨 오른쪽"
            print("맨 오른쪽")


    print("The number of label 2 in result1 : ", caution['clist1'])
    print("The number of label 2 in result2 : ", caution['clist2'])
    print("The number of label 2 in result3 : ", caution['clist3'])
    print("The number of label 2 in result4 : ", caution['clist4'])
    print("The number of label 2 in result5 : ", caution['clist5'])
    print("The number of label 2 in result6 : ", caution['clist6'])

    direction_guidance()




# @api_view(['GET','POST'])
# def direction(request):
#     if request.method == "GET":
#         DL()
#         mydata = [{"result": result}]
#         direction_serializers = DirectionSerializer(mydata, many=True)
#         return Response(direction_serializers.data)


@api_view(['GET','POST'])
def direction(request):
    if request.method == "POST":
        bytes = request.FILES['file'].file.getvalue()
        image = Image.open(io.BytesIO(bytes)).convert("RGB")
        image.save('/Users/kim-yulhee/SmartCane-Back-end/SmartCane/smartcane_app/surface_img/test.png', format='PNG')
        return Response("OK")
    else:
        DL()
        mydata = [{"result": result}]
        direction_serializers = DirectionSerializer(mydata, many=True)
        return Response(direction_serializers.data)
