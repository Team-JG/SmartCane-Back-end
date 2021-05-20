from rest_framework import serializers
#from django.core.serializers import serialize


class DirectionSerializer(serializers.Serializer):
    result = serializers.CharField()

    class Meta:
        fields = ('result')


class Image(object):
    def __init__(self,):
        self.image = image

class ImageSerializer(serializers.Serializer):
    image = serializers.FileField()

    class Meta:
        fields = ('image')

    # def create(self, validated_data):
    #     return Image(**validated_data)
    
