from rest_framework import serializers
#from django.core.serializers import serialize


class DirectionSerializer(serializers.Serializer):
    result = serializers.CharField()

    class Meta:
        fields = ('result')

class ImageSerializer(serializers.Serializer):
    image = serializers.ImageField()

    class Meta:
        fields = ('image')
