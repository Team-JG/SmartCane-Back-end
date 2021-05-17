from django.urls import path
from django.conf.urls import url 
from . import views


urlpatterns = [
#    path('direct/', views.MyView.as_view(), name='post-dl')
    path('direct/', views.direction, name='post-dl')
]