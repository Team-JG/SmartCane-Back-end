from django.urls import path
from django.conf.urls import url 
from . import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('segmentation/', views.direction, name='post-dl')
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)