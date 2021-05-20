from django.urls import path
from django.conf.urls import url 
from . import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
#    path('file/', views.FileUploadView, name='file-upload'),
    path('direct/', views.direction, name='post-dl'),
    path('image/', views.image, name='image')
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)