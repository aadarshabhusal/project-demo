from django.urls import path
from . import views

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("upload/", views.upload_video, name="upload_video"),
    path("stream/<str:session_id>/", views.stream, name="stream"),
    path("violations/", views.violations, name="violations"),
]