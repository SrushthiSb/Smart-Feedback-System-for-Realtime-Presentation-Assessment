from django.urls import path
from . import views

urlpatterns = [
    path('',views.Home_page,name='home'),
    path('upload/', views.upload_presentation, name='upload_presentation'),
    path('video_feed_page/<int:video_id>/', views.video_feed_page, name='video_feed_page'),
    path('video_feed/<int:video_id>/', views.video_feed, name='video_feed'),
    path('feedback_status/<int:video_id>/', views.feedback_status, name='feedback_status'),
    path('feedback/<int:video_id>/', views.feedback, name='feedback'),
    
]
