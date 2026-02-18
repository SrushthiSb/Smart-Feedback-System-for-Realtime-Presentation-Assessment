
# presentations/models.py
from django.db import models

class Presentation(models.Model):
    video = models.FileField(upload_to='videos/')

class Feedback(models.Model):
    presentation = models.OneToOneField(Presentation, on_delete=models.CASCADE)
    gesture_score = models.FloatField(default=0)
    posture_score = models.FloatField(default=0)
    volume_score = models.FloatField(default=0)
    pace_score = models.FloatField(default=0)
    emotion_score = models.FloatField(default=0)
    overall_score = models.FloatField(default=0)
    processing_complete = models.BooleanField(default=False) 

    def __str__(self):
        return f"Feedback for Presentation {self.presentation.id}"
