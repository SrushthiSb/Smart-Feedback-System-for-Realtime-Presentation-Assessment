from django.shortcuts import redirect, render, get_object_or_404
from django.http import StreamingHttpResponse, JsonResponse
from .models import Presentation, Feedback
from .forms import PresentationForm
import cv2
import mediapipe as mp
import speech_recognition as sr
import moviepy.editor as mp_editor
import audioop
import numpy as np
from keras.models import load_model  # type: ignore
import threading

# Load the pre-trained emotion detection model
emotion_model = load_model('presentations/models/model.h5')
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
def Home_page(request):
    return render(request,'presentations/home.html')
def upload_presentation(request):
    if request.method == 'POST':
        form = PresentationForm(request.POST, request.FILES)
        if form.is_valid():
            presentation = form.save()
            # Start the video processing in a separate thread
            thread = threading.Thread(target=process_video_and_save_feedback, args=(presentation.video.path, presentation))
            thread.start()
            # Redirect to video feed page with the new video's ID
            return redirect('video_feed_page', video_id=presentation.id)
    else:
        form = PresentationForm()
    return render(request, 'presentations/upload.html', {'form': form})
def video_feed_page(request, video_id):
    return render(request, 'presentations/video_feed.html', {'video_id': video_id})

def video_feed(request, video_id):
    presentation = get_object_or_404(Presentation, id=video_id)
    video_path = presentation.video.path

    def generate():
        cap = cv2.VideoCapture(video_path)
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Annotate the frame with gesture and posture data
            annotated_frame = annotate_frame(frame, pose)

            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        cap.release()

    return StreamingHttpResponse(generate(), content_type='multipart/x-mixed-replace; boundary=frame')

def annotate_frame(frame, pose):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    
    if results.pose_landmarks:
        key_points = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in results.pose_landmarks.landmark]
        
        # Draw gestures
        for lm in results.pose_landmarks.landmark:
            x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Draw posture lines
        left_shoulder = key_points[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = key_points[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = key_points[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
        right_hip = key_points[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
        
        cv2.line(frame, (int(left_shoulder['x'] * frame.shape[1]), int(left_shoulder['y'] * frame.shape[0])),
                 (int(right_shoulder['x'] * frame.shape[1]), int(right_shoulder['y'] * frame.shape[0])), (255, 0, 0), 2)
        cv2.line(frame, (int(left_hip['x'] * frame.shape[1]), int(left_hip['y'] * frame.shape[0])),
                 (int(right_hip['x'] * frame.shape[1]), int(right_hip['y'] * frame.shape[0])), (255, 0, 0), 2)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    for (x, y, w, h) in faces:
        face = gray_frame[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        img = extract_features(face)
        pred = emotion_model.predict(img)
        emotion_label = labels[pred.argmax()]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def process_video_and_save_feedback(video_path, presentation):
    video_feedback = process_video(video_path)
    audio_feedback = process_audio(video_path)

    gesture_score = score_gestures(video_feedback['gestures'])
    posture_score = score_posture(video_feedback['posture'])
    volume_score = score_volume(audio_feedback['volume'])
    pace_score = score_pace(audio_feedback['pace'])
    emotion_score = score_emotions(video_feedback['expressions'])

    overall_score = (gesture_score + posture_score + volume_score + pace_score + emotion_score) / 5

    # Create or update feedback record
    feedback_record, created = Feedback.objects.get_or_create(presentation=presentation)
    feedback_record.gesture_score = gesture_score
    feedback_record.posture_score = posture_score
    feedback_record.volume_score = volume_score
    feedback_record.pace_score = pace_score
    feedback_record.emotion_score = emotion_score
    feedback_record.overall_score = overall_score
    feedback_record.processing_complete = True
    feedback_record.save()

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    feedback = {
        "gestures": [],
        "posture": [],
        "expressions": []
    }
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        if results.pose_landmarks:
            key_points = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in results.pose_landmarks.landmark]
            feedback['gestures'].append(key_points)
            feedback['posture'].append(key_points)  # Collecting posture data as well

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
        for (x, y, w, h) in faces:
            face = gray_frame[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            img = extract_features(face)
            pred = emotion_model.predict(img)
            emotion_label = labels[pred.argmax()]
            feedback['expressions'].append(emotion_label)

    cap.release()
    return feedback

def process_audio(video_path):
    video_clip = mp_editor.VideoFileClip(video_path)
    audio_path = video_path.replace('.mp4', '.wav')
    video_clip.audio.write_audiofile(audio_path)

    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
    
    feedback = {
        "transcription": text,
        "volume": analyze_volume(audio_data),
        "pace": analyze_pace(text)
    }
    return feedback

def analyze_volume(audio_data):
    rms = audio_data.frame_data
    volume = abs(audioop.rms(rms, 2))
    if volume < 500:
        return "Too quiet"
    elif volume > 2000:
        return "Too loud"
    else:
        return "Normal"

def analyze_pace(text):
    words = text.split()
    duration = len(text) / 60  # Assuming length of text as duration in seconds
    pace = len(words) / duration if duration > 0 else 0
    return pace

def score_gestures(gestures):
    total_frames = len(gestures)
    score = 0
    
    for frame in gestures:
        if frame[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]['y'] < frame[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value]['y']:
            score += 1
        if frame[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]['y'] < frame[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value]['y']:
            score += 1
    
    max_score = total_frames * 2
    percentage = (score / max_score) * 100 if max_score > 0 else 0
    return percentage

def score_posture(posture_data):
    total_frames = len(posture_data)
    correct_posture_frames = 0

    for frame in posture_data:
        left_shoulder = frame[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = frame[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = frame[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
        right_hip = frame[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]

        if abs(left_shoulder['y'] - right_shoulder['y']) < 0.05 and abs(left_hip['y'] - right_hip['y']) < 0.05:
            correct_posture_frames += 1

    percentage = (correct_posture_frames / total_frames) * 100 if total_frames > 0 else 0
    return percentage

def score_volume(volume):
    if volume == "Normal":
        return 100
    elif volume == "Too quiet":
        return 50
    elif volume == "Too loud":
        return 50
    else:
        return 0

def score_pace(pace):
    ideal_pace = 130  # Ideal words per minute
    if ideal_pace * 0.9 < pace < ideal_pace * 1.1:
        return 100
    elif ideal_pace * 0.8 < pace < ideal_pace * 1.2:
        return 75
    elif ideal_pace * 0.7 < pace < ideal_pace * 1.3:
        return 50
    else:
        return 25

def score_emotions(emotions):
    emotion_counts = {emotion: emotions.count(emotion) for emotion in set(emotions)}
    positive_emotions = emotion_counts.get('happy', 0) + emotion_counts.get('surprise', 0)
    neutral_emotions = emotion_counts.get('neutral', 0)
    negative_emotions = emotion_counts.get('angry', 0) + emotion_counts.get('disgust', 0) + emotion_counts.get('fear', 0) + emotion_counts.get('sad', 0)

    total_emotions = positive_emotions + neutral_emotions + negative_emotions
    if total_emotions == 0:
        return 0

    emotion_score = (positive_emotions * 1 + neutral_emotions * 0.5) / total_emotions * 100
    return emotion_score

def feedback_status(request, video_id):
    feedback = Feedback.objects.filter(presentation_id=video_id).first()
    if feedback:
        if feedback.processing_complete:
            return JsonResponse({'status': 'complete'})
    return JsonResponse({'status': 'processing'})
def feedback(request, video_id):
    feedback = get_object_or_404(Feedback, presentation_id=video_id)
    return render(request, 'presentations/feedback.html', {'feedback': feedback, 'video_id': video_id})
