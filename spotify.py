import os
import tkinter as tk
import threading
import cv2
import numpy as np
from PIL import Image, ImageTk
from textblob import TextBlob
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import speech_recognition as sr
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import customtkinter

SPOTIPY_CLIENT_ID = 'c488ceb00d564c15b7926b3ff8a6fe69'
SPOTIPY_CLIENT_SECRET = '9708652d5d3742738a20a3c25d559681'
SPOTIPY_REDIRECT_URI = 'http://localhost:5000/callback'

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET,
    redirect_uri=SPOTIPY_REDIRECT_URI,
    scope="user-read-playback-state,user-modify-playback-state"
))

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("blue")
root = customtkinter.CTk()
root.title("Personality Detector")
root.geometry('1000x1000')

face_cascade = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
classifier = load_model(r'model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

frame_1 = customtkinter.CTkFrame(master=root)
frame_1.pack(pady=20, padx=60, fill="both", expand=True)

video_label = customtkinter.CTkLabel(master=frame_1)
video_label.pack(anchor='nw')

emotion_label = customtkinter.CTkLabel(master=frame_1, text="Emotions: ", font=("Helvetica", 20))
emotion_label.place(x=700, y=0)

mic_switch_value = tk.BooleanVar()
mic_switch = customtkinter.CTkSwitch(master=frame_1, text="Microphone", variable=mic_switch_value)
mic_switch.place(x=10, y=400)

voice_data = customtkinter.CTkTextbox(master=frame_1, width=500, height=100)
voice_data.place(x=10, y=450)

sentiment_label = customtkinter.CTkLabel(master=frame_1, text="Sentiment: ", font=("Helvetica", 20))
sentiment_label.place(x=10, y=600)

mic_active = False

def toggle_microphone():
    global mic_active
    mic_active = mic_switch_value.get()
    if mic_active:
        threading.Thread(target=listen_to_speech).start()
    else:
        voice_data.delete(1.0, tk.END)

def listen_to_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        voice_data.delete(1.0, tk.END)
        voice_data.insert(tk.END, "Listening...")
        try:
            audio = recognizer.listen(source, timeout=5)
            voice_data.delete(1.0, tk.END)
            voice_text = recognizer.recognize_google(audio)
            voice_data.insert(tk.END, voice_text)
            analyze_sentiment(voice_text)
        except sr.WaitTimeoutError:
            voice_data.delete(1.0, tk.END)
            voice_data.insert(tk.END, "Listening timed out.")
        except sr.UnknownValueError:
            voice_data.delete(1.0, tk.END)
            voice_data.insert(tk.END, "Could not understand audio.")
        except sr.RequestError:
            voice_data.delete(1.0, tk.END)
            voice_data.insert(tk.END, "Network error occurred.")

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        sentiment_label.configure(text="Sentiment: Positive")
    elif sentiment < 0:
        sentiment_label.configure(text="Sentiment: Negative")
    else:
        sentiment_label.configure(text="Sentiment: Neutral")

def recommend_music_based_on_emotion(emotion):
    emotion_genre_map = {
        'Angry': 'rock',
        'Disgust': 'grunge',
        'Fear': 'ambient',
        'Happy': 'pop',
        'Neutral': 'indie',
        'Sad': 'acoustic',
        'Surprise': 'electronic'
    }
    genre = emotion_genre_map.get(emotion, 'pop')
    try:
        results = sp.recommendations(seed_genres=[genre], limit=10)
        tracks = [f"{track['name']} by {track['artists'][0]['name']}" for track in results['tracks']]
        return tracks
    except Exception as e:
        print(f"Error fetching recommendations: {e}")
        return ["Error fetching recommendations"]

def update_frame():
    ret, frame = cap.read()
    if not ret:
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = img_to_array(roi_gray.astype('float') / 255.0)
        roi = np.expand_dims(roi, axis=0)

        prediction = classifier.predict(roi)[0]
        label = emotion_labels[prediction.argmax()]
        emotion_label.configure(text="Emotions: " + label)

        recommended_tracks = recommend_music_based_on_emotion(label)
        voice_data.delete(1.0, tk.END)
        voice_data.insert(tk.END, "Recommended Music:\n" + "\n".join(recommended_tracks))

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=im)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk, text="")

    root.after(15, update_frame)

mic_switch.configure(command=toggle_microphone)

cap = cv2.VideoCapture(0)
update_frame()

root.mainloop()

cap.release()
cv2.destroyAllWindows()
