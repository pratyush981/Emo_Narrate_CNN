import tkinter as tk
import speech_recognition as sr
import threading
from PIL import Image, ImageTk
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import customtkinter
from textblob import TextBlob
import openai
customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("blue")

openai.api_key = 'XXk-proj-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-N_94abEqfK0GXXXXXXXXXXXXXXXXXXXXXXXX1fOaNaypZFC0RKVcANsuuw_aMA'

face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
classifier = load_model(r'model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

root = customtkinter.CTk()
root.title("Personality Detector")
root.geometry('1000x1000')

frame_1 = customtkinter.CTkFrame(master=root)
frame_1.pack(pady=20, padx=60, fill="both", expand=True)

video_label = customtkinter.CTkLabel(master=frame_1)
video_label.pack(anchor='nw')

emotion_label = customtkinter.CTkLabel(master=frame_1, text="Emotions: ", font=("Helvetica", 20))
emotion_label.place(x=700, y=0)

mic_switch_value = tk.BooleanVar()
mic_switch = customtkinter.CTkSwitch(master=frame_1, text="Microphone", variable=mic_switch_value)
mic_switch.place(x=10, y=400)

# Adding the text box for displaying recognized text
voice_data = customtkinter.CTkTextbox(master=frame_1, width=500, height=100)
voice_data.place(x=10, y=450)

sentiment_label = customtkinter.CTkLabel(master=frame_1, text="Sentiment: ", font=("Helvetica", 20))
sentiment_label.place(x=10, y=600)

gpt_label = customtkinter.CTkLabel(master=frame_1, text="GPT Response: ", font=("Helvetica", 20))
gpt_label.place(x=10, y=700)

story_label = customtkinter.CTkLabel(master=frame_1, text="Generated Story: ", font=("Helvetica", 20))
story_label.place(x=10, y=800)

# Load pre-trained model and cascade classifier
classifier = load_model('model.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# Function to toggle microphone and perform speech recognition
def toggle_microphone():
    if mic_switch_value.get():
        # Microphone is turned on, start listening
        threading.Thread(target=listen_to_speech).start()
    else:
        # Microphone is turned off, clear the text box
        voice_data.delete(1.0, tk.END)


# Function to listen to speech, update the text box, perform sentiment analysis, and get GPT response
def listen_to_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        voice_data.delete(1.0, tk.END)  # Clear previous text
        voice_data.insert(tk.END, "Listening...")
        try:
            audio = recognizer.listen(source, timeout=5)  # Listen for up to 5 seconds
            voice_data.delete(1.0, tk.END)  # Clear previous text
            voice_text = recognizer.recognize_google(audio)
            voice_data.insert(tk.END, voice_text)

            analyze_sentiment(voice_text)

            # Get GPT response based on the recognized text
            get_gpt_response(voice_text)

        except sr.WaitTimeoutError:
            voice_data.delete(1.0, tk.END)
            voice_data.insert(tk.END, "Listening timed out.")
        except sr.UnknownValueError:
            voice_data.delete(1.0, tk.END)
            voice_data.insert(tk.END, "Could not understand audio.")
        except sr.RequestError:
            voice_data.delete(1.0, tk.END)
            voice_data.insert(tk.END, "Network error occurred.")


# Function to perform sentiment analysis using TextBlob
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity

    # Determine the sentiment as Positive, Negative, or Neutral
    if sentiment > 0:
        sentiment_label.configure(text="Sentiment: Positive")
    elif sentiment < 0:
        sentiment_label.configure(text="Sentiment: Negative")
    else:
        sentiment_label.configure(text="Sentiment: Neutral")


# Function to get response from the GPT model
def get_gpt_response(text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use the appropriate model
            messages=[{"role": "user", "content": text}]
        )
        gpt_response = response['choices'][0]['message']['content']
        gpt_label.configure(text="GPT Response: " + gpt_response)
    except Exception as e:
        gpt_label.configure(text="GPT Error: " + str(e))


# Function to generate a story based on detected emotion
def generate_story_based_on_emotion(emotion):
    prompt = f"Generate a short story based on the emotion: {emotion}."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        story = response['choices'][0]['message']['content']
        story_label.configure(text="Generated Story: " + story)
    except Exception as e:
        story_label.configure(text="Story Generation Error: " + str(e))


# Function to update the video frame
def update_frame():
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        prediction = classifier.predict(roi)[0]
        label = emotion_labels[prediction.argmax()]
        emotion_label.configure(text="Emotions: " + label)

        # Generate story based on the detected emotion
        generate_story_based_on_emotion(label)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Convert the frame to a PhotoImage and display it
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=im)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk, text="")

    # Schedule the next frame update
    root.after(15, update_frame)


# Bind the microphone switch to toggle_microphone function
mic_switch.configure(command=toggle_microphone)

# Start the video capture and frame updates
cap = cv2.VideoCapture(0)
update_frame()

root.mainloop()

cap.release()
cv2.destroyAllWindows()
