import tkinter as tk
import threading
from PIL import Image, ImageTk
import cv2
import numpy as np
import customtkinter
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from textblob import TextBlob
from google.cloud import aiplatform

# Initialize Google AI Platform with project ID
key = "XXk-proj-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-N_94abEqfK0GXXXXXXXXXXXXXXXXXXXXXXXX1fOaNaypZFC0RKVcANsuuw_aMA"
location = "us-central1"  # Replace with your location if different

aiplatform.init(project=key, location=location)

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("blue")

# Load the emotion detection model and face cascade
face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
classifier = load_model(r'model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Set up the GUI window
root = customtkinter.CTk()
root.title("Personality Detector")
root.geometry('1000x800')

# Create GUI layout frames and widgets
frame_1 = customtkinter.CTkFrame(master=root)
frame_1.pack(pady=20, padx=60, fill="both", expand=True)

video_label = customtkinter.CTkLabel(master=frame_1)
video_label.pack(anchor='nw')

emotion_label = customtkinter.CTkLabel(master=frame_1, text="Emotions: ", font=("Helvetica", 20))
emotion_label.place(x=700, y=0)

story_textbox = customtkinter.CTkTextbox(master=frame_1, width=250, height=300)
story_textbox.place(x=700, y=50)

mic_switch_value = tk.BooleanVar()
mic_switch = customtkinter.CTkSwitch(master=frame_1, text="Microphone", variable=mic_switch_value)
mic_switch.place(x=10, y=400)

voice_data = customtkinter.CTkTextbox(master=frame_1, width=500, height=100)
voice_data.place(x=10, y=450)

# Function to use Google AI to generate a dynamic story based on emotion
def generate_dynamic_story_with_gemini(emotion):
    prompt = f"Write a creative, short story that conveys a sense of {emotion}."
    try:
        # Use text generation model from Vertex AI
        model = aiplatform.TextPredictor(endpoint="YOUR_MODEL_ENDPOINT")  # Replace with your model endpoint ID
        response = model.predict(instances=[{"content": prompt}])
        story = response.predictions[0]["content"].strip()
        return story
    except Exception as e:
        print(f"Error generating story: {e}")
        return "Could not generate story due to an error."

# Function to perform sentiment analysis using TextBlob
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        sentiment_text = "Positive"
    elif sentiment < 0:
        sentiment_text = "Negative"
    else:
        sentiment_text = "Neutral"
    return sentiment_text

# Function to update the video frame
def update_frame():
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        prediction = classifier.predict(roi)[0]
        label = emotion_labels[prediction.argmax()]
        emotion_label.configure(text="Emotions: " + label)

        # Generate a dynamic story based on detected emotion
        story = generate_dynamic_story_with_gemini(label)
        story_textbox.delete(1.0, tk.END)
        story_textbox.insert(tk.END, story)

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Convert the frame to a PhotoImage and display it
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=im)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk, text="")

    # Schedule the next frame update
    root.after(15, update_frame)

# Function to listen to the microphone and analyze text sentiment
def listen_to_speech():
    import speech_recognition as sr
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        voice_data.delete(1.0, tk.END)  # Clear previous text
        voice_data.insert(tk.END, "Listening...")
        try:
            audio = recognizer.listen(source, timeout=5)  # Listen for up to 5 seconds
            voice_data.delete(1.0, tk.END)  # Clear previous text
            voice_text = recognizer.recognize_google(audio)
            voice_data.insert(tk.END, voice_text)

            # Perform sentiment analysis on the recognized text
            sentiment = analyze_sentiment(voice_text)
            print("Sentiment:", sentiment)

        except sr.WaitTimeoutError:
            voice_data.delete(1.0, tk.END)
            voice_data.insert(tk.END, "Listening timed out.")
        except sr.UnknownValueError:
            voice_data.delete(1.0, tk.END)
            voice_data.insert(tk.END, "Could not understand audio.")
        except sr.RequestError:
            voice_data.delete(1.0, tk.END)
            voice_data.insert(tk.END, "Network error occurred.")

# Function to toggle microphone listening
def toggle_microphone():
    if mic_switch_value.get():
        threading.Thread(target=listen_to_speech).start()
    else:
        voice_data.delete(1.0, tk.END)

# Bind the microphone switch to toggle_microphone function
mic_switch.configure(command=toggle_microphone)

# Start the video capture and frame updates
cap = cv2.VideoCapture(0)
update_frame()

root.mainloop()
cap.release()
cv2.destroyAllWindows()
