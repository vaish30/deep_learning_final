# Importing required libraries 
# TensorFlow Keras
import tensorflow as tf
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model, model_from_json
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm
import importlib
import signal

# Other  
import librosa
import librosa.display
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import pandas as pd
import seaborn as sns
import glob 
import os
import pickle
import IPython.display as ipd  # To play sound in the notebook

import sounddevice as sd
import numpy as np
import threading
import queue
import time
import keyboard  # For detecting key press
from tensorflow.keras.models import load_model

from tensorflow.keras.metrics import F1Score

# import win32com.client as win32
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import seaborn as sns
import argparse
from voice_activity_detection import is_speech 

# import pythoncom

# pythoncom.CoInitialize()


mean = np.load("mean_new1002.npy")
std = np.load("std_new1002.npy")

prediction_list = []

# Global variables
q = queue.Queue()
stop_event = threading.Event() 
recipients = ["jyvaishu@gmail.com", "vj1@fordham.edu"]

def signal_handler(sig, frame):
    print('Ctrl+C pressed, stopping recording...')
    global stop_event
    stop_event.set()  # Signal the stop event to terminate threads

def audio_callback(indata, frames, time, status):
    """Callback function that continuously receives audio chunks."""
    try:
        if status:
            print(status)
        q.put(indata.copy())
    except Exception as e:
        print(f"Error in audio callback: {e}")


# Function to send email using Outlook for windows
# def send_email_alert(subject, body, recipients):
#     outlook = win32.Dispatch('outlook.application')
#     mail = outlook.CreateItem(0)  # 0: olMailItem
#     mail.Subject = subject
#     mail.Body = body
#     mail.To = ";".join(recipients)  # Multiple recipients separated by semicolons
#     mail.Send()
#     print(f"Email alert sent to: {', '.join(recipients)}")

to_email = ['aakrutikatre13@outlook.com', 'jyvaishu@gmail.com']
def send_email_alert(subject, body, to_email):
    try:
        # Connect to Outlook's SMTP server
        smtpserver = smtplib.SMTP("smtp.office365.com", 587)  # Use the Outlook SMTP server
        smtpserver.ehlo()  # Identify ourselves to the SMTP server
        smtpserver.starttls()  # Start TLS encryption for security
        smtpserver.ehlo()  # Re-identify ourselves after encryption
        
        # Log in to the Outlook account
        smtpserver.login()  # Replace with your Outlook credentials
        
        # Construct the email
        message = f"Subject: {subject}\n\n{body}"
        
        # Send the email
        smtpserver.sendmail('aakrutikatre13@outlook.com', to_email, message)
        
        print("Email alert sent successfully.")
    
    except Exception as e:
        print(f"Error sending email: {e}")
    
    finally:
        smtpserver.quit()  # Close the connection to the SMTP server




def normalize_real_time_data(real_time_df, mean, std):
    normalized_real_time_df = (real_time_df - mean) / std
    return np.array(normalized_real_time_df)

def extract_features(audio_data, sample_rate):
    # Extract features
    mfccs = np.mean(librosa.feature.mfcc(y=audio_data.flatten(), sr=sample_rate, n_mfcc=13), axis=0)
    mel_spec = np.mean(librosa.feature.melspectrogram(y=audio_data.flatten(), sr=sample_rate, n_mels=128), axis=0)
    # spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=audio_data.flatten(), sr=sample_rate), axis=0)

    # Combine all features into a single array
    features = np.concatenate((mfccs, mel_spec))
    return pd.DataFrame(features).T  # Transpose to match expected DataFrame structure

def predict_features(df, model):
    prediction = model.predict([df])
    return prediction

def load_model_from_file(model_path, custom_objects):
    """Load the pre-trained model from the given file path."""
    return load_model(model_path, custom_objects=custom_objects)

def run_model_on_gpu(model, features):
    try:
        with tf.device('/device:GPU:0'):
            return model.predict(features)
    except RuntimeError as e:
        print(f"GPU error: {e}. Falling back to CPU.")
        return model.predict(features)
    
def pad_audio(audio, target_length):
    """Pad audio to a target length with zeros."""
    pad_length = target_length - len(audio)
    padding = np.zeros(pad_length)
    audio = np.concatenate([audio, padding])
    return audio
    
def predict_from_audio_file(file_path, model, sample_rate=32000, chunk_samples=None, hop_samples=None):
    processed_features = []
    """Predict emotion from a pre-recorded audio file."""
    try:
        # Load audio file
        audio_data, _ = librosa.load(file_path, sr=sample_rate)

        # Check if audio is long enough
        if len(audio_data) < chunk_samples:
            audio_data = pad_audio(audio_data, chunk_samples)

        # Calculate the total number of windows
        total_windows = (len(audio_data) - chunk_samples) // hop_samples + 1
        # print(f'Total windows: {total_windows}')

        # Extract features using sliding windows
        for i in range(0, len(audio_data) - chunk_samples + 1, hop_samples):
            window = audio_data[i:i + chunk_samples]


            # If the window is smaller than the window size (for the last window), pad it
            if len(window) < chunk_samples:
                window = pad_audio(window, chunk_samples)
                # print(window.shape)

            # Extract features and normalize them
            features_df = extract_features(window, sample_rate)
            normalized_features = normalize_real_time_data(features_df, mean, std)
            features_reshaped = normalized_features.reshape(1, normalized_features.shape[1], 1)  # Reshape for model input

            processed_features.append(features_reshaped)    

        # Get prediction
        classes = ['Gunshot_or_gunfire', 'Screams', 'Shatter','background_noise']
        print(f"Processed Features Shape: {np.array(processed_features).shape}")
        prediction = run_model_on_gpu(model, processed_features)
        prediction = list(prediction.argmax(axis=1))[0]
        prediction_label = classes[prediction]
        # print(f"Prediction for {file_path}: {prediction_label}")
        return prediction_label,  np.array(processed_features)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None
    
def autopct_format(pct):
            return ('%1.1f%%' % pct) if pct > 0 else ''

def record_continuous_chunks(filename,  window_size=3.0, hop_length=1.5, sample_rate=32000, model=None, real_time=True, audio_file=None):
    chunk_samples = int(window_size * sample_rate)
    hop_samples = int(hop_length * sample_rate)

    """Record real-time audio or load from a file and predict emotion."""
    if real_time:
        print("Real-time audio recording and prediction...")
        
        
        audio_buffer = np.zeros((0, 1), dtype='int16')
        start_time = time.time()
        max_duration = 30  # Maximum recording duration in seconds

        def save_chunk(chunk_data, chunk_index):
            filename_chunk = f"{filename}_{chunk_index}.wav"
            wavio.write(filename_chunk, chunk_data, sample_rate, sampwidth=2)
            # print(f"Saved {filename_chunk}")

            features_df = extract_features(chunk_data.astype(np.float32), sample_rate)
            normalized_features = normalize_real_time_data(features_df, mean, std)
            features_reshaped = normalized_features.reshape(1, normalized_features.shape[1], 1)  # Reshape to (120, 1)
            print(f"Extracted Features Shape: {features_reshaped.shape}")

            # print(f"Extracted Features Shape: {features_reshaped.shape}")

            classes =  ['Gunshot_or_gunfire', 'Screams', 'Shatter','background_noise']
            prediction = run_model_on_gpu(model, features_reshaped)
            prediction = list(prediction.argmax(axis=1))[0]
            prediction_label = classes[prediction]
            print(f"Prediction for {filename_chunk}: {prediction_label}")
            # prediction_list.append(prediction_label)
            try:
                # Send email alert if a critical sound is detected
                if prediction_label in ['Gunshot_or_gunfire', 'Screams', 'Shatter']:
                    print(f"Detected {prediction_label}, sending email alert...")
                    subject = f"Alert: Detected {prediction_label}"
                    body = f"An alert sound has been detected: {prediction_label}. Please investigate immediately!"
                    # send_email_alert(subject, body, recipients)
                    # print(f"Email alert sent for {prediction_label}")
            except Exception as e:
                print(f"Error sending email alert: {e}")
            
            prediction_list.append(prediction_label)

        def process_chunks():
            chunk_index = 0
            while not stop_event.is_set() and (time.time() - start_time) < max_duration:
                try:
                    data = q.get(timeout=0.1)
                    nonlocal audio_buffer
                    audio_buffer = np.concatenate((audio_buffer, data), axis=0)

                    while len(audio_buffer) >= chunk_samples:
                        chunk_data = audio_buffer[:chunk_samples]

                        # Check if voice activity is detected before saving and predicting
                        # if is_speech(chunk_data.flatten().astype(np.int16), sample_rate):
                        #     print("Voice detected, processing chunk...")
                        save_chunk(chunk_data, chunk_index)
                        chunk_index += 1
                        # elsss("No voice activity detected.")
                        
                        audio_buffer = audio_buffer[hop_samples:]

                except queue.Empty:
                    continue

        stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate, dtype='int16')
        
        try:
            stream.start()
            print("Recording started... Will stop after 30 seconds.")

            processing_thread = threading.Thread(target=process_chunks)
            processing_thread.start()

            while (time.time() - start_time) < max_duration:
                time.sleep(0.1)

            stop_event.set()  # Stop recording after 30 seconds

        except KeyboardInterrupt:
            print("Ctrl+C pressed, stopping recording...")
            stop_event.set()

        finally:
            stream.stop()
            stream.close()
            processing_thread.join()
            print("Recording stopped.")
    else:
    # Handle prediction from a recorded file
        if audio_file:
            prediction, fea = predict_from_audio_file(audio_file, model, sample_rate, chunk_samples, hop_samples)
            print(f"Processed Features Shape: {np.array(fea).shape}")
            # prediction_list.append(prediction)
            print(f"Prediction for {audio_file}: {prediction_list}")
            try:
            # Send email alert if a critical sound is detected
                if prediction in ['Gunshot_or_gunfire', 'Screams', 'Shatter']:
                    print(f"Detected {prediction}, sending email alert...")
                    subject = f"Alert: Detected {prediction}"
                    body = f"Alert!!! {prediction} Sound has been detected. Please investigate immediately!"
                    # send_email_alert(subject, body, recipients)
                    # print(f"Email alert sent for {prediction}")
            except Exception as e:
                print(f"Error sending email alert: {e}")
        
            prediction_list.append(prediction)

    # # Plotting the final pie chart with a legend on the right-hand side
    # classes = ['angry', 'fear', 'happy', 'sad', 'surprise']
    # class_counts = {cls: prediction_list.count(cls) for cls in classes}
    # labels = list(class_counts.keys())
    # sizes = list(class_counts.values())
    # colors = sns.color_palette('pastel', n_colors=len(classes))

    # plt.figure(figsize=(8, 8))
    # # Create pie chart without 0% labels
    # patches, texts, autotexts = plt.pie(sizes, colors=colors, autopct=autopct_format, startangle=140)

    # # Add legend outside the pie chart on the right-hand side
    # plt.legend(patches, labels, loc="center left", bbox_to_anchor=(1, 0.5), title="Emotion Classes")

    # plt.title('Real-time Emotion Prediction')
    # plt.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle
    # plt.tight_layout()  # Adjusts the layout to fit everything
    # plt.show()

def main():
    # print("Main function triggered")
    parser = argparse.ArgumentParser(description="Real-time or pre-recorded audio sentiment analysis.")
    parser.add_argument('--mode', type=str, choices=['real_time', 'recorded'], required=True,
                        help="Choose whether to analyze real-time audio or a pre-recorded audio file.")
    parser.add_argument('--audio_file', type=str, default=None, help="Path to the pre-recorded audio file.")
    
        
    args = parser.parse_args()
    # print(f"Mode selected: {args.mode}")

    # Load pre-trained model
    model_path = r"/Users/jyvaishu/Downloads/vandalism_detection-main/van_detector_1004.h5"  # Replace with your actual model path
    # model = load_model(model_path, custom_objects={'f1_score': f1_score})
    # model = load_model(model_path, custom_objects={'f1_score': F1Score()})/
    model = load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', F1Score()])




    signal.signal(signal.SIGINT, signal_handler)

    # Handle real-time or recorded prediction based on user input
    if args.mode == 'real_time':
        print("Running real-time audio prediction...")
        record_continuous_chunks('output',  window_size=3.0, hop_length=1.5, sample_rate=32000, model=model, real_time=True)
    elif args.mode == 'recorded':
        if not args.audio_file:
            print("Error: Please provide the --audio_file argument for recorded mode.")
        else:
            print(f"Running prediction on recorded audio file: {args.audio_file}")
            record_continuous_chunks('output', window_size=3.0, hop_length=1.5, sample_rate=32000, model=model, real_time=False, audio_file=args.audio_file)

if __name__ == '__main__':
    main()
