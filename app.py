import streamlit as st
import numpy as np

st.title("Deepfake Audio Detection ")
st.write("Upload a FLAC audio file to check if it's a deepfake. ")

uploaded_file = st.file_uploader("Choose a FLAC file", type=".flac")

if uploaded_file is not None:
  
  from keras.models import load_model
  import librosa

  # Load the model
  model = load_model('cnn_audio.h5')

  def predict_voice(model, audio_file_path, genre_mapping):

    
    signal, sample_rate = librosa.load(audio_file_path, sr=22050)

    
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13, n_fft=2048, hop_length=512)
    mfcc = mfcc.T

    # Resize MFCCs to appropriate size
    mfcc = np.resize(mfcc, (130, 13, 1))

    # Reshape MFCCs to appropriate size
    mfcc = mfcc[np.newaxis, ...]

   
    prediction = model.predict(mfcc)
    predicted_index = np.argmax(prediction, axis=1)

    
    genre_label = genre_mapping[predicted_index[0]]
    st.write("Raw prediction:")
    st.write(prediction)

    return genre_label


  audio_file_path = uploaded_file

  genre_mapping = {0: "spoof", 1: "bonafide"}


  predicted_voice = predict_voice(model, audio_file_path, genre_mapping)

  st.write("Predicted label:")
  st.write(predicted_voice)
else:
  st.info("Upload a FLAC file to proceed.")
