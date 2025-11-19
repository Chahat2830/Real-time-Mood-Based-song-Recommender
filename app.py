###########################################################
# REAL-TIME MOOD RECOMMENDER (ONNX Emotion Model)
# Streamlit Cloud Compatible - NO TensorFlow / NO DeepFace
###########################################################

import streamlit as st
import numpy as np
import pandas as pd
import cv2
import onnxruntime as ort
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


##########################################################################
# 1. STREAMLIT SETUP
##########################################################################
st.set_page_config(page_title="Real-time Mood Recommender (ONNX)", layout="wide")
st.title("🎵 Real-Time Mood Detection + Song Recommender (ONNX Model)")


##########################################################################
# 2. LOAD ONNX EMOTION MODEL
##########################################################################
MODEL_PATH = "emotion-ferplus-8.onnx"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Missing ONNX model 'emotion-ferplus-8.onnx'. Place it beside app.py.")
    st.stop()

# ONNX runtime session
sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

emotion_labels = ["neutral", "happy", "sad", "surprise", "anger", "disgust", "fear", "contempt"]


def onnx_predict_emotion(face_gray):
    resized = cv2.resize(face_gray, (64, 64))
    tensor = resized.astype("float32")[None, None, :, :]  # shape: (1,1,64,64)

    preds = sess.run(None, {"Input3": tensor})[0][0]
    emotion = emotion_labels[np.argmax(preds)]
    return emotion


##########################################################################
# 3. SETUP SPOTIFY
##########################################################################
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

if SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET:
    try:
        auth = SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET)
        sp = spotipy.Spotify(auth_manager=auth)
        USE_SPOTIFY = True
    except:
        sp = None
        USE_SPOTIFY = False
else:
    sp = None
    USE_SPOTIFY = False


def get_spotify_info(track_name, artist_name):
    if not USE_SPOTIFY:
        return "https://via.placeholder.com/300?text=No+Image", None

    try:
        query = f"track:{track_name} artist:{artist_name}"
        results = sp.search(query, limit=1, type="track")
        items = results["tracks"]["items"]
        if not items:
            return "https://via.placeholder.com/300?text=Not+Found", None
        info = items[0]
        img = info["album"]["images"][0]["url"]
        preview = info.get("preview_url", None)
        return img, preview
    except:
        return "https://via.placeholder.com/300?text=Error", None


##########################################################################
# 4. LOAD DATASET
##########################################################################
@st.cache_data
def load_data():
    if os.path.exists("spotify_tracks.csv"):
        return pd.read_csv("spotify_tracks.csv")
    return pd.DataFrame()

df = load_data()

feature_cols = ["acousticness","danceability","energy","instrumentalness",
                "liveness","loudness","speechiness","valence","tempo"]


@st.cache_data
def scale_features(df_local):
    if df_local.empty:
        return None, None
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_local[feature_cols].fillna(0.0))
    return scaled, scaler


scaled_features, scaler = scale_features(df)


##########################################################################
# 5. RECOMMENDATION SYSTEM
##########################################################################
def recommend(song_name, mood):
    if df.empty:
        return pd.DataFrame()

    if song_name not in df["track_name"].values:
        return pd.DataFrame()

    idx = df[df["track_name"] == song_name].index[0]

    sims = cosine_similarity([scaled_features[idx]], scaled_features)[0]
    top_idx = sims.argsort()[::-1][1:30]  # top 30

    recs = df.iloc[top_idx].copy()

    if mood == "happy":
        recs = recs[recs["valence"] > 0.6]
    elif mood == "sad":
        recs = recs[recs["valence"] < 0.4]
    elif mood == "energetic":
        recs = recs[recs["energy"] > 0.6]
    elif mood == "calm":
        recs = recs[recs["energy"] < 0.45]

    return recs.head(10)


##########################################################################
# 6. EMOTION → MOOD MAPPING
##########################################################################
def map_mood(emotion):
    mapping = {
        "happy": "happy",
        "sad": "sad",
        "anger": "energetic",
        "surprise": "happy",
        "fear": "calm",
        "disgust": "sad",
        "neutral": "calm",
        "contempt": "calm"
    }
    return mapping.get(emotion.lower(), "calm")


##########################################################################
# 7. VIDEO TRANSFORMER (REAL-TIME EMOTION)
##########################################################################
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


class EmotionTransformer(VideoTransformerBase):
    def __init__(self):
        self.last_emotion = "neutral"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_gray = gray[y:y+h, x:x+w]

            emotion = onnx_predict_emotion(face_gray)
            self.last_emotion = emotion

            mood = map_mood(emotion)
            st.session_state.latest_mood = mood

            # Draw green box
            cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 3)
            cv2.putText(img, mood, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        return img


##########################################################################
# 8. STREAMLIT UI
##########################################################################
rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

tab1, tab2 = st.tabs(["📷 Live Mood Detection", "🎶 Song Recommendations"])


##########################################################################
#  TAB 1: LIVE EMOTION DETECTION
##########################################################################
with tab1:
    st.header("📷 Live Emotion Detection (ONNX - Fast)")

    webrtc_streamer(
        key="emotion-stream",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        video_transformer_factory=EmotionTransformer,
        media_stream_constraints={
            "video": {"width": 1280, "height": 720, "frameRate": 30},
            "audio": False
        }
    )

    if "latest_mood" in st.session_state:
        st.success(f"Detected Mood: **{st.session_state.latest_mood}**")
        if st.button("✔ Use This Mood"):
            st.session_state.confirmed_mood = st.session_state.latest_mood
            st.success(f"Mood saved: **{st.session_state.confirmed_mood}**")


##########################################################################
#  TAB 2: SONG RECOMMENDER
##########################################################################
with tab2:
    st.header("🎶 Mood-Based Song Recommendations")

    mood = st.session_state.get("confirmed_mood", None)

    if mood:
        st.success(f"Using Mood: **{mood}**")
    else:
        st.warning("No mood selected! Detect your mood first in the camera tab.")

    song = st.text_input("Enter a song name:")

    if st.button("Recommend Songs"):
        recs = recommend(song, mood)
        if recs.empty:
            st.error("No recommendations found.")
        else:
            for _, row in recs.iterrows():
                img, preview = get_spotify_info(row["track_name"], row["artist_name"])
                st.image(img, width=200)
                st.write(f"🎵 **{row['track_name']}** — {row['artist_name']}")
                if preview:
                    st.audio(preview)
                st.write("---")
