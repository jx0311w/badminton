import streamlit as st
import cv2
import mediapipe as mp
import tempfile
import numpy as np

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Define function to analyze pose
def analyze_pose(landmarks):
    feedback = []
    # Example rule: Check arm angle for a smash
    if landmarks and len(landmarks) > 0:
        # Example: Check wrist and elbow position
        # Add your own detailed analysis logic here
        wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        if wrist.y > elbow.y:
            feedback.append("Lower your wrist for a stronger smash.")
        else:
            feedback.append("Good wrist position for smash.")
    return feedback

# Define function to process video
def process_video(file_path):
    cap = cv2.VideoCapture(file_path)
    feedback_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame_rgb)

        if result.pose_landmarks:
            feedback = analyze_pose(result.pose_landmarks.landmark)
            feedback_list.extend(feedback)

            mp_drawing.draw_landmarks(
                frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

    cap.release()
    return feedback_list

# Streamlit Web App
st.title("Badminton Technique Analyzer")
st.write("Upload a video (smash, push, serve) to analyze your technique.")

uploaded_file = st.file_uploader("Upload your video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        video_path = temp_file.name

    st.video(uploaded_file)

    st.write("Processing video...")
    feedback = process_video(video_path)

    st.write("Analysis Complete:")
    for point in feedback:
        st.write(f"- {point}")
