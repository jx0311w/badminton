import streamlit as st
import cv2
import mediapipe as mp
import tempfile
import numpy as np
from scipy.spatial.distance import euclidean

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Function to extract pose landmarks from a video
def extract_landmarks(video_path):
    cap = cv2.VideoCapture(video_path)
    landmarks_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame_rgb)

        if result.pose_landmarks:
            landmarks = [(lmk.x, lmk.y, lmk.z) for lmk in result.pose_landmarks.landmark]
            landmarks_list.append(landmarks)

    cap.release()
    return landmarks_list

# Function to calculate similarity between two sets of landmarks
def calculate_similarity(reference_landmarks, test_landmarks):
    if len(reference_landmarks) != len(test_landmarks):
        min_length = min(len(reference_landmarks), len(test_landmarks))
        reference_landmarks = reference_landmarks[:min_length]
        test_landmarks = test_landmarks[:min_length]

    similarity_scores = []

    for ref_frame, test_frame in zip(reference_landmarks, test_landmarks):
        frame_similarity = []
        for ref_point, test_point in zip(ref_frame, test_frame):
            distance = euclidean(ref_point, test_point)
            frame_similarity.append(distance)
        avg_similarity = np.mean(frame_similarity)
        similarity_scores.append(avg_similarity)

    # Normalize the similarity score to a percentage (lower is better)
    max_possible_distance = 1  # Normalized landmark positions range from 0 to 1
    similarity_percentage = 100 - (np.mean(similarity_scores) / max_possible_distance * 100)
    return similarity_percentage

# Streamlit Web App
st.title("Badminton Technique Analyzer")
st.write("Upload a reference video (perfect smash) and your video to compare.")

# Step 1: Upload Reference Video
st.header("Step 1: Upload Perfect Smash Video")
reference_video = st.file_uploader("Upload the perfect smash video", type=["mp4", "mov", "avi"], key="reference")

if reference_video:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(reference_video.read())
        reference_video_path = temp_file.name

    st.video(reference_video)
    st.write("Processing the perfect smash video...")
    reference_landmarks = extract_landmarks(reference_video_path)
    st.write("Perfect smash video processed!")

# Step 2: Upload User Video
st.header("Step 2: Upload Your Smash Video")
uploaded_video = st.file_uploader("Upload your smash video", type=["mp4", "mov", "avi"], key="uploaded")

if uploaded_video and reference_video:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_video.read())
        uploaded_video_path = temp_file.name

    st.video(uploaded_video)
    st.write("Processing your video...")
    uploaded_landmarks = extract_landmarks(uploaded_video_path)

    # Step 3: Compare Landmarks
    st.write("Comparing your video with the perfect smash video...")
    similarity_score = calculate_similarity(reference_landmarks, uploaded_landmarks)
    st.write(f"Similarity Score: {similarity_score:.2f}%")

    if similarity_score > 90:
        st.success("Great job! Your smash is very similar to the perfect smash.")
    else:
        st.warning("Your smash could use some improvement. Check your posture and technique.")
