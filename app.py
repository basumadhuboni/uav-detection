import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mimetypes
import subprocess
import os
import uuid
import shutil
import sys
import torch
from norfair import Detection, Tracker, draw_tracked_objects

# Set page config
st.set_page_config(
    page_title="UAV Behaviour Detection System",
    page_icon="✈️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.stApp {
    background-color: #f5f5f5;
}
.uploadfile {
    border: 2px dashed #4c4c4c;
    padding: 20px;
    border-radius: 10px;
}
.stButton button {
    width: 100%;
    border-radius: 5px;
    height: 3em;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Title and description
st.title('🛸 UAV Detection System')
st.markdown('''
### Upload a video to detect and analyze drone activity
''')

# Initialize session state
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'unique_id' not in st.session_state:
    st.session_state.unique_id = None
if 'video_filename' not in st.session_state:
    st.session_state.video_filename = None
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'output_video_path' not in st.session_state:
    st.session_state.output_video_path = None
if 'trajectory_output_path' not in st.session_state:
    st.session_state.trajectory_output_path = None

def is_valid_video(file):
    if file is None:
        return False
    mime_type = mimetypes.guess_type(file.name)[0]
    return mime_type is not None and mime_type.startswith('video/')

def analyze_trajectory(trajectories, fps):
    """
    Analyze the drone trajectory to determine if the behavior is suspicious based on sharp turns and speed variability.
    Returns a tuple: (flight_pattern, behavior_classification, confidence_score)
    """
    if not trajectories:
        return "Unknown", "Unknown", 0

    # For simplicity, assume only one drone (first track_id)
    track_id = list(trajectories.keys())[0]
    points = trajectories[track_id]
    
    if len(points) < 3:
        return "Unknown", "Unknown", 0

    # 1. Calculate number of sharp turns (direction changes > 90 degrees)
    sharp_turns = 0
    for i in range(2, len(points)):
        # Get three consecutive points to compute the angle
        p1 = np.array(points[i-2])
        p2 = np.array(points[i-1])
        p3 = np.array(points[i])
        
        # Vectors
        v1 = p2 - p1  # Vector from p1 to p2
        v2 = p3 - p2  # Vector from p2 to p3
        
        # Compute the angle between vectors using dot product
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            continue
        
        cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Avoid numerical errors
        angle = np.degrees(np.arccos(cos_angle))
        
        # Consider it a sharp turn if angle > 90 degrees
        if angle > 90:
            sharp_turns += 1

    # 2. Calculate speed variability
    speeds = []
    for i in range(1, len(points)):
        # Distance between consecutive points
        p1 = np.array(points[i-1])
        p2 = np.array(points[i])
        distance = np.linalg.norm(p2 - p1)
        
        # Time interval between frames (in seconds) = 1/fps
        time_interval = 1.0 / fps
        
        # Speed = distance / time (pixels per second)
        speed = distance / time_interval
        speeds.append(speed)
    
    # Compute speed variance
    if len(speeds) > 1:
        speed_variance = np.var(speeds)
        mean_speed = np.mean(speeds)
        # Normalize variance relative to mean speed to get a coefficient of variation
        speed_cv = (speed_variance ** 0.5) / mean_speed if mean_speed > 0 else 0
    else:
        speed_cv = 0

    # 3. Classify behavior based on rules
    # - More than 3 sharp turns indicates suspicious movement
    # - High speed variability (coefficient of variation > 0.5) indicates suspicious behavior
    suspicious_score = 0
    if sharp_turns > 3:
        suspicious_score += 40  # Sharp turns contribute to suspiciousness
    if speed_cv > 0.5:
        suspicious_score += 40  # High speed variability contributes to suspiciousness

    # Determine flight pattern and behavior
    flight_pattern = "Suspicious" if sharp_turns > 3 or speed_cv > 0.5 else "Normal"
    behavior_classification = "Threatening" if suspicious_score >= 60 else "Non-threatening"
    confidence_score = min(suspicious_score + 20, 100)  # Base confidence + bonus for clarity

    return flight_pattern, behavior_classification, confidence_score

# Create directories if they don’t exist
os.makedirs('uploads', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# File uploader
uploaded_file = st.file_uploader(
    "Choose a video file",
    type=['mp4', 'avi', 'mov', 'mkv'],
    help="Upload a video file (MP4, AVI, MOV, or MKV format)"
)

# Handle file upload
if uploaded_file is not None and uploaded_file != st.session_state.uploaded_file:
    # Reset session state when a new file is uploaded
    st.session_state.uploaded_file = uploaded_file
    st.session_state.analysis_done = False
    st.session_state.unique_id = None
    st.session_state.video_filename = None
    st.session_state.video_path = None
    st.session_state.output_video_path = None
    st.session_state.trajectory_output_path = None

# Process uploaded file if present in session state
if st.session_state.uploaded_file is not None:
    if is_valid_video(st.session_state.uploaded_file):
        try:
            # Generate a unique ID and paths if not already set
            if st.session_state.unique_id is None:
                st.session_state.unique_id = str(uuid.uuid4())
                st.session_state.video_filename = f"{st.session_state.unique_id}.mp4"
                st.session_state.video_path = os.path.join('uploads', st.session_state.video_filename)

                # Save the uploaded video
                with open(st.session_state.video_path, 'wb') as f:
                    f.write(st.session_state.uploaded_file.getvalue())

            st.success(f"Video uploaded successfully: {st.session_state.uploaded_file.name}")

            # Analyze button
            if not st.session_state.analysis_done and st.button('🔍 Analyze Video', key='analyze'):
                with st.spinner('Processing video for detection...'):
                    # Define paths
                    current_dir = os.getcwd()
                    weights_path = os.path.join(current_dir, 'weights', 'best.pt')
                    output_project = os.path.join(current_dir, 'outputs')

                    # Run YOLOv5 detection using the same Python interpreter
                    command = [
                        sys.executable, 'yolov5/detect.py',
                        '--weights', weights_path,
                        '--img', '640',
                        '--conf', '0.4',
                        '--source', st.session_state.video_path,
                        '--project', output_project,
                        '--name', st.session_state.unique_id,
                        '--exist-ok'
                    ]
                    subprocess.run(command, check=True, cwd=current_dir)

                    # Define output video path for detection
                    st.session_state.output_video_path = os.path.join('outputs', st.session_state.unique_id, st.session_state.video_filename)

                with st.spinner('Processing video for trajectory tracing...'):
                    # Load YOLOv5 model
                    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)

                    # Set up video capture and output for trajectory tracing
                    cap = cv2.VideoCapture(st.session_state.video_path)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    st.session_state.trajectory_output_path = os.path.join('outputs', st.session_state.unique_id, f"trajectory_{st.session_state.video_filename}")
                    out = cv2.VideoWriter(st.session_state.trajectory_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

                    # Initialize Norfair tracker
                    tracker = Tracker(distance_function='mean_euclidean', distance_threshold=50)

                    # Initialize trajectories dictionary
                    trajectories = {}

                    # Process video frames
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        # Run YOLOv5 detection
                        results = model(frame)
                        detections = []
                        for *xyxy, conf, cls in results.xyxy[0]:
                            if cls == 0 and conf > 0.5:
                                x1, y1, x2, y2 = map(int, xyxy)
                                center = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]])
                                detections.append(Detection(points=center, data={'bbox': [x1, y1, x2, y2]}))

                        # Update tracker
                        tracked_objects = tracker.update(detections=detections)

                        # Update trajectories
                        for obj in tracked_objects:
                            if obj.id not in trajectories:
                                trajectories[obj.id] = []
                            trajectories[obj.id].append(obj.estimate[0])

                        # Draw tracked objects (bounding boxes and IDs)
                        draw_tracked_objects(frame, tracked_objects)

                        # Draw trajectories as green lines
                        for track_id, points in trajectories.items():
                            if len(points) > 1:
                                pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                                cv2.polylines(frame, [pts], isClosed=False, color=(0, 255, 0), thickness=2)

                        # Write the annotated frame to the output video
                        out.write(frame)

                    # Release video objects
                    cap.release()
                    out.release()

                    # Analyze the trajectory
                    flight_pattern, behavior_classification, confidence_score = analyze_trajectory(trajectories, fps)

                    # Store analysis results in session state
                    st.session_state.flight_pattern = flight_pattern
                    st.session_state.behavior_classification = behavior_classification
                    st.session_state.confidence_score = confidence_score

                # Mark analysis as done
                st.session_state.analysis_done = True

            # Display download buttons if analysis is done
            if st.session_state.analysis_done:
                # Detection output download button
                if st.session_state.output_video_path and os.path.exists(st.session_state.output_video_path):
                    with open(st.session_state.output_video_path, 'rb') as f:
                        st.download_button(
                            label="⬇️ Download detection output video",
                            data=f,
                            file_name="detection_output.mp4",
                            mime="video/mp4"
                        )
                else:
                    st.error("Detection output video not found. Processing may have failed.")

                # Trajectory output download button
                if st.session_state.trajectory_output_path and os.path.exists(st.session_state.trajectory_output_path):
                    with open(st.session_state.trajectory_output_path, 'rb') as f:
                        st.download_button(
                            label="⬇️ Download trajectory output video",
                            data=f,
                            file_name="trajectory_output.mp4",
                            mime="video/mp4"
                        )
                else:
                    st.error("Trajectory output video not found. Processing may have failed.")

                # Display analysis results
                st.subheader("Analysis Results")
                with st.expander("View Detailed Analysis", expanded=True):
                    st.markdown(f'''
                    🔍 **Analysis Results:**
                    - Flight Pattern: {st.session_state.flight_pattern}
                    - Behavior Classification: {st.session_state.behavior_classification}
                    - Confidence Score: {st.session_state.confidence_score}%
                    ''')

        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            st.info("Please try uploading a different video file")
    else:
        st.error("Please upload a valid video file (MP4, AVI, MOV, or MKV format)")
else:
    st.info('👆 Upload a video file to begin analysis')

# Add footer
st.markdown('''
---
💡 This system uses YOLOv5 for UAV detection and custom algorithms for trajectory analysis and threat assessment.
''')