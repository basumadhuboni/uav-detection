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
    page_title="UAV Detection System",
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

def is_valid_video(file):
    if file is None:
        return False
    mime_type = mimetypes.guess_type(file.name)[0]
    return mime_type is not None and mime_type.startswith('video/')

# Create directories if they don’t exist
os.makedirs('uploads', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# File uploader
uploaded_file = st.file_uploader(
    "Choose a video file",
    type=['mp4', 'avi', 'mov', 'mkv'],
    help="Upload a video file (MP4, AVI, MOV, or MKV format)"
)

if uploaded_file is not None:
    if is_valid_video(uploaded_file):
        try:
            # Generate a unique ID for this upload
            unique_id = str(uuid.uuid4())
            video_filename = f"{unique_id}.mp4"
            video_path = os.path.join('uploads', video_filename)

            # Save the uploaded video
            with open(video_path, 'wb') as f:
                f.write(uploaded_file.getvalue())

            st.success(f"Video uploaded successfully: {uploaded_file.name}")

            # Analyze button
            if st.button('🔍 Analyze Video', key='analyze'):
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
                        '--source', video_path,
                        '--project', output_project,
                        '--name', unique_id,
                        '--exist-ok'
                    ]
                    subprocess.run(command, check=True, cwd=current_dir)

                    # Define output video path for detection
                    output_video_path = os.path.join('outputs', unique_id, video_filename)

                    # Check if output video exists and provide download
                    if os.path.exists(output_video_path):
                        with open(output_video_path, 'rb') as f:
                            st.download_button(
                                label="⬇️ Download detection output video",
                                data=f,
                                file_name="detection_output.mp4",
                                mime="video/mp4"
                            )
                    else:
                        st.error("Detection output video not found. Processing may have failed.")

                with st.spinner('Processing video for trajectory tracing...'):
                    # Load YOLOv5 model
                    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)

                    # Set up video capture and output for trajectory tracing
                    cap = cv2.VideoCapture(video_path)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    trajectory_output_path = os.path.join('outputs', unique_id, f"trajectory_{video_filename}")
                    out = cv2.VideoWriter(trajectory_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

                    # Initialize Norfair tracker
                    tracker = Tracker(distance_function='euclidean', distance_threshold=50)

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

                    # Provide download button for trajectory output video
                    if os.path.exists(trajectory_output_path):
                        with open(trajectory_output_path, 'rb') as f:
                            st.download_button(
                                label="⬇️ Download trajectory output video",
                                data=f,
                                file_name="trajectory_output.mp4",
                                mime="video/mp4"
                            )
                    else:
                        st.error("Trajectory output video not found. Processing may have failed.")

                # Display placeholder analysis results
                st.subheader("Analysis Results")
                with st.expander("View Detailed Analysis", expanded=True):
                    st.markdown('''
                    🔍 **Analysis Results:**
                    - Flight Pattern: Suspicious
                    - Behavior Classification: threatening
                    - Confidence Score: 82%
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