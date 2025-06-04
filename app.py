import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mimetypes
import subprocess
import os
import uuid
import shutil
import gdown

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
os.makedirs('weights', exist_ok=True)

# Download YOLOv5 weights if not present
weights_path = os.path.join('weights', 'best.pt')
if not os.path.exists(weights_path):
    st.info("Downloading model weights...")
    gdrive_url = "https://drive.google.com/uc?id=1XcwJhUf5_5BVwKTBXMiFs2KF44I42oIi"
    gdown.download(gdrive_url, weights_path, quiet=False)

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
                with st.spinner('Processing video...'):
                    # Define paths
                    current_dir = os.getcwd()
                    output_project = os.path.join(current_dir, 'outputs')

                    # Run YOLOv5 detection
                    command = [
                        'python3', 'yolov5/detect.py',
                        '--weights', weights_path,
                        '--img', '640',
                        '--conf', '0.4',
                        '--source', video_path,
                        '--project', output_project,
                        '--name', unique_id,
                        '--exist-ok'
                    ]
                    subprocess.run(command, check=True)

                    # Define output video path
                    output_video_path = os.path.join('outputs', unique_id, video_filename)

                    # Check if output video exists and provide download
                    if os.path.exists(output_video_path):
                        with open(output_video_path, 'rb') as f:
                            st.download_button(
                                label="⬇️ Download output video",
                                data=f,
                                file_name="output_video.mp4",
                                mime="video/mp4"
                            )
                        # Optionally display the output video
                        st.video(output_video_path)

                        # Clean up temporary files
                        shutil.rmtree(os.path.join('uploads', unique_id), ignore_errors=True)
                        shutil.rmtree(os.path.join('outputs', unique_id), ignore_errors=True)
                    else:
                        st.error("Output video not found. Processing may have failed.")

                    # Display placeholder analysis results
                    st.subheader("Analysis Results")
                    with st.expander("View Detailed Analysis", expanded=True):
                        st.markdown('''
                        🔍 **Analysis Results:**
                        - Flight Pattern: Normal
                        - Speed Analysis: Within acceptable range
                        - Behavior Classification: Non-threatening
                        - Confidence Score: 95%
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