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
import torch.nn as nn
from norfair import Detection, Tracker
from io import BytesIO
from scipy.interpolate import interp1d
import math
import warnings

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Set page config
st.set_page_config(
    page_title="UAV Behaviour Detection System",
    page_icon="✈️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.st straApp {
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
st.title('🛸 UAV Behaviour Detection System')
st.markdown('''
### Upload a video to detect and analyze drone activity
''')

# Define LSTM Autoencoder class
class LSTMAutoencoder(nn.Module):
    def __init__(self, timesteps, num_features, hidden_size=32):
        super(LSTMAutoencoder, self).__init__()
        self.timesteps = timesteps
        self.num_features = num_features
        self.hidden_size = hidden_size

        self.encoder_lstm1 = nn.LSTM(num_features, hidden_size, batch_first=True)
        self.encoder_lstm2 = nn.LSTM(hidden_size, hidden_size // 2, batch_first=True)
        self.encoder_lstm3 = nn.LSTM(hidden_size // 2, hidden_size // 4, batch_first=True)

        self.decoder_dense1 = nn.Linear(hidden_size // 4, hidden_size // 2)
        self.decoder_dense2 = nn.Linear(hidden_size // 2, timesteps * num_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        out, _ = self.encoder_lstm1(x)
        out, _ = self.encoder_lstm2(out)
        out, _ = self.encoder_lstm3(out)
        encoded = out[:, -1, :]
        out = self.relu(self.decoder_dense1(encoded))
        out = self.decoder_dense2(out)
        out = out.view(batch_size, self.timesteps, self.num_features)
        return out

# Load LSTM model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lstm_model_path = 'lstm_autoencoder.pth'  # Adjust path if in a subdirectory, e.g., 'models/lstm_autoencoder.pth'
try:
    lstm_model = LSTMAutoencoder(timesteps=20, num_features=10, hidden_size=32)
    lstm_model.load_state_dict(torch.load(lstm_model_path, map_location=device))
    lstm_model = lstm_model.to(device)
    lstm_model.eval()
    st.success("LSTM Autoencoder model loaded successfully.")
except Exception as e:
    st.error(f"Failed to load LSTM model: {str(e)}")
    st.stop()

# Trajectory processing functions
def resimulate_trajectory(traj, original_fps, target_interval=1.0/30):
    if len(traj) < 2:
        return []

    time_original = np.array([frame_idx / original_fps for frame_idx, _ in traj])
    points = np.array([point for _, point in traj])

    start_time = time_original[0]
    end_time = time_original[-1]
    new_time = np.arange(start_time, end_time, target_interval)

    if len(new_time) == 0:
        return []

    interp_x = interp1d(time_original, points[:, 0], kind='linear', bounds_error=False, fill_value='extrapolate')
    interp_y = interp1d(time_original, points[:, 1], kind='linear', bounds_error=False, fill_value='extrapolate')

    new_x = interp_x(new_time)
    new_y = interp_y(new_time)

    return [[x, y] for x, y in zip(new_x, new_y)]

def normalize_trajectory(points, width, height):
    return [[x / width, y / height] for [x, y] in points]

def calculate_trajectory_features(points, time_interval):
    if len(points) < 3:
        return {
            'velocity_x': [], 'velocity_y': [],
            'linear_acceleration_x': [], 'linear_acceleration_y': [],
            'orientation_x': [], 'orientation_y': [],
            'angular_x': [], 'angular_y': []
        }

    points = np.array(points)
    num_points = len(points)

    velocity_x = [0]
    velocity_y = [0]
    linear_acceleration_x = [0, 0]
    linear_acceleration_y = [0, 0]
    orientation_x = [0]
    orientation_y = [0]
    angular_x = [0, 0]
    angular_y = [0, 0]

    for t in range(1, num_points):
        dx = points[t, 0] - points[t-1, 0]
        dy = points[t, 1] - points[t-1, 1]
        vx = dx / time_interval
        vy = dy / time_interval
        velocity_x.append(vx)
        velocity_y.append(vy)

    for t in range(2, num_points):
        dvx = velocity_x[t] - velocity_x[t-1]
        dvy = velocity_y[t] - velocity_y[t-1]
        ax = dvx / time_interval
        ay = dvy / time_interval
        linear_acceleration_x.append(ax)
        linear_acceleration_y.append(ay)

    for t in range(1, num_points):
        dx = points[t, 0] - points[t-1, 0]
        dy = points[t, 1] - points[t-1, 1]
        heading = math.atan2(dy, dx) if dx != 0 or dy != 0 else 0
        orientation_x.append(math.cos(heading))
        orientation_y.append(math.sin(heading))

    for t in range(2, num_points):
        heading_t = math.atan2(points[t, 1] - points[t-1, 1], points[t, 0] - points[t-1, 0])
        heading_t_minus_1 = math.atan2(points[t-1, 1] - points[t-2, 1], points[t-1, 0] - points[t-2, 0])
        angular_vel = (heading_t - heading_t_minus_1) / time_interval
        angular_x.append(angular_vel)
        angular_y.append(0)  # Assuming 2D plane

    def normalize_feature(feature):
        if not feature or max(feature) == min(feature):
            return [0] * len(feature)
        min_val, max_val = min(feature), max(feature)
        return [(x - min_val) / (max_val - min_val) for x in feature]

    return {
        'velocity_x': normalize_feature(velocity_x),
        'velocity_y': normalize_feature(velocity_y),
        'linear_acceleration_x': normalize_feature(linear_acceleration_x),
        'linear_acceleration_y': normalize_feature(linear_acceleration_y),
        'orientation_x': normalize_feature(orientation_x),
        'orientation_y': normalize_feature(orientation_y),
        'angular_x': normalize_feature(angular_x),
        'angular_y': normalize_feature(angular_y)
    }

def prepare_feature_sequence(points, features):
    num_points = len(points)
    feature_sequence = np.zeros((num_points, 10))
    feature_sequence[:, 0:2] = points
    feature_keys = ['velocity_x', 'velocity_y', 'linear_acceleration_x', 'linear_acceleration_y',
                    'orientation_x', 'orientation_y', 'angular_x', 'angular_y']
    for i, key in enumerate(feature_keys):
        feature_values = features[key]
        feature_values = feature_values + [0] * (num_points - len(feature_values))
        feature_sequence[:, i + 2] = feature_values[:num_points]
    return feature_sequence

def analyze_trajectory(trajectories, fps, width, height, model, device, threshold=0.03):
    if not trajectories:
        return "Unknown", "Unknown", 0

    # Analyze the first trajectory (extend for multiple if needed)
    track_id = list(trajectories.keys())[0]
    traj = trajectories[track_id]

    if len(traj) < 3:
        return "Unknown", "Unknown", 0

    # Resample trajectory
    resampled_traj = resimulate_trajectory(traj, fps)

    # Normalize trajectory
    normalized_traj = normalize_trajectory(resampled_traj, width, height)

    # Calculate features
    features = calculate_trajectory_features(normalized_traj, time_interval=1.0/30)

    # Prepare feature sequence
    feature_sequence = prepare_feature_sequence(normalized_traj, features)

    # Analyze with LSTM
    if len(feature_sequence) < 20:
        st.warning(f"Track {track_id}: Insufficient data for LSTM analysis.")
        is_anomalous = False
        confidence = 0.0
    else:
        feature_sequence = np.array(feature_sequence)
        num_sequences = len(feature_sequence) // 20
        if num_sequences == 0:
            st.warning(f"Track {track_id}: Not enough sequences for LSTM analysis.")
            is_anomalous = False
            confidence = 0.0
        else:
            sequences = []
            for i in range(num_sequences):
                seq = feature_sequence[i*20:(i+1)*20]
                sequences.append(seq)
            sequences = np.array(sequences)
            sequences_tensor = torch.tensor(sequences, dtype=torch.float32).to(device)

            with torch.no_grad():
                reconstructions = model(sequences_tensor)

            mse = torch.mean((sequences_tensor - reconstructions) ** 2, dim=[1, 2]).cpu().numpy()
            avg_mse = np.mean(mse)
            is_anomalous = avg_mse > threshold
            k = 100  # Scaling factor
            if is_anomalous:
                confidence = 100 / (1 + np.exp(-k * (avg_mse - threshold)))
            else:
                confidence = 100 / (1 + np.exp(k * (avg_mse - threshold)))

    flight_pattern = "Suspicious" if is_anomalous else "Normal"
    behavior_classification = "Threatening" if is_anomalous else "Non-threatening"
    confidence_score = confidence

    return flight_pattern, behavior_classification, confidence_score

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

# Create directories if they don’t exist
os.makedirs('uploads', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# File uploader
uploaded_file = st.file_uploader(
    "Choose a video file",
    type=['mp4', 'avi', 'mov', 'mkv'],
    help="Upload a video file (MP4, AVI, MOV, or MKV format)"
)

# Add button to use sample video
sample_video_path = os.path.join(os.getcwd(), 'drone_flying.mp4')
if os.path.exists(sample_video_path):
    if st.button('Use Sample Video (drone_flying.mp4)', key='use_sample'):
        with open(sample_video_path, 'rb') as f:
            sample_video_bytes = f.read()
        sample_file = BytesIO(sample_video_bytes)
        sample_file.name = 'drone_flying.mp4'
        st.session_state.uploaded_file = sample_file
        st.session_state.analysis_done = False
        st.session_state.unique_id = None
        st.session_state.video_filename = None
        st.session_state.video_path = None
        st.session_state.output_video_path = None
        st.session_state.trajectory_output_path = None
else:
    st.warning("Sample video (drone_flying.mp4) not found in the root directory.")

# Handle file upload
if uploaded_file is not None and uploaded_file != st.session_state.uploaded_file:
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
            if st.session_state.unique_id is None:
                st.session_state.unique_id = str(uuid.uuid4())
                st.session_state.video_filename = f"{st.session_state.unique_id}.mp4"
                st.session_state.video_path = os.path.join('uploads', st.session_state.video_filename)

                with open(st.session_state.video_path, 'wb') as f:
                    f.write(st.session_state.uploaded_file.getvalue())

            st.success(f"Video uploaded successfully: {st.session_state.uploaded_file.name}")

            if not st.session_state.analysis_done and st.button('🔍 Analyze Video', key='analyze'):
                with st.spinner('Processing video for detection...'):
                    current_dir = os.getcwd()
                    weights_path = os.path.join(current_dir, 'weights', 'best.pt')
                    output_project = os.path.join(current_dir, 'outputs')

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

                    st.session_state.output_video_path = os.path.join('outputs', st.session_state.unique_id, st.session_state.video_filename)

                with st.spinner('Processing video for trajectory tracing and LSTM analysis...'):
                    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
                    cap = cv2.VideoCapture(st.session_state.video_path)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    st.session_state.trajectory_output_path = os.path.join('outputs', st.session_state.unique_id, f"trajectory_{st.session_state.video_filename}")
                    out = cv2.VideoWriter(st.session_state.trajectory_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

                    tracker = Tracker(distance_function='mean_euclidean', distance_threshold=50)
                    trajectories = {}
                    frame_idx = 0

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        results = model(frame)
                        detections = []
                        for *xyxy, conf, cls in results.xyxy[0]:
                            if cls == 0 and conf > 0.5:
                                x1, y1, x2, y2 = map(int, xyxy)
                                center = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]])
                                detections.append(Detection(points=center, data={'bbox': [x1, y1, x2, y2]}))

                        tracked_objects = tracker.update(detections=detections)

                        for obj in tracked_objects:
                            if obj.id not in trajectories:
                                trajectories[obj.id] = []
                            trajectories[obj.id].append((frame_idx, obj.estimate[0]))

                        # Custom drawing to avoid deprecation warning
                        for obj in tracked_objects:
                            center = obj.estimate[0].astype(int)
                            cv2.circle(frame, (center[0], center[1]), 5, (0, 0, 255), -1)
                            if 'bbox' in obj.last_detection.data:
                                x1, y1, x2, y2 = obj.last_detection.data['bbox']
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"ID: {obj.id}", (center[0] + 10, center[1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        for track_id, points in trajectories.items():
                            if len(points) > 1:
                                pts = np.array([p[1] for p in points], dtype=np.int32).reshape((-1, 1, 2))
                                cv2.polylines(frame, [pts], isClosed=False, color=(0, 255, 0), thickness=2)

                        out.write(frame)
                        frame_idx += 1

                    cap.release()
                    out.release()

                    flight_pattern, behavior_classification, confidence_score = analyze_trajectory(
                        trajectories, fps, width, height, lstm_model, device, threshold=0.03
                    )

                    st.session_state.flight_pattern = flight_pattern
                    st.session_state.behavior_classification = behavior_classification
                    st.session_state.confidence_score = confidence_score

                st.session_state.analysis_done = True

            if st.session_state.analysis_done:
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

                st.subheader("Analysis Results")
                with st.expander("View Detailed Analysis", expanded=True):
                    st.markdown(f'''
                    🔍 **Analysis Results:**
                    - Flight Pattern: {st.session_state.flight_pattern}
                    - Behavior Classification: {st.session_state.behavior_classification}
                    - Confidence Score: {st.session_state.confidence_score:.2f}%
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
💡 This system uses YOLOv5 for UAV detection, Norfair for tracking, and an LSTM Autoencoder for trajectory analysis.
''')