import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class CoordinateSmoother:
    """Applies a moving average filter over a sliding window to smooth landmark coordinates."""
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = []  # List of np.arrays of shape (33, 4)
        
    def smooth(self, landmarks_array):
        self.history.append(landmarks_array)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        # Calculate mean over the sliding window
        return np.mean(np.array(self.history), axis=0)

def calculate_angle(a, b, c):
    """
    Calculates the angle at vertex b given three 2D coordinates.
    a: [x, y] of first point (e.g., Hip)
    b: [x, y] of vertex point (e.g., Knee)
    c: [x, y] of end point (e.g., Ankle)
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    angle_rad = np.arccos(cosine_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def draw_pose_skeleton(frame, landmarks_normalized, color=(0, 255, 0), thickness=3):
    """
    Draws custom pose connections on the frame using normalized landmarks.
    landmarks_normalized: np.array of shape (33, 4) with x, y, z, visibility
    """
    h, w = frame.shape[:2]
    pixel_coords = {}
    
    # Precompute pixel coordinates and draw joint circles
    for idx in range(33):
        x_px = int(landmarks_normalized[idx, 0] * w)
        y_px = int(landmarks_normalized[idx, 1] * h)
        pixel_coords[idx] = (x_px, y_px)
        
        # Only draw circle if visibility is high enough
        if landmarks_normalized[idx, 3] > 0.5:
            cv2.circle(frame, (x_px, y_px), 5, (255, 50, 50), -1)
            
    # Define pose skeleton connections (index mapping based on MediaPipe)
    connections = [
        (11, 12),  # Shoulder to shoulder
        (11, 13), (13, 15),  # Left arm
        (12, 14), (14, 16),  # Right arm
        (11, 23), (12, 24), (23, 24),  # Torso (Shoulders to Hips)
        (23, 25), (25, 27),  # Left leg
        (24, 26), (26, 28)   # Right leg
    ]
    
    # Draw connections
    for start, end in connections:
        # Check if both keypoints have decent visibility
        if landmarks_normalized[start, 3] > 0.5 and landmarks_normalized[end, 3] > 0.5:
            pt1 = pixel_coords[start]
            pt2 = pixel_coords[end]
            cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)

def process_video(input_video_path, output_video_path, model_path="pose_landmarker.task", window_size=5):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {input_video_path}")
        return None
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {input_video_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps:.2f}, Total Frames: {total_frames}")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Configure MediaPipe Pose Landmarker
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        output_segmentation_masks=False
    )
    
    # Initialize smoother and data storage
    smoother = CoordinateSmoother(window_size=window_size)
    frame_data = []
    
    # Tracks last valid landmarks to interpolate if detection is temporarily lost
    last_valid_raw_lm = None
    
    sum_left_vis = 0.0
    sum_right_vis = 0.0
    valid_frames_count = 0
    
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            # Manually calculate timestamp in ms
            timestamp_ms = int((frame_idx / fps) * 1000)
            
            # Detect
            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            # Initialize default values
            row = {
                'frame_idx': frame_idx,
                'left_knee': np.nan, 'left_hip': np.nan, 'left_elbow': np.nan,
                'right_knee': np.nan, 'right_hip': np.nan, 'right_elbow': np.nan,
                'left_knee_smooth': np.nan, 'left_hip_smooth': np.nan, 'left_elbow_smooth': np.nan,
                'right_knee_smooth': np.nan, 'right_hip_smooth': np.nan, 'right_elbow_smooth': np.nan,
                'left_vis': 0.0, 'right_vis': 0.0,
                'detected': 0
            }
            
            pose_landmarks_list = detection_result.pose_landmarks
            
            if pose_landmarks_list and len(pose_landmarks_list) > 0:
                landmarks = pose_landmarks_list[0]
                # Convert to np.array [x, y, z, visibility]
                raw_lm_arr = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks])
                last_valid_raw_lm = raw_lm_arr.copy()
                row['detected'] = 1
            elif last_valid_raw_lm is not None:
                # Use last valid detection to bridge small gaps
                raw_lm_arr = last_valid_raw_lm.copy()
            else:
                raw_lm_arr = None
                
            if raw_lm_arr is not None:
                # Apply smoothing
                smooth_lm_arr = smoother.smooth(raw_lm_arr)
                
                # Raw coordinates for angle calculation
                l_shoulder = raw_lm_arr[11, :2]
                l_elbow    = raw_lm_arr[13, :2]
                l_wrist    = raw_lm_arr[15, :2]
                l_hip      = raw_lm_arr[23, :2]
                l_knee     = raw_lm_arr[25, :2]
                l_ankle    = raw_lm_arr[27, :2]
                
                r_shoulder = raw_lm_arr[12, :2]
                r_elbow    = raw_lm_arr[14, :2]
                r_wrist    = raw_lm_arr[16, :2]
                r_hip      = raw_lm_arr[24, :2]
                r_knee     = raw_lm_arr[26, :2]
                r_ankle    = raw_lm_arr[28, :2]
                
                # Calculate raw angles
                row['left_knee'] = calculate_angle(l_hip, l_knee, l_ankle)
                row['left_hip'] = calculate_angle(l_shoulder, l_hip, l_knee)
                row['left_elbow'] = calculate_angle(l_shoulder, l_elbow, l_wrist)
                
                row['right_knee'] = calculate_angle(r_hip, r_knee, r_ankle)
                row['right_hip'] = calculate_angle(r_shoulder, r_hip, r_knee)
                row['right_elbow'] = calculate_angle(r_shoulder, r_elbow, r_wrist)
                
                # Smoothed coordinates for angle calculation
                sl_shoulder = smooth_lm_arr[11, :2]
                sl_elbow    = smooth_lm_arr[13, :2]
                sl_wrist    = smooth_lm_arr[15, :2]
                sl_hip      = smooth_lm_arr[23, :2]
                sl_knee     = smooth_lm_arr[25, :2]
                sl_ankle    = smooth_lm_arr[27, :2]
                
                sr_shoulder = smooth_lm_arr[12, :2]
                sr_elbow    = smooth_lm_arr[14, :2]
                sr_wrist    = smooth_lm_arr[16, :2]
                sr_hip      = smooth_lm_arr[24, :2]
                sr_knee     = smooth_lm_arr[26, :2]
                sr_ankle    = smooth_lm_arr[28, :2]
                
                # Calculate smoothed angles
                row['left_knee_smooth'] = calculate_angle(sl_hip, sl_knee, sl_ankle)
                row['left_hip_smooth'] = calculate_angle(sl_shoulder, sl_hip, sl_knee)
                row['left_elbow_smooth'] = calculate_angle(sl_shoulder, sl_elbow, sl_wrist)
                
                row['right_knee_smooth'] = calculate_angle(sr_hip, sr_knee, sr_ankle)
                row['right_hip_smooth'] = calculate_angle(sr_shoulder, sr_hip, sr_knee)
                row['right_elbow_smooth'] = calculate_angle(sr_shoulder, sr_elbow, sr_wrist)
                
                # Calculate side visibilities (average of hip, knee, ankle visibilities)
                row['left_vis'] = float(np.mean(raw_lm_arr[[23, 25, 27], 3]))
                row['right_vis'] = float(np.mean(raw_lm_arr[[24, 26, 28], 3]))
                
                sum_left_vis += row['left_vis']
                sum_right_vis += row['right_vis']
                valid_frames_count += 1
                
                # Draw customized pose skeleton on the frame
                # Green connections indicate smoothed tracker active
                draw_pose_skeleton(frame, smooth_lm_arr, color=(0, 220, 0), thickness=3)
                
                # Determine tracking side based on visibility
                if row['left_vis'] > row['right_vis']:
                    side = "Left"
                    knee_ang = row['left_knee_smooth']
                    hip_ang = row['left_hip_smooth']
                    elbow_ang = row['left_elbow_smooth']
                else:
                    side = "Right"
                    knee_ang = row['right_knee_smooth']
                    hip_ang = row['right_hip_smooth']
                    elbow_ang = row['right_elbow_smooth']
                
                # Rule-based classification
                # In squats, both knee and hip angles flex significantly
                is_squatting = knee_ang < 140 and hip_ang < 140
                activity = "Squatting" if is_squatting else "Standing"
                
                # Draw overlays
                cv2.putText(frame, f"Tracking Side: {side}", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Knee Angle: {knee_ang:.1f} deg", (30, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Hip Angle:  {hip_ang:.1f} deg", (30, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Elbow Angle:{elbow_ang:.1f} deg", (30, 200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                
                color = (0, 0, 255) if activity == "Squatting" else (0, 255, 255)
                cv2.putText(frame, f"Activity: {activity}", (30, 270), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3, cv2.LINE_AA)
            
            out.write(frame)
            frame_data.append(row)
            frame_idx += 1
            
            if frame_idx % 50 == 0:
                print(f"Processed frame {frame_idx}/{total_frames}...")
                
    cap.release()
    out.release()
    print("Video processing completed!")
    
    avg_left_vis = sum_left_vis / valid_frames_count if valid_frames_count > 0 else 0
    avg_right_vis = sum_right_vis / valid_frames_count if valid_frames_count > 0 else 0
    print(f"Average Left Side Keypoint Visibility: {avg_left_vis:.3f}")
    print(f"Average Right Side Keypoint Visibility: {avg_right_vis:.3f}")
    
    df = pd.DataFrame(frame_data)
    df.to_csv("pose_data.csv", index=False)
    print("Metrics written to pose_data.csv")

if __name__ == "__main__":
    process_video("squat_sample.mp4", "output_pose.mp4")
