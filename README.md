# Pose Estimation & Activity Classification Pipeline

This repository implements a complete Computer Vision pipeline to detect human body poses, perform coordinate smoothing, track joint geometry, and classify physical activities (Standing vs. Squatting) in real-time. It satisfies the criteria for the Course Learning Outcome **CLO-3** (Complex Computing Problem).

## Suggested Repository Name
`Pose-Estimation-Activity-Classifier`

---

## Project Structure
```text
├── pose_landmarker.task   # Downloaded MediaPipe Pose Landmarker model bundle (Not uploaded on githup due to its large size)
├── squat_sample.mp4       # Raw input video clip
├── output_pose.mp4        # Annotated video output with skeleton overlay & live stats
├── pose_data.csv          # Extracted raw and smoothed joint angle metrics per frame
├── angle_tracking.png     # High-resolution joint angle tracking visualization
├── screenshot_20.png      # Representative screenshot (Standing state)
├── screenshot_90.png      # Representative screenshot (Squatting state)
├── report.pdf             # Academic 2-page report (PDF format)
├── download_sample.py     # Script to download the video and pose landmarker model
├── pose_analysis.py       # Core pose landmark detection & angle calculation script
└── evaluate_and_plot.py   # Code for manual ground truth comparison, plotting, & metrics
```

---

## Methodology

### 1. Pose Detection & Coordinate Smoothing
- We use the state-of-the-art **Google MediaPipe Tasks API (`PoseLandmarker`)** with the heavy accuracy model (`pose_landmarker_heavy.task`) to extract 33 landmarks frame-by-frame.
- A **5-frame Simple Moving Average (SMA) filter** is applied to the coordinates of key landmarks. This temporal smoothing removes high-frequency jitter in keypoint tracking, creating stable and reliable angle measurements.
- We draw a custom skeleton overlay (green bones and red joint nodes) on each frame and output the annotated video to `output_pose.mp4`.

### 2. Joint Angle Computation
- Using vector geometry, we compute 2D angles at key joints. For a joint vertex $B$ with adjacent points $A$ and $C$:
  $$\vec{BA} = A - B, \quad \vec{BC} = C - B$$
  $$\theta = \arccos\left(\frac{\vec{BA} \cdot \vec{BC}}{\|\vec{BA}\| \|\vec{BC}\|}\right) \times \frac{180}{\pi}$$
- The pipeline tracks three key angles over time: **Knee Angle** (Hip-Knee-Ankle), **Hip Angle** (Shoulder-Hip-Knee), and **Elbow Angle** (Shoulder-Elbow-Wrist).
- The pipeline automatically tracks the side facing the camera based on visibility. In the sample video, the **Right Side** has **96.8% visibility** and is used as the primary tracking source.

### 3. Rule-Based Classification
- A heuristic classifier distinguishes between **Standing** and **Squatting**:
  - **Squatting Rule**: `(Knee Angle < 140°)` AND `(Hip Angle < 140°)`
  - **Standing Rule**: Otherwise

---

## Performance Evaluation
The classifier predictions were compared frame-by-frame against a manually labeled ground truth representing the 5 squat repetitions:

- **Accuracy**: 82.29%
- **Precision**: 100.00% (Zero false positives; standing state is never misclassified as squatting)
- **Recall**: 72.73% (Conservative classification during the start/end of the descent)
- **F1-Score**: 84.21%

### Detected Transitions
- **Frame 64**: Transitioned to Squatting
- **Frame 114**: Transitioned to Standing
- **Frame 170**: Transitioned to Squatting
- **Frame 214**: Transitioned to Standing
- **Frame 272**: Transitioned to Squatting
- **Frame 329**: Transitioned to Standing
- **Frame 385**: Transitioned to Squatting
- **Frame 440**: Transitioned to Standing
- **Frame 494**: Transitioned to Squatting
- **Frame 552**: Transitioned to Standing

---

## Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/MKS-24/Pose-Estimation-Activity-Classifier.git
   cd Pose-Estimation-Activity-Classifier
   ```

2. **Install Dependencies**:
   ```bash
   pip install opencv-python mediapipe numpy pandas matplotlib reportlab scikit-learn requests
   ```
---

## Running the Pipeline

To run the pipeline from scratch and regenerate all outputs:

1. **Process Video and Extract Keypoints**:
   ```bash
   python pose_analysis.py
   ```
   *Creates `output_pose.mp4` and `pose_data.csv`.*

2. **Evaluate Performance & Generate Plot**:
   ```bash
   python evaluate_and_plot.py
   ```
   *Prints metrics, prints transitions, and saves `angle_tracking.png`.*
