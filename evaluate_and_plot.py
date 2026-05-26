import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_and_plot():
    # Load data
    df = pd.read_csv("pose_data.csv")
    
    # We will use the Right side as it has 96.8% visibility (Left side has only 47.0%)
    knee_key = "right_knee_smooth"
    hip_key = "right_hip_smooth"
    elbow_key = "right_elbow_smooth"
    
    # Check if there are missing values and fill them
    df[knee_key] = df[knee_key].ffill().bfill()
    df[hip_key] = df[hip_key].ffill().bfill()
    df[elbow_key] = df[elbow_key].ffill().bfill()
    
    # Define Ground Truth (manually labeled based on movement transitions)
    # The user starts the squat motion around frame 55, reaches the bottom around frame 90, and recovers by frame 125.
    # This pattern repeats for 5 repetitions.
    gt_intervals = [(55, 125), (160, 225), (265, 340), (375, 450), (485, 558)]
    
    num_frames = len(df)
    ground_truth = np.zeros(num_frames, dtype=int)
    for start, end in gt_intervals:
        # Clamp to bounds
        start_idx = max(0, start)
        end_idx = min(num_frames - 1, end)
        ground_truth[start_idx:end_idx + 1] = 1 # 1 for Squatting, 0 for Standing
        
    df['ground_truth'] = ground_truth
    
    # Rule-Based Classifier
    # We classify as Squatting if the Knee Angle < 140 AND Hip Angle < 140
    # Otherwise Standing.
    predictions = []
    for idx, row in df.iterrows():
        knee = row[knee_key]
        hip = row[hip_key]
        if knee < 140 and hip < 140:
            predictions.append(1) # Squatting
        else:
            predictions.append(0) # Standing
            
    df['predicted'] = predictions
    
    # Evaluation Metrics
    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions)
    recall = recall_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions)
    cm = confusion_matrix(ground_truth, predictions)
    
    print("=== Evaluation Metrics ===")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1-Score:  {f1:.4f} ({f1 * 100:.2f}%)")
    print("\nConfusion Matrix:")
    print(f"True Negative (Standing):  {cm[0, 0]} | False Positive: {cm[0, 1]}")
    print(f"False Negative:            {cm[1, 0]} | True Positive (Squatting):  {cm[1, 1]}")
    
    # Find transition frames
    transitions = []
    prev_pred = df['predicted'].iloc[0]
    for idx, row in df.iterrows():
        curr_pred = row['predicted']
        if curr_pred != prev_pred:
            state = "Squatting" if curr_pred == 1 else "Standing"
            transitions.append((row['frame_idx'], state))
            prev_pred = curr_pred
            
    print("\nDetected Activity Transitions:")
    for frame, state in transitions:
        print(f"Frame {int(frame):3d}: Transitioned to {state}")
        
    # Plotting code - High Quality Visualizations
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    frames = df['frame_idx']
    
    # Plot 1: Joint Angles over Time (Raw and Smoothed)
    # Right Knee Angle
    ax1.plot(frames, df['right_knee'], color='salmon', alpha=0.3, label='Raw Knee Angle')
    ax1.plot(frames, df[knee_key], color='crimson', linewidth=2, label='Smoothed Knee Angle')
    
    # Right Hip Angle
    ax1.plot(frames, df['right_hip'], color='skyblue', alpha=0.3, label='Raw Hip Angle')
    ax1.plot(frames, df[hip_key], color='dodgerblue', linewidth=2, label='Smoothed Hip Angle')
    
    # Right Elbow Angle
    ax1.plot(frames, df['right_elbow'], color='lightgreen', alpha=0.3, label='Raw Elbow Angle')
    ax1.plot(frames, df[elbow_key], color='forestgreen', linewidth=2, label='Smoothed Elbow Angle')
    
    # Add classification threshold line
    ax1.axhline(y=140, color='gray', linestyle='--', linewidth=1.5, label='Squat Threshold (140°)')
    
    ax1.set_title("Joint Angles Tracking & Keypoint Smoothing", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Angle (Degrees)", fontsize=12)
    ax1.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    ax1.set_ylim(0, 200)
    
    # Plot 2: Activity Classification (Ground Truth vs Prediction)
    ax2.plot(frames, df['ground_truth'], color='forestgreen', linewidth=2.5, label='Manual Ground Truth')
    ax2.plot(frames, df['predicted'], color='crimson', linestyle=':', linewidth=2.5, label='Rule-Based Prediction')
    
    # Shade the regions where Squatting occurs (Ground Truth = 1)
    # We can find the intervals and fill them
    in_squat = False
    start_f = 0
    for idx, val in enumerate(ground_truth):
        if val == 1 and not in_squat:
            start_f = idx
            in_squat = True
        elif val == 0 and in_squat:
            ax2.axvspan(start_f, idx-1, color='green', alpha=0.15, label='Ground Truth Squat' if 'Ground Truth Squat' not in ax2.get_legend_handles_labels()[1] else "")
            in_squat = False
    if in_squat:
        ax2.axvspan(start_f, len(ground_truth)-1, color='green', alpha=0.15)
        
    ax2.set_title("Activity Classification: Standing (0) vs Squatting (1)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Frame Index (30 FPS)", fontsize=12)
    ax2.set_ylabel("Class Label", fontsize=12)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Standing (0)', 'Squatting (1)'])
    ax2.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    ax2.set_ylim(-0.2, 1.2)
    
    plt.tight_layout()
    plt.savefig("angle_tracking.png", dpi=300)
    print("\nTracking plot saved to angle_tracking.png")
    
    # Save evaluation summary to a text file for the report
    with open("evaluation_summary.txt", "w") as f:
        f.write("=== ACTIVITY CLASSIFICATION PERFORMANCE ===\n")
        f.write(f"Accuracy:  {accuracy * 100:.2f}%\n")
        f.write(f"Precision: {precision * 100:.2f}%\n")
        f.write(f"Recall:    {recall * 100:.2f}%\n")
        f.write(f"F1-Score:  {f1 * 100:.2f}%\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"True Negative (Standing):  {cm[0, 0]}\n")
        f.write(f"False Positive (Standing classified as Squatting): {cm[0, 1]}\n")
        f.write(f"False Negative (Squatting classified as Standing): {cm[1, 0]}\n")
        f.write(f"True Positive (Squatting classified as Squatting): {cm[1, 1]}\n\n")
        f.write("Detected Activity Transitions:\n")
        for frame, state in transitions:
            f.write(f"Frame {int(frame):3d}: Transitioned to {state}\n")

if __name__ == "__main__":
    evaluate_and_plot()
