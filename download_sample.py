import os
import requests

def download_file(url, output_filename):
    print(f"Attempting to download from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        
        print(f"Saving to {os.path.abspath(output_filename)} (Size: {total_size / (1024*1024):.2f} MB)...")
        
        with open(output_filename, 'wb') as file:
            downloaded = 0
            for data in response.iter_content(block_size):
                file.write(data)
                downloaded += len(data)
                if total_size > 0 and downloaded % (500 * block_size) == 0:
                    percent = (downloaded / total_size) * 100
                    print(f"Progress: {percent:.1f}% ({downloaded / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB)", end='\r')
        
        print(f"\nDownload of {output_filename} complete successfully!")
        return True
    except Exception as e:
        print(f"\nFailed to download {output_filename}. Error: {e}")
        return False

def download_video_and_model():
    video_url = "https://raw.githubusercontent.com/Furkan-Gulsen/Sport-With-AI/main/videos/squat.mp4"
    model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
    
    # Download video
    if not os.path.exists("squat_sample.mp4"):
        download_file(video_url, "squat_sample.mp4")
    else:
        print("squat_sample.mp4 already exists.")
        
    # Download model
    if not os.path.exists("pose_landmarker.task"):
        download_file(model_url, "pose_landmarker.task")
    else:
        print("pose_landmarker.task already exists.")

if __name__ == "__main__":
    download_video_and_model()
