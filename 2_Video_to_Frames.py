import cv2
import time
import os

file = r"【聽牌戰】第14屆棋王挑戰賽第六局 許皓鋐棋王 vs. 林君諺八段_720p.mp4"
output_dir = './go_images_2fps'

# Check if file exists
if not os.path.isfile(file):
    print(f"Error: Video file not found: {file}")
    exit()

# Ensure output directory exists
if not os.path.exists(output_dir):
    try:
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    except Exception as e:
        print(f"Unable to create directory: {e}")
        exit()

# Try to open video
cap = cv2.VideoCapture(file)
if not cap.isOpened():
    print("Error: Unable to open video file")
    exit()

# Get video information
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = int(total_frames / fps)
print(f"Video FPS: {fps}, Total Frames: {total_frames}, Duration: {duration} seconds")

# Test reading first frame
ret, frame = cap.read()
if not ret:
    print("Cannot read first frame, video might be corrupted or unsupported")
    exit()
else:
    print("First frame read successfully")
    # Reset to first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

frame_count = 0
start_time = time.time()

# Modify to save one frame per half-second
current_half_sec = -1
frame_index = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video ended or reading failed")
        break
    
    # Calculate current half-second 
    # Multiply by 2 to create units for every 0.5 seconds
    # e.g., 0.0-0.49 seconds → 0, 0.5-0.99 seconds → 1, 1.0-1.49 seconds → 2, and so on
    current_time_sec = frame_index / fps
    current_frame_half_sec = int(current_time_sec * 2)
    
    # Save frame if entering a new half-second
    if current_frame_half_sec > current_half_sec:
        current_half_sec = current_frame_half_sec
        try:
            # Convert half-second count back to actual seconds
            seconds = current_half_sec // 2
            is_half = (current_half_sec % 2) == 1  # Check if it's x.5 seconds
            
            # Generate timestamp, format: HH-MM-SS-5 (add "-5" if half-second)
            timestamp = time.strftime('%H-%M-%S', time.gmtime(seconds))
            if is_half:
                timestamp += "-5"  # Add "-5" to represent 0.5 seconds
                
            filename = f'frame_{timestamp}.png'
            save_path = os.path.join(output_dir, filename)
            
            success = cv2.imwrite(save_path, frame)
            if success:
                frame_count += 1
                print(f"Saved: {save_path}")
            else:
                print(f"Unable to write file: {save_path}")
        except Exception as e:
            print(f"Error during file saving: {e}")
    
    frame_index += 1

end_time = time.time()
print(f"Extracted {frame_count} images, Time taken: {end_time - start_time:.2f} seconds")
cap.release()