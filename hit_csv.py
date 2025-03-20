import pandas as pd
import numpy as np
import cv2
import tqdm
from pathlib import Path

def generate_hit(folder: Path, video_path: Path, result: pd.Series, frames: pd.Series, fps = 25):
    
    # Convert time to frame number
    def time_to_frame(time_str, offset):
        hours, minutes, seconds = map(float, time_str.split(':'))
        seconds += offset
        total_seconds = hours * 3600 + minutes * 60 + seconds
        return int(total_seconds * fps)
    
    # Save clips in videos/{name}
    output_dir = folder / 'hits'
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Extract clips
    for rally, times in tqdm.tqdm(result.items(), desc='Generating hit labels', total=len(result)):
        start = time_to_frame(times[0], -1) # Start 1 second before the rally
        end = time_to_frame(times[-1], 3)   # End 2 seconds after the last ball round

        cap.set(cv2.CAP_PROP_POS_FRAMES, start) 

        hit_labels = []
        idx = 0
        
        for frame_idx in range(start, end):
            ret, frame = cap.read()
            if not ret:
                break
            hit = 1 if frame_idx in frames[rally] else 0
            hit_labels.append([idx, hit])
            idx += 1

        hit_df = pd.DataFrame(hit_labels, columns=['frame', 'hit'])
        filename = output_dir / f'clip_{rally}_hit.csv'
        hit_df.to_csv(filename, index=False)
        #print(f"[FINISH] hit_{rally}.csv saved.")


    cap.release()
    print(f'[FINISH] Clips saved to {output_dir}')

# Leading to the folder
fps = [30, 30, 30, 30, 25, 25, 25, 30, 30, 30, 30, 30, 30, 25, 25, 25, 30, 30, 30, 30, 30, 30, 25, 25, 30, 30, 30, 30, 30, 30, 30]
for i in range(2, 32):
    folder = Path(f"match{i}/")
    data = pd.read_csv(folder / 'set.csv', encoding='ISO-8859-1')
    time = data.groupby('rally')['time'].apply(list)
    frames = data.groupby('rally')['frame_num'].apply(list)
    #print(frames)
    video_path = folder / f"videoplayback.mp4"
    generate_hit(folder, video_path, time, frames, fps[i-1])