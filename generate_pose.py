import os
from detect_pose import process

log_file = "logTest.txt"

with open(log_file, "w") as f:
    for id in range(1, 5):
        court_dir = f'matches/test_match{id}/court'
        rally_dir = f'matches/test_match{id}/rally_video'
        poses_dir = f'matches/test_match{id}/poses'

        f.write(f'[Match {id}]\n\t')

        for filename in os.listdir(court_dir):
            name, _ = os.path.splitext(filename)

            f.write(f'{name} ')

            court_file = f'{court_dir}/{filename}'
            video_path = f'{rally_dir}/{name}.mp4'
            process(video_path, poses_dir, court_file)

            f.flush()

        f.write("\n\n")
        