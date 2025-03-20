import cv2
import csv
import os
import argparse
import numpy as np
from mmpose.apis import MMPoseInferencer
from mayavi import mlab
from tqdm import tqdm
from pathlib import Path


def normalize_joints(
    arr: np.ndarray,
    bbox: np.ndarray,
    v_height=None,
    center_align=False,
):
    '''
    - `arr`: (m, J, 2), m=2.
    - `bbox`: (m, 4), m=2.
    
    Output: (m, J, 2), m=2.
    '''
    # If v_height == None and center_align == False,
    # this normalization method is same as that used in TemPose.
    if v_height:
        dist = v_height / 4
    else:  # bbox diagonal dist
        dist = np.linalg.norm(bbox[:, 2:] - bbox[:, :2], axis=-1, keepdims=True)
    
    arr_x = arr[:, :, 0]
    arr_y = arr[:, :, 1]
    x_normalized = np.where(arr_x != 0.0, (arr_x - bbox[:, None, 0]) / dist, 0.0)
    y_normalized = np.where(arr_y != 0.0, (arr_y - bbox[:, None, 1]) / dist, 0.0)

    if center_align:
        center = (bbox[:, :2] + bbox[:, 2:]) / 2
        c_normalized = (center - bbox[:, :2]) / dist
        x_normalized -= c_normalized[:, None, 0]
        y_normalized -= c_normalized[:, None, 1]

    return np.stack((x_normalized, y_normalized), axis=-1)


def demo_human_2d_and_3d(img_path):
    inferencer_2d = MMPoseInferencer('human')
    inferencer_3d = MMPoseInferencer(pose3d='human3d')
    result_generator_2d = inferencer_2d(img_path, show=False)
    result_generator_3d = inferencer_3d(img_path, show=False)
    for result_2d, result_3d in zip(result_generator_2d, result_generator_3d):
        for e_2d, e_3d in zip(result_2d['predictions'][0],
                              result_3d['predictions'][0]):  # batch_size=1 (default)
            
            ## 2d
            keypoints_2d = np.array(e_2d['keypoints'])[None, :]
            bbox = np.concatenate([
                keypoints_2d.min(1),
                keypoints_2d.max(1)
            ], axis=-1)
            
            keypoints_2d_normalized = normalize_joints(
                keypoints_2d, bbox,
                center_align=True
            )[0]

            coords = np.concatenate([
                keypoints_2d_normalized[:, 0:1],
                keypoints_2d_normalized[:, 1:],
                np.zeros((len(e_2d['keypoints']), 1)),
            ], axis=1)
            
            # 創建一個 3D 圖形
            fig = mlab.figure(figure='My Figure')

            # 繪製散點圖
            pts = mlab.points3d(coords[:, 0], coords[:, 1], coords[:, 2], scale_factor=0.03)

            # 為每個點添加文字標籤
            for i, (x, y, z) in enumerate(coords):
                mlab.text3d(x, y, z, str(i), scale=0.02)

            # 顯示圖形
            mlab.show(stop=True)

            ## 3d
            coords = np.array(e_3d['keypoints'])

            # x
            print('0 -> 1 :', coords[1] - coords[0])
            print('0 -> 4 :', coords[4] - coords[0])

            # y
            print('12 -> 13 :', coords[13] - coords[12])

            # z
            print('0 -> 7 :', coords[7] - coords[0])

            # 創建一個 3D 圖形
            fig = mlab.figure(figure='My Figure')

            # 繪製散點圖
            pts = mlab.points3d(coords[:, 0], coords[:, 1], coords[:, 2], scale_factor=0.1)

            # 為每個點添加文字標籤
            for i, (x, y, z) in enumerate(coords):
                mlab.text3d(x, y, z, str(i), scale=0.1)

            # 顯示圖形
            mlab.show(stop=True)


def demo_human_2d(img_path):
    inferencer = MMPoseInferencer('human')
    x = []
    y = []
    result_generator = inferencer(img_path, show=False)
    for result in result_generator:
        for e in result['predictions'][0]:  # batch_size=1 (default)
            keypoints_2d = np.array(e['keypoints'])[None, :]
            bbox = np.concatenate([
                keypoints_2d.min(1),
                keypoints_2d.max(1)
            ], axis=-1)
            
            keypoints_2d_normalized = normalize_joints(
                keypoints_2d, bbox,
                center_align=True
            )[0]

            coords = np.concatenate([
                keypoints_2d_normalized[:, 0:1],
                keypoints_2d_normalized[:, 1:],
                np.zeros((len(e['keypoints']), 1)),
            ], axis=1)
            
            # 創建一個 3D 圖形
            fig = mlab.figure(figure='My Figure')

            # 繪製散點圖
            pts = mlab.points3d(coords[:, 0], coords[:, 1], coords[:, 2], scale_factor=0.03)

            # 為每個點添加文字標籤
            for i, (x, y, z) in enumerate(coords):
                mlab.text3d(x, y, z, str(i), scale=0.02)

            # 顯示圖形
            mlab.show(stop=True)


def demo_human_3d(img_path):
    inferencer = MMPoseInferencer(pose3d='human3d')
    result_generator = inferencer(img_path, show=False)

    for result in result_generator:
        for e in result['predictions'][0]:  # batch_size=1 (default)
            coords = np.array(e['keypoints'])

            # x
            print('0 -> 1 :', coords[1] - coords[0])
            print('0 -> 4 :', coords[4] - coords[0])

            # y
            print('12 -> 13 :', coords[13] - coords[12])

            # z
            print('0 -> 7 :', coords[7] - coords[0])

            # 創建一個 3D 圖形
            fig = mlab.figure(figure='My Figure')

            # 繪製散點圖
            pts = mlab.points3d(coords[:, 0], coords[:, 1], coords[:, 2], scale_factor=0.05)

            # 為每個點添加文字標籤
            for i, (x, y, z) in enumerate(coords):
                mlab.text3d(x, y, z, str(i), scale=0.05)

            # 顯示圖形
            mlab.show(stop=True)


def test_bug(inferencer, p):
    result_generator = inferencer(str(p), show=False)
    for result in result_generator:
        pass


def no_bug(p):
    inferencer = MMPoseInferencer(pose3d='human3d')
    result_generator = inferencer(str(p), show=False)
    for result in result_generator:
        pass

def save_player_keypoints(player_data, output_path):
    """Save player keypoints into a CSV file."""
    with open(output_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        headers = ['frame']
        num_keypoints = len(player_data[0]['keypoints']) if player_data else 0
        headers += [f'x{i}' for i in range(num_keypoints)] + [f'y{i}' for i in range(num_keypoints)]
        csv_writer.writerow(headers)
        
        for frame_data in player_data:
            frame_idx = frame_data['frame']
            keypoints = frame_data['keypoints']
            row = [frame_idx]
            for kp in keypoints:
                row.extend(kp)  # Append x, y, and confidence
            csv_writer.writerow(row)

    print(f"Keypoints saved to {output_path}")

def read_court_corners(file_path):
    """
    Read the first four court corner coordinates from a .txt file.
    The first four lines should contain two numbers separated by a semicolon:
    1st line: Upper-left corner
    2nd line: Bottom-left corner
    3rd line: Bottom-right corner
    4th line: Upper-right corner
    """
    corners = []
    
    with open(file_path, 'r') as file:
        for _ in range(4):  # Read only the first 4 lines
            line = file.readline().strip()
            if not line:
                break  # Stop if there are fewer than 4 lines
            x, y = map(float, line.split(';'))
            corners.append((x, y))
    
    if len(corners) == 4:
        ordered_corners = [corners[0], corners[3], corners[1], corners[2]]  # Reorder to UL, UR, BL, BR
        return np.array(ordered_corners, dtype=np.int32)
    else:
        print("Error: Could not read all 4 court corners.")
        return np.array(corners, dtype=np.int32)



def is_inside_court(foot_positions, court_polygon, threshold=30):
    """Check if a player is inside the court based on their foot positions."""
    right_heel, left_heel = foot_positions
    
    # Get signed distance from the court for each foot
    dist_right = cv2.pointPolygonTest(court_polygon, right_heel, True)
    dist_left = cv2.pointPolygonTest(court_polygon, left_heel, True)

    # Check if inside OR within threshold distance
    return (dist_right >= -threshold or dist_left >= -threshold)

def visualize_video_estimated(in_path, csv_output_dir='pose_data.csv', cap=None, court_corners=None):
    inferencer = MMPoseInferencer('human')
    result_generator = inferencer(in_path, csv_output_dir)

    video_name = os.path.splitext(os.path.basename(in_path))[0]
    os.makedirs(csv_output_dir, exist_ok=True)
    output_video_path = os.path.join(csv_output_dir, f"{video_name}_visualized_output.mp4")

    #Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Data for CSVs
    bottom_player_data = []
    top_player_data = []

    # Process each frame
    for frame_idx, result in enumerate(result_generator):
        ret, frame = cap.read()
        if not ret:
            print(f"Frame {frame_idx}: Could not read frame, outputting null frame...")
            null_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            out_video.write(null_frame)
            continue

        predictions = result['predictions'][0] 
        pred_instances = predictions.pred_instances

        # Get keypoints for all detected people
        keypoints_list = pred_instances.keypoints

        players = []
        for i in range(len(pred_instances.bboxes)):
            keypoints = keypoints_list[i]  # Keypoints for the i-th person

            # Store the player's foot position and keypoints
            players.append({
                'foot_position': (keypoints[15], keypoints[16]),  # Store both heels
                'keypoints': keypoints
            })

        # Filter players inside the court using their foot positions
        players_inside_court = [
            player for player in players
            if is_inside_court(player['foot_position'], court_corners)
        ]

        # ** Consideration **
        if len(players_inside_court) < 2:
            out_video.write(frame)
            bottom_player_data.append({'frame': frame_idx, 'keypoints': [(0, 0) for _ in range(17)]})
            top_player_data.append({'frame': frame_idx, 'keypoints': [(0, 0) for _ in range(17)]})
            continue
        players_inside_court = sorted(players_inside_court, key=lambda p: p['foot_position'][1][1], reverse=True)
        bottom_player = players_inside_court[0]
        top_player = players_inside_court[1]

        # Save keypoints to respective lists
        bottom_player_data.append({'frame': frame_idx, 'keypoints': bottom_player['keypoints']})
        top_player_data.append({'frame': frame_idx, 'keypoints': top_player['keypoints']})
        
        # Draw keypoints and bounding boxes
        for player, color in [(top_player, (0, 255, 0)), (bottom_player, (0, 0, 255))]:
            keypoints = player['keypoints']

            # Draw keypoints
            for kp in keypoints:
                x, y = map(int, kp[:2])
                cv2.circle(frame, (x, y), 5, color, -1)

        corner_colors = [
            (0, 0, 255),   # Upper-left (Red)
            (0, 255, 0),   # Upper-right (Green)
            (255, 0, 0),   # Bottom-left (Blue)
            (255, 255, 0)  # Bottom-right (Cyan)
        ]

        # Draw each corner with a different color
        for i, corner in enumerate(court_corners):
            x, y = map(int, corner)
            cv2.circle(frame, (x, y), 8, corner_colors[i], -1)
        out_video.write(frame)

    # Release resources
    cap.release()
    out_video.release()
    print(f"Visualized video saved to {output_video_path}")

    bottom_file = os.path.join(csv_output_dir, f"{video_name}_bottom.csv")
    top_file = os.path.join(csv_output_dir, f"{video_name}_top.csv")

    save_player_keypoints(bottom_player_data, bottom_file)
    save_player_keypoints(top_player_data, top_file)  

def process(video_path, csv_output_dir, court_file):

    # Read court corners
    court_corners = read_court_corners(court_file)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
    else:
        visualize_video_estimated(
            in_path=video_path,
            csv_output_dir = csv_output_dir,
            cap=cap,
            court_corners=court_corners
        )
    cap.release()

