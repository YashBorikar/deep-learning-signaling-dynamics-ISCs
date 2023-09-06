# Import libraries
import supervision
from supervision.draw.color import Color
from supervision.video.dataclasses import VideoInfo                     # Video Processing
from supervision.video.source import get_video_frames_generator         # Frame generation function
from supervision.video.sink import VideoSink                            # Video Processing
from supervision.tools.detections import Detections, BoxAnnotator       # Visualising Detections

import os
import sys
import pandas as pd
import numpy as np
import random
import cv2                    # Trajectories
from ultralytics import YOLO  # for YoloV8 algorithm
import ByteTrack.yolox                  # ByteTrack Dependacy
from tqdm.notebook import tqdm
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
from pathlib import Path


# Optimization and Customization ByteTrack Algorithm

@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh = 0.0001          # High_threshold
    track_buffer = 30              # Number of frame lost tracklets are kept
    match_thresh = 0.8             # Matching threshold for first stage linear assignment
    aspect_ratio_thresh = 1.0      # Minimum bounding box aspect ratio
    min_box_area = 1.0             # Minimum bounding box area
    mot20 = True                  # If used, bounding boxes are not clipped.

# Code to combine cell detection model (YoloV8) output box with corresponding tracking algorithm (ByteTrack) box
def detections2boxes(detections):#
  '''Converts detections format to use in match_detections_with_tracks function'''
  boxes = np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))
  return boxes

def tracks2boxes(tracks):
  '''Convert List[STrack] format to use in match_detections_with_tracks function'''
  boxes = []
  for track in tracks:
    boxes.append(track.tlbr)
  boxes = np.array(boxes, dtype=float)
  return boxes

def match_detections_with_tracks(detections, tracks):
  '''Comparing bounding boxes with predictions'''
  iou_values = []
  true_positive = 0   # True detection by tracker
  false_positive = 0  # True detection by tracker, false detection by Yolo
  threshold= 0.5      # threshold value of IoU

  if not np.any(detections.xyxy) or len(tracks) == 0:
      return np.empty((0,))

  tracks_boxes = tracks2boxes(tracks=tracks)
  iou = box_iou_batch(tracks_boxes, detections.xyxy)
  track2detection = np.argmax(iou, axis=1)
  tracker_ids = [None] * len(detections)

  for tracker_index, detection_index in enumerate(track2detection):
      if iou[tracker_index, detection_index] != 0:
        iou_values.append(iou[tracker_index, detection_index])
        tracker_ids[detection_index] = tracks[tracker_index].track_id
        if iou[tracker_index, detection_index] < 0.50:
          false_positive += 1
        else:
          true_positive += 1
  return tracker_ids, iou_values, true_positive, false_positive

# Importing Cell Detection Model (YoloV8)

# Using YOLOV8-L Model Trained weights
detection_model = 'best.pt'

model = YOLO(detection_model)
model_class = model.model.names
cell_index = [0]
model.fuse()
# print(model_class)

# ByteTracker Initialisation
byte_tracker = BYTETracker(BYTETrackerArgs())

def draw_trajectory(frame, trajectories) -> np.ndarray:
  '''Function to draw trajectory following coordinates
   of individual Cell/Tracking ID'''
  for cell_id, coordinates in trajectories.items():
    for i in range(1, len(coordinates)):
      start = coordinates[i]
      end = coordinates[i - 1]
      cv2.line(
          frame,
          start,
          end,
          color = (255, 255, 255),
          thickness = 1,
      )
  return frame

# Code to Track Cell and save to output file
def main():
    if len(sys.argv) != 2:
        print("Usage: python script_name.py video_directory")
        return
    
    video_dir = os.path.join(os.getcwd(), sys.argv[1])

    # Output file directory
    tracking_output = 'output.avi'

    #### Implementtion of tracking algorithmn (ByteTrack)
    # Video File Details
    print('Video Details')
    VideoInfo.from_video_path(video_dir)

    # Video initialisation
    video_info = VideoInfo.from_video_path(video_dir)

    # Video to frame generator
    generator = get_video_frames_generator(video_dir)

    # Box annotator configurations
    box_annotator = BoxAnnotator(color=Color(255, 255, 255),
                                thickness=1,
                                text_thickness=1,
                                text_scale=0.3,
                                text_padding=3)

    with VideoSink(tracking_output, video_info) as sink:
        frame_count = 0
        trajectories = {}
        tracker_id_iou_values = {}
        track_missed = 0
        total_detection = 0
        total_tracker_detection = 0
        true_positive = 0
        false_positive = 0
        # Loop over video frames
        for frame in tqdm(generator, total=video_info.total_frames):
            # Model prediction on single frame and conversion to supervision Detections
            try:
                results = model(frame)
                detections = Detections(
                    xyxy=results[0].boxes.xyxy.cpu().numpy(),
                    confidence=results[0].boxes.conf.cpu().numpy(),
                    class_id=results[0].boxes.cls.cpu().numpy().astype(int)
                )

                # Filter detections with unwanted classes
                mask = np.array([class_id in cell_index for class_id in detections.class_id], dtype=bool)
                detections.filter(mask=mask, inplace=True)

                # Tracking detections
                tracks = byte_tracker.update(
                    output_results=detections2boxes(detections=detections),
                    img_info=frame.shape,
                    img_size=frame.shape
                )

                tracker_id, iou_values, true_positive_each_frame, false_positive_each_frame = match_detections_with_tracks(detections=detections, tracks=tracks)
                true_positive += true_positive_each_frame
                false_positive += false_positive_each_frame
                track_missed += tracker_id.count(None)
                total_detection += len(tracker_id)
                total_tracker_detection += len(tracker_id) - tracker_id.count(None)

                detections.tracker_id = np.array(tracker_id)

                # Filter detections without trackers
                mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
                detections.filter(mask=mask, inplace=True)

                # format custom labels
                labels = [
                    f"{tracker_id} {model_class[class_id]} {confidence:0.2f}" for _, confidence, class_id, tracker_id in detections
                ]

                frame = box_annotator.annotate(frame, detections=detections, labels=labels)

                detect_coordinates = detections.xyxy
                for i, coordinate in enumerate(detect_coordinates):
                    track_id = (detections.tracker_id)[i]
                    value = [(int((coordinate[0] + coordinate[2])/2), int((coordinate[1] + coordinate[3])/2))]
                    value = sorted(value, key=lambda tuple: tuple[1], reverse=False)
                    if track_id in trajectories:
                        trajectories[track_id] += value if value[0] not in trajectories[track_id] else trajectories[track_id]
                        tracker_id_iou_values[track_id] += [iou_values[i]]
                    else:
                        trajectories[track_id] = value
                        tracker_id_iou_values[track_id] = [iou_values[i]]

                if frame_count > 0:
                    frame = draw_trajectory(frame, trajectories)

                frame_count += 1

                sink.write_frame(frame)
            except ValueError:
                frame = draw_trajectory(frame, trajectories)
                frame_count += 1
                sink.write_frame(frame)
    ## Performance Evaluation and Metrics

    # Code to calculate IOU of each Tracker
    mean_iou = 0
    total_iou = 0
    tracking_ids = []
    iou_values_tracker = []

    for track_id, IoU_values in tracker_id_iou_values.items():
        iou_value = round(sum(IoU_values) / len(IoU_values), 4)
        mean_iou += iou_value
        total_iou += sum(IoU_values)
        tracking_ids.append(track_id)
        iou_values_tracker.append(iou_value)

    # # Calculate Mean IoU
    mean_iou_final = f'{mean_iou/len(tracker_id_iou_values):.4f}'

    # # Calculate MOTP
    motp = f'{total_iou/total_tracker_detection:.2%}'

    # # Calculate MOTA
    mota = f'{(1 - (false_positive + track_missed) / total_detection):.2%}'

    data = {'Cell/Tracker ID': tracking_ids,
        'IoU Values':iou_values_tracker}
    df = pd.DataFrame(data)
    print(df)

    df_overall = pd.DataFrame({
    "Metrics": ["Mean IoU", "MOTP", 'MOTA'],
    "Values": [mean_iou_final, motp, mota],
    })

    print(df_overall)
if __name__ == "__main__":
    main()