import argparse
from time import time

import os

import cv2

import logging

import numpy as np

from OpenPersonDetector import OpenPersonDetector

logging.basicConfig()
logger = logging.getLogger("DetectorPreviewLog")
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preview videos using detectors.')

    parser.add_argument('path', metavar='path', type=str, nargs=1, help='path to video file')

    parser.add_argument('--scale', metavar='scale', type=float, nargs=1, help='scale of video', default=[1])
    parser.add_argument('--vdup', metavar='vdup', type=int, nargs=1, help='vertical duplication of video', default=[1])

    parser.add_argument('--hdup', metavar='hdup', type=int, nargs=1,
                        help='horizontal duplication of video', default=[1])

    parser.add_argument('--preview', metavar='preview', type=bool, nargs=1,
                        help='detector preview of video', default=[False])

    parser.add_argument('--url', metavar='url', type=str, nargs=1, help='url to load from', default=[None])

    args = parser.parse_args()

    video_path = os.path.realpath(args.path[0])
    video_scale = args.scale[0]
    preview_detector = args.preview[0]
    x_dup_count = args.hdup[0]
    y_dup_count = args.vdup[0]

    url = args.url[0]

    if url is not None:
        video_path = url

    logger.info("File Name: " + video_path)

    detector = OpenPersonDetector(preview=preview_detector)

    cap = cv2.VideoCapture(video_path)

    cv2.namedWindow("Detector Output", cv2.WINDOW_FREERATIO)

    last_frame_time = None
    video_frame_time = None

    while True:
        if video_frame_time is None:
            r, frame = cap.read()
        else:
            skip_count = 0
            while video_frame_time + last_frame_time > cap.get(cv2.CAP_PROP_POS_MSEC) / 1000:
                r, new_frame = cap.read()
                if r:
                    frame = new_frame
                if skip_count > 24:
                    break
                skip_count += 1
            logger.info("Skipped Frames: " + str(skip_count - 1))

        video_frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        frame_start_time = time()

        frame = cv2.resize(frame, (0, 0), fx=video_scale, fy=video_scale)

        if frame is not None:
            new_frame = np.zeros((int(frame.shape[0] * y_dup_count), int(frame.shape[1] * x_dup_count), 3),
                                 dtype=frame.dtype)
            for y_d in range(y_dup_count):
                for x_d in range(x_dup_count):
                    new_frame[y_d * frame.shape[0]:(((y_d + 1) * frame.shape[0])),
                    x_d * frame.shape[1]:(((x_d + 1) * frame.shape[1])), :] = np.array(frame, copy=True)[:, :, :]

            frame = new_frame
        # fgmask = fgbg.apply(frame)

        cv2.putText(frame, "Time: " + str(video_frame_time), (20, 25), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))
        cv2.putText(frame, "Frame Time: " + str(last_frame_time), (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))

        person_detections = detector.detectPersons(frame, None)

        for detection in person_detections:
            cv2.rectangle(frame, (detection.person_bound[0], detection.person_bound[1]),
                          (detection.person_bound[2], detection.person_bound[3]), (0, 0, 255), 2)

            if hasattr(detection, "tracked_points"):
                tracked_points = detection.tracked_points
                for name, tracked_point in tracked_points.items():
                    cv2.drawMarker(frame, (int(tracked_point[0]), int(tracked_point[1])), (0, 255, 255), markerSize=10)

        if hasattr(detector, "draw_patches"):
            frame = detector.draw_patches(frame)

        cv2.imshow("Detector Output", frame)

        k = cv2.waitKey(1)
        if k & 0xFF == ord("q"):
            break

        frame_end_time = time()

        last_frame_time = frame_end_time - frame_start_time
        logger.info("Last Frame Time: " + str(frame_end_time - frame_start_time))
