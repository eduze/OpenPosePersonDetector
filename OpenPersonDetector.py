'''
Person Detector and Leg Estimator based on OpenPose Project.
OpenPose Project Repo: https://github.com/CMU-Perceptual-Computing-Lab/openpose
'''
# NOTE:
# LD_LIBRARY_PATH should be set as [OPENPOSE_CAFFE_HOME]/build/lib/:[CUDA_HOME]/lib64:[OPENPOSE_HOME]build/lib:[Python3.5_Path]/dist-packages
import logging
from threading import RLock

import cv2


def initLibrary(preview, net_width, net_height):
    '''
    Loads Python wrapper on OpenPose (OpenPersonDetectorAPI)
    :param preview: True to enable OP Preview
    :param net_width: Detector net width
    :param net_height: Detector net height
    :return: 
    '''
    if "person_detector_api" in globals():
        logging.info("Library Init Already")
        return
    global person_detector_api
    person_detector_api = None
    import libOpenPersonDetectorAPI
    person_detector_api = libOpenPersonDetectorAPI
    person_detector_api.setup(preview, int(net_width), int(net_height))


# OP Mappings Index
coordinates_mapping = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "RHip",
                       "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar", "Bkg"]
centralBodyKeywords = ["Nose", "Neck", "RShoulder", "LShoulder", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
                       "REye", "LEye", "REar", "LEar", "Bkg"]
upperBodyKeywords = ["Neck", "RShoulder", "LShoulder", "RHip", "LHip", "Bkg"]

MAX_VALUE = 1000000

# Static Rules for detection of head orientation
head_direction_map = {}
# Nose, LEye, REye, LEar, REar
head_direction_map[(True, True, True, True, True)] = (0, 45)
head_direction_map[(True, False, False, True, True)] = (0, 45)
head_direction_map[(True, True, True, False, False)] = (0, 45)

head_direction_map[(True, True, True, True, False)] = (45, 45)
head_direction_map[(True, True, False, True, False)] = (90, 45)
head_direction_map[(True, True, False, True, True)] = (135, 45)

head_direction_map[(False, False, False, True, True)] = (180, 45)
head_direction_map[(True, False, True, True, True)] = (225, 45)
head_direction_map[(True, False, True, False, True)] = (270, 45)
head_direction_map[(True, True, True, False, True)] = (315, 45)

# Supplementary Rules
head_direction_map[(False, True, False, True, True)] = (135, 45)
head_direction_map[(True, False, False, False, True)] = (135, 45)

head_direction_map[(False, False, True, True, True)] = (225, 45)
head_direction_map[(True, False, False, True, False)] = (225, 45)

lock = RLock()

class OpenPersonDetector:
    '''
    Person Detector and Leg Estimator based on OpenPose Project.
    OpenPose Project Repo: https://github.com/CMU-Perceptual-Computing-Lab/openpose
    '''

    def __init__(self, preview=True, net_width=656, net_height=368):
        '''
        Initialize OpenPose and Detector
        :param preview: 
        :param net_width: 
        :param net_height: 
        '''
        lock.acquire()
        global person_detector_api
        self.preview = preview
        initLibrary(preview, net_width, net_height)
        lock.release()

    def _detectHeadDirection(self, human):
        '''
        Estimate head direction of human
        :param human: 
        :return: 
        '''

        lock.acquire()

        key = ("Nose" in human.tracked_points.keys(), "LEye" in human.tracked_points.keys(),
               "REye" in human.tracked_points.keys(), "LEar" in human.tracked_points.keys(),
               "REar" in human.tracked_points.keys())
        if key in head_direction_map.keys():
            human.head_direction = head_direction_map[key][0]
            human.head_direction_error = head_direction_map[key][1]
        else:
            # Failed to estimate
            human.head_direction = None
            human.head_direction_error = None

        lock.release()

    def detectPersons(self, colour_frame, gray_frame):
        '''
        Detect persons in frame
        :param colour_frame: colour frame
        :param gray_frame: gray frame (unused)
        :return: 
        '''

        lock.acquire()

        global person_detector_api

        # Obtain results
        results = person_detector_api.detect(colour_frame)

        # Obtain scale factors
        results_height, results_width = person_detector_api.getOutputHeight(), person_detector_api.getOutputHeight()
        frame_height, frame_width = colour_frame.shape[:2]
        scale_factor = frame_height / results_height

        # Preview output
        if self.preview:
            outputImage = person_detector_api.getOutputImage()
            cv2.imshow("pose", outputImage)

        person_detections = []

        # For none detections
        if results is None:
            return []

        # Process results
        for i in range(len(results)):
            # Add Detection
            _person = results[i]
            person_detection = PersonDetection()
            person_detections.append(person_detection)

            # Default bounds
            minX = MAX_VALUE
            maxX = -MAX_VALUE
            minY = MAX_VALUE
            maxY = -MAX_VALUE

            minCentralX = MAX_VALUE
            maxCentralX = -MAX_VALUE
            minCentralY = MAX_VALUE
            maxCentralY = -MAX_VALUE

            minUpperX = MAX_VALUE
            maxUpperX = -MAX_VALUE
            minUpperY = MAX_VALUE
            maxUpperY = -MAX_VALUE

            # Update bounds
            for j in range(len(_person)):
                # Do scale correction
                coordinate = _person[j]
                #coordinate[0] = coordinate[0] * scale_factor
                #coordinate[1] = coordinate[1] * scale_factor

                if coordinate[2] > 0:  # In presence of point
                    person_detection.tracked_points[coordinates_mapping[j]] = coordinate[:]
                    minX = min(coordinate[0], minX)
                    minY = min(coordinate[1], minY)
                    maxX = max(coordinate[0], maxX)
                    maxY = max(coordinate[1], maxY)
                    # Update bounds
                    if coordinates_mapping[j] in centralBodyKeywords:
                        minCentralX = min(coordinate[0], minCentralX)
                        minCentralY = min(coordinate[1], minCentralY)
                        maxCentralX = max(coordinate[0], maxCentralX)
                        maxCentralY = max(coordinate[1], maxCentralY)

                    if coordinates_mapping[j] in upperBodyKeywords:
                        minUpperX = min(coordinate[0], minUpperX)
                        minUpperY = min(coordinate[1], minUpperY)
                        maxUpperX = max(coordinate[0], maxUpperX)
                        maxUpperY = max(coordinate[1], maxUpperY)

            def findAverage(names):
                '''
                Finds average of coordinates with corresponding names
                :param names: 
                :return: 
                '''
                total = None
                count = 0
                for name in names:
                    if name in person_detection.tracked_points.keys():
                        if total is None:
                            total = [0, 0]
                        total[0] += person_detection.tracked_points[name][0]
                        total[1] += person_detection.tracked_points[name][1]
                        count += 1

                if total is not None:
                    total[0] = total[0] / count
                    total[1] = total[1] / count
                return total

            # Update result fields
            person_detection.person_bound = (minX, minY, maxX, maxY)
            person_detection.central_bound = (minCentralX, minCentralY, maxCentralX, maxCentralY)
            person_detection.central_point = ((minCentralX + maxCentralX) / 2, (minCentralY + maxCentralY) / 2)
            person_detection.upper_body_bound = (minUpperX, minUpperY, maxUpperX, maxUpperY)

            legCount = 0  # Number of visible legs
            legSet = False  # Is leg position known?
            legY = 0  # Y coordinate of leg
            legX = 0
            # Obtain leg point
            if "LAnkle" in person_detection.tracked_points.keys():
                legY = max(legY, person_detection.tracked_points["LAnkle"][1])
                if legY ==  person_detection.tracked_points["LAnkle"][1]:
                    legX =  person_detection.tracked_points["LAnkle"][0]
                legSet = True
                legCount += 1

            if "RAnkle" in person_detection.tracked_points.keys():
                legY = max(legY, person_detection.tracked_points["RAnkle"][1])
                if legY == person_detection.tracked_points["RAnkle"][1]:
                    legX = person_detection.tracked_points["RAnkle"][0]
                legSet = True
                legCount += 1

            # find other points of interest
            hip = findAverage(["RHip", "LHip"])
            neck = findAverage(["Neck"])
            ankle = findAverage(["RAnkle", "LAnkle"])
            knee = findAverage(["RKnee", "LKnee"])

            elbow = findAverage(["RElbow","LElbow"])

            if elbow is not None:
                person_detection.elbow_point = elbow

            if hip is not None:
                person_detection.hip_point = hip

            if hip is not None and neck is not None:
                # Estimate leg Y using hip,neck,leg ratio
                person_detection.estimated_leg_point = (
                person_detection.central_point[0], neck[1] + 2.3 * (hip[1] - neck[1]))
                # person_detection.estimated_leg_point = (person_detection.central_point[0],
                #                                        person_detection.upper_body_bound[1] +
                #                                        (person_detection.upper_body_bound[3] - person_detection.upper_body_bound[1])* 2)
            else:
                # Estimate leg y using bounds
                logging.info("Poor Estimate of Leg Point")
                person_detection.estimated_leg_point = (person_detection.central_point[0],
                                                        person_detection.central_bound[1] +
                                                        (person_detection.central_bound[3] -
                                                         person_detection.central_bound[1]) * 1)
            if not legSet:
                # If leg point is not known, use estimate
                legX = person_detection.central_point[0]
                legY = person_detection.estimated_leg_point[1]

            person_detection.leg_point = (legX, legY)
            person_detection.leg_count = legCount

            # Calculate neck_hip_ankle ratio
            if hip is not None and neck is not None and ankle is not None:
                person_detection.neck_hip_ankle_ratio = (ankle[1] - neck[1]) / (hip[1] - neck[1])

            if hip is not None and neck is not None and knee is not None:
                person_detection.neck_hip_knee_ratio = (knee[1] - neck[1]) / (hip[1] - neck[1])

            self._detectHeadDirection(person_detection)

        lock.release()
        return person_detections


class PersonDetection:
    '''
    Detection of a person
    '''

    def __init__(self):
        self.tracked_points = {}  # Points detected by OP
        self.person_bound = None  # Boundary of person
        self.central_bound = None  # Boundary of central body of person (no hands and feet for X coordinate)
        self.upper_body_bound = None  # Boundary of upper body of person
        self.central_point = None  # Central point of person
        self.leg_point = None  # Average Feet point of person
        self.leg_count = None  # Number of detected feet
        self.estimated_leg_point = None  # Estimated feet point of person
        self.neck_hip_ankle_ratio = None
        self.neck_hip_knee_ratio = None
        self.head_direction = None
        self.head_direction_error = None
        self.hip_point = None
        self.elbow_point = None