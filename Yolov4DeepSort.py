import os

# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time

import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import cv2
import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
from PIL import Image
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
from tensorflow.python.saved_model import tag_constants

import core.utils as utils
from core.config import cfg
from core.yolov4 import filter_boxes
# deep sort imports
from deep_sort import nn_matching, preprocessing
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')
flags.DEFINE_boolean('csv', False, 'store data into csv file')
flags.DEFINE_list('Region1', [(575, 420), (1185, 420), (1185, 460), (575, 460)], 'Change Region 1 placement')
flags.DEFINE_list('Region2', [(475, 765), (1320, 765), (1320, 880), (475, 880)], 'Change Region 2 placement')

def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video


    saved_model_loaded = tf.saved_model.load('./checkpoints/yolov4-416', tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    # region_1 = [(575, 420), (1185, 420), (1185, 460), (575, 460)]
    # region_2 = [(475, 765), (1320, 765), (1320, 880), (475, 880)]

    region_1 = FLAGS.Region1
    region_2 = FLAGS.Region2

    region_1_ids = set()
    region_2_ids = set()
    speed_1_list = set()
    speed_2_list = set()

    vehicle_entering_1 = {}
    vehicle_entering_2 = {}
    vehicles_elapsed_time_1 = {}
    vehicles_elapsed_time_2 = {}

    frame_num = 0
    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        ih, iw = frame_size
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()


        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['car', 'truck', 'bus', 'bicycle']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue  
            bbox = track.to_tlbr()
            class_name = track.get_class()
            
            # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            cv2.polylines(frame, [np.array(region_1)], True, (0, 255, 255), 4)
            cv2.polylines(frame, [np.array(region_2)], True, (255, 0, 255), 4)

            if class_name == "car" or class_name == "bus" or class_name == "truck":
                mid_x = (bbox[0]+bbox[2])/2
                mid_y = (bbox[1]+bbox[3])/2

                inside_region_1 = cv2.pointPolygonTest(np.array(region_1), (int(mid_x),int(mid_y)), False)
                inside_region_2 = cv2.pointPolygonTest(np.array(region_2), (int(mid_x),int(mid_y)), False)

                if inside_region_1 > 0:
                    vehicle_entering_1[track.track_id] = time.time()
                
                if inside_region_2 > 0:
                    vehicle_entering_2[track.track_id] = time.time()

            if track.track_id in vehicle_entering_1:
                result_1 = cv2.pointPolygonTest(np.array(region_2, np.int32), (int(mid_x), int(mid_y)), False)
                
                if result_1 >= 0:
                    elapsed_time_1 = time.time() - vehicle_entering_1[track.track_id]
                    region_2_ids.add(track.track_id)
                    
                    if track.track_id not in vehicles_elapsed_time_1:
                        vehicles_elapsed_time_1[track.track_id] = elapsed_time_1
                    
                    if track.track_id in vehicles_elapsed_time_1:
                        elapsed_time_1 = vehicles_elapsed_time_1[track.track_id]
                    
                    # Calculation of avg speed
                    distance_1 = 125
                    avg_speed_ms_1 = distance_1/elapsed_time_1
                    avg_speed_kmh_1 = avg_speed_ms_1 * 3.6
                    speed_2_list.add(avg_speed_kmh_1)
                    cv2.rectangle(frame, (int(bbox[2])-30, int(bbox[3])), (int(bbox[2]), int(bbox[3])-30), color, -1)
                    cv2.putText(frame, str(int(avg_speed_kmh_1)) + "km/h", (int(bbox[2]-20), int(bbox[3])), 0, 0.75, (255, 255, 255), 2)

                if track.track_id in vehicle_entering_2:
                    result_2 = cv2.pointPolygonTest(np.array(region_1, np.int32), (int(mid_x), int(mid_y)), False)
                
                    if result_2 >= 0:
                        elapsed_time_2 = time.time() - vehicle_entering_2[track.track_id]
                        region_1_ids.add(track.track_id)
                        
                        if track.track_id not in vehicles_elapsed_time_2:
                            vehicles_elapsed_time_2[track.track_id] = elapsed_time_2
                        
                        if track.track_id in vehicles_elapsed_time_2:
                            elapsed_time_2 = vehicles_elapsed_time_2[track.track_id]
                        
                        # Calculation of avg speed
                        distance_2 = 125
                        avg_speed_ms_2 = distance_2/elapsed_time_2
                        avg_speed_kmh_2 = avg_speed_ms_2 * 3.6
                        speed_1_list.add(avg_speed_kmh_2)
                        cv2.rectangle(frame, (int(bbox[2])-30, int(bbox[3])), (int(bbox[2]), int(bbox[3])-30), color, -1)
                        cv2.putText(frame, str(int(avg_speed_kmh_2)) + "km/h", (int(bbox[2]-20), int(bbox[3])), 0, 0.75, (255, 255, 255), 2)


            # Show count
            vehicle_count_up = len(region_1_ids)
            vehicle_count_down = len(region_2_ids)
            print(region_2_ids)
            print(speed_2_list)
            cv2.putText(frame, "Vehicles Up: " + str(vehicle_count_up), (10, 40), 0, 1.5, (255, 0, 0), 2)
            cv2.putText(frame, "Vehicles Down: " + str(vehicle_count_down), (10, 80), 0, 1.5, (255, 0, 0), 2)

            # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

            # if FLAGS.csv:
            #     with 

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
