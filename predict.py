#! /usr/bin/env python

import os
import argparse
import json
import cv2
import time
from utils.utils import get_yolo_boxes, makedirs
from utils.bbox import draw_boxes
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from keras.models import load_model
from tqdm import tqdm
import numpy as np

def read_pb_return_tensors(graph, pb_file, return_elements):

    with tf.gfile.FastGFile(pb_file, 'rb') as f:
        frozen_graph_def = tf.GraphDef()
        frozen_graph_def.ParseFromString(f.read())

    with graph.as_default():
        return_elements = tf.import_graph_def(frozen_graph_def,
                                              return_elements=return_elements)
        input_tensor, output_tensors = return_elements[0], return_elements[1:]

    return input_tensor, output_tensors
def _main_(args):
    config_path  = args.conf
    input_path   = args.input
    output_path  = args.output
    config_path ="config424.json"
    input_path = "webcam"
    output_path = "outvideo"
    input_model = "tftrt_int8_fire424.pb"

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    makedirs(output_path)

    ###############################
    #   Set some parameter
    ###############################       
    net_h, net_w = 416, 416 # a multiple of 32, the smaller the faster
    obj_thresh, nms_thresh = 0.35, 0.8

    ###############################
    #   Load the model
    ###############################
    # os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    # infer_model = load_model(config['train']['saved_weights_name'])
    input_tensor, output_tensors = \
            read_pb_return_tensors(tf.get_default_graph(),
                                     input_model,
                                     ["input_1:0", 'conv_81/BiasAdd:0', 'conv_93/BiasAdd:0', 'conv_105/BiasAdd:0'])

    # perform inference
    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.3))) as sess:
        ###############################
        #   Predict bounding boxes
        ###############################
        if 'webcam' in input_path: # do detection on the first webcam
            video_reader = cv2.VideoCapture("videoplayback (1).mp4")

            # the main loop
            batch_size  = 1
            images      = []
            while True:
                ret_val, image = video_reader.read()
                if ret_val == False:
                    print('ret:', ret_val)
                    video_reader = cv2.VideoCapture("videoplayback (1).mp4")
                    ret_val, image = video_reader.read()
                if ret_val == True: images += [image]

                if (len(images)==batch_size) or (ret_val==False and len(images)>0):
                    prev_time = time.time()
                    batch_boxes = get_yolo_boxes(sess,input_tensor,output_tensors, images, net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)
                    curr_time = time.time()
                    exec_time = curr_time - prev_time
                    result = np.asarray(image)
                    info = "time:" + str(round(1000 * exec_time, 2)) + " ms, FPS: " + str(round((1000 / (1000 * exec_time)), 1))
                    cv2.putText(result, text=info, org=(50, 70),
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=1, color=(255, 0, 0), thickness=2)
                    for i in range(len(images)):
                        draw_boxes(images[i], batch_boxes[i], config['model']['labels'], obj_thresh)
                        cv2.imshow('video with bboxes', images[i])
                    images = []
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
            cv2.destroyAllWindows()
        elif input_path[-4:] == '.mp4': # do detection on a video
            video_out = output_path + input_path.split('/')[-1]
            video_reader = cv2.VideoCapture(input_path)

            nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

            video_writer = cv2.VideoWriter(video_out,
                                   cv2.VideoWriter_fourcc(*'MPEG'),
                                   50.0,
                                   (frame_w, frame_h))
            # the main loop
            batch_size  = 1
            images      = []
            start_point = 0 #%
            show_window = True
            for i in tqdm(range(nb_frames)):
                _, image = video_reader.read()

                if (float(i+1)/nb_frames) > start_point/100.:
                    images += [image]

                    if (i%batch_size == 0) or (i == (nb_frames-1) and len(images) > 0):
                        # predict the bounding boxes
                        batch_boxes = get_yolo_boxes(infer_model, images, net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)

                        for i in range(len(images)):
                            # draw bounding boxes on the image using labels
                            draw_boxes(images[i], batch_boxes[i], config['model']['labels'], obj_thresh)

                            # show the video with detection bounding boxes
                            if show_window: cv2.imshow('video with bboxes', images[i])

                            # write result to the output video
                            video_writer.write(images[i])
                        images = []
                    if show_window and cv2.waitKey(1) == 27: break  # esc to quit

            if show_window: cv2.destroyAllWindows()
            video_reader.release()
            video_writer.release()
        else: # do detection on an image or a set of images
            image_paths = []

            if os.path.isdir(input_path):
                for inp_file in os.listdir(input_path):
                    image_paths += [input_path + inp_file]
            else:
                image_paths += [input_path]

            image_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.jpg', '.png', 'JPEG'])]

            # the main loop
            for image_path in image_paths:
                image = cv2.imread(image_path)
                print(image_path)

                # predict the bounding boxes
                boxes = get_yolo_boxes(infer_model, [image], net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)[0]

                # draw bounding boxes on the image using labels
                draw_boxes(image, boxes, config['model']['labels'], obj_thresh)

                # write the image with bounding boxes to file
                cv2.imwrite(output_path + image_path.split('/')[-1], np.uint8(image))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    argparser.add_argument('-i', '--input', help='path to an image, a directory of images, a video, or webcam')    
    argparser.add_argument('-o', '--output', default='output/', help='path to output directory')   
    
    args = argparser.parse_args()
    _main_(args)
