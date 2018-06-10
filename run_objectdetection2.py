#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:01:40 2017

@author: www.github.com/GustavZ
"""
import numpy as np
import tensorflow as tf
import os
from rod.helper import FPS, WebcamVideoStream, SessionWorker, conv_detect2track, conv_track2detect, vis_detection, Timer
from rod.model import Model
from rod.config import Config
from rod.utils import ops as utils_ops
import cv2, datetime
from easy_memmap import EasyMemmap, MultiImagesMemmap

class FPS2:
    def __init__(self, interval):
        self._glob_start = None
        self._glob_end = None
        self._glob_numFrames = 0
        self._local_start = None
        self._local_numFrames = 0
        self._interval = interval
        self.curr_local_elapsed = None
        self.first = False

    def start(self):
        self._glob_start = datetime.datetime.now()
        self._local_start = self._glob_start
        return self

    def stop(self):
        self._glob_end = datetime.datetime.now()

    def update(self):
        self.first = True
        curr_time = datetime.datetime.now()
        self.curr_local_elapsed = (curr_time - self._local_start).total_seconds()
        self._glob_numFrames += 1
        self._local_numFrames += 1
        if self.curr_local_elapsed > self._interval:
          print("FPS: {}".format(self.fps_local()))
          self._local_numFrames = 0
          self._local_start = curr_time
    def elapsed(self):
        return (self._glob_end - self._glob_start).total_seconds()

    def fps(self):
        return self._glob_numFrames / self.elapsed()
    
    def fps_local(self):
        if self.first:
            return round(self._local_numFrames / self.curr_local_elapsed,1)
        else:
            return 0.0

def detection(model,config):

    print("> Building Graph")
    # tf Session Config
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth=True
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.1
    detection_graph = model.detection_graph
    category_index = model.category_index
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph,config=tf_config) as sess:
            # start Videostream
            # vs = WebcamVideoStream(config.VIDEO_INPUT,config.WIDTH,config.HEIGHT).start()
            vs = MultiImagesMemmap(mode = "r", name = "main_stream", memmap_path = os.getenv("MEMMAP_PATH", "/tmp"))
            vs.wait_until_available() #initialize and find video data
            # Define Input and Ouput tensors
            tensor_dict = model.get_tensordict(['num_detections', 'detection_boxes', 'detection_scores','detection_classes', 'detection_masks'])
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            score_out = detection_graph.get_tensor_by_name('Postprocessor/convert_scores:0')
            expand_out = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1:0')
            score_in = detection_graph.get_tensor_by_name('Postprocessor/convert_scores_1:0')
            expand_in = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1_1:0')
            # Threading
            score = model.score
            expand = model.expand
            gpu_worker = SessionWorker("GPU",detection_graph,tf_config)
            cpu_worker = SessionWorker("CPU",detection_graph,tf_config)
            gpu_opts = [score_out, expand_out]
            cpu_opts = [tensor_dict['detection_boxes'], tensor_dict['detection_scores'], tensor_dict['detection_classes'], tensor_dict['num_detections']]
            gpu_counter = 0
            cpu_counter = 0

            fps = FPS2(config.FPS_INTERVAL).start()
            print('> Starting Detection')
            frame = vs.read("C")
            # frame = vs.read()
            h,w,_ = frame.shape
            vs.real_width, vs.real_height = w,h
            while True:
                # Detection

                # split model in seperate gpu and cpu session threads
                masks = None # No Mask Detection possible yet
                if gpu_worker.is_sess_empty():
                    # read video frame, expand dimensions and convert to rgb
                    frame = vs.read("C")
                    # frame = vs.read()
                    # put new queue
                    image_expanded = np.expand_dims(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), axis=0)
                    gpu_feeds = {image_tensor: image_expanded}
                    if config.VISUALIZE:
                        gpu_extras = frame # for visualization frame
                    else:
                        gpu_extras = None
                    gpu_worker.put_sess_queue(gpu_opts,gpu_feeds,gpu_extras)
                g = gpu_worker.get_result_queue()
                if g is None:
                    # gpu thread has no output queue. ok skip, let's check cpu thread.
                    gpu_counter += 1
                else:
                    # gpu thread has output queue.
                    gpu_counter = 0
                    score,expand,frame = g["results"][0],g["results"][1],g["extras"]

                    if cpu_worker.is_sess_empty():
                        # When cpu thread has no next queue, put new queue.
                        # else, drop gpu queue.
                        cpu_feeds = {score_in: score, expand_in: expand}
                        cpu_extras = frame
                        cpu_worker.put_sess_queue(cpu_opts,cpu_feeds,cpu_extras)
                c = cpu_worker.get_result_queue()
                if c is None:
                    # cpu thread has no output queue. ok, nothing to do. continue
                    cpu_counter += 1
                    continue # If CPU RESULT has not been set yet, no fps update
                else:
                    cpu_counter = 0
                    boxes, scores, classes, num, frame = c["results"][0],c["results"][1],c["results"][2],c["results"][3],c["extras"]


                    # reformat detection
                    num = int(num)
                    boxes = np.squeeze(boxes)
                    classes = np.squeeze(classes).astype(np.uint8)
                    scores = np.squeeze(scores)

                    # Visualization
                    # print frame.shape
                    if frame is not None:
                        vis = vis_detection(frame.copy(), boxes, classes, scores, masks, category_index, fps.fps_local(),
                                            config.VISUALIZE, config.DET_INTERVAL, config.DET_TH, config.MAX_FRAMES,
                                            fps._glob_numFrames, config.OD_MODEL_NAME)
                        if not vis:
                            break

                fps.update()

    # End everything
    # vs.stop()
    fps.stop()
    if config.SPLIT_MODEL:
        gpu_worker.stop()
        cpu_worker.stop()


if __name__ == '__main__':
    config = Config()
    model = Model('od',config.OD_MODEL_NAME,config.OD_MODEL_PATH,config.LABEL_PATH,
                config.NUM_CLASSES,config.SPLIT_MODEL, config.SSD_SHAPE).prepare_od_model()
    detection(model, config)
