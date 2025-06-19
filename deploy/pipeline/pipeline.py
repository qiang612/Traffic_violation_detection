# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# deploy/pipeline/py 的顶部
import os
import requests 
import json     
import base64   
import cv2      
import numpy as np 

# --- 全局API及模型配置 ---
# 警告：这种方法不安全，请勿在共享或公开的代码中使用
ARK_API_KEY = "baab7418-edcc-44e5-bc06-4f9de8240876" # 直接将密钥写在这里
# os.environ['ARK_API_KEY'] = "baab7418-edcc-44e5-bc06-4f9de8240876"
# ARK_API_KEY = os.environ.get('ARK_API_KEY')
ARK_VQA_ENDPOINT = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
ARK_VQA_MODEL = "ep-20250618010033-g4898"

# --- 全自动分析的“大师级”Prompt ---
MASTER_PROMPT = """
你是一位资深的交通警察和AI分析专家。你的任务是仔细审查这张包含多个被白色方框和ID标记的车辆的交通监控截图，找出所有正在发生的交通违法行为。

请重点关注但不限于以下行为：
1.  **压黄实线**: 车辆车轮是否碾压或越过黄色的实线。
2.  **违章停车**: 车辆是否在禁止停车的区域（如路肩、行车道）处于长时间静止状态。
3.  **占用应急车道**: 车辆是否行驶在最右侧的应急车道内。
4.  **未戴头盔**: 骑行电动车或摩托车的人员是否未佩戴头盔。

请严格按照以下JSON数组格式返回你的分析结果，不要添加任何额外的解释和文字。
- 如果发现违法行为，返回一个包含多个对象的JSON数组，每个对象必须包含 "track_id" (整数型) 和 "violation_type" (字符串)。
- 如果没有发现任何违法行为，请只返回一个空数组 `[]`。

示例返回:
[
  {"track_id": 15, "violation_type": "压黄实线"},
  {"track_id": 21, "violation_type": "未戴头盔"}
]
"""

import os
import yaml
import glob
import cv2
import numpy as np
import math
import paddle
import sys
import copy
import threading
import queue
import time
from collections import defaultdict
from datacollector import DataCollector, Result
try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

# add deploy path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

from cfg_utils import argsparser, print_arguments, merge_cfg
from pipe_utils import PipeTimer
from pipe_utils import get_test_images, crop_image_with_det, crop_image_with_mot, parse_mot_res, parse_mot_keypoint
from pipe_utils import PushStream

from python.infer import Detector, DetectorPicoDet
from python.keypoint_infer import KeyPointDetector
from python.keypoint_postprocess import translate_to_ori_images
from python.preprocess import decode_image, ShortSizeScale
from python.visualize import visualize_box_mask, visualize_attr, visualize_pose, visualize_action, visualize_vehicleplate, visualize_vehiclepress, visualize_lane, visualize_vehicle_retrograde

from pptracking.python.mot_sde_infer import SDE_Detector
from pptracking.python.mot.visualize import plot_tracking_dict
from pptracking.python.mot.utils import flow_statistic, update_object_info

from pphuman.attr_infer import AttrDetector
from pphuman.video_action_infer import VideoActionRecognizer
from pphuman.action_infer import SkeletonActionRecognizer, DetActionRecognizer, ClsActionRecognizer
from pphuman.action_utils import KeyPointBuff, ActionVisualHelper
from pphuman.reid import ReID
from pphuman.mtmct import mtmct_process

from ppvehicle.vehicle_plate import PlateRecognizer
from ppvehicle.vehicle_attr import VehicleAttr
from ppvehicle.vehicle_pressing import VehiclePressingRecognizer
from ppvehicle.vehicle_retrograde import VehicleRetrogradeRecognizer
from ppvehicle.lane_seg_infer import LaneSegPredictor

from download import auto_download_model

def analyze_image_with_ark_vlm(frame, prompt_text):
    """
    Calls the Volcano Engine Ark platform's VQA model to analyze an image,
    with a fix for UTF-8 encoding.
    """
    if not ARK_API_KEY:
        print("Error: The API key (ARK_API_KEY) is not configured.")
        return "Error: API key not configured"

    # Header now explicitly states UTF-8 charset
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {ARK_API_KEY}"
    }

    _, buffer = cv2.imencode('.jpg', frame)
    base64_frame = base64.b64encode(buffer).decode('utf-8')
    image_url_for_api = f"data:image/jpeg;base64,{base64_frame}"

    payload = {
        "model": ARK_VQA_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": image_url_for_api}}
                ]
            }
        ]
    }

    try:
        # This is the critical fix:
        # We manually dump the json with Chinese characters and encode it to utf-8 bytes.
        response = requests.post(
            ARK_VQA_ENDPOINT,
            headers=headers,
            data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
            timeout=60
        )
        response.raise_for_status()
        answer = response.json().get("choices")[0].get("message").get("content")
        return answer
    except Exception as e:
        print(f"Error calling Volcano Engine VLM API: {e}")
        if 'response' in locals() and hasattr(response, 'text'):
            print(f"API raw response content: {response.text}")
        return f"Error: {e}"
# class Pipeline(object):
# ...

class Pipeline(object):
    """
    Pipeline

    Args:
        args (argparse.Namespace): arguments in pipeline, which contains environment and runtime settings
        cfg (dict): config of models in pipeline
    """

    def __init__(self, args, cfg):
        self.multi_camera = False
        reid_cfg = cfg.get('REID', False)
        self.enable_mtmct = reid_cfg['enable'] if reid_cfg else False
        self.is_video = False
        self.output_dir = args.output_dir
        self.vis_result = cfg['visual']
        self.input = self._parse_input(args.image_file, args.image_dir,
                                       args.video_file, args.video_dir,
                                       args.camera_id, args.rtsp)
        if self.multi_camera:
            self.predictor = []
            for name in self.input:
                predictor_item = PipePredictor(
                    args, cfg, is_video=True, multi_camera=True)
                predictor_item.set_file_name(name)
                self.predictor.append(predictor_item)

        else:
            self.predictor = PipePredictor(args, cfg, self.is_video)
            if self.is_video:
                self.predictor.set_file_name(self.input)

    def _parse_input(self, image_file, image_dir, video_file, video_dir,
                     camera_id, rtsp):

        # parse input as is_video and multi_camera

        if image_file is not None or image_dir is not None:
            input = get_test_images(image_dir, image_file)
            self.is_video = False
            self.multi_camera = False

        elif video_file is not None:
            assert os.path.exists(
                video_file
            ) or 'rtsp' in video_file, "video_file not exists and not an rtsp site."
            self.multi_camera = False
            input = video_file
            self.is_video = True

        elif video_dir is not None:
            videof = [os.path.join(video_dir, x) for x in os.listdir(video_dir)]
            if len(videof) > 1:
                self.multi_camera = True
                videof.sort()
                input = videof
            else:
                input = videof[0]
            self.is_video = True

        elif rtsp is not None:
            if len(rtsp) > 1:
                rtsp = [rtsp_item for rtsp_item in rtsp if 'rtsp' in rtsp_item]
                self.multi_camera = True
                input = rtsp
            else:
                self.multi_camera = False
                input = rtsp[0]
            self.is_video = True

        elif camera_id != -1:
            self.multi_camera = False
            input = camera_id
            self.is_video = True

        else:
            raise ValueError(
                "Illegal Input, please set one of ['video_file', 'camera_id', 'image_file', 'image_dir']"
            )

        return input

    def run_multithreads(self):
        if self.multi_camera:
            multi_res = []
            threads = []
            for idx, (predictor,
                      input) in enumerate(zip(self.predictor, self.input)):
                thread = threading.Thread(
                    name=str(idx).zfill(3),
                    target=predictor.run,
                    args=(input, idx))
                threads.append(thread)

            for thread in threads:
                thread.start()

            for predictor, thread in zip(self.predictor, threads):
                thread.join()
                collector_data = predictor.get_result()
                multi_res.append(collector_data)

            if self.enable_mtmct:
                mtmct_process(
                    multi_res,
                    self.input,
                    mtmct_vis=self.vis_result,
                    output_dir=self.output_dir)

        else:
            self.predictor.run(self.input)

    def run(self):
        if self.multi_camera:
            multi_res = []
            for predictor, input in zip(self.predictor, self.input):
                predictor.run(input)
                collector_data = predictor.get_result()
                multi_res.append(collector_data)
            if self.enable_mtmct:
                mtmct_process(
                    multi_res,
                    self.input,
                    mtmct_vis=self.vis_result,
                    output_dir=self.output_dir)

        else:
            self.predictor.run(self.input)

def get_model_dir(cfg):
    """
        Auto download inference model if the model_path is a url link.
        Otherwise it will use the model_path directly.
    """
    for key in cfg.keys():
        if type(cfg[key]) ==  dict and \
            ("enable" in cfg[key].keys() and cfg[key]['enable']
                or "enable" not in cfg[key].keys()):

            if "model_dir" in cfg[key].keys():
                model_dir = cfg[key]["model_dir"]
                downloaded_model_dir = auto_download_model(model_dir)
                if downloaded_model_dir:
                    model_dir = downloaded_model_dir
                    cfg[key]["model_dir"] = model_dir
                print(key, " model dir: ", model_dir)
            elif key == "VEHICLE_PLATE":
                det_model_dir = cfg[key]["det_model_dir"]
                downloaded_det_model_dir = auto_download_model(det_model_dir)
                if downloaded_det_model_dir:
                    det_model_dir = downloaded_det_model_dir
                    cfg[key]["det_model_dir"] = det_model_dir
                print("det_model_dir model dir: ", det_model_dir)

                rec_model_dir = cfg[key]["rec_model_dir"]
                downloaded_rec_model_dir = auto_download_model(rec_model_dir)
                if downloaded_rec_model_dir:
                    rec_model_dir = downloaded_rec_model_dir
                    cfg[key]["rec_model_dir"] = rec_model_dir
                print("rec_model_dir model dir: ", rec_model_dir)

        elif key == "MOT":  # for idbased and skeletonbased actions
            model_dir = cfg[key]["model_dir"]
            downloaded_model_dir = auto_download_model(model_dir)
            if downloaded_model_dir:
                model_dir = downloaded_model_dir
                cfg[key]["model_dir"] = model_dir
            print("mot_model_dir model_dir: ", model_dir)

class PipePredictor(object):
    """
    Predictor in single camera

    The pipeline for image input:

        1. Detection
        2. Detection -> Attribute

    The pipeline for video input:

        1. Tracking
        2. Tracking -> Attribute
        3. Tracking -> KeyPoint -> SkeletonAction Recognition
        4. VideoAction Recognition

    Args:
        args (argparse.Namespace): arguments in pipeline, which contains environment and runtime settings
        cfg (dict): config of models in pipeline
        is_video (bool): whether the input is video, default as False
        multi_camera (bool): whether to use multi camera in pipeline,
            default as False
    """

    def __init__(self, args, cfg, is_video=True, multi_camera=False):
        # general module for pphuman and ppvehicle
        self.with_mot = cfg.get('MOT', False)['enable'] if cfg.get(
            'MOT', False) else False
        self.with_human_attr = cfg.get('ATTR', False)['enable'] if cfg.get(
            'ATTR', False) else False
        if self.with_mot:
            print('Multi-Object Tracking enabled')
        if self.with_human_attr:
            print('Human Attribute Recognition enabled')

        # only for pphuman
        self.with_skeleton_action = cfg.get(
            'SKELETON_ACTION', False)['enable'] if cfg.get('SKELETON_ACTION',
                                                           False) else False
        self.with_video_action = cfg.get(
            'VIDEO_ACTION', False)['enable'] if cfg.get('VIDEO_ACTION',
                                                        False) else False
        self.with_idbased_detaction = cfg.get(
            'ID_BASED_DETACTION', False)['enable'] if cfg.get(
                'ID_BASED_DETACTION', False) else False
        self.with_idbased_clsaction = cfg.get(
            'ID_BASED_CLSACTION', False)['enable'] if cfg.get(
                'ID_BASED_CLSACTION', False) else False
        self.with_mtmct = cfg.get('REID', False)['enable'] if cfg.get(
            'REID', False) else False

        if self.with_skeleton_action:
            print('SkeletonAction Recognition enabled')
        if self.with_video_action:
            print('VideoAction Recognition enabled')
        if self.with_idbased_detaction:
            print('IDBASED Detection Action Recognition enabled')
        if self.with_idbased_clsaction:
            print('IDBASED Classification Action Recognition enabled')
        if self.with_mtmct:
            print("MTMCT enabled")

        # only for ppvehicle
        self.with_vehicleplate = cfg.get(
            'VEHICLE_PLATE', False)['enable'] if cfg.get('VEHICLE_PLATE',
                                                         False) else False
        if self.with_vehicleplate:
            print('Vehicle Plate Recognition enabled')

        self.with_vehicle_attr = cfg.get(
            'VEHICLE_ATTR', False)['enable'] if cfg.get('VEHICLE_ATTR',
                                                        False) else False
        if self.with_vehicle_attr:
            print('Vehicle Attribute Recognition enabled')

        self.with_vehicle_press = cfg.get(
            'VEHICLE_PRESSING', False)['enable'] if cfg.get('VEHICLE_PRESSING',
                                                            False) else False
        if self.with_vehicle_press:
            print('Vehicle Pressing Recognition enabled')

        self.with_vehicle_retrograde = cfg.get(
            'VEHICLE_RETROGRADE', False)['enable'] if cfg.get(
                'VEHICLE_RETROGRADE', False) else False
        if self.with_vehicle_retrograde:
            print('Vehicle Retrograde Recognition enabled')

        self.modebase = {
            "framebased": False,
            "videobased": False, 
            "idbased": False,
            "skeletonbased": False
        }

        self.basemode = {
            "MOT": "idbased",
            "ATTR": "idbased",
            "VIDEO_ACTION": "videobased",
            "SKELETON_ACTION": "skeletonbased",
            "ID_BASED_DETACTION": "idbased",
            "ID_BASED_CLSACTION": "idbased",
            "REID": "idbased",
            "VEHICLE_PLATE": "idbased",
            "VEHICLE_ATTR": "idbased",
            "VEHICLE_PRESSING": "idbased",
            "VEHICLE_RETROGRADE": "idbased",
        }

        self.is_video = is_video
        self.multi_camera = multi_camera
        self.cfg = cfg

        self.output_dir = args.output_dir
        self.draw_center_traj = args.draw_center_traj
        self.secs_interval = args.secs_interval
        self.do_entrance_counting = args.do_entrance_counting
        self.do_break_in_counting = args.do_break_in_counting
        self.region_type = args.region_type
        self.region_polygon = args.region_polygon
        self.illegal_parking_time = args.illegal_parking_time

        self.warmup_frame = self.cfg['warmup_frame']
        self.pipeline_res = Result()
        self.pipe_timer = PipeTimer()
        self.file_name = None
        self.collector = DataCollector()

        self.pushurl = args.pushurl

        # auto download inference model
        get_model_dir(self.cfg)

        if self.with_vehicleplate:
            vehicleplate_cfg = self.cfg['VEHICLE_PLATE']
            self.vehicleplate_detector = PlateRecognizer(args, vehicleplate_cfg)
            basemode = self.basemode['VEHICLE_PLATE']
            self.modebase[basemode] = True

        if self.with_human_attr:
            attr_cfg = self.cfg['ATTR']
            basemode = self.basemode['ATTR']
            self.modebase[basemode] = True
            self.attr_predictor = AttrDetector.init_with_cfg(args, attr_cfg)

        if self.with_vehicle_attr:
            vehicleattr_cfg = self.cfg['VEHICLE_ATTR']
            basemode = self.basemode['VEHICLE_ATTR']
            self.modebase[basemode] = True
            self.vehicle_attr_predictor = VehicleAttr.init_with_cfg(
                args, vehicleattr_cfg)

        if self.with_vehicle_press:
            vehiclepress_cfg = self.cfg['VEHICLE_PRESSING']
            basemode = self.basemode['VEHICLE_PRESSING']
            self.modebase[basemode] = True
            self.vehicle_press_predictor = VehiclePressingRecognizer(
                vehiclepress_cfg)

        if self.with_vehicle_press or self.with_vehicle_retrograde:
            laneseg_cfg = self.cfg['LANE_SEG']
            self.laneseg_predictor = LaneSegPredictor(
                laneseg_cfg['lane_seg_config'], laneseg_cfg['model_dir'])

        if not is_video:

            det_cfg = self.cfg['DET']
            model_dir = det_cfg['model_dir']
            batch_size = det_cfg['batch_size']
            self.det_predictor = Detector(
                model_dir, args.device, args.run_mode, batch_size,
                args.trt_min_shape, args.trt_max_shape, args.trt_opt_shape,
                args.trt_calib_mode, args.cpu_threads, args.enable_mkldnn)
        else:
            if self.with_idbased_detaction:
                idbased_detaction_cfg = self.cfg['ID_BASED_DETACTION']
                basemode = self.basemode['ID_BASED_DETACTION']
                self.modebase[basemode] = True

                self.det_action_predictor = DetActionRecognizer.init_with_cfg(
                    args, idbased_detaction_cfg)
                self.det_action_visual_helper = ActionVisualHelper(1)

            if self.with_idbased_clsaction:
                idbased_clsaction_cfg = self.cfg['ID_BASED_CLSACTION']
                basemode = self.basemode['ID_BASED_CLSACTION']
                self.modebase[basemode] = True

                self.cls_action_predictor = ClsActionRecognizer.init_with_cfg(
                    args, idbased_clsaction_cfg)
                self.cls_action_visual_helper = ActionVisualHelper(1)

            if self.with_skeleton_action:
                skeleton_action_cfg = self.cfg['SKELETON_ACTION']
                display_frames = skeleton_action_cfg['display_frames']
                self.coord_size = skeleton_action_cfg['coord_size']
                basemode = self.basemode['SKELETON_ACTION']
                self.modebase[basemode] = True
                skeleton_action_frames = skeleton_action_cfg['max_frames']

                self.skeleton_action_predictor = SkeletonActionRecognizer.init_with_cfg(
                    args, skeleton_action_cfg)
                self.skeleton_action_visual_helper = ActionVisualHelper(
                    display_frames)

                kpt_cfg = self.cfg['KPT']
                kpt_model_dir = kpt_cfg['model_dir']
                kpt_batch_size = kpt_cfg['batch_size']
                self.kpt_predictor = KeyPointDetector(
                    kpt_model_dir,
                    args.device,
                    args.run_mode,
                    kpt_batch_size,
                    args.trt_min_shape,
                    args.trt_max_shape,
                    args.trt_opt_shape,
                    args.trt_calib_mode,
                    args.cpu_threads,
                    args.enable_mkldnn,
                    use_dark=False)
                self.kpt_buff = KeyPointBuff(skeleton_action_frames)

            if self.with_vehicleplate:
                vehicleplate_cfg = self.cfg['VEHICLE_PLATE']
                self.vehicleplate_detector = PlateRecognizer(args,
                                                             vehicleplate_cfg)
                basemode = self.basemode['VEHICLE_PLATE']
                self.modebase[basemode] = True

            if self.with_mtmct:
                reid_cfg = self.cfg['REID']
                basemode = self.basemode['REID']
                self.modebase[basemode] = True
                self.reid_predictor = ReID.init_with_cfg(args, reid_cfg)

            if self.with_vehicle_retrograde:
                vehicleretrograde_cfg = self.cfg['VEHICLE_RETROGRADE']
                basemode = self.basemode['VEHICLE_RETROGRADE']
                self.modebase[basemode] = True
                self.vehicle_retrograde_predictor = VehicleRetrogradeRecognizer(
                    vehicleretrograde_cfg)

            if self.with_mot or self.modebase["idbased"] or self.modebase[
                    "skeletonbased"]:
                mot_cfg = self.cfg['MOT']
                model_dir = mot_cfg['model_dir']
                tracker_config = mot_cfg['tracker_config']
                batch_size = mot_cfg['batch_size']
                skip_frame_num = mot_cfg.get('skip_frame_num', -1)
                basemode = self.basemode['MOT']
                self.modebase[basemode] = True
                self.mot_predictor = SDE_Detector(
                    model_dir,
                    tracker_config,
                    args.device,
                    args.run_mode,
                    batch_size,
                    args.trt_min_shape,
                    args.trt_max_shape,
                    args.trt_opt_shape,
                    args.trt_calib_mode,
                    args.cpu_threads,
                    args.enable_mkldnn,
                    skip_frame_num=skip_frame_num,
                    draw_center_traj=self.draw_center_traj,
                    secs_interval=self.secs_interval,
                    do_entrance_counting=self.do_entrance_counting,
                    do_break_in_counting=self.do_break_in_counting,
                    region_type=self.region_type,
                    region_polygon=self.region_polygon)

            if self.with_video_action:
                video_action_cfg = self.cfg['VIDEO_ACTION']
                basemode = self.basemode['VIDEO_ACTION']
                self.modebase[basemode] = True
                self.video_action_predictor = VideoActionRecognizer.init_with_cfg(
                    args, video_action_cfg)

    def set_file_name(self, path):
        if type(path) == int:
            self.file_name = path
        elif path is not None:
            self.file_name = os.path.split(path)[-1]
            if "." in self.file_name:
                self.file_name = self.file_name.split(".")[-2]
        else:
            # use camera id
            self.file_name = None

    def get_result(self):
        return self.collector.get_res()

    def run(self, input, thread_idx=0):
        if self.is_video:
            self.predict_video(input, thread_idx=thread_idx)
        else:
            self.predict_image(input)
        self.pipe_timer.info()
        if hasattr(self, 'mot_predictor'):
            self.mot_predictor.det_times.tracking_info(average=True)

    def predict_image(self, input):
        # det
        # det -> attr
        batch_loop_cnt = math.ceil(
            float(len(input)) / self.det_predictor.batch_size)
        self.warmup_frame = min(10, len(input) // 2) - 1
        for i in range(batch_loop_cnt):
            start_index = i * self.det_predictor.batch_size
            end_index = min((i + 1) * self.det_predictor.batch_size, len(input))
            batch_file = input[start_index:end_index]
            batch_input = [decode_image(f, {})[0] for f in batch_file]

            if i > self.warmup_frame:
                self.pipe_timer.total_time.start()
                self.pipe_timer.module_time['det'].start()
            # det output format: class, score, xmin, ymin, xmax, ymax
            det_res = self.det_predictor.predict_image(
                batch_input, visual=False)
            det_res = self.det_predictor.filter_box(det_res,
                                                    self.cfg['crop_thresh'])
            if i > self.warmup_frame:
                self.pipe_timer.module_time['det'].end()
                self.pipe_timer.track_num += len(det_res['boxes'])
            self.pipeline_res.update(det_res, 'det')

            if self.with_human_attr:
                crop_inputs = crop_image_with_det(batch_input, det_res)
                attr_res_list = []

                if i > self.warmup_frame:
                    self.pipe_timer.module_time['attr'].start()

                for crop_input in crop_inputs:
                    attr_res = self.attr_predictor.predict_image(
                        crop_input, visual=False)
                    attr_res_list.extend(attr_res['output'])

                if i > self.warmup_frame:
                    self.pipe_timer.module_time['attr'].end()

                attr_res = {'output': attr_res_list}
                self.pipeline_res.update(attr_res, 'attr')

            if self.with_vehicle_attr:
                crop_inputs = crop_image_with_det(batch_input, det_res)
                vehicle_attr_res_list = []

                if i > self.warmup_frame:
                    self.pipe_timer.module_time['vehicle_attr'].start()

                for crop_input in crop_inputs:
                    attr_res = self.vehicle_attr_predictor.predict_image(
                        crop_input, visual=False)
                    vehicle_attr_res_list.extend(attr_res['output'])

                if i > self.warmup_frame:
                    self.pipe_timer.module_time['vehicle_attr'].end()

                attr_res = {'output': vehicle_attr_res_list}
                self.pipeline_res.update(attr_res, 'vehicle_attr')

            if self.with_vehicleplate:
                if i > self.warmup_frame:
                    self.pipe_timer.module_time['vehicleplate'].start()
                crop_inputs = crop_image_with_det(batch_input, det_res)
                platelicenses = []
                for crop_input in crop_inputs:
                    platelicense = self.vehicleplate_detector.get_platelicense(
                        crop_input)
                    platelicenses.extend(platelicense['plate'])
                if i > self.warmup_frame:
                    self.pipe_timer.module_time['vehicleplate'].end()
                vehicleplate_res = {'vehicleplate': platelicenses}
                self.pipeline_res.update(vehicleplate_res, 'vehicleplate')

            if self.with_vehicle_press:
                vehicle_press_res_list = []
                if i > self.warmup_frame:
                    self.pipe_timer.module_time['vehicle_press'].start()

                lanes, direction = self.laneseg_predictor.run(batch_input)
                if len(lanes) == 0:
                    print(" no lanes!")
                    continue

                lanes_res = {'output': lanes, 'direction': direction}
                self.pipeline_res.update(lanes_res, 'lanes')

                vehicle_press_res_list = self.vehicle_press_predictor.run(
                    lanes, det_res)
                vehiclepress_res = {'output': vehicle_press_res_list}
                self.pipeline_res.update(vehiclepress_res, 'vehicle_press')

            self.pipe_timer.img_num += len(batch_input)
            if i > self.warmup_frame:
                self.pipe_timer.total_time.end()

            if self.cfg['visual']:
                self.visualize_image(batch_file, batch_input, self.pipeline_res)

    def capturevideo(self, capture, queue):
        frame_id = 0
        while (1):
            if queue.full():
                time.sleep(0.1)
            else:
                ret, frame = capture.read()
                if not ret:
                    return
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                queue.put(frame_rgb)

      # =================== 3. 最终版 predict_video 函数 ===================
    def predict_video(self, video_file, thread_idx=0):
        # 1. Initialization
        final_results = {} 
        all_tracked_ids_with_plates = {}

        capture = cv2.VideoCapture(video_file)
        frame_id = 0
        ANALYSIS_INTERVAL = 30 

        if not self.with_mot:
            print("错误：多目标跟踪(MOT)在该配置文件中被禁用，但当前代码逻辑需要它。请使用启用了MOT的配置文件。")
            return

        while (True):
            ret, frame = capture.read()
            # --- DEBUG PRINT 1: Check if video frame was read ---
            print(f"[DEBUG] Reading next frame. Success (ret): {ret}")

            if not ret: break
            
            print(f'Thread: {thread_idx}; frame id: {frame_id}')

            # 2. Perception Layer
            res = self.mot_predictor.predict_image([frame.copy()], visual=False)
            # --- DEBUG PRINT 2: Check if the MOT model ran ---
            print(f"[DEBUG] MOT predictor finished for frame {frame_id}.")
            mot_res = parse_mot_res(res)

            
            if mot_res is None or len(mot_res['boxes']) == 0:
                frame_id += 1
                continue

            online_ids = mot_res['boxes'][:, 0].astype('int').tolist()
            online_boxes = mot_res['boxes'][:, 1:5].astype('int').tolist()

            # 3. License Plate Recognition (Corrected)
            # --- License Plate Recognition ---
            print("[DEBUG] Starting License Plate Recognition block...")
            # --- License Plate Recognition (Corrected and Robust Version) ---
            # --- License Plate Recognition (Final Debugging Version) ---
            if self.with_vehicleplate:
                print("[DEBUG] Starting License Plate Recognition block...")
                # Loop through each detected vehicle from the tracker one by one
                for i, track_id in enumerate(online_ids):
                    if track_id not in all_tracked_ids_with_plates:
                        box = online_boxes[i]
                        
                        # --- Add a check for valid box dimensions ---
                        if box[2] <= box[0] or box[3] <= box[1]:
                            print(f"[DEBUG] Invalid box dimensions for track_id {track_id}: {box}. Skipping.")
                            continue

                        cropped_image = frame[box[1]:box[3], box[0]:box[2]]
                        
                        if cropped_image.size == 0:
                            continue

                        # --- Add print statements to inspect the crop ---
                        print(f"[DEBUG] Processing track_id: {track_id}, Box: {box}, Crop Shape: {cropped_image.shape}")

                        # --- Save the crop to a file for visual inspection ---
                        # This will save the image that is about to be processed.
                        # If the script crashes, the last image saved is the one that caused it.
                        cv2.imwrite(f"/content/debug_crop_{track_id}.png", cropped_image)

                        if len(cropped_image.shape) == 2:
                            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR)

                        try:
                            platelicense_result = self.vehicleplate_detector.get_platelicense(cropped_image)
                            plate_text = platelicense_result.get('plate')
                            
                            if plate_text:
                                plate_text = plate_text[0] 
                                print(f"Frame {frame_id}: Found Plate '{plate_text}' for Track ID {track_id}")
                                all_tracked_ids_with_plates[track_id] = plate_text
                        except Exception as e:
                            # Add a try-except block to catch any possible Python-level errors
                            print(f"[DEBUG] ERROR during get_platelicense for track_id {track_id}: {e}")

                print("[DEBUG] Finished License Plate Recognition block.")
            # 4. Trigger Cognition Layer

            # ... (your license plate code) ...
            print("[DEBUG] Finished License Plate Recognition block.")
            if frame_id % ANALYSIS_INTERVAL == 0:
                print(f"\n--- Global Review Triggered --- [Frame:{frame_id}] Preparing to call VLM to analyze the current scene... ---")
                print(f"[DEBUG] VLM TRIGGERED for frame {frame_id}. Preparing to call API...")
                frame_for_vlm = frame.copy()
                for i, track_id in enumerate(online_ids):
                    box = online_boxes[i]
                    cv2.rectangle(frame_for_vlm, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), 2)
                    cv2.putText(frame_for_vlm, f"ID:{track_id}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                print("[DEBUG] >>> Calling VLM API now...")
                vlm_response_str = analyze_image_with_ark_vlm(frame_for_vlm, MASTER_PROMPT)
                print("[DEBUG] <<< VLM API call finished. Response received.") # If you see this, the call was successful.
                print(f"VLM Global Review Response: {vlm_response_str}")

                # 5. Decision Layer
                try:
                    violations_found = json.loads(vlm_response_str)
                    for violation in violations_found:
                        v_track_id = violation.get("track_id")
                        v_type = violation.get("violation_type")
                        if v_track_id and v_type and v_track_id in online_ids:
                            plate = all_tracked_ids_with_plates.get(v_track_id, "未知车牌")
                            final_results[v_track_id] = {"license_plate": plate, "violation": v_type}
                except Exception as e:
                    print(f"解析VLM返回的JSON时出错: {e}。原始回复: {vlm_response_str}")

            frame_id += 1
            # --- DEBUG PRINT 3: Confirm the end of the loop iteration ---
            print(f"[DEBUG] End of loop for frame {frame_id - 1}.")
            print(f"[DEBUG] End of loop for frame {frame_id - 1}. Continuing to next frame.")
        # 6. After the loop, finalize the results
        output_list = []
        all_processed_ids = set(final_results.keys())
        
        for track_id, data in final_results.items():
            output_list.append(data)
            
        for track_id, plate in all_tracked_ids_with_plates.items():
            if track_id not in all_processed_ids:
                output_list.append({"license_plate": plate, "violation": "无违法"})

        # 7. Save the JSON file

        if not os.path.exists(self.output_dir):
          os.makedirs(self.output_dir) 
        output_json_path = os.path.join(self.output_dir, 'auto_analysis_results.json')
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_list, f, ensure_ascii=False, indent=4)
        print(f"所有分析完成！结果已保存至: {output_json_path}")
        
        capture.release()

    def visualize_video(self,
                        image_rgb,
                        result,
                        collector,
                        frame_id,
                        fps,
                        entrance=None,
                        records=None,
                        center_traj=None,
                        do_illegal_parking_recognition=False,
                        illegal_parking_dict=None):
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        mot_res = copy.deepcopy(result.get('mot'))

        if mot_res is not None:
            ids = mot_res['boxes'][:, 0]
            scores = mot_res['boxes'][:, 2]
            boxes = mot_res['boxes'][:, 3:]
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        else:
            boxes = np.zeros([0, 4])
            ids = np.zeros([0])
            scores = np.zeros([0])

        # single class, still need to be defaultdict type for ploting
        num_classes = 1
        online_tlwhs = defaultdict(list)
        online_scores = defaultdict(list)
        online_ids = defaultdict(list)
        online_tlwhs[0] = boxes
        online_scores[0] = scores
        online_ids[0] = ids

        if mot_res is not None:
            image = plot_tracking_dict(
                image,
                num_classes,
                online_tlwhs,
                online_ids,
                online_scores,
                frame_id=frame_id,
                fps=fps,
                ids2names=self.mot_predictor.pred_config.labels,
                do_entrance_counting=self.do_entrance_counting,
                do_break_in_counting=self.do_break_in_counting,
                do_illegal_parking_recognition=do_illegal_parking_recognition,
                illegal_parking_dict=illegal_parking_dict,
                entrance=entrance,
                records=records,
                center_traj=center_traj)

        human_attr_res = result.get('attr')
        if human_attr_res is not None:
            boxes = mot_res['boxes'][:, 1:]
            human_attr_res = human_attr_res['output']
            image = visualize_attr(image, human_attr_res, boxes)
            image = np.array(image)

        vehicle_attr_res = result.get('vehicle_attr')
        if vehicle_attr_res is not None:
            boxes = mot_res['boxes'][:, 1:]
            vehicle_attr_res = vehicle_attr_res['output']
            image = visualize_attr(image, vehicle_attr_res, boxes)
            image = np.array(image)

        lanes_res = result.get('lanes')
        if lanes_res is not None:
            lanes = lanes_res['output'][0]
            image = visualize_lane(image, lanes)
            image = np.array(image)

        vehiclepress_res = result.get('vehicle_press')
        if vehiclepress_res is not None:
            press_vehicle = vehiclepress_res['output']
            if len(press_vehicle) > 0:
                image = visualize_vehiclepress(
                    image, press_vehicle, threshold=self.cfg['crop_thresh'])
                image = np.array(image)

        if mot_res is not None:
            vehicleplate = False
            plates = []
            for trackid in mot_res['boxes'][:, 0]:
                plate = collector.get_carlp(trackid)
                if plate != None:
                    vehicleplate = True
                    plates.append(plate)
                else:
                    plates.append("")
            if vehicleplate:
                boxes = mot_res['boxes'][:, 1:]
                image = visualize_vehicleplate(image, plates, boxes)
                image = np.array(image)

        kpt_res = result.get('kpt')
        if kpt_res is not None:
            image = visualize_pose(
                image,
                kpt_res,
                visual_thresh=self.cfg['kpt_thresh'],
                returnimg=True)

        video_action_res = result.get('video_action')
        if video_action_res is not None:
            video_action_score = None
            if video_action_res and video_action_res["class"] == 1:
                video_action_score = video_action_res["score"]
            mot_boxes = None
            if mot_res:
                mot_boxes = mot_res['boxes']
            image = visualize_action(
                image,
                mot_boxes,
                action_visual_collector=None,
                action_text="SkeletonAction",
                video_action_score=video_action_score,
                video_action_text="Fight")

        vehicle_retrograde_res = result.get('vehicle_retrograde')
        if vehicle_retrograde_res is not None:
            mot_retrograde_res = copy.deepcopy(result.get('mot'))
            image = visualize_vehicle_retrograde(image, mot_retrograde_res,
                                                 vehicle_retrograde_res)
            image = np.array(image)

        visual_helper_for_display = []
        action_to_display = []

        skeleton_action_res = result.get('skeleton_action')
        if skeleton_action_res is not None:
            visual_helper_for_display.append(self.skeleton_action_visual_helper)
            action_to_display.append("Falling")

        det_action_res = result.get('det_action')
        if det_action_res is not None:
            visual_helper_for_display.append(self.det_action_visual_helper)
            action_to_display.append("Smoking")

        cls_action_res = result.get('cls_action')
        if cls_action_res is not None:
            visual_helper_for_display.append(self.cls_action_visual_helper)
            action_to_display.append("Calling")

        if len(visual_helper_for_display) > 0:
            image = visualize_action(image, mot_res['boxes'],
                                     visual_helper_for_display,
                                     action_to_display)

        return image

    def visualize_image(self, im_files, images, result):
        start_idx, boxes_num_i = 0, 0
        det_res = result.get('det')
        human_attr_res = result.get('attr')
        vehicle_attr_res = result.get('vehicle_attr')
        vehicleplate_res = result.get('vehicleplate')
        lanes_res = result.get('lanes')
        vehiclepress_res = result.get('vehicle_press')

        for i, (im_file, im) in enumerate(zip(im_files, images)):
            if det_res is not None:
                det_res_i = {}
                boxes_num_i = det_res['boxes_num'][i]
                det_res_i['boxes'] = det_res['boxes'][start_idx:start_idx +
                                                      boxes_num_i, :]
                im = visualize_box_mask(
                    im,
                    det_res_i,
                    labels=['target'],
                    threshold=self.cfg['crop_thresh'])
                im = np.ascontiguousarray(np.copy(im))
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            if human_attr_res is not None:
                human_attr_res_i = human_attr_res['output'][start_idx:start_idx
                                                            + boxes_num_i]
                im = visualize_attr(im, human_attr_res_i, det_res_i['boxes'])
            if vehicle_attr_res is not None:
                vehicle_attr_res_i = vehicle_attr_res['output'][
                    start_idx:start_idx + boxes_num_i]
                im = visualize_attr(im, vehicle_attr_res_i, det_res_i['boxes'])
            if vehicleplate_res is not None:
                plates = vehicleplate_res['vehicleplate']
                det_res_i['boxes'][:, 4:6] = det_res_i[
                    'boxes'][:, 4:6] - det_res_i['boxes'][:, 2:4]
                im = visualize_vehicleplate(im, plates, det_res_i['boxes'])
            if vehiclepress_res is not None:
                press_vehicle = vehiclepress_res['output'][i]
                if len(press_vehicle) > 0:
                    im = visualize_vehiclepress(
                        im, press_vehicle, threshold=self.cfg['crop_thresh'])
                    im = np.ascontiguousarray(np.copy(im))
            if lanes_res is not None:
                lanes = lanes_res['output'][i]
                im = visualize_lane(im, lanes)
                im = np.ascontiguousarray(np.copy(im))

            img_name = os.path.split(im_file)[-1]
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            out_path = os.path.join(self.output_dir, img_name)
            cv2.imwrite(out_path, im)
            print("save result to: " + out_path)
            start_idx += boxes_num_i

def main():
    cfg = merge_cfg(FLAGS)  # use command params to update config
    print_arguments(cfg)

    pipeline = Pipeline(FLAGS, cfg)
    # pipeline.run()
    pipeline.run_multithreads()

if __name__ == '__main__':
    paddle.enable_static()

    # parse params from command
    parser = argsparser()

    FLAGS = parser.parse_args()
    FLAGS.device = FLAGS.device.upper()
    assert FLAGS.device in ['CPU', 'GPU', 'XPU', 'NPU', 'GCU'
                            ], "device should be CPU, GPU, XPU, NPU or GCU"

    main()





