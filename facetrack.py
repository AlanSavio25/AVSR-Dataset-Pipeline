#!/usr/bin/python
import sys, time, os, subprocess, glob, cv2
import numpy as np
import scenedetect
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector
from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy import signal
from detectors import S3FD
import logging
logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)
        
class FaceTrack:
    
    def __init__(self, device='cuda:0'):
        self.facedet_scale = 0.25
        self.crop_scale = 0.40
        self.min_track = 100
        self.frame_rate = 25
        self.num_failed_det = 25
        self.min_face_size = 100
        self.DET = S3FD(device)
        
    def bb_intersection_over_union(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def track_shot(self,scenefaces):
        iouThres = 0.75    # Minimum IOU between consecutive face detections
        tracks = []
        while True:
            track = []
            for framefaces in scenefaces:
                for face in framefaces:
                    if track == []:
                        track.append(face)
                        framefaces.remove(face)
                    elif face['frame'] - track[-1]['frame'] <= self.num_failed_det:
                        iou = self.bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
                        if iou > iouThres:
                            track.append(face)
                            framefaces.remove(face)
                            continue
                    else:
                        break
            if track == []:
                break
            elif len(track) > self.min_track:
                framenum = np.array([ f['frame'] for f in track ])
                bboxes = np.array([np.array(f['bbox']) for f in track])
                frame_i = np.arange(framenum[0],framenum[-1]+1)
                bboxes_i = []
                for ij in range(0,4):
                    interpfn  = interp1d(framenum, bboxes[:,ij])
                    bboxes_i.append(interpfn(frame_i))
                bboxes_i  = np.stack(bboxes_i, axis=1)
                if max(np.mean(bboxes_i[:,2]-bboxes_i[:,0]), np.mean(bboxes_i[:,3]-bboxes_i[:,1])) > self.min_face_size:
                    tracks.append({'frame':frame_i,'bbox':bboxes_i})
        return tracks

    def crop_video(self,track,cropfile,frames):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        vOut = cv2.VideoWriter(cropfile+'t.avi', fourcc, self.frame_rate, (224,224))
        dets = {'x':[], 'y':[], 's':[]}
        for det in track['bbox']:
            dets['s'].append(max((det[3]-det[1]),(det[2]-det[0]))/2) 
            dets['y'].append((det[1]+det[3])/2) # crop center x 
            dets['x'].append((det[0]+det[2])/2) # crop center y
        # Smooth detections
        dets['s'] = signal.medfilt(dets['s'],kernel_size=13)
        dets['x'] = signal.medfilt(dets['x'],kernel_size=13)
        dets['y'] = signal.medfilt(dets['y'],kernel_size=13)
        for fidx, f in enumerate(track['frame']):
            cs  = self.crop_scale
            bs  = dets['s'][fidx]     # Detection box size
            bsi = int(bs*(1+2*cs))    # Pad videos by this amount
            image = frames[f].numpy()
            frame = np.pad(image,((bsi,bsi),(bsi,bsi),(0,0)), 'constant', constant_values=(110,110))
            my  = dets['y'][fidx]+bsi  # BBox center Y
            mx  = dets['x'][fidx]+bsi  # BBox center X
            face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
            vOut.write(cv2.resize(face,(224,224)))
        audiotmp    = os.path.join(self.tmp_dir,'audio.wav')
        audiostart  = (track['frame'][0])/self.frame_rate
        audioend    = (track['frame'][-1]+1)/self.frame_rate
        with open(os.path.join(self.crop_dir, "trimtimes.txt"), "a") as f:
            f.write(f"\n{audiostart}, {audioend}")
        vOut.release()
        # Crop audio file
        command = f"ffmpeg -loglevel quiet -y -i {os.path.join(self.avi_dir,'audio.wav')} -ss {audiostart:.3f} -to {audioend:.3f} {audiotmp}"
        output = subprocess.call(command, shell=True, stdout=None)
        if output != 0:
            logging.exception("Error cropping audio.", exc_info=True)
        sample_rate, audio = wavfile.read(audiotmp)
      # Combine audio and video files
        command = ("ffmpeg -loglevel quiet -y -i %st.avi -i %s -c:v copy -c:a copy %s.avi" % (cropfile,audiotmp,cropfile))
        output = subprocess.call(command, shell=True, stdout=None)
        if output != 0:
            logging.exception("Error combining audio and video files", exc_info=True)
        os.remove(cropfile+'t.avi')
        return {'track':track, 'proc_track':dets}

    
    def face_detection(self, frames):
        dets = []
        for fidx, frame in enumerate(frames):
            frame = cv2.cvtColor(frame.numpy(), cv2.COLOR_BGR2RGB)
            bboxes = self.DET.detect_faces(frame, conf_th=0.9, scales=[self.facedet_scale])
            dets.append([])
            for bbox in bboxes:
                dets[-1].append({'frame':fidx, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]})
        return dets

    def scene_detection(self):
        # Note: Below is not optimal as it reads the video from disk and internally converts it
        # into images, instead of simply using previously converted images
        video_manager = VideoManager([os.path.join(self.avi_dir,'video.avi')])
        stats_manager = StatsManager()
        scene_manager = SceneManager(stats_manager)
        scene_manager.add_detector(ContentDetector())
        base_timecode = video_manager.get_base_timecode()
        video_manager.set_downscale_factor()
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list(base_timecode)
        if scene_list == []:
            scene_list = [(video_manager.get_base_timecode(),video_manager.get_current_timecode())]
        return scene_list

    
    def run(self, data_dir='',frames=None):

        setattr(self, 'data_dir', data_dir)
        setattr(self,'avi_dir', os.path.join(self.data_dir,'pyavi'))
        setattr(self,'tmp_dir', os.path.join(self.data_dir,'pytmp'))
        setattr(self,'crop_dir', os.path.join(self.data_dir,'pycrop'))
        setattr(self,'frames_dir', os.path.join(self.data_dir,'pyframes'))
        if os.path.exists(self.crop_dir):
            rmtree(self.crop_dir)
        if os.path.exists(self.tmp_dir):
            rmtree(self.tmp_dir)
        os.makedirs(self.crop_dir)
        os.makedirs(self.tmp_dir)
        
        command = ("ffmpeg -loglevel quiet -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (os.path.join(self.avi_dir,'video.avi'),os.path.join(self.avi_dir,'audio.wav'))) 
        output = subprocess.call(command, shell=True, stdout=None)
        
        faces = self.face_detection(frames)
        scene = self.scene_detection()

        alltracks = []
        vidtracks = []
        for shot in scene:
            if shot[1].frame_num - shot[0].frame_num >= self.min_track:
                alltracks.extend(self.track_shot(faces[shot[0].frame_num:shot[1].frame_num]))
        for ii, track in enumerate(alltracks):
            vidtracks.append(self.crop_video(track,os.path.join(self.crop_dir,'%05d'%ii),frames))
        rmtree(self.tmp_dir)                                                                                                    
