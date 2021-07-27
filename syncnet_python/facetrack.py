#!/usr/bin/python
import sys, time, os, pdb, argparse, pickle, subprocess, glob, cv2, random
import numpy as np
from shutil import rmtree
import GPUtil
import scenedetect
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector
from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy import signal
from syncnet_python.detectors import S3FD      
        
class FaceTrack:
    
    def __init__(self, device='cuda:0'):
        self.facedet_scale = 0.25
        self.crop_scale = 0.40
        self.min_track = 100
        self.frame_rate = 25
        self.num_failed_det = 25
        self.min_face_size = 100
        self.reference = 'tracks'
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
                framenum    = np.array([ f['frame'] for f in track ])
                bboxes      = np.array([np.array(f['bbox']) for f in track])
                frame_i   = np.arange(framenum[0],framenum[-1]+1)
                bboxes_i    = []
                for ij in range(0,4):
                    interpfn  = interp1d(framenum, bboxes[:,ij])
                    bboxes_i.append(interpfn(frame_i))
                bboxes_i  = np.stack(bboxes_i, axis=1)
                if max(np.mean(bboxes_i[:,2]-bboxes_i[:,0]), np.mean(bboxes_i[:,3]-bboxes_i[:,1])) > self.min_face_size:
                    tracks.append({'frame':frame_i,'bbox':bboxes_i})
        return tracks

    def crop_video(self,track,cropfile,frames):
        flist = glob.glob(os.path.join(self.frames_dir,self.reference,'*.jpg'))
        flist.sort()
        print(f"The shape of flist is: {len(flist)}")
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
            bs  = dets['s'][fidx]   # Detection box size
            bsi = int(bs*(1+2*cs))  # Pad videos by this amount
            image = frames[f].numpy() #  cv2.imread(flist[f]) #
#             print(f"flist[frame] image when read through cv2 gives: {type(cv2.imread(flist[f]))},{cv2.imread(flist[f]).shape}, {cv2.imread(flist[f])[0]}")
#             print(f"frames[frame].numpy(): {type(frames[f].numpy())}, {frames[f].numpy().shape}, {frames[f].numpy()[0]}")
#             print(f"is true? {frames[f].numpy() == cv2.imread(flist[f])}")
            frame = np.pad(image,((bsi,bsi),(bsi,bsi),(0,0)), 'constant', constant_values=(110,110))
            my  = dets['y'][fidx]+bsi  # BBox center Y
            mx  = dets['x'][fidx]+bsi  # BBox center X
            face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
            vOut.write(cv2.resize(face,(224,224)))
            cv2.imwrite('frames f numpy broken.png', frames[f].numpy())
            cv2.imwrite('good cv2 imread flist.png', cv2.imread(flist[f]))

        audiotmp    = os.path.join(self.tmp_dir,self.reference,'audio.wav')
        audiostart  = (track['frame'][0])/self.frame_rate
        audioend    = (track['frame'][-1]+1)/self.frame_rate
        f = open(os.path.join(self.crop_dir,self.reference, "trimtimes.txt"), "a")
        f.write(f"\n{audiostart}, {audioend}")
        f.close()
        vOut.release()
        # Crop audio file
        command = ("ffmpeg -loglevel quiet -y -i %s -ss %.3f -to %.3f %s" % (os.path.join(self.avi_dir,self.reference,'audio.wav'),audiostart,audioend,audiotmp))
        output = subprocess.call(command, shell=True, stdout=None)
        if output != 0:
            pdb.set_trace()
        sample_rate, audio = wavfile.read(audiotmp)
      # Combine audio and video files
        command = ("ffmpeg -loglevel quiet -y -i %st.avi -i %s -c:v copy -c:a copy %s.avi" % (cropfile,audiotmp,cropfile))
        output = subprocess.call(command, shell=True, stdout=None)
        if output != 0:
            pdb.set_trace()
        os.remove(cropfile+'t.avi')
        return {'track':track, 'proc_track':dets}

    
    def face_detection(self, frames):
        
        dets = []
        for fidx, frame in enumerate(frames):
            frame = cv2.cvtColor(frame.numpy(), cv2.COLOR_BGR2RGB)
            start_time = time.time()
            bboxes = self.DET.detect_faces(frame, conf_th=0.9, scales=[self.facedet_scale])
            dets.append([])
            for bbox in bboxes:
                dets[-1].append({'frame':fidx, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]})
            elapsed_time = time.time() - start_time
        return dets

    def scene_detection(self):
        video_manager = VideoManager([os.path.join(self.avi_dir,self.reference,'video.avi')])
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
        setattr(self,'work_dir', os.path.join(self.data_dir,'pywork'))
        setattr(self,'crop_dir', os.path.join(self.data_dir,'pycrop'))
        setattr(self,'frames_dir', os.path.join(self.data_dir,'pyframes'))
        if os.path.exists(os.path.join(self.work_dir,self.reference)):
            rmtree(os.path.join(self.work_dir,self.reference))
        if os.path.exists(os.path.join(self.crop_dir,self.reference)):
            rmtree(os.path.join(self.crop_dir,self.reference))
#         if os.path.exists(os.path.join(self.frames_dir,self.reference)):
#             rmtree(os.path.join(self.frames_dir,self.reference))
        if os.path.exists(os.path.join(self.tmp_dir,self.reference)):
            rmtree(os.path.join(self.tmp_dir,self.reference))
        os.makedirs(os.path.join(self.work_dir,self.reference))
        os.makedirs(os.path.join(self.crop_dir,self.reference))
#         os.makedirs(os.path.join(self.frames_dir,self.reference))
        os.makedirs(os.path.join(self.tmp_dir,self.reference))
        
#         command = ("ffmpeg -loglevel quiet -y -i %s -qscale:v 2 -threads 1 -f image2 %s" % (os.path.join(self.avi_dir,self.reference,'video.avi'),os.path.join(self.frames_dir,self.reference,'%06d.jpg'))) 
#         output = subprocess.call(command, shell=True, stdout=None)

        command = ("ffmpeg -loglevel quiet -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (os.path.join(self.avi_dir,self.reference,'video.avi'),os.path.join(self.avi_dir,self.reference,'audio.wav'))) 
        output = subprocess.call(command, shell=True, stdout=None)
        
        faces = self.face_detection(frames)
        scene = self.scene_detection()

        alltracks = []
        vidtracks = []
        for shot in scene:
            if shot[1].frame_num - shot[0].frame_num >= self.min_track:
                alltracks.extend(self.track_shot(faces[shot[0].frame_num:shot[1].frame_num]))
        for ii, track in enumerate(alltracks):
            vidtracks.append(self.crop_video(track,os.path.join(self.crop_dir,self.reference,'%05d'%ii),frames))
        rmtree(os.path.join(self.tmp_dir,self.reference))                                                                                                            