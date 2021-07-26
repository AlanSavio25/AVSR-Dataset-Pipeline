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
        
       
        self.DET = S3FD(device)

        
        
    # IOU Function
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

    # Face Tracking
    def track_shot(self,scenefaces):

      iouThres  = 0.5     # Minimum IOU between consecutive face detections
      tracks    = []

      while True:
        track     = []
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


    # VIDEO CROP AND SAVE
    def crop_video(self,track,cropfile):

      flist = glob.glob(os.path.join(self.frames_dir,self.reference,'*.jpg'))
      flist.sort()

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

      for fidx, frame in enumerate(track['frame']):

        cs  = self.crop_scale

        bs  = dets['s'][fidx]   # Detection box size
        bsi = int(bs*(1+2*cs))  # Pad videos by this amount

        image = cv2.imread(flist[frame])

        frame = np.pad(image,((bsi,bsi),(bsi,bsi),(0,0)), 'constant', constant_values=(110,110))
        my  = dets['y'][fidx]+bsi  # BBox center Y
        mx  = dets['x'][fidx]+bsi  # BBox center X

        face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
        
        vOut.write(cv2.resize(face,(224,224)))

      audiotmp    = os.path.join(self.tmp_dir,self.reference,'audio.wav')
      audiostart  = (track['frame'][0])/self.frame_rate
      audioend    = (track['frame'][-1]+1)/self.frame_rate

      f = open(os.path.join(self.crop_dir,self.reference, "trimtimes.txt"), "a")
#       print(os.path.join(self.crop_dir,self.reference, "trimtimes.txt"))
      f.write(f"\n{audiostart}, {audioend}")
      f.close()
      vOut.release()

      # Crop audio file

      command = ("ffmpeg -loglevel quiet -y -i %s -ss %.3f -to %.3f %s" % (os.path.join(self.avi_dir,self.reference,'audio.wav'),audiostart,audioend,audiotmp)) 
      output = subprocess.call(command, shell=True, stdout=None)
#       print(f"audiostart: {audiostart}, audioend: {audioend}")
      if output != 0:
        pdb.set_trace()

      sample_rate, audio = wavfile.read(audiotmp)

      # Combine audio and video files

      command = ("ffmpeg -loglevel quiet -y -i %st.avi -i %s -c:v copy -c:a copy %s.avi" % (cropfile,audiotmp,cropfile))
      output = subprocess.call(command, shell=True, stdout=None)

      if output != 0:
        pdb.set_trace()

#       print('Written %s'%cropfile)

      os.remove(cropfile+'t.avi')

#       print('Mean pos: x %.2f y %.2f s %.2f'%(np.mean(dets['x']),np.mean(dets['y']),np.mean(dets['s'])))

      return {'track':track, 'proc_track':dets}

    
    def face_detection(self):

      flist = glob.glob(os.path.join(self.frames_dir,self.reference,'*.jpg'))
      flist.sort()

      dets = []
      for fidx, fname in enumerate(flist):

        start_time = time.time()

        image = cv2.imread(fname)

        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = self.DET.detect_faces(image_np, conf_th=0.9, scales=[self.facedet_scale])
        dets.append([]);
        for bbox in bboxes:
          dets[-1].append({'frame':fidx, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]})

        elapsed_time = time.time() - start_time

#         print('%s-%05d; %d dets; %.2f Hz' % (os.path.join(self.avi_dir,self.reference,'video.avi'),fidx,len(dets[-1]),(1/elapsed_time))) 

      savepath = os.path.join(self.work_dir,self.reference,'faces.pckl')

      with open(savepath, 'wb') as fil:
        pickle.dump(dets, fil)

      return dets

    # Scene detection

    def scene_detection(self):
#       print(self.avi_dir, self.reference)
      video_manager = VideoManager([os.path.join(self.avi_dir,self.reference,'video.avi')])
      stats_manager = StatsManager()
      scene_manager = SceneManager(stats_manager)
      # Add ContentDetector algorithm (constructor takes detector selfions like threshold).
      scene_manager.add_detector(ContentDetector())
      base_timecode = video_manager.get_base_timecode()

      video_manager.set_downscale_factor()

      video_manager.start()

      scene_manager.detect_scenes(frame_source=video_manager)

      scene_list = scene_manager.get_scene_list(base_timecode)

      savepath = os.path.join(self.work_dir,self.reference,'scene.pckl')

      if scene_list == []:
        scene_list = [(video_manager.get_base_timecode(),video_manager.get_current_timecode())]

      with open(savepath, 'wb') as fil:
        pickle.dump(scene_list, fil)

#       print('%s - scenes detected %d'%(os.path.join(self.avi_dir,self.reference,'video.avi'),len(scene_list)))

      return scene_list

    def delete_existing_directories(self):
        
        if os.path.exists(os.path.join(self.work_dir,self.reference)):
          rmtree(os.path.join(self.work_dir,self.reference))

        if os.path.exists(os.path.join(self.crop_dir,self.reference)):
          rmtree(os.path.join(self.crop_dir,self.reference))

#         if os.path.exists(os.path.join(self.avi_dir,self.reference)):
#           rmtree(os.path.join(self.avi_dir,self.reference))

        if os.path.exists(os.path.join(self.frames_dir,self.reference)):
          rmtree(os.path.join(self.frames_dir,self.reference))

        if os.path.exists(os.path.join(self.tmp_dir,self.reference)):
          rmtree(os.path.join(self.tmp_dir,self.reference))

    def make_new_directories(self):
        os.makedirs(os.path.join(self.work_dir,self.reference))
        os.makedirs(os.path.join(self.crop_dir,self.reference))
#         os.makedirs(os.path.join(self.avi_dir,self.reference))
        os.makedirs(os.path.join(self.frames_dir,self.reference))
        os.makedirs(os.path.join(self.tmp_dir,self.reference))
    
    def run(self, data_dir='',videofile='',reference=''):
        
        # Initial Setup
        setattr(self, 'data_dir', data_dir)
#       setattr(self, 'videofile', videofile)
        setattr(self, 'reference', reference)
        
        setattr(self,'avi_dir', os.path.join(self.data_dir,'pyavi'))
        setattr(self,'tmp_dir', os.path.join(self.data_dir,'pytmp'))
        setattr(self,'work_dir', os.path.join(self.data_dir,'pywork'))
        setattr(self,'crop_dir', os.path.join(self.data_dir,'pycrop'))
        setattr(self,'frames_dir', os.path.join(self.data_dir,'pyframes'))
        
        self.delete_existing_directories()
        self.make_new_directories()

        # Convert video and extract frames

#         command = ("ffmpeg -loglevel quiet -y -i %s -qscale:v 2 -async 1 -r 25 %s" % (self.videofile,os.path.join(self.avi_dir,self.reference,'video.avi')))
#         output = subprocess.call(command, shell=True, stdout=None)

        command = ("ffmpeg -loglevel quiet -y -i %s -qscale:v 2 -threads 1 -f image2 %s" % (os.path.join(self.avi_dir,self.reference,'video.avi'),os.path.join(self.frames_dir,self.reference,'%06d.jpg'))) 
        output = subprocess.call(command, shell=True, stdout=None)

        command = ("ffmpeg -loglevel quiet -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (os.path.join(self.avi_dir,self.reference,'video.avi'),os.path.join(self.avi_dir,self.reference,'audio.wav'))) 
        output = subprocess.call(command, shell=True, stdout=None)



        faces = self.face_detection()
#         print("Face detection complete.")
        scene = self.scene_detection()
#         print("Scene detection complete.")

    # ========== FACE TRACKING ==========

        alltracks = []
        vidtracks = []

        for shot in scene:

          if shot[1].frame_num - shot[0].frame_num >= self.min_track :
            alltracks.extend(self.track_shot(faces[shot[0].frame_num:shot[1].frame_num]))

        # ========== FACE TRACK CROP ==========

        for ii, track in enumerate(alltracks):
            vidtracks.append(self.crop_video(track,os.path.join(self.crop_dir,self.reference,'%05d'%ii)))

        # ========== SAVE RESULTS ==========

#         savepath = os.path.join(self.work_dir,self.reference,'tracks.pckl')

#         with open(savepath, 'wb') as fil:
#           pickle.dump(vidtracks, fil)

        rmtree(os.path.join(self.tmp_dir,self.reference))

    
    
    
    
    
    
          #try:
#       gpu_list = GPUtil.getAvailable(order='memory', limit=3, excludeID=[0], maxLoad=0.90, maxMemory=0.85)
#       if (len(gpu_list)==0):
#         gpu = GPUtil.getFirstAvailable(order='memory', maxLoad=0.55, maxMemory=0.55, attempts=3, interval=7)[0]
#       else:
#         gpu = random.choice(gpu_list)
        
#       print(gpu)
      #except Exception as e:
       # print("Error finding free gpu")
       # print(e)
       # raise Exception("Could not find any available GPU after 3 attempts")                                                                                                                   