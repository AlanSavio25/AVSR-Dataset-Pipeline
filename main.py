import torch 
import numpy as np
import cv2
import time, glob, shutil, datetime
from matplotlib import pyplot as plt
import csv, json
import xml.etree.ElementTree as ET 
import pandas as pd
from collections import defaultdict
from scipy.io import wavfile
from syncnet_python.facetrack import *
from syncnet_python.syncnet import *
from itertools import cycle
from utils.functions import cut_into_utterances, get_genre, \
prepare_output_directory, create_transcript_from_XML, cleanup

class VideoIterableDataset(torch.utils.data.IterableDataset):
    
    def __init__(self, data_dir):
        super(VideoIterableDataset).__init__()
        self.utts = []
        self.avis = []
        for utt in glob.glob(data_dir+'*'):
            self.utts.append(utt)
            frames_dir = utt+'/pyframes/'
            if os.path.exists(frames_dir):
                rmtree(frames_dir)
            os.makedirs(frames_dir)
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            offset = 0
            shift = 1
        else:
            offset = worker_info.id
            shift = worker_info.num_workers
        for i in range(offset, len(self.utts), shift):
            yield self.utts[i], self.load_frames(self.utts[i])
    
    def load_frames(self, utt):
        
        videofile = os.path.join(utt,'pyavi','video.avi')
        output = os.path.join(utt,'pyframes','%06d.jpg')
        command = f"ffmpeg -loglevel quiet -y -i {videofile} -qscale:v 2 -threads 1 -f image2 {output}"
        output = subprocess.call(command, shell=True, stdout=None)
        flist = glob.glob(utt+'/pyframes/*.jpg')
        flist.sort()
        frames = []
        for fname in flist:
            image = cv2.imread(fname)
            frames.append(image)
        return np.array(frames)
    
class SyncNetIterableDataset(torch.utils.data.IterableDataset):    
   
    def __init__(self, path):
        super(SyncNetIterableDataset).__init__()
        self.avis = []
        for avi in glob.glob(path+'*/pycrop/*.avi'):
            self.avis.append(avi)
            tmp_dir = avi.split('pycrop')[0]+'/pytmp/'
            if os.path.exists(tmp_dir):
                rmtree(tmp_dir)
            os.makedirs(tmp_dir)
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            offset = 0
            shift = 1
        else:
            offset = worker_info.id
            shift = worker_info.num_workers
        for i in range(offset, len(self.avis), shift):
            utt = self.avis[i].split('/pycrop/')[0]
            yield utt, self.avis[i], self.load_frames(self.avis[i]), self.load_audio(utt, self.avis[i])
    
    def load_audio(self, utt, videofile):
        command = f"ffmpeg -loglevel quiet -y -i {videofile} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {os.path.join(utt,'pytmp/audio.wav')}"
        output = subprocess.call(command, shell=True, stdout=None)
        sample_rate, audio = wavfile.read(os.path.join(utt,'pytmp/audio.wav'))
        return (sample_rate, audio)
    
    def load_frames(self, videofile):
        cap = cv2.VideoCapture(videofile)
        frame_num = 1;
        frames = []
        while frame_num:
            frame_num += 1
            ret, image = cap.read()
            if ret == 0:
                break
            frames.append(cv2.resize(image, (224, 224)))
        cap.release()
        cv2.destroyAllWindows()
        frames = [frames[0], frames[0]] + frames + [frames[-1], frames[-1]]
        frames = np.stack(frames, axis=3)
        frames = np.transpose(frames, (2,3,0,1))
        frames = np.array([frames[:,i:i+5,:,:] for i in range(0, frames.shape[1] - 4)], dtype='float32')
        return frames

def main(data_dir='/disk/scratch/s1768177/pipeline/output_data/'):
    
    filelist = "/afs/inf.ed.ac.uk/group/cstr/datawww/asru/MGB1/scripts/dev.full"
    desired_genres = ["drama", "childrens", "news", "documentary"]
    print(f"{datetime.datetime.now()}\n")
    script_start = time.time()
    
    # 1. Crop utterances
    count = 1
    with open(filelist, "r") as f:
        files = f.read().split()
#     files = files[9:11]
    print(f"Cutting utterances from raw videos.")
    total_utterances_processed = 0
    for filename in files:
        genre = get_genre(filename)
        if (genre in desired_genres):
            print(f"{count}. {filename}. ({genre}) ")
            count += 1
            utterance_items = cut_into_utterances(filename, data_dir)
            total_utterances_processed += len(utterance_items)
    print(f"\nFinished cutting total {total_utterances_processed} utterances from {count-1} videos\n")
    
    # 2. Generate face tracks
    start = time.time()
    dataset = VideoIterableDataset(data_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=False, num_workers=24)
    facetrack = FaceTrack()
    for i, (utt, frames) in enumerate(dataloader):
        print(i, utt.split('/')[-1], len(frames))
        facetrack.run(data_dir=utt, frames=frames)
        no_faces_found = len(os.listdir(utt + "/pycrop/")) == 0
        if(no_faces_found):
            shutil.rmtree(utt)
    print(f"Time taken: {(time.time()-start)/60:.2f} minutes\n")
    
    # 3. Compute Confidence scores
    start = time.time()
    dataset = SyncNetIterableDataset(data_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=False, num_workers=24)
    syncnet = SyncNet()
    for i, (utt, avi, frames, (sample_rate, audio)) in enumerate(dataloader):
        print(i, utt, avi.split('/')[-1], len(frames))
        syncnet.setup(utt)
        offset, conf, dist = syncnet.evaluate(avi,frames,sample_rate,audio)
        print(offset, conf)
    print(f"Time taken: {(time.time()-start)/60:.2f} minutes\n")

    cleanup(data_dir)
    
    print(f"Script running time: {time.time() - script_start}")
    
if __name__ == '__main__':
    main()