#!/usr/bin/python
#-*- coding: utf-8 -*-

# Video 25 FPS, Audio 16000HZ
import torch
import numpy
import time, pdb, argparse, subprocess, os, math, glob, pickle, gzip
import cv2
import python_speech_features
from scipy import signal
from scipy.io import wavfile
from syncnet_python.SyncNetModel import *
from shutil import rmtree
import pandas as pd

def calc_pdist(feat1, feat2, vshift=10):
    win_size = vshift*2+1
    feat2p = torch.nn.functional.pad(feat2,(0,0,vshift,vshift))
    dists = []
    for i in range(0,len(feat1)):
        dists.append(torch.nn.functional.pairwise_distance(feat1[[i],:].repeat(win_size, 1), feat2p[i:i+win_size,:]))
    return dists

class SyncNetInstance(torch.nn.Module):

#     def __init__(self, dropout = 0, num_layers_in_fc_layers = 1024):
#         super(SyncNetInstance, self).__init__();
#         self.__S__ = S(num_layers_in_fc_layers = num_layers_in_fc_layers).cuda('cuda:3');
        
#     def loadParameters(self, path):
#         loaded_state = torch.load(path, map_location=lambda storage, loc: storage);
#         model_state = self.__S__.state_dict();
#         for name, param in loaded_state.items():
#             model_state[name].copy_(param);

    def extract_feature(self, opt, videofile):

        self.__S__.eval();       
        cap = cv2.VideoCapture(videofile)
        frame_num = 1;
        images = []
        while frame_num:
            frame_num += 1
            ret, image = cap.read()
            if ret == 0:
                break
            images.append(image)
        im = numpy.stack(images,axis=3)
        im = numpy.expand_dims(im,axis=0)
        im = numpy.transpose(im,(0,3,4,1,2))
        imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())
        lastframe = len(images)-4
        im_feat = []
        tS = time.time()
        for i in range(0,lastframe,opt.batch_size):
            im_batch = [ imtv[:,:,vframe:vframe+5,:,:] for vframe in range(i,min(lastframe,i+opt.batch_size)) ]
            im_in = torch.cat(im_batch,0)
            im_out  = self.__S__.forward_lipfeat(im_in.cuda());
            im_feat.append(im_out.data.cpu())
        im_feat = torch.cat(im_feat,0)
        return im_feat

class SyncNet:

    def __init__(self, device='cuda:0'):
        
        modelpath = 'syncnet_python/data/syncnet_v2.model'
        model = S().cuda(device)
        model_state = model.state_dict();
        loaded_state = torch.load(modelpath, map_location=lambda storage, loc: storage);
        for name, param in loaded_state.items():
            model_state[name].copy_(param);
        self.model = model
        print(f"Model {modelpath} loaded.")

        self.batch_size = 20
        self.vshift = 15
        self.reference = 'tracks'
        
    def setup(self, data_dir=''):
        self.data_dir = data_dir
        setattr(self,'avi_dir',os.path.join(self.data_dir,'pyavi'))
        setattr(self,'tmp_dir',os.path.join(self.data_dir,'pytmp'))
        setattr(self,'work_dir',os.path.join(self.data_dir,'pywork'))
        setattr(self,'crop_dir',os.path.join(self.data_dir,'pycrop'))
        
#         df = pd.DataFrame.from_dict(info)
#         df.to_csv(os.path.join(self.crop_dir,self.reference,"activespeakerinfo.csv"), index = False)
#         self.save_results(dists)

    def evaluate(self, videofile, frames):
        self.model.eval()
        if os.path.exists(os.path.join(self.tmp_dir,self.reference)):
            rmtree(os.path.join(self.tmp_dir,self.reference))
        os.makedirs(os.path.join(self.tmp_dir,self.reference))
        
        command = ("ffmpeg -loglevel quiet -y -i %s -threads 1 -f image2 %s" % (videofile,os.path.join(self.tmp_dir,self.reference,'%06d.jpg'))) 
        output = subprocess.call(command, shell=True, stdout=None)

        command = ("ffmpeg -loglevel quiet -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (videofile,os.path.join(self.tmp_dir,self.reference,'audio.wav'))) 
        output = subprocess.call(command, shell=True, stdout=None)
        
        images = []
        flist = glob.glob(os.path.join(self.tmp_dir,self.reference,'*.jpg'))
        flist.sort()
        for fname in flist:
            images.append(cv2.imread(fname))
        im = numpy.stack(images,axis=3)
        im = numpy.expand_dims(im,axis=0)
        im = numpy.transpose(im,(0,3,4,1,2))
        imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())
        # Load audio
        sample_rate, audio = wavfile.read(os.path.join(self.tmp_dir,self.reference,'audio.wav'))
        mfcc = zip(*python_speech_features.mfcc(audio,sample_rate))
        mfcc = numpy.stack([numpy.array(i) for i in mfcc])
        cc = numpy.expand_dims(numpy.expand_dims(mfcc,axis=0),axis=0)
        cct = torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())
        
        # Check audio and video input length
        if (float(len(audio))/16000) != (float(len(images))/25) :
            print("WARNING: Audio (%.4fs) and video (%.4fs) lengths are different."%(float(len(audio))/16000,float(len(images))/25))
        min_length = min(len(images),math.floor(len(audio)/640))
        
        # Generate video and audio feats
        lastframe = min_length-5
        im_feat = []
        cc_feat = []
        cuda0='cuda:0'

        tS = time.time()
        for i in range(0,lastframe,self.batch_size):
            
            im_batch = [ imtv[:,:,vframe:vframe+5,:,:] for vframe in range(i,min(lastframe,i+self.batch_size)) ]
            im_in = torch.cat(im_batch,0)
            im_out  = self.model.forward_lip(im_in.cuda(cuda0));
            im_feat.append(im_out.data.cpu())

            cc_batch = [ cct[:,:,:,vframe*4:vframe*4+20] for vframe in range(i,min(lastframe,i+self.batch_size)) ]
            cc_in = torch.cat(cc_batch,0)
            cc_out  = self.model.forward_aud(cc_in.cuda(cuda0))
            cc_feat.append(cc_out.data.cpu())

        im_feat = torch.cat(im_feat,0)
        cc_feat = torch.cat(cc_feat,0)
        # Compute Offset
        dists = calc_pdist(im_feat,cc_feat,vshift=self.vshift)
        mdist = torch.mean(torch.stack(dists,1),1)
        minval, minidx = torch.min(mdist,0)
        offset = self.vshift-minidx
        conf   = torch.median(mdist) - minval
        fdist   = numpy.stack([dist[minidx].numpy() for dist in dists])
        fconf   = torch.median(mdist).numpy() - fdist
        fconfm  = signal.medfilt(fconf,kernel_size=9)
        numpy.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        dists_npy = numpy.array([ dist.numpy() for dist in dists ])
        return offset.numpy(), conf.numpy(), dists_npy

