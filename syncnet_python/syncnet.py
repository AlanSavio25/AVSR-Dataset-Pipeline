#!/usr/bin/python
#-*- coding: utf-8 -*-

import time, pdb, argparse, subprocess, pickle, os, gzip, glob
import pandas as pd
from syncnet_python.SyncNetInstance import *

class SyncNet:
    

    def __init__(self):
        self.batch_size= 20
        self.vshift = 15
        self.initial_model = 'syncnet_python/data/syncnet_v2.model'
        
        self.s = SyncNetInstance()
        self.s.loadParameters(self.initial_model)
        print(f"Model {self.initial_model} loaded.")
        
    def setup(self,reference='',data_dir=''):
        
        self.reference = reference
        self.data_dir = data_dir
        
        setattr(self,'avi_dir',os.path.join(self.data_dir,'pyavi'))
        setattr(self,'tmp_dir',os.path.join(self.data_dir,'pytmp'))
        setattr(self,'work_dir',os.path.join(self.data_dir,'pywork'))
        setattr(self,'crop_dir',os.path.join(self.data_dir,'pycrop'))
        
    def compute_offsets(self):
        flist = glob.glob(os.path.join(self.crop_dir,self.reference,'0*.avi'))
        flist.sort()
        dists = []
        info = {'filename': [], 'Offset': [], 'Confidence' : []} 
        for idx, fname in enumerate(flist):

            offset, conf, dist = self.s.evaluate(self,videofile=fname)
            info['filename'].append(fname)
            info['Offset'].append(offset)
            info['Confidence'].append(conf)
            df = pd.DataFrame.from_dict(info)
            df.to_csv(os.path.join(self.crop_dir,self.reference,"activespeakerinfo.csv"), index = False)
        self.save_results(dists)
        
        
    def test_compute_offsets(self, videofile=''):

            offset, conf, dist = self.s.evaluate(self,videofile=videofile)
            info = {'filename': videofile, 'Offset': offset, 'Confidence' : conf} 
            df = pd.DataFrame.from_dict(info)
       
#             print(df)
                
    def save_results(self, dists):
        with open(os.path.join(self.work_dir,self.reference,'activesd.pckl'), 'wb') as fil:
            pickle.dump(dists, fil)
