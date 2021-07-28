import torch 
import numpy as np
import cv2
import time, glob, shutil, datetime
from matplotlib import pyplot as plt
import csv, json
import xml.etree.ElementTree as ET 
import pandas as pd
from collections import defaultdict
from syncnet_python.facetrack import *
from syncnet_python.syncnet import *
from itertools import cycle

def cut_into_utterances(filename, output_dir, maxWMER=1000):
    
    xmldir = "/afs/inf.ed.ac.uk/group/cstr/datawww/asru/MGB1/data/xml"
    xmlfile = os.path.join(xmldir, filename+'.xml')
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    utterance_items = []
    paths = glob.glob(f"/afs/inf.ed.ac.uk/group/project/nst/bbcdata/ptn*/**/{filename}*.ts") \
    + glob.glob(f"/afs/inf.ed.ac.uk/group/project/nst/bbcdata/raw/{filename}*.ts")
    inputVideo = paths[0]
    command_elems = ["ffmpeg -loglevel quiet -y -i " + inputVideo]
    for item in root.findall('./body/segments/segment'):
        if (item.attrib['id'].split('_')[-1]=='align' and float(item.attrib['WMER'])<=maxWMER):
            if (float(item.attrib['endtime']) - float(item.attrib['starttime'])<2):
                continue                        
            location = output_dir + item.attrib['id']
            ready_to_crop = prepare_output_directory(location)
            if ready_to_crop:
                utterance_items.append(item)
                data = item.attrib
                start = datetime.timedelta(seconds=float(data['starttime']))
                end = datetime.timedelta(seconds=float(data['endtime']))
                output = os.path.join(location, 'pyavi', 'video.avi')
                command_elems.append(" -ss " + str(start) + " -to " + str(end) + " -c copy " + output) # -c:a mp3 -c:v mpeg4
                create_transcript_from_XML(location, item)
    command = "".join(command_elems)
    s = time.time()
    result = subprocess.run(command, shell=True, stdout=None)
    if result.returncode != 0:
        print(f"ERROR: ffmpeg failed to trim video: {filename}")
        print(f"result: {result}")
    t = time.time() - s
    print(f"Took {t} seconds to trim {len(command_elems)-1} utterances")
    return utterance_items

def get_genre(filename):
    xmldir = "/afs/inf.ed.ac.uk/group/cstr/datawww/asru/MGB1/data/xml/"
    xmlfile = os.path.join(xmldir, filename+'.xml')
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    head = root.find('./head/recording')
    genre = head.attrib["genre"]
    return genre
    
def prepare_output_directory(location):
    ready = True
    incomplete_directory_exists = os.path.isdir(location) and not os.path.exists(f"{location}/utterance_info.csv")
    if(incomplete_directory_exists):
        shutil.rmtree(location)
    elif(os.path.isdir(location)):
        ready = False
        return ready  #  This utterance has been processed already.
    else:
        pass
    subprocess.run("mkdir -p " + location + "/pyavi/", stdout=subprocess.DEVNULL, shell=True)    
    return ready

def create_transcript_from_XML(location, item):
    # TODO: the transcript should only contain the words spoken in the final cropped video. 
    utterance = ""
    for child in item:
        utterance+=child.text + " "
    data = item.attrib
    data.update({"utterance": utterance})
    with open(location + '/transcript.txt', 'w') as outfile:
        outfile.write(str(data))
        
def cleanup(dataset_dir):
    for utterance in os.listdir(dataset_dir):
        source = os.path.join(dataset_dir, utterance, 'pycrop')
        dest = os.path.join(dataset_dir, utterance)
        for f in os.listdir(source):
            new_path = shutil.move(f"{source}/{f}", f"{dest}/{f}")
        for f in glob.glob(f"{dest}/py*"):
            shutil.rmtree(f)
            
