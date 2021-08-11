import torch
import numpy as np
import time, glob, shutil, datetime
import xml.etree.ElementTree as ET 
from facetrack import *
from syncnet import *
import logging
logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)

def cut_into_utterances(filename, output_dir, genre, source_dir=None):
    
    xmldir = "/afs/inf.ed.ac.uk/group/cstr/datawww/asru/MGB1/data/xml"
    xmlfile = os.path.join(xmldir, filename+'.xml')
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    utterance_items = []
    paths = glob.glob(f"/afs/inf.ed.ac.uk/group/project/nst/bbcdata/ptn*/**/{filename}*.ts") \
    + glob.glob(f"/afs/inf.ed.ac.uk/group/project/nst/bbcdata/raw/{filename}*.ts")
    if source_dir:
        paths += glob.glob(f"{source_dir}/*{filename}*webm")
#        (paths[0])
    if len(paths)==0:
        raise Exception(f"Could not find {filename} in any of the source directories")
    inputVideo = paths[0]
    command_elems = ["ffmpeg -loglevel quiet -y -i " + inputVideo]
    for item in root.findall('./body/segments/segment'):
        if (item.attrib['id'].split('_')[-1]=='human'):
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
                create_transcript_from_XML(location, item, genre)
    s = time.time()
    if len(command_elems) == 1:
        logging.warning("No utterances found.")
        logging.warning(f"The available segments are: {list(root.findall('./body/segments'))}")
    else:
        command = "".join(command_elems)
        result = subprocess.run(command, shell=True, stdout=None)
        if result.returncode != 0:
            logging.error(f"ERROR: ffmpeg failed to trim video: {filename}")
            logging.error(f"result: {result}")
    t = time.time() - s
    logging.info(f"Took {t} seconds to trim {len(command_elems)-1} utterances")
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
    incomplete_directory_exists = os.path.isdir(location) and not os.path.exists(f"{location}/0*.txt")
    if(incomplete_directory_exists):
        shutil.rmtree(location)
    elif(os.path.isdir(location)):
        ready = False
        return ready  #  This utterance has been processed already.
    else:
        pass
    subprocess.run("mkdir -p " + location + "/pyavi/", stdout=subprocess.DEVNULL, shell=True)    
    if not os.path.isdir(location):
        return False
    return ready

def create_transcript_from_XML(location, item, genre):
    utterance = ""
    for child in item:
        if child.text:
            utterance+=child.text + " "
    data = item.attrib
    data.update({"utterance": utterance})
    data.update({"genre": genre})
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