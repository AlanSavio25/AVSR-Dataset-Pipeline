import csv, json
import sys, pdb, os
import traceback
import xml.etree.ElementTree as ET 
import datetime, time
import shutil
import glob
import subprocess
import pandas as pd
from collections import defaultdict
from syncnet_python.facetrack import *
from syncnet_python.syncnet import *
from itertools import cycle

def prepare_output_directory(location):
    ready = True
    does_not_require_processing = False
    incomplete_directory_exists = os.path.isdir(location) and not os.path.exists(f"{location}/utterance_info.csv")
    if(incomplete_directory_exists):
        shutil.rmtree(location)
    elif(os.path.isdir(location)):
        return does_not_require_processing  #  This utterance has been processed already. Continuing to next utterance..
    else:
        pass
    subprocess.run("mkdir -p " + location + "/pyavi/tracks/", stdout=subprocess.DEVNULL, shell=True)    
    return ready

def calculate_active_speaker_confidence(location):
    syncnet.setup(data_dir=location+'/')
    syncnet.compute_offsets()
    
def cleanup(location):
    source = location + "/pycrop/tracks/" 
    dest = location + "/"
    fs = os.listdir(source)
    for f in fs:
        new_path = shutil.move(f"{source}/{f}", f"{dest}/{f}")
    #   transcript = open(f"{dest}/transcript.txt", "w")
    #   transcript.write(str(item.attrib))
    #   transcript.close()
    for f in glob.glob(f"{dest}/py*"):
        shutil.rmtree(f)
        
def getGenre(filename):
    xmldir = "/afs/inf.ed.ac.uk/group/cstr/datawww/asru/MGB1/data/xml/"
    xmlfile = xmldir + filename + ".xml"
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    head = root.find('./head/recording')
    genre = head.attrib["genre"]
    return genre

def cut_into_utterances(filename, maxWMER=1000):
    
    xmldir = "/afs/inf.ed.ac.uk/group/cstr/datawww/asru/MGB1/data/xml/"
    xmlfile = xmldir + filename + ".xml"
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    utterance_items = []
    paths = glob.glob(f"/afs/inf.ed.ac.uk/group/project/nst/bbcdata/ptn*/**/{filename}*.ts") + glob.glob(f"/afs/inf.ed.ac.uk/group/project/nst/bbcdata/raw/{filename}*.ts")
    inputVideo = paths[0]
    command_elems = ["ffmpeg -loglevel quiet -y -i " + inputVideo]
    for item in root.findall('./body/segments/segment'):
        if (item.attrib['id'].split('_')[-1]=='align' and float(item.attrib['WMER'])<=maxWMER):
            if (float(item.attrib['endtime']) - float(item.attrib['starttime'])<2):
                continue                        
            output_dir = "/disk/scratch/s1768177/pipeline/output_data/"
            location = output_dir + item.attrib['id']
            reference = "tracks"
            status = prepare_output_directory(location)
            if status:
                utterance_items.append(item)
                data = item.attrib
                start = datetime.timedelta(seconds=float(data['starttime']))
                end = datetime.timedelta(seconds=float(data['endtime']))
                output = location + '/pyavi/tracks/video.avi'
                command_elems.append(" -ss " + str(start) + " -to " + str(end) + " -c copy " + output)
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

def get_video_durations(videos, speaker_info_location):
    
    durations_and_confidences = defaultdict(list)
    for video in videos:
        durations_and_confidences[video.split('/')[-1]] = {"duration": [], 'Confidence': []}
        duration = subprocess.run(f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {video}",
                                capture_output=True,
                                text=True,
                                shell=True)
        duration = str(duration.stdout)
        #results.append(duration) 
        durations_and_confidences[video.split('/')[-1]]["duration"] = duration
        
    # Read CSV and if this particular video has a high confidence then add to results_active 
    speaker_confidences = pd.read_csv(speaker_info_location)

    for i, row in speaker_confidences.iterrows():
        durations_and_confidences[row["filename"].split('/')[-1]]['Confidence'] = row['Confidence']
        #active_confidences.append(row['Confidence'])
    return  dict(durations_and_confidences)  #results, active_results, active_confidences  

def create_transcript_from_XML(location, item):
    # TODO: the transcript should only contain the words spoken in the final cropped video. 

    utterance = ""
    for child in item:
        utterance+=child.text + " "
    data = item.attrib
    data.update({"utterance": utterance})
    with open(location + '/transcript.txt', 'w') as outfile:
        outfile.write(str(data))


def updateInfo(utt):
    info['ID'] = utt
    info['# Heads detected'] = len(glob.glob(utt + '/0*'))
    durations_and_confidences = get_video_durations(videos = glob.glob(utt + '/0*'), speaker_info_location = utt + "/activespeakerinfo.csv")
    info['Duration containing heads'] = durations_and_confidences 

    return info


def generate_face_tracks(args):
    facetracker, utt = args[0], args[1]
    print(f"Starting face tracking for: {utt}. OS pid: {os.getpid()}")
    facetracker.run(reference='tracks',data_dir=utt+'/')
    print(f"Finished face tracking for: {utt}")
    no_faces_found = len(os.listdir(utt + "/pycrop/tracks/")) == 0
    if(no_faces_found):
        shutil.rmtree(utt)
#         utt_with_no_faces += 1

if __name__ == '__main__':
    
    info = {'ID':[], 
        '# Heads detected': [], 
        'Duration containing heads': [],
        }

    count = 1
    start_time = time.time()
    filelist = "/afs/inf.ed.ac.uk/group/cstr/datawww/asru/MGB1/scripts/train.short"
    with open(filelist, "r") as f:
        lines = f.read()
        files = lines.split()



    output_dir = "/disk/scratch/s1768177/pipeline/output_data/"

    files = files[:1]
    total_utterances_processed = 0
    print(f"\n{datetime.datetime.now()}. CROPPING utterances from raw videos.")
    
    for filename in files:

        start = time.time()
              
        genre = getGenre(filename)
        desired_genres = ["drama", "childrens", "news", "documentary"]
        
        if (genre in desired_genres):
            
            print(f"{count}. {filename}. ({genre}) ")
            count += 1
            utterance_items = cut_into_utterances(filename)
            
            total_utterances_processed += len(utterance_items)

    
    print(f"\nFinished Cutting total {total_utterances_processed} utterances from {count-1} videos")

    total_utterances_before_face_tracking = len(os.listdir(output_dir))
    sys.exit(1)

    # ------------------FACE TRACKS ---------------------
    
    print("Initializing Face detection model on all GPUs")
    facetrackers = []
    num_gpus = 4
    for i in range(num_gpus):
        facetrackers.append(FaceTrack(f'cuda:{i}'))
    facetrackers_iter = cycle(facetrackers)
    print("\nCreating FACE TRACKS.")
    print(f"Main process id: {os.getpid()}")
    utt_with_no_faces = 0
    detect_time = 0
    mp.set_start_method('spawn')


    
    processes = []
    for index,utt in enumerate(glob.glob(output_dir+'*')):
        p = mp.Process(target=generate_face_tracks, args=([next(facetrackers_iter),utt,],))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    print("Done with processes")
    sys.exit(1)
    for utt in glob.iglob(output_dir+'*'):
                
        if os.path.exists(f"{utt}/utterance_info.csv"):
            continue
        
        start_detect = time.time()
#         print(f"utt: {utt}")
        generate_face_tracks([facetracker, utt])
#         pool.apply(generate_face_tracks, args=(utt))

        detect_time += time.time() - start_detect
    
        
#             print(f"No faces found in {utt}. Deleted.")

    utterances_in_dir_after_face_tracking = len(os.listdir(output_dir))
    average_detect_time = detect_time/total_utterances_before_face_tracking
    face_tracks_created_now = len(glob.glob(output_dir+'*/pycrop/tracks/0*.avi'))
    face_tracks_created_before_running_script = len(glob.glob(output_dir+'*/0*.avi'))
    print(f"\nFace track creation complete. \n\tTotal time taken for face track creation: {detect_time/60:.2f} minutes\n\tAverage processing time per utterance: {average_detect_time:.2f} s\n\tUtterances processed: {total_utterances_processed}.\n\tUtterances deleted (had no faces): {utt_with_no_faces}.\n\tFace tracks created: {face_tracks_created_now}.")
    print(f"\nTotal Utterances in directory: {utterances_in_dir_after_face_tracking}\n\tTotal Face tracks: {face_tracks_created_now} (now) + {face_tracks_created_before_running_script} (created earlier) = {face_tracks_created_now+face_tracks_created_before_running_script}\n")
    
    sys.exit(1)
    # ------------------ SYNCNET ------------------------
    
    syncnet = SyncNet() # Initialize Syncnet pretrained model
    activespeakertime = 0
    for utt in glob.iglob(output_dir+'*'):
                
        if os.path.exists(f"{utt}/utterance_info.csv"):
            continue
        start_active = time.time()
        calculate_active_speaker_confidence(utt)
        activespeakertime += time.time() - start_active
#         create_transcript_from_XML(utt)
        cleanup(utt)
    
        # Write results
        info = updateInfo(utt)
        df = pd.DataFrame([info])
        df = df.transpose()
        os.remove(f"{utt}/activespeakerinfo.csv")
        df.to_csv(f'{utt}/utterance_info.csv')
    
    average_activespeaker_time = activespeakertime/len(glob.glob(output_dir+'*'))
    print(f"\nActive Speaker Confidence computation complete.\n\tTotal time taken for active speaker: {activespeakertime/60:.2f} minutes.\\n\tAverage processing time per utterance: {average_activespeaker_time} s")
    

    total_time = time.time() - start_time
    print(f"Total time taken to run the main script: {total_time/60:.2f} minutes")
    

    

#             for num, item in enumerate(utterance_items):
# #                 result = process_utterance(item)
#                 print(f"Finished item {num+1}/{len(utterance_items)} ")
#             # Multiprocessing: calling run on different utterance items, and splitting the work across several cores of the CPU
#             # p = Pool(25)
#             # result = p.imap_unordered(run, utterance_items)
#             # p.close()
#             # p.join()
    
#             print(f"Finished processing filename: {filename}. Completed Percentage: {(count-1)*100/len(files):.2f}%")
        
#             f = open("timetaken.txt", "a")
#             f.write(f"\n{filename} - {(time.time() - start)/60} mins - {len(utterance_items)} number of utterances")
#             f.close()



# def process_utterance(item):
    
#     try:
# #       if(item.attrib['id']!="ID20080505_000500_bbcone_weatherview_utt_13_align"):
# #           return
#         start = time.time()

#         output_dir = "/disk/scratch/s1768177/pipeline/output_data/"
#         location = output_dir + item.attrib['id']
#         reference = "tracks"
        
        
# #         st_trimming = time.time()
# #         video_duration = crop_utterance_from_video(filename, item)
# #         time_to_trim = time.time() - st_trimming
        
#         st_detect = time.time()
#         generate_face_tracks(location, item)
#         time_to_detect = time.time() - st_detect
#         print("Face track generation complete.")
        
#         no_faces_found = len(os.listdir(location + "/pycrop/tracks/")) == 0
#         if(no_faces_found):
#             shutil.rmtree(location)
#             print(f"No faces found in {location}. Deleted.")
#             return
    
#         st_active = time.time()
#         calculate_active_speaker_confidence(location, item)
#         time_to_active = time.time() - st_active
#         print("Active Speaker Confidence computation complete.")
#         create_transcript_from_XML(location, item)
#         cleanup(location, item)
#         print("Cleanup complete.")
#         total_processing_time = time.time() - start
        
#         st_writing = time.time()
#         video_duration = 0
#         info = updateInfo(item, video_duration, total_processing_time, time_to_trim, time_to_detect, time_to_active, st_writing)
#         df = pd.DataFrame([info])
#         df = df.transpose()
#         os.remove(f"{location}/activespeakerinfo.csv")
#         df.to_csv(f'{location}/utterance_info.csv')
        
#     except Exception as e:
#         print("Error in process_utterance()")
#         print(e)
#         print(traceback.format_exc())
#         sys.exit(1)
        
#         f = open("/afs/inf.ed.ac.uk/group/ug4-projects/s1768177/AVSR/fileswitherrortrainshort.txt", "a") #        f.write(f"\n{item.attrib['id']}")
#         print(f"\n{item.attrib['id']}")
#         f.close()



#     output = subprocess.run("cd syncnet_python && python3.7 run_pipeline.py --videofile " + location + "/segment.ts" + " --reference tracks --data_dir " + location + "/", capture_output=True, shell=True)
    #                 print(output)
    #                 print(f"Error: {output.stderr[:100]} \n .\n .\n .\n {output.stderr[-100:]}")
    #                 error = ('frames/s' not in str(output.stderr)) and ('Video file(s) not found' not in str(output.stderr))
    #                 if error:
    #                     pass
    #                 else:
    #                     print("Finished running face tracker")

        #print(f"Is an actual cuda error: {'frames/s' not in str(output.stderr)}")
        # Deletes the directory created for this utterance if no face was detected and moves onto the next utterance.
        #if(error):
         #   print(os.listdir("/disk/scratch1/s1768177/trainshort/ID20080505_180000_bbcone_the_one_show_utt_151_align"))
       # return

    #                 if(error): #error):

    #                     sys.exit(1)
    #                     print(output.stderr)
    #                     print("CUDA error! Waiting for 10 seconds before retrying..")
    #                     time.sleep(10)

    #                     output = subprocess.run("cd syncnet_python && python3.7 run_pipeline.py --videofile" + " /disk/scratch/s1768177/pipeline/output_data/" + item.attrib['id'] + "/segment.ts" + " --reference tracks --data_dir /disk/scratch/s1768177/pipeline/output_data/" + item.attrib['id'] + "/", capture_output=True, shell=True)
    #                     error = 'frames/s' not in str(output.stderr)
    #                     if(error):
    #                         raise Exception("Tried twice but there seems to be an error: likely CUDA")
    
    # error logging has to be revamped completely
#     fe = open("fileswitherrortrainshort.txt", "r") #files with error
#     error_list_lines = fe.read()
#     error_list = error_list_lines.split()
#     error_list_filenames = map(lambda x: x[2:].split('_utt')[0], error_list)
#     error_list_filenames = list(set(error_list_filenames))
#     print(len(error_list_filenames))

