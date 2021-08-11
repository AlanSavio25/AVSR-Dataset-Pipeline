import pandas as pd
import ast
import os
import json
import sys
import glob
import subprocess
import logging
logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)

def read_syncnet_output(logfile):
    with open(logfile, 'r') as log:
        lines = log.read().split("\n")
    return lines

def get_syncnet_scores(syncnet_output, utt_id):
    for i in range(len(syncnet_output)):
        if utt_id in syncnet_output[i]:
            return syncnet_output[i], syncnet_output[i+1]

def get_video_duration(video):
    output = subprocess.run(f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {video}", 
                            capture_output=True, text=True, shell=True)
    duration = str(output.stdout)
    return duration
    
def create_transcripts(data_dir, ctm_dict, score_file):
    syncnet_output = read_syncnet_output(score_file)
    for line in syncnet_output:
        if "WARNING" in line:
            syncnet_output.remove(line)
    j = open(ctm_dict)
    ctm_dict = json.load(j)
    count = 1
    for f in os.listdir(data_dir):
        transcriptfile = open(os.path.join(data_dir, f, 'transcript.txt'), 'r')
        transcript = transcriptfile.read().replace('\\', '\\\\')
        transcript = ast.literal_eval(transcript)
        trimtimesfile = open(os.path.join(data_dir, f, 'trimtimes.txt'), 'r')
        trimtimes = trimtimesfile.read().split("\n")[1:]

        facetracks_list = glob.glob(f"{data_dir}/{f}/*.avi")
        facetracks_list.sort()
        videoname = transcript['id'][2:].split('_speaker')[0]
        utt_start_time = (float(transcript['starttime']))
        utterance = transcript['utterance']
        genre = transcript["genre"]

        for i, face in enumerate(facetracks_list):
            text = ["Text: "]
            trimstart = float(trimtimes[i].split(', ')[0])
            trimend = float(trimtimes[i].split(', ')[1])
            for (start,duration,word) in ctm_dict[videoname]:
                if utt_start_time+trimstart<start and start+duration<utt_start_time+trimend:
                    text.append(word)
                if start+duration>utt_start_time+trimend:
                    break
            utt_id = '/'.join(face.split('/')[-2:-1]) + " "+  face.split('/')[-1]
            if len(text)==1:
                text.append("[N/A]")
            logging.info(get_syncnet_scores(syncnet_output, utt_id))
            offset, conf = get_syncnet_scores(syncnet_output, utt_id)[1].split(" ")
            duration = get_video_duration(face)
            text.append(f"\nGenre: {genre}\nSyncnet Conf: {conf}\nOffset: {offset}\nDuration: {float(duration):.2f} s")
            text = " ".join(text)
            logging.info(f"Text for {os.path.splitext(face)[0]+'.txt'} is: {text}")
            with open(os.path.splitext(face)[0]+'.txt', "w") as out:
                out.write(text)


if __name__ == '__main__':
    create_transcripts(sys.argv[1], ctm_dict=sys.argv[2], score_file=sys.argv[3])
