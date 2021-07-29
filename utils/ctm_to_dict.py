# This script converts all transcripts to the word-level timed trimmed transcript, which is essential for feeding into the neural network (for training and testing)
# Pre-processing of the word level timings file, convert to json and store in memory (if too expensive, write into file once, and read into dict in subsequent processing)
from collections import defaultdict
import json
import sys
import os

def main(input_ctm):
    
    f = open(input_ctm, 'r')
    lines = f.read().split("\n")
    f.close()
    ctm = defaultdict(list)
    # Loop through ctm and build dict.
    count = 1
    for line in lines:
        if not line:
            continue
    #    print(f"Line is: {line}")
        linesplit = line.split(' ')
        videoname = linesplit[0]
#         print(f"Videoname is: {videoname}")
#         speaker = int(line.split('_spk-')[1].split('_seg-')[0])
#         seg = line.split('_seg-')[1].split()[0].split(':')
    #    print(f"Seg: {seg}")
        start = float(linesplit[2])
        duration = float(linesplit[3])
    #    print(f"Segstart is : {segstart}")
    #    print(f"Segstart is : {segend}")
        linesplit = line.split()
        wordstart = float(linesplit[2])
        wordduration = float(linesplit[3])
    #    print(f"Word start is : {wordstart}")
    #    print(f"Word wordduration is : {wordduration}")
        word = linesplit[4]
        print(count)
        count += 1
#         print(type(ctm[videoname]))
        ctm[videoname].append((start, duration, word))

    print(count)
    print("Done")
    with open(os.path.splitext(input_ctm)[0]+'.json', 'w') as outfile:
        json.dump(dict(ctm), outfile, indent=4)

    
if __name__ == '__main__':
    main(input_ctm=sys.argv[1])