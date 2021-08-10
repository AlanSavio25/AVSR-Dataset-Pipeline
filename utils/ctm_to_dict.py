# This script converts all transcripts to the word-level timed trimmed transcript, which is essential for feeding into the neural network (for training and testing)
# Pre-processing of the word level timings file, convert to json and store in memory (if too expensive, write into file once, and read into dict in subsequent processing)


from collections import defaultdict
import json, sys, os

def ctm_to_dict(input_ctm, output_dict):
    
    f = open(input_ctm, 'r')
    lines = f.read().split("\n")
    f.close()
    ctm = defaultdict(list)
    # Loop through ctm and build dict.
    count = 1
    for line in lines:
        if not line:
            continue
        linesplit = line.split(' ')
        videoname = linesplit[0]
        start = float(linesplit[2])
        duration = float(linesplit[3])
        linesplit = line.split()
        wordstart = float(linesplit[2])
        wordduration = float(linesplit[3])
        word = linesplit[4]
        count += 1
        ctm[videoname].append((start, duration, word))

    print(count)
    print("Done")
    with open(output_dict, 'w') as outfile:
        json.dump(dict(ctm), outfile, indent=4)

    
if __name__ == '__main__':
    ctm_to_dict(input_ctm=sys.argv[1])