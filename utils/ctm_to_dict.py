# Convert ctm to dictionary for quick access during transcript creation
from collections import defaultdict
import json, sys, os
import logging
logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)

def ctm_to_dict(input_ctm, output_dict):
    
    f = open(input_ctm, 'r')
    lines = f.read().split("\n")
    f.close()
    ctm = defaultdict(list)
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
    logging.info(count)
    logging.info("Done")
    with open(output_dict, 'w') as outfile:
        json.dump(dict(ctm), outfile, indent=4)
    
if __name__ == '__main__':
    ctm_to_dict(input_ctm=sys.argv[1])