# AVSR-Dataset-Pipeline

Multi-stage pipeline for generating an AVSR dataset consisting of active-speaker face tracks with their transcriptions from widely available videos (such as TV data). 

A sample output of the pipeline can be found in the sample/ folder. 

The models and code in this project are adapted from the following work: Paper: [Out of time: automated lip sync in the wild](https://www.robots.ox.ac.uk/~vgg/publications/2016/Chung16a/chung16a.pdf). Code: https://github.com/joonson/syncnet_python


## Installation

```
1. $ git clone git@github.com:AlanSavio25/AVSR-Dataset-Pipeline.git  
2. $ cd AVSR-Dataset-Pipeline
3. $ conda create -n pipeline_env python=3.7
4. $ source activate pipeline_env
5. (pipeline_env)$ pip install -r ./requirements.txt 
6. (pipeline_env)$ source deactivate pipeline_env # When you want to leave virtual environment

```

### Prequisite Files

Todo


## Usage

1. Modify the default configuration file 'config.yml' to set up the directories.
2. Run the following command.
```
python main.py > main.log
```

## Tasks

[] Split the work between available GPUs 

[] Change SyncNet input shape to avoid loading repeated frames

[] Modify Scene Detection input to utilize existing frames instead of re-loading the video each time.




