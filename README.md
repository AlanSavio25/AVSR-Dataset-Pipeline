# AVSR-Dataset-Pipeline

Multi-stage pipeline for generating an AVSR dataset consisting of speaking face tracks with their transcriptions from raw "in-the-wild" videos (such as TV data).

A sample output of the pipeline can be found in the sample/ folder. 

## Installation

```
1. $ git clone git@github.com:AlanSavio25/AVSR-Dataset-Pipeline.git  # Cloning project repository
2. $ cd AVSR-Dataset-Pipeline  # Enter to project directory
3. $ python3 -m venv my_venv # If not created, creating virtualenv
4. $ source ./my_venv/bin/activate # Activating virtualenv
5. (my_venv)$ pip3 install -r ./requirements.txt # Installing dependencies
6. (my_venv)$ deactivate # When you want to leave virtual environment

```

## Prequisite Files



## Usage

Modify the config file.

```python main.py > main.log
```

## Tasks

Pending tasks include:

[] Split the work between available GPUs to speed-up processing

[] Change SyncNet input shape to avoid loading repeated frames

[] Modify Scene Detection input to utilize existing frames instead of re-loading the video each time.
