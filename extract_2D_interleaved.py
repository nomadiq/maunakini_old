#!/usr/bin/env python
import numpy as np
#import maunakini as mk
import matplotlib.pyplot as plt
import sys
from os import path
from os import mkdir

# this extracts NUS spectra from the Forbidden3Q Experiment

xN = sys.argv[1] # we need to know how many r+i points in each FID
try:
    xN = int(xN)
except:
    print('xN points not an integer. Bad news!')
    raise

# first lets inspect the vdlist or vclist and figure out how many spectra are present
if path.exists('vdlist'):
    vdlist_length = sum(1 for _ in open('vdlist').readlines())
else:
    vdlist_length = 0

if path.exists('vclist'):
    vclist_length = sum(1 for _ in open('vclist').readlines())
else:
    vclist_length = 0

num_delays = np.maximum(vdlist_length, vclist_length)

# figure out how many different 'types' of 2D spectra are present by reading hte nuslist
if path.exists('nuslist'):
    nuslist_file = open('nuslist', 'r')
    firstline = nuslist_file.readline().split()
    num_states = len(firstline) - 1
    num_samples = sum(1 for _ in open('nuslist').readlines())
else: 
    num_states = 0
    num_samples = 0

print(f'num delays = {num_delays}')
print(f'num states = {num_states}')
print(f'num samples = {num_samples}')

data = np.fromfile('ser', dtype=np.int32)
try:
    data = data.reshape((num_samples, 2*xN))
except:
    print('Your number of samples and your xN value don\'t match the amount of data in the ser file')
    raise


# there are four spectra, encoded as 0 0, 0 1, 1 0, 1 1
# we will store 2 FIDs (as numpy arrays) per point in lists
spec_datas = [[] for _ in range(num_delays*num_states)]
spec_lists = [[] for _ in range(num_delays*num_states)]

i = 0
with open("nuslist") as fp:
    while True:
        line = fp.readline()

        if not line:
            break

        points = line.split()

        if len(points) == 3:
            spec_delay = int(points[1])
            spec_state = int(points[0])
            spec_datas[spec_delay*num_states+spec_state].append(data[i,:])
            spec_lists[spec_delay*num_states+spec_state].append(points[2])
        
        if len(points) == 2:
            spec_delay = int(points[0])
            spec_state = 0
            spec_datas[spec_delay*num_states+spec_state].append(data[i,:])
            spec_lists[spec_delay*num_states+spec_state].append(points[1])
        
        i += 1

if not path.isdir('PROC'):
    mkdir('PROC')

proc_dir = 'PROC/'

for i in range(len(spec_datas)): # for each spectrum and its nuslist
    data_list = spec_lists[i]
    data_data = spec_datas[i]
    data = np.zeros((len(data_list), 2*xN), dtype=np.int32)
    for j in range(len(data_list)):
        data[j,:] = data_data[j]
    #print(data.shape)
    data = data.flatten()
    #print(data[10000])
    data.tofile(proc_dir+'ser_'+str(i+1), format='int32')
    with open(proc_dir+'nuslist_'+str(i+1), 'w') as f:
        for item in data_list:
            f.write('%s\n' % item)



