# import matplotlib.pyplot as plt
import numpy as np
import random
import argparse
from collections import OrderedDict
# import collections
import copy
from scipy.stats import truncnorm
# import time



def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array-value)).argmin()
    nearest_index = np.where(array == array[idx])[0]
    return random.choice(nearest_index)



def get_truncated_normal(mean, sd, low, upp):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)



parser = argparse.ArgumentParser(description='Sampling 2048 and save bounding box and SDF points')

parser.add_argument('in_file', type=str,
                    help='Location of SDF input file')

parser.add_argument('out_file', type=str,
                    help='Location of output file')

args = parser.parse_args()

sdfFile = open(args.in_file, 'r')
SDF_list = []
j = 1
for line in sdfFile:
    split = line.split()
    #if blank line, skip
    if not len(split):
      continue

    if j == 1:
        dimensionX = int(split[0])
    if j == 2:
        dimensionY = int(split[0])
    if j == 3:
        dimensionZ = int(split[0])

    if j == 4:
        bminx = float(split[0])
    if j == 5:
        bminy = float(split[0])
    if j == 6:
        bminz = float(split[0])

    if j == 7:
        bmaxx = float(split[0])
    if j == 8:
        bmaxy = float(split[0])
    if j == 9:
        bmaxz = float(split[0])


    if j>9:
        SDF_list.append(float(split[0]))

    j = j+1

sdfFile.close()

print(bminx, bminy, bminz)
print(bmaxx, bmaxy, bmaxz)


gridsizeX = (bmaxx - bminx)/dimensionX
gridsizeY = (bmaxy - bminy)/dimensionY
gridsizeZ = (bmaxz - bminz)/dimensionZ



mu = 0.0
sigma = max(SDF_list)/3


# count, bins, ignored = plt.hist(SDF_list, 40 , alpha = 0.5, label = 'input', density = True) #density = True
# plt.show()

rand = []

X = get_truncated_normal(mu, sigma, min(SDF_list), max(SDF_list))

rand = X.rvs(2048)

SDF_value = np.asarray(SDF_list)

# count, bins, ignored = plt.hist(rand, 40 , alpha = 0.5, label = 'input', density = True) #density = True
# plt.show()



array1 = copy.copy(SDF_value)
sample = []


for i in range(len(rand)):
    print(i)
    index = find_nearest(array1 ,rand[i])
    array1[index] = 10000 # to remove selected items in order to don't have repetetive sampling
    sample.append(index)



print(len(sample))



SDF = OrderedDict() #{}

for ii in range(len(sample)):

    index = sample[ii]

    i = index/((dimensionX+1)*(dimensionY+1))
    index = index % ((dimensionX+1)*(dimensionY+1))

    j = index/(dimensionX+1)
    index = index % (dimensionX+1)

    k = index

    x = i*gridsizeX + bminx
    y = j*gridsizeY + bminy
    z = k*gridsizeZ + bminz

    SDF[tuple([i, j, k])] = SDF_list[sample[ii]]


final = {'bmin':[bminx, bminy, bminz], 'bmax':[bmaxx, bmaxy, bmaxz], 'SDF': SDF}


# Save
#np.save(args.out_file, final)
np.save(args.out_file, final)
