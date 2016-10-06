#!/usr/bin/env python


import matplotlib.pyplot as plt
import glob
import sys
import numpy as np


results = glob.glob('*RESULTS')
results.sort()




means = []
selfn = []
psds = []
orients = []

for curfile in results:
    with open(curfile,"r") as f:
        for line in f:
            line = line.rstrip()
            if 'Mean Ratio' in line:
                line = line.split(' ')
                
                nums = line[7].split('+/-')
                means.append(line[3] + ', ' + line[5].replace(':','') + ', ' + line[6] + ', ' + nums[0] + ', '+ nums[1])
            elif 'Self-Noise' in line:
                line = line.split(' ')
                nums = line[8].split('+/-')
                selfn.append(line[6].replace(':','') + ', ' + line[7] + ', ' + nums[0] + ', ' + nums[1])
            elif 'PSD Noise'  in line:
                line = line.split(' ')
                nums = line[9].split('+/-')
                psds.append(line[7].replace(':','') + ', '+ line[8] + ', ' + nums[0] + ', ' + nums[1])
            elif 'Orientation' in line:
                line = line.replace(':', ' ' )
                line = line.split(' ')
                
                orients.append(line[2] + ', ' + line[4] + ', ' + line[5] + ', ' + line[7])
                


#### Here we plot the self noise


# Do BH0
sn1 = []
err1 = []
sn2 = []
err2 = []
sn3 = []
err3 = []
for sn in selfn:
    if 'BH0' in sn:
        if '1, B' in sn:
            sn1.append(float(sn.split(', ')[2]))
            err1.append(2.5*float(sn.split(', ')[3]))
        elif '2, B' in sn:
            sn2.append(float(sn.split(', ')[2]))
            err2.append(2.5*float(sn.split(', ')[3]))
        elif '3, B' in sn:
            sn3.append(float(sn.split(', ')[2]))
            err3.append(2.5*float(sn.split(', ')[3]))
            


# Here we plot the mean ratio
plt.figure(1)
plt.subplot(311)
plt.errorbar(range(1,len(sn1)+1),sn1,err1, fmt='o', label='Sensor 1')
plt.errorbar(range(1,len(sn2)+1),sn2,err2, fmt='o', label='Sensor 2')
plt.errorbar(range(1,len(sn3)+1),sn3,err3, fmt='o', label='Sensor 3')
plt.title('Self-Noise 30 to 100 s Period BH0 Channel')
plt.xlim((0,len(sn3)+4))
plt.legend(loc=4, frameon=True)
plt.ylim((-160,-120))
plt.xticks([])



# Do BH1
sn1 = []
err1 = []
sn2 = []
err2 = []
sn3 = []
err3 = []
for sn in selfn:
    if 'BH1' in sn:
        if '1, B' in sn:
            sn1.append(float(sn.split(', ')[2]))
            err1.append(2.5*float(sn.split(', ')[3]))
        elif '2, B' in sn:
            sn2.append(float(sn.split(', ')[2]))
            err2.append(2.5*float(sn.split(', ')[3]))
        elif '3, B' in sn:
            sn3.append(float(sn.split(', ')[2]))
            err3.append(2.5*float(sn.split(', ')[3]))

# Here we plot the mean ratio
plt.figure(1)
plt.subplot(312)
plt.errorbar(range(1,len(sn1)+1),sn1,err1, fmt='o', label='Sensor 1')
plt.errorbar(range(1,len(sn1)+1),sn2,err2, fmt='o', label='Sensor 2')
plt.errorbar(range(1,len(sn1)+1),sn3,err3, fmt='o', label='Sensor 3')
plt.title('Self-Noise 30 to 100 s Period BH1 Channel')
plt.legend(loc=4, frameon=True)
plt.xticks([])
plt.xlim((0,len(sn1)+4))
plt.ylim((-160,-120))
plt.ylabel('Mean Self-Noise (dB)')

# Do BH2
sn1 = []
err1 = []
sn2 = []
err2 = []
sn3 = []
err3 = []
for sn in selfn:
    if 'BH2' in sn:
        if '1, B' in sn:
            sn1.append(float(sn.split(', ')[2]))
            err1.append(2.5*float(sn.split(', ')[3]))
        elif '2, B' in sn:
            sn2.append(float(sn.split(', ')[2]))
            err2.append(2.5*float(sn.split(', ')[3]))
        elif '3, B' in sn:
            sn3.append(float(sn.split(', ')[2]))
            err3.append(2.5*float(sn.split(', ')[3]))

# Here we plot the mean ratio
plt.figure(1)
plt.subplot(313)
plt.errorbar(range(1,len(sn1)+1),sn1,err1, fmt='o', label='Sensor 1')
plt.errorbar(range(1,len(sn2)+1),sn2,err2, fmt='o', label='Sensor 2')
plt.errorbar(range(1,len(sn3)+1),sn3,err3, fmt='o', label='Sensor 3')
plt.title('Self-Noise 30 to 100 s Period BH2 Channel')
plt.xlim((0,len(sn3)+4))
plt.xticks(range(1,len(sn3)+1))
plt.ylim((-160,-120))
plt.xlabel('Trial Number')
plt.legend(loc=4, frameon=True)
plt.savefig("SelfNoise.jpg",dpi=300)
plt.clf()

######################################################################################################################################3

#### Here we plot the PSD Noise


# Do BH0
psdn1 = []
err1 = []
psdn2 = []
err2 = []
psdn3 = []
err3 = []
for sn in psds:
    if 'BH0' in sn:
        if '1, B' in sn:
            psdn1.append(float(sn.split(', ')[2]))
            err1.append(float(sn.split(', ')[3]))
        elif '2, B' in sn:
            psdn2.append(float(sn.split(', ')[2]))
            err2.append(float(sn.split(', ')[3]))
        elif '3, B' in sn:
            psdn3.append(float(sn.split(', ')[2]))
            err3.append(float(sn.split(', ')[3]))
            

# Here we plot the mean ratio
plt.figure(1)
plt.subplot(311)
plt.errorbar(range(1,len(psdn1)+1),psdn1,err1, fmt='o', label='Sensor 1')
plt.errorbar(range(1,len(psdn2)+1),psdn2,err2, fmt='o', label='Sensor 2')
plt.errorbar(range(1,len(psdn3)+1),psdn3,err3, fmt='o', label='Sensor 3')
plt.title('PSD 30 to 100 s Period BH0 Channel')
plt.xlim((0,len(sn3)+4))
plt.legend(loc=4, frameon=True)
plt.ylim((-170,-120))
plt.xticks([])



# Do BH1
psdn1 = []
err1 = []
psdn2 = []
err2 = []
psdn3 = []
err3 = []
for sn in psds:
    if 'BH1' in sn:
        if '1, B' in sn:
            psdn1.append(float(sn.split(', ')[2]))
            err1.append(2.5*float(sn.split(', ')[3]))
        elif '2, B' in sn:
            psdn2.append(float(sn.split(', ')[2]))
            err2.append(2.5*float(sn.split(', ')[3]))
        elif '3, B' in sn:
            psdn3.append(float(sn.split(', ')[2]))
            err3.append(2.5*float(sn.split(', ')[3]))

# Here we plot the mean ratio
plt.figure(1)
plt.subplot(312)
plt.errorbar(range(1,len(psdn1)+1),psdn1,err1, fmt='o', label='Sensor 1')
plt.errorbar(range(1,len(psdn1)+1),psdn2,err2, fmt='o', label='Sensor 2')
plt.errorbar(range(1,len(psdn1)+1),psdn3,err3, fmt='o', label='Sensor 3')
plt.title('PSD 30 to 100 s Period BH1 Channel')
plt.legend(loc=4, frameon=True)
plt.xticks([])
plt.xlim((0,len(sn1)+4))
plt.ylim((-170,-120))
plt.ylabel('Mean PSD (dB)')

# Do BH2
psdn1 = []
err1 = []
psdn2 = []
err2 = []
psdn3 = []
err3 = []
for sn in psds:
    if 'BH2' in sn:
        if '1, B' in sn:
            psdn1.append(float(sn.split(', ')[2]))
            err1.append(2.5*float(sn.split(', ')[3]))
        elif '2, B' in sn:
            psdn2.append(float(sn.split(', ')[2]))
            err2.append(2.5*float(sn.split(', ')[3]))
        elif '3, B' in sn:
            psdn3.append(float(sn.split(', ')[2]))
            err3.append(2.5*float(sn.split(', ')[3]))

# Here we plot the mean ratio
plt.figure(1)
plt.subplot(313)
plt.errorbar(range(1,len(psdn1)+1),psdn1,err1, fmt='o', label='Sensor 1')
plt.errorbar(range(1,len(psdn2)+1),psdn2,err2, fmt='o', label='Sensor 2')
plt.errorbar(range(1,len(psdn3)+1),psdn3,err3, fmt='o', label='Sensor 3')
plt.title('PSD 30 to 100 s Period BH2 Channel')
plt.xlim((0,len(sn3)+4))
plt.xticks(range(1,len(sn3)+1))
plt.ylim((-170,-120))
plt.xlabel('Trial Number')
plt.legend(loc=4, frameon=True)
plt.savefig("PSDNoise.jpg",dpi=300)
plt.clf()

######################################################################################################################################3



#### Here we plot the PSD Noise From the mean


# Do BH0
psdn1 = []
err1 = []
psdn2 = []
err2 = []
psdn3 = []
err3 = []
for sn in psds:
    if 'BH0' in sn:
        if '1, B' in sn:
            psdn1.append(float(sn.split(', ')[2]))
            err1.append(2.5*float(sn.split(', ')[3]))
        elif '2, B' in sn:
            psdn2.append(float(sn.split(', ')[2]))
            err2.append(2.5*float(sn.split(', ')[3]))
        elif '3, B' in sn:
            psdn3.append(float(sn.split(', ')[2]))
            err3.append(2.5*float(sn.split(', ')[3]))


psd = psdn1 + psdn2 + psdn3
err = err1 + err2 + err3
psdn1 = psdn1-np.mean(psd)
psdn2 = psdn2 -np.mean(psd)
psdn3 = psdn3-np.mean(psd)
err1 = err1 -np.mean(err)
err2 = err2 -np.mean(err)
err3 = err3 - np.mean(err)
            

# Here we plot the mean ratio
plt.figure(1)
plt.subplot(311)
plt.errorbar(range(1,len(psdn1)+1),psdn1,err1, fmt='o', label='Sensor 1')
plt.errorbar(range(1,len(psdn2)+1),psdn2,err2, fmt='o', label='Sensor 2')
plt.errorbar(range(1,len(psdn3)+1),psdn3,err3, fmt='o', label='Sensor 3')
plt.title('Mean removed PSD 30 to 100 s Period BH0 Channel')
plt.xlim((0,len(sn3)+4))
plt.legend(loc=4, frameon=True)
plt.ylim((-5,5))
plt.xticks([])



# Do BH1
psdn1 = []
err1 = []
psdn2 = []
err2 = []
psdn3 = []
err3 = []
for sn in psds:
    if 'BH1' in sn:
        if '1, B' in sn:
            psdn1.append(float(sn.split(', ')[2]))
            err1.append(2.5*float(sn.split(', ')[3]))
        elif '2, B' in sn:
            psdn2.append(float(sn.split(', ')[2]))
            err2.append(2.5*float(sn.split(', ')[3]))
        elif '3, B' in sn:
            psdn3.append(float(sn.split(', ')[2]))
            err3.append(2.5*float(sn.split(', ')[3]))
psd = psdn1 + psdn2 + psdn3
err = err1 + err2 + err3
psdn1 = psdn1-np.mean(psd)
psdn2 = psdn2 -np.mean(psd)
psdn3 = psdn3-np.mean(psd)
err1 = err1 -np.mean(err)
err2 = err2 -np.mean(err)
err3 = err3 - np.mean(err)
# Here we plot the mean ratio
plt.figure(1)
plt.subplot(312)
plt.errorbar(range(1,len(psdn1)+1),psdn1,err1, fmt='o', label='Sensor 1')
plt.errorbar(range(1,len(psdn1)+1),psdn2,err2, fmt='o', label='Sensor 2')
plt.errorbar(range(1,len(psdn1)+1),psdn3,err3, fmt='o', label='Sensor 3')
plt.title('Mean Removed PSD 30 to 100 s Period BH1 Channel')
plt.legend(loc=4, frameon=True)
plt.xticks([])
plt.xlim((0,len(sn1)+4))
plt.ylim((-5,5))
plt.ylabel('Mean Removed PSD (dB)')

# Do BH2
psdn1 = []
err1 = []
psdn2 = []
err2 = []
psdn3 = []
err3 = []
for sn in psds:
    if 'BH2' in sn:
        if '1, B' in sn:
            psdn1.append(float(sn.split(', ')[2]))
            err1.append(2.5*float(sn.split(', ')[3]))
        elif '2, B' in sn:
            psdn2.append(float(sn.split(', ')[2]))
            err2.append(2.5*float(sn.split(', ')[3]))
        elif '3, B' in sn:
            psdn3.append(float(sn.split(', ')[2]))
            err3.append(2.5*float(sn.split(', ')[3]))
            
psd = psdn1 + psdn2 + psdn3
err = err1 + err2 + err3
psdn1 = psdn1-np.mean(psd)
psdn2 = psdn2 -np.mean(psd)
psdn3 = psdn3-np.mean(psd)
err1 = err1 -np.mean(err)
err2 = err2 -np.mean(err)
err3 = err3 - np.mean(err)

# Here we plot the mean ratio
plt.figure(1)
plt.subplot(313)
plt.errorbar(range(1,len(psdn1)+1),psdn1,err1, fmt='o', label='Sensor 1')
plt.errorbar(range(1,len(psdn2)+1),psdn2,err2, fmt='o', label='Sensor 2')
plt.errorbar(range(1,len(psdn3)+1),psdn3,err3, fmt='o', label='Sensor 3')
plt.title('Mean Removed PSD 30 to 100 s Period BH2 Channel')
plt.xlim((0,len(sn3)+4))
plt.xticks(range(1,len(sn3)+1))
plt.ylim((-5,5))
plt.xlabel('Trial Number')
plt.legend(loc=4, frameon=True)
plt.savefig("PSDNoiseMeanRemoved.jpg",dpi=300)
#plt.show()
plt.clf()

######################################################################################################################################3
######################################################################################################################################3

#### Here we plot the self noise Mean Removed


# Do BH0
sn1 = []
err1 = []
sn2 = []
err2 = []
sn3 = []
err3 = []
for sn in selfn:
    if 'BH0' in sn:
        if '1, B' in sn:
            sn1.append(float(sn.split(', ')[2]))
            err1.append(2.5*float(sn.split(', ')[3]))
        elif '2, B' in sn:
            sn2.append(float(sn.split(', ')[2]))
            err2.append(2.5*float(sn.split(', ')[3]))
        elif '3, B' in sn:
            sn3.append(float(sn.split(', ')[2]))
            err3.append(2.5*float(sn.split(', ')[3]))
            
sn = sn1 + sn2 + sn3
err = err1 + err2 + err3
sn1 = sn1 - np.mean(sn)
sn2 = sn2 - np.mean(sn)
sn3 = sn3 - np.mean(sn)
err1 = err1 - np.mean(err)
err2 = err2 - np.mean(err)
err3 = err3 - np.mean(err)

# Here we plot the mean ratio
plt.figure(1)
plt.subplot(311)
plt.errorbar(range(1,len(sn1)+1),sn1,err1, fmt='o', label='Sensor 1')
plt.errorbar(range(1,len(sn2)+1),sn2,err2, fmt='o', label='Sensor 2')
plt.errorbar(range(1,len(sn3)+1),sn3,err3, fmt='o', label='Sensor 3')
plt.title('Mean Removed Self-Noise 30 to 100 s Period BH0 Channel')
plt.xlim((0,len(sn3)+4))
plt.legend(loc=4, frameon=True)
plt.ylim((-5,5))
plt.xticks([])



# Do BH1
sn1 = []
err1 = []
sn2 = []
err2 = []
sn3 = []
err3 = []
for sn in selfn:
    if 'BH1' in sn:
        if '1, B' in sn:
            sn1.append(float(sn.split(', ')[2]))
            err1.append(2.5*float(sn.split(', ')[3]))
        elif '2, B' in sn:
            sn2.append(float(sn.split(', ')[2]))
            err2.append(2.5*float(sn.split(', ')[3]))
        elif '3, B' in sn:
            sn3.append(float(sn.split(', ')[2]))
            err3.append(2.5*float(sn.split(', ')[3]))
            
sn = sn1 + sn2 + sn3
err = err1 + err2 + err3
sn1 = sn1 - np.mean(sn)
sn2 = sn2 - np.mean(sn)
sn3 = sn3 - np.mean(sn)
err1 = err1 - np.mean(err)
err2 = err2 - np.mean(err)
err3 = err3 - np.mean(err)

# Here we plot the mean ratio
plt.figure(1)
plt.subplot(312)
plt.errorbar(range(1,len(sn1)+1),sn1,err1, fmt='o', label='Sensor 1')
plt.errorbar(range(1,len(sn1)+1),sn2,err2, fmt='o', label='Sensor 2')
plt.errorbar(range(1,len(sn1)+1),sn3,err3, fmt='o', label='Sensor 3')
plt.title('Mean Removed Self-Noise 30 to 100 s Period BH1 Channel')
plt.legend(loc=4, frameon=True)
plt.xticks([])
plt.xlim((0,len(sn1)+4))
plt.ylim((-5,5))
plt.ylabel('Mean Removed Self-Noise (dB)')

# Do BH2
sn1 = []
err1 = []
sn2 = []
err2 = []
sn3 = []
err3 = []
for sn in selfn:
    if 'BH2' in sn:
        if '1, B' in sn:
            sn1.append(float(sn.split(', ')[2]))
            err1.append(2.5*float(sn.split(', ')[3]))
        elif '2, B' in sn:
            sn2.append(float(sn.split(', ')[2]))
            err2.append(2.5*float(sn.split(', ')[3]))
        elif '3, B' in sn:
            sn3.append(float(sn.split(', ')[2]))
            err3.append(2.5*float(sn.split(', ')[3]))

sn = sn1 + sn2 + sn3
err = err1 + err2 + err3
sn1 = sn1 - np.mean(sn)
sn2 = sn2 - np.mean(sn)
sn3 = sn3 - np.mean(sn)
err1 = err1 - np.mean(err)
err2 = err2 - np.mean(err)
err3 = err3 - np.mean(err)



# Here we plot the mean ratio
plt.figure(1)
plt.subplot(313)
plt.errorbar(range(1,len(sn1)+1),sn1,err1, fmt='o', label='Sensor 1')
plt.errorbar(range(1,len(sn2)+1),sn2,err2, fmt='o', label='Sensor 2')
plt.errorbar(range(1,len(sn3)+1),sn3,err3, fmt='o', label='Sensor 3')
plt.title('Mean Removed Self-Noise 30 to 100 s Period BH2 Channel')
plt.xlim((0,len(sn3)+4))
plt.xticks(range(1,len(sn3)+1))
plt.ylim((-5,5))
plt.xlabel('Trial Number')
plt.legend(loc=4, frameon=True)
plt.savefig("SelfNoiseMeanRemoved.jpg",dpi=300)
plt.clf()

######################################################################################################################################3
