#!/usr/bin/env python


import matplotlib.pyplot as plt
import glob
import sys
import numpy as np
import matplotlib as mpl
mpl.rc('font',family='serif')
mpl.rc('font',serif='Times') 
mpl.rc('text', usetex=True)
mpl.rc('font',size=12)
results = glob.glob('*RESULTSLP')
results.sort()

from scipy import stats

import pylab

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
                
# We now have the results

print(psds)
print(selfn)





######################################################################################################################################3

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
            

print 'Here is the mean of the Self-Noise BH0: ' + str(np.mean(sn1+sn2+sn3))
print 'Here is the std of the Self-Noise BH0: ' + str(np.std(sn1+sn2+sn3))
allvals = sn1 + sn2 +sn3

# Here we plot the mean ratio
plt.figure(1)
ax=plt.subplot(311)
ax.errorbar(range(1,len(sn1)+1),sn1,err1, fmt='+', label='Sensor 1')
ax.errorbar(range(1,len(sn2)+1),sn2,err2, fmt='+', label='Sensor 2')
ax.errorbar(range(1,len(sn3)+1),sn3,err3, fmt='+', label='Sensor 3')
plt.title('BH0')
plt.xlim((0,len(sn3)+1))
plt.xticks(range(1,len(sn3)+1))
#plt.legend(loc=4, frameon=True)
plt.ylim((-180,-150))
plt.yticks([-180, -165, -150],[-180, -165,-150])
#plt.xticks([])
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.3,
                 box.width, box.height * 0.8])


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

print 'Here is the mean of the Self-Noise BH1: ' + str(np.mean(sn1+sn2+sn3))
print 'Here is the std of the Self-Noise BH1: ' + str(np.std(sn1+sn2+sn3))
allvals += sn1 + sn2 +sn3
# Here we plot the mean ratio
plt.figure(1)
ax=plt.subplot(312)
ax.errorbar(range(1,len(sn1)+1),sn1,err1, fmt='+', label='Sensor 1')
ax.errorbar(range(1,len(sn1)+1),sn2,err2, fmt='+', label='Sensor 2')
ax.errorbar(range(1,len(sn1)+1),sn3,err3, fmt='+', label='Sensor 3')
plt.title('BH1')
#plt.legend(loc=4, frameon=True)
plt.xticks(range(1,len(sn3)+1))
#plt.xticks([])
plt.xlim((0,len(sn1)+1))
plt.ylim((-180,-150))
plt.yticks([-180, -165, -150],[-180, -165,-150])
plt.ylabel('Mean Self-Noise 30 to 100 s Period (dB rel. 1 $(m/s^2)^2/Hz$)')
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.3,
                 box.width, box.height * 0.8])


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


print 'Here is the mean of the Self-Noise BH2: ' + str(np.mean(sn1+sn2+sn3))
print 'Here is the std of the Self-Noise BH2: ' + str(np.std(sn1+sn2+sn3))
allvals += sn1 + sn2 +sn3
# Here we plot the mean ratio
plt.figure(1)
ax=plt.subplot(313)
ax.errorbar(range(1,len(sn1)+1),sn1,err1, fmt='+', label='Sensor 1')
ax.errorbar(range(1,len(sn2)+1),sn2,err2, fmt='+', label='Sensor 2')
ax.errorbar(range(1,len(sn3)+1),sn3,err3, fmt='+', label='Sensor 3')
plt.title('BH2')
plt.xlim((0,len(sn3)+1))
plt.xticks(range(1,len(sn3)+1))
plt.ylim((-180,-150))
plt.yticks([-180, -165, -150],[-180, -165,-150])
plt.xlabel('Trial Number')
#plt.legend(loc=4, frameon=True)


box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.3,
                 box.width, box.height * 0.8])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5),
          fancybox=False, shadow=False, ncol=3)


conf_int = stats.t.interval(0.99, len(allvals)-1, loc=np.mean(allvals), scale=stats.sem(allvals) )

print  'Here is the confidence internval Self-noise 30 to 100 ' + str(conf_int)


plt.savefig("SelfNoiseLP.pdf",dpi=400, format='pdf')
plt.savefig("SelfNoiseLP.jpg",dpi=400)
plt.clf()


allvals = allvals - np.mean(allvals)

#plt.hist(allvals)
#plt.show()
#stats.probplot(allvals,dist="t", sparams=(5,), plot=pylab)
fig = plt.figure(1)
ax = fig.add_subplot(111)
stats.probplot(allvals,plot=pylab)
pylab.xlabel('Quantiles')
pylab.ylabel('Mean Removed Ordered Values')
pylab.title('Probability Plot Self-Noise 30 s to 100 s Period')
ax.get_lines()[0].set_marker('o')
ax.get_lines()[0].set_markerfacecolor('k')
ax.get_lines()[0].set_markeredgecolor('k')
ax.get_lines()[0].set_markersize(10.0)
ax.get_lines()[1].set_color('k')
ax.get_lines()[1].set_linewidth(2.)
ax.get_children()[2].set_fontsize(18.0)
pylab.savefig('SelfNoiseQQLP.pdf',dpi=400, format='pdf')
pylab.savefig('SelfNoiseQQLP.jpg',dpi=400, format='jpg')
pylab.clf()


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
            
print 'Here is the mean of the PSD BH0: ' + str(np.mean(psdn1+psdn2+psdn3))
print 'Here is the std of the PSD BH0: ' + str(np.std(psdn1+psdn2+psdn3))
allvals = psdn1+psdn2+psdn3

# Here we plot the mean ratio
plt.figure(1)
ax=plt.subplot(311)
ax.errorbar(range(1,len(psdn1)+1),psdn1,err1, fmt='+', label='Sensor 1')
ax.errorbar(range(1,len(psdn2)+1),psdn2,err2, fmt='+', label='Sensor 2')
ax.errorbar(range(1,len(psdn3)+1),psdn3,err3, fmt='+', label='Sensor 3')
plt.title('BH0')

#plt.legend(loc=4, frameon=True)
plt.ylim((-180,-130))
plt.yticks([-180, -155, -130],[-180, -155,-130])
plt.xlim((0,len(sn3)+1))
plt.xticks(range(1,len(sn3)+1))
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.3,
                 box.width, box.height * 0.8])


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
print 'Here is the mean of the PSD BH1: ' + str(np.mean(psdn1+psdn2+psdn3))
print 'Here is the std of the PSD BH1: ' + str(np.std(psdn1+psdn2+psdn3))

allvals += psdn1+psdn2+psdn3

# Here we plot the mean ratio
plt.figure(1)
ax=plt.subplot(312)
ax.errorbar(range(1,len(psdn1)+1),psdn1,err1, fmt='+', label='Sensor 1')
ax.errorbar(range(1,len(psdn1)+1),psdn2,err2, fmt='+', label='Sensor 2')
ax.errorbar(range(1,len(psdn1)+1),psdn3,err3, fmt='+', label='Sensor 3')
plt.title('BH1')
#plt.legend(loc=4, frameon=True)
plt.xlim((0,len(sn3)+1))
plt.xticks(range(1,len(sn3)+1))
plt.ylim((-180,-130))
plt.yticks([-180, -155, -130],[-180, -155,-130])
plt.ylabel('Mean PSD 30 to 100 s Period (dB rel. 1 $(m/s^2)^2/Hz$)')
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.3,
                 box.width, box.height * 0.8])
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
print 'Here is the mean of the PSD BH2: ' + str(np.mean(psdn1+psdn2+psdn3))
print 'Here is the std of the PSD BH2: ' + str(np.std(psdn1+psdn2+psdn3))


allvals += psdn1+psdn2+psdn3

# Here we plot the mean ratio
plt.figure(1)
ax=plt.subplot(313)
ax.errorbar(range(1,len(psdn1)+1),psdn1,err1, fmt='+', label='Sensor 1')
ax.errorbar(range(1,len(psdn2)+1),psdn2,err2, fmt='+', label='Sensor 2')
ax.errorbar(range(1,len(psdn3)+1),psdn3,err3, fmt='+', label='Sensor 3')
plt.title('BH2')
plt.xlim((0,len(sn3)+1))
plt.xticks(range(1,len(sn3)+1))
plt.ylim((-180,-130))
plt.yticks([-180, -155, -130],[-180, -155,-130])
plt.xlabel('Trial Number')
#plt.legend(loc=4, frameon=True)
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.3,
                 box.width, box.height * 0.8])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5),
          fancybox=False, shadow=False, ncol=3)




plt.savefig("PSDNoiseLP.pdf",dpi=400,format='pdf')
plt.savefig("PSDNoiseLP.jpg",dpi=400)
plt.clf()


allvals = allvals -np.mean(allvals)

fig = plt.figure(1)
ax = fig.add_subplot(111)
stats.probplot(allvals,plot=pylab)
pylab.xlabel('Quantiles')
pylab.ylabel('Mean Removed Ordered Values')
pylab.title('Probability Plot PSD 30 s to 100 s Period')
ax.get_lines()[0].set_marker('o')
ax.get_lines()[0].set_markerfacecolor('k')
ax.get_lines()[0].set_markeredgecolor('k')
ax.get_lines()[0].set_markersize(10.0)
ax.get_lines()[1].set_color('k')
ax.get_lines()[1].set_linewidth(2.)
ax.get_children()[2].set_fontsize(18.0)
pylab.savefig('PSDQQLP.pdf',dpi=400, format='pdf')
pylab.savefig('PSDQQLP.jpg',dpi=400, format='jpg')
#pylab.show()
pylab.clf()



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
            err1.append(float(sn.split(', ')[3]))
        elif '2, B' in sn:
            psdn2.append(float(sn.split(', ')[2]))
            err2.append(float(sn.split(', ')[3]))
        elif '3, B' in sn:
            psdn3.append(float(sn.split(', ')[2]))
            err3.append(float(sn.split(', ')[3]))


psd = psdn1 + psdn2 + psdn3
err = err1 + err2 + err3
psdn1 = psdn1-np.mean(psd)
psdn2 = psdn2 -np.mean(psd)
psdn3 = psdn3-np.mean(psd)
#err1 = err1 -np.mean(err)
#err2 = err2 -np.mean(err)
#err3 = err3 - np.mean(err)
            

# Here we plot the mean ratio
plt.figure(1)
ax=plt.subplot(311)
ax.errorbar(range(1,len(psdn1)+1),psdn1,err1, fmt='+', label='Sensor 1')
ax.errorbar(range(1,len(psdn2)+1),psdn2,err2, fmt='+', label='Sensor 2')
ax.errorbar(range(1,len(psdn3)+1),psdn3,err3, fmt='+', label='Sensor 3')
plt.title('BH0')

#plt.legend(loc=4, frameon=True)
plt.ylim((-10,10))
plt.yticks([-10,0,10],[-10, 0,10])
plt.xlim((0,len(sn3)+1))
plt.xticks(range(1,len(sn3)+1))
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.3,
                 box.width, box.height * 0.8])


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
            err1.append(float(sn.split(', ')[3]))
        elif '2, B' in sn:
            psdn2.append(float(sn.split(', ')[2]))
            err2.append(float(sn.split(', ')[3]))
        elif '3, B' in sn:
            psdn3.append(float(sn.split(', ')[2]))
            err3.append(float(sn.split(', ')[3]))
psd = psdn1 + psdn2 + psdn3
err = err1 + err2 + err3
psdn1 = psdn1-np.mean(psd)
psdn2 = psdn2 -np.mean(psd)
psdn3 = psdn3-np.mean(psd)
#err1 = err1 -np.mean(err)
#err2 = err2 -np.mean(err)
#err3 = err3 - np.mean(err)
# Here we plot the mean ratio
plt.figure(1)
ax=plt.subplot(312)
ax.errorbar(range(1,len(psdn1)+1),psdn1,err1, fmt='+', label='Sensor 1')
ax.errorbar(range(1,len(psdn1)+1),psdn2,err2, fmt='+', label='Sensor 2')
ax.errorbar(range(1,len(psdn1)+1),psdn3,err3, fmt='+', label='Sensor 3')
plt.title('BH1')
#plt.legend(loc=4, frameon=True)
plt.xlim((0,len(sn3)+1))
plt.xticks(range(1,len(sn3)+1))
plt.ylim((-10,10))
plt.yticks([-10,0,10],[-10, 0,10])
plt.ylabel('Mean Removed PSD 30 to 100 s Period (dB rel. 1 $(m/s^2)^2/Hz$)')
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.3,
                 box.width, box.height * 0.8])
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
            err1.append(float(sn.split(', ')[3]))
        elif '2, B' in sn:
            psdn2.append(float(sn.split(', ')[2]))
            err2.append(float(sn.split(', ')[3]))
        elif '3, B' in sn:
            psdn3.append(float(sn.split(', ')[2]))
            err3.append(float(sn.split(', ')[3]))
            
psd = psdn1 + psdn2 + psdn3
err = err1 + err2 + err3
psdn1 = psdn1-np.mean(psd)
psdn2 = psdn2 -np.mean(psd)
psdn3 = psdn3-np.mean(psd)
#err1 = err1 -np.mean(err)
#err2 = err2 -np.mean(err)
#err3 = err3 - np.mean(err)

# Here we plot the mean ratio
plt.figure(1)
ax=plt.subplot(313)
ax.errorbar(range(1,len(psdn1)+1),psdn1,err1, fmt='+', label='Sensor 1')
ax.errorbar(range(1,len(psdn2)+1),psdn2,err2, fmt='+', label='Sensor 2')
ax.errorbar(range(1,len(psdn3)+1),psdn3,err3, fmt='+', label='Sensor 3')
plt.title('BH2')
plt.xlim((0,len(sn3)+1))
plt.xticks(range(1,len(sn3)+1))
plt.ylim((-10,10))
plt.yticks([-10,0,10],[-10, 0,10])
plt.xlabel('Trial Number')
plt.legend(loc=4, frameon=True)





box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.3,
                 box.width, box.height * 0.8])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5),
          fancybox=False, shadow=False, ncol=3)







plt.savefig("PSDNoiseMeanRemovedLP.pdf",dpi=400, format= 'pdf')
plt.savefig("PSDNoiseMeanRemovedLP.jpg",dpi=400)
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
            err1.append(float(sn.split(', ')[3]))
        elif '2, B' in sn:
            sn2.append(float(sn.split(', ')[2]))
            err2.append(float(sn.split(', ')[3]))
        elif '3, B' in sn:
            sn3.append(float(sn.split(', ')[2]))
            err3.append(float(sn.split(', ')[3]))
            
sn = sn1 + sn2 + sn3
err = err1 + err2 + err3
sn1 = sn1 - np.mean(sn)
sn2 = sn2 - np.mean(sn)
sn3 = sn3 - np.mean(sn)
#err1 = err1 - np.mean(err)
#err2 = err2 - np.mean(err)
#err3 = err3 - np.mean(err)

# Here we plot the mean ratio
plt.figure(1)
ax=plt.subplot(311)
ax.errorbar(range(1,len(sn1)+1),sn1,err1, fmt='+', label='Sensor 1')
ax.errorbar(range(1,len(sn2)+1),sn2,err2, fmt='+', label='Sensor 2')
ax.errorbar(range(1,len(sn3)+1),sn3,err3, fmt='+', label='Sensor 3')
plt.title('BH0')
plt.xlim((0,len(sn3)+1))
plt.xticks(range(1,len(sn3)+1))
#plt.legend(loc=4, frameon=True)
plt.ylim((-10,10))
plt.yticks([-10,0,10],[-10, 0,10])
#plt.xticks([])
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.3,
                 box.width, box.height * 0.8])


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
            err1.append(float(sn.split(', ')[3]))
        elif '2, B' in sn:
            sn2.append(float(sn.split(', ')[2]))
            err2.append(float(sn.split(', ')[3]))
        elif '3, B' in sn:
            sn3.append(float(sn.split(', ')[2]))
            err3.append(float(sn.split(', ')[3]))
            
sn = sn1 + sn2 + sn3
err = err1 + err2 + err3
sn1 = sn1 - np.mean(sn)
sn2 = sn2 - np.mean(sn)
sn3 = sn3 - np.mean(sn)
#err1 = err1 - np.mean(err)
#err2 = err2 - np.mean(err)
#err3 = err3 - np.mean(err)

# Here we plot the mean ratio
plt.figure(1)
ax=plt.subplot(312)
ax.errorbar(range(1,len(sn1)+1),sn1,err1, fmt='+', label='Sensor 1')
ax.errorbar(range(1,len(sn1)+1),sn2,err2, fmt='+', label='Sensor 2')
ax.errorbar(range(1,len(sn1)+1),sn3,err3, fmt='+', label='Sensor 3')
plt.title('BH1')
#plt.legend(loc=4, frameon=True)
plt.xlim((0,len(sn3)+1))
plt.xticks(range(1,len(sn3)+1))
plt.ylim((-10,10))
plt.yticks([-10,0,10],[-10, 0,10])
plt.ylabel('Mean Removed Self-Noise 30 to 100 s Period (dB rel. 1 $(m/s^2)^2/Hz$)',fontsize=10)
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.3,
                 box.width, box.height * 0.8])
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
            err1.append(float(sn.split(', ')[3]))
        elif '2, B' in sn:
            sn2.append(float(sn.split(', ')[2]))
            err2.append(float(sn.split(', ')[3]))
        elif '3, B' in sn:
            sn3.append(float(sn.split(', ')[2]))
            err3.append(float(sn.split(', ')[3]))

# Do not remove the mean of the std
sn = sn1 + sn2 + sn3
err = err1 + err2 + err3
sn1 = sn1 - np.mean(sn)
sn2 = sn2 - np.mean(sn)
sn3 = sn3 - np.mean(sn)
#err1 = err1 - np.mean(err)
#err2 = err2 - np.mean(err)
#err3 = err3 - np.mean(err)



# Here we plot the mean ratio
plt.figure(1)
ax=plt.subplot(313)
ax.errorbar(range(1,len(sn1)+1),sn1,err1, fmt='+', label='Sensor 1')
ax.errorbar(range(1,len(sn2)+1),sn2,err2, fmt='+', label='Sensor 2')
ax.errorbar(range(1,len(sn3)+1),sn3,err3, fmt='+', label='Sensor 3')
plt.title('BH2')
plt.xlim((0,len(sn3)+1))
plt.xticks(range(1,len(sn3)+1))
plt.ylim((-10,10))
plt.yticks([-10,0,10],[-10, 0,10])
plt.xlabel('Trial Number')
#plt.legend(loc=4, frameon=True)

box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.3,
                 box.width, box.height * 0.8])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5),
          fancybox=False, shadow=False, ncol=3)




plt.savefig("SelfNoiseMeanRemovedLP.pdf",dpi=400, format='pdf')
plt.savefig("SelfNoiseMeanRemovedLP.jpg",dpi=400)
plt.clf()

######################################################################################################################################3
