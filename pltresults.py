#!/usr/bin/env python


import matplotlib.pyplot as plt
import glob
import sys
import numpy as np
import matplotlib as mpl

from scipy import stats
import pylab


mpl.rc('font',family='serif')
mpl.rc('font',serif='Times') 
mpl.rc('text', usetex=True)
mpl.rc('font',size=12)

Flag ='LP'


results = glob.glob('*RESULTS' + Flag)
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
                
# We now have the results

#print(psds)
#print(selfn)


































#########################################################################################################################
comp12 = []
err12 = []
comp23 = []
err23 = []
comp13 = []
err13 = []
for mean in means:
    if 'BH0' in mean:
        if '1, 2' in mean:
            comp12.append(float(mean.split(', ')[3]))
            err12.append(2.5*float(mean.split(', ')[4]))
        elif '1, 3' in mean:
            comp13.append(float(mean.split(', ')[3]))
            err13.append(2.5*float(mean.split(', ')[4]))
        else:
            comp23.append(float(mean.split(', ')[3]))
            err23.append(2.5*float(mean.split(', ')[4]))

allvalsZ = comp12+comp13+comp23  
allvals = allvalsZ         

print 'Here is the mean of the mean ratios BH0: ' + str(np.mean(comp12+comp13+comp23))
print 'Here is the std of the mean ratios BH0: ' + str(np.std(comp12+comp13+comp23))
# Here we plot the mean ratio
plt.figure(1)
ax=plt.subplot(311)
ax.errorbar(range(1,len(comp23)+1),comp23,err23, fmt='+', label='Sensor 2 to 3')
ax.errorbar(range(1,len(comp13)+1),comp13,err13, fmt='+', label='Sensor 1 to 3')
ax.errorbar(range(1,len(comp12)+1),comp12,err12, fmt='+', label='Sensor 1 to 2')
plt.title('BH0')
plt.ylim((0.999, 1.001))
plt.xlim((0,len(comp23)+1))
plt.yticks([.999, 1.000, 1.001], [0.999, 1.000, 1.001])
#plt.legend(loc=4, frameon=True)
#plt.xticks([])
plt.xticks(range(1,len(comp23)+1))
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.3,
                 box.width, box.height * 0.8])




# Do BH1
comp12 = []
err12 = []
comp23 = []
err23 = []
comp13 = []
err13 = []
for mean in means:
    if 'BH1' in mean:
        if '1, 2' in mean:
            comp12.append(float(mean.split(', ')[3]))
            err12.append(2.5*float(mean.split(', ')[4]))
        elif '1, 3' in mean:
            comp13.append(float(mean.split(', ')[3]))
            err13.append(2.5*float(mean.split(', ')[4]))
        else:
            comp23.append(float(mean.split(', ')[3]))
            err23.append(2.5*float(mean.split(', ')[4]))
            
            
            
            
          
            
            
            
            
allvals += comp12+comp13+comp23        
            
            
            
            
            
print 'Here is the mean of the mean ratios BH1: ' + str(np.mean(comp12+comp13+comp23))
print 'Here is the std of the mean ratios BH1: ' + str(np.std(comp12+comp13+comp23))
# Here we plot the mean ratio
plt.figure(1)
ax=plt.subplot(312)
ax.errorbar(range(1,len(comp23)+1),comp23,err23, fmt='+', label='Sensor 2 to 3')
ax.errorbar(range(1,len(comp13)+1),comp13,err13, fmt='+', label='Sensor 1 to 3')
ax.errorbar(range(1,len(comp12)+1),comp12,err12, fmt='+', label='Sensor 1 to 2')
plt.title('BH1')
plt.ylim((0.999, 1.001))
plt.yticks([.999, 1.000, 1.001], [0.999, 1.000, 1.001])
#plt.legend(loc=4, frameon=True)
#plt.xticks([])
plt.xticks(range(1,len(comp23)+1))
plt.xlim((0,len(comp23)+1))
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.3,
                 box.width, box.height * 0.8])
plt.ylabel('Mean Ratio 4 to 8 s Period')

# Do BH2
comp12 = []
err12 = []
comp23 = []
err23 = []
comp13 = []
err13 = []
for mean in means:
    if 'BH2' in mean:
        if '1, 2' in mean:
            comp12.append(float(mean.split(', ')[3]))
            err12.append(2.5*float(mean.split(', ')[4]))
        elif '1, 3' in mean:
            comp13.append(float(mean.split(', ')[3]))
            err13.append(2.5*float(mean.split(', ')[4]))
        else:
            comp23.append(float(mean.split(', ')[3]))
            err23.append(2.5*float(mean.split(', ')[4]))

allvals += comp12+comp13+comp23
            
print 'Here is the mean of the mean ratios BH2: ' + str(np.mean(comp12+comp13+comp23))
print 'Here is the std of the mean ratios BH2: ' + str(np.std(comp12+comp13+comp23))

#conf_int = stats.norm.interval(0.99, loc=np.mean(allvals), scale=np.std(allvals) / np.sqrt(len(allvals)))

#print  'Here is the confidence internval Ratios:' + str(conf_int)
print 'Here is the confidence of the ratio: ' + str(np.mean(allvals)) + '+/-' + str(2.576*np.std(allvals))







# Here we plot the mean ratio
plt.figure(1)
ax=plt.subplot(313)
ax.errorbar(range(1,len(comp23)+1),comp23,err23, fmt='+', label='Sensor 2 to 3')
ax.errorbar(range(1,len(comp13)+1),comp13,err13, fmt='+', label='Sensor 1 to 3')
ax.errorbar(range(1,len(comp12)+1),comp12,err12, fmt='+', label='Sensor 1 to 2')
plt.title('BH2')
plt.ylim((0.999, 1.001))
plt.xlim((0,len(comp23)+1))
plt.xticks(range(1,len(comp23)+1))
plt.yticks([.999, 1.000, 1.001], [0.999, 1.000, 1.001])
plt.xlabel('Trial Number')
plt.legend(loc=4, frameon=True)
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.3,
                 box.width, box.height * 0.8])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5),
          fancybox=False, shadow=False, ncol=3)



plt.savefig("MeanRatio" + Flag + ".jpg",dpi=400)
plt.savefig("MeanRatio" + Flag + ".pdf",dpi=400)
plt.clf()

allvals = allvals - np.mean(allvals)

fig = plt.figure(1)
ax = fig.add_subplot(111)
stats.probplot(allvals,plot=pylab)
pylab.xlabel('Quantiles')
pylab.ylabel('Mean Removed Ordered Values')
pylab.title('Probability Plot Mean Ratio 4 s to 8 s Period')
ax.get_lines()[0].set_marker('o')
ax.get_lines()[0].set_markerfacecolor('k')
ax.get_lines()[0].set_markeredgecolor('k')
ax.get_lines()[0].set_markersize(10.0)
ax.get_lines()[1].set_color('k')
ax.get_lines()[1].set_linewidth(2.)
ax.get_children()[2].set_fontsize(18.0)
ax = plt.gca()
ax.yaxis.labelpad = -2
plt.tight_layout() 
pylab.savefig('RatioQQ' + Flag + '.pdf',dpi=400, format='pdf')
pylab.savefig('RatioQQ' + Flag + '.jpg',dpi=400, format='jpg')
pylab.clf()



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
            
allvals = sn1+ sn2+sn3
print 'Here is the mean of the Self-Noise BH0: ' + str(np.mean(sn1+sn2+sn3))
print 'Here is the std of the Self-Noise BH0: ' + str(np.std(sn1+sn2+sn3))
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
allvals += sn1+ sn2+sn3
print 'Here is the mean of the Self-Noise BH1: ' + str(np.mean(sn1+sn2+sn3))
print 'Here is the std of the Self-Noise BH1: ' + str(np.std(sn1+sn2+sn3))

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
if Flag == 'LP':
    plt.ylabel('Mean Self-Noise 30 to 100 s Period (dB rel. 1 $(m/s^2)^2/Hz$)')
else:        
    plt.ylabel('Mean Self-Noise 0.1 to 30 s Period (dB rel. 1 $(m/s^2)^2/Hz$)')
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

allvals += sn1+ sn2+sn3
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



plt.savefig("SelfNoise" + Flag + ".pdf",dpi=400, format='pdf')
plt.savefig("SelfNoise" + Flag + ".jpg",dpi=400)
plt.clf()


conf_int = stats.norm.interval(0.99, loc=np.mean(allvals), scale=np.std(allvals) / np.sqrt(len(allvals)))

print  'Here is the confidence internval self-noise 0.1 to 30 s' + str(conf_int)




allvals = allvals - np.mean(allvals)

#plt.hist(allvals)
#plt.show()
#stats.probplot(allvals,dist="t", sparams=(5,), plot=pylab)
fig = plt.figure(1)
ax = fig.add_subplot(111)
stats.probplot(allvals,plot=pylab)
pylab.xlabel('Quantiles')
pylab.ylabel('Mean Removed Ordered Values')
if Flag == 'LP':
    pylab.title('Probability Plot Self-Noise 30 s to 100 s Period')
else:
    pylab.title('Probability Plot Self-Noise 0.1 s to 30 s Period')
ax.get_lines()[0].set_marker('o')
ax.get_lines()[0].set_markerfacecolor('k')
ax.get_lines()[0].set_markeredgecolor('k')
ax.get_lines()[0].set_markersize(10.0)
ax.get_lines()[1].set_color('k')
ax.get_lines()[1].set_linewidth(2.)
ax.get_children()[2].set_fontsize(18.0)
pylab.savefig('SelfNoiseQQ' +  Flag + '.pdf',dpi=400, format='pdf')
pylab.savefig('SelfNoiseQQ' + Flag + '.jpg',dpi=400, format='jpg')
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
allvals = psdn1 + psdn2 + psdn3            
print 'Here is the mean of the PSD BH0: ' + str(np.mean(psdn1+psdn2+psdn3))
print 'Here is the std of the PSD BH0: ' + str(np.std(psdn1+psdn2+psdn3))
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
allvals +=psdn1 + psdn2 + psdn3            
print 'Here is the mean of the PSD BH1: ' + str(np.mean(psdn1+psdn2+psdn3))
print 'Here is the std of the PSD BH1: ' + str(np.std(psdn1+psdn2+psdn3))
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
if Flag == "LP":
    plt.ylabel('Mean PSD 30 to 100 s Period (dB rel. 1 $(m/s^2)^2/Hz$)')
else:
    plt.ylabel('Mean PSD 0.1 to 30 s Period (dB rel. 1 $(m/s^2)^2/Hz$)')
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
allvals += psdn1 + psdn2 + psdn3            
print 'Here is the mean of the PSD BH2: ' + str(np.mean(psdn1+psdn2+psdn3))
print 'Here is the std of the PSD BH2: ' + str(np.std(psdn1+psdn2+psdn3))
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




plt.savefig("PSDNoise" + Flag + ".pdf",dpi=400,format='pdf')
plt.savefig("PSDNoise" + Flag + ".jpg",dpi=400)
plt.clf()

allvals = allvals - np.mean(allvals)
x = stats.shapiro(allvals)
print(x)


fig = plt.figure(1)
ax = fig.add_subplot(111)
stats.probplot(allvals,plot=pylab)
pylab.xlabel('Quantiles')
pylab.ylabel('Mean Removed Ordered Values')
if Flag == "LP":
    pylab.title('Probability Plot PSD 30 s to 100 s Period')
else:
    pylab.title('Probability Plot PSD 0.1 s to 30 s Period')
ax.get_lines()[0].set_marker('o')
ax.get_lines()[0].set_markerfacecolor('k')
ax.get_lines()[0].set_markeredgecolor('k')
ax.get_lines()[0].set_markersize(10.0)
ax.get_lines()[1].set_color('k')
ax.get_lines()[1].set_linewidth(2.)
ax.get_children()[2].set_fontsize(18.0)
pylab.savefig('PSDQQ' + Flag + '.pdf',dpi=400, format='pdf')
pylab.savefig('PSDQQ' + Flag + '.jpg',dpi=400, format='jpg')
#pylab.show()
pylab.clf()


######################################################################################################################################3

#### Here we plot the Orientations
#### Stacked Histogram in 1/4 degree bins
#### side by side orientation on x-axis

# Do BH0
if Flag == '':
    or12 = []
    err12 = []
    or13 = []
    err13 = []
    or23 = []
    err23 = []
    for sn in orients:
            if '1, 2' in sn:
                or12.append(float(sn.split(', ')[2]))
                err12.append(2.5*float(sn.split(', ')[3]))
            elif '1, 3' in sn:
                or13.append(float(sn.split(', ')[2]))
                err13.append(2.5*float(sn.split(', ')[3]))
            elif '2, 3' in sn:
                or23.append(float(sn.split(', ')[2]))
                err23.append(2.5*float(sn.split(', ')[3]))
               

    # Here we plot the mean ratio
    plt.figure(1)
    plt.title('Relative Orientation')
    plt.subplot(121)
    plt.plot(range(1,len(or12)+1),or12,'o', label='Sensor 1 to 2')
    plt.plot(range(1,len(or13)+1),or13,'o', label='Sensor 1 to 3')
    plt.plot(range(1,len(or23)+1),or23,'o', label='Sensor 2 to 3')

    plt.ylim((-5,5))
    plt.ylabel('Orientation (Degrees)')
    plt.xlim((0,len(or13)+1))
    plt.xticks(range(1,len(or13)+1,2))
    plt.xlabel('Trial Number')
    plt.legend(frameon=True)
    plt.subplot(122)
    binval = np.linspace(-3.5,3.5,29)
    ors = or12+or13+or23
    plt.hist(ors, bins=binval)
    plt.xlabel('Relative Orientation (Degrees)')
    plt.ylabel('Frequency')
    ax = plt.gca()
    ax.yaxis.labelpad = -2 
    plt.xlim((-3.5,3.5))
    plt.xticks([-3,-2,-1,0,1,2,3])
    plt.savefig("Orient.jpg",dpi=400)
    plt.savefig("Orient.pdf",dpi=400, format ='pdf')
    plt.clf()

    print 'Here is the mean relative orientation: ' + str(np.mean(ors))
    print 'Here is the std relative orientation: ' + str(np.std(ors))

    allvals = ors


    conf_int = stats.t.interval(0.99, len(allvals)-1, loc=np.mean(allvals), scale=stats.sem(allvals) )

    print  'Here is the confidence interval orientations' + str(conf_int)



    allvals = allvals - np.mean(allvals)

    x = stats.shapiro(allvals)
    print(x)

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    stats.probplot(allvals,plot=pylab)
    #stats.probplot(allvals,dist="t", sparams=(len(ors),), plot=pylab)
    pylab.xlabel('Quantiles')
    pylab.ylabel('Mean Removed Ordered Values')
    pylab.title('Probability Plot Relative Orientation')
    ax.get_lines()[0].set_marker('o')
    ax.get_lines()[0].set_markerfacecolor('k')
    ax.get_lines()[0].set_markeredgecolor('k')
    ax.get_lines()[0].set_markersize(10.0)
    ax.get_lines()[1].set_color('k')
    ax.get_lines()[1].set_linewidth(2.)
    ax.get_children()[2].set_fontsize(18.0)
    pylab.savefig('OrientationQQ.pdf',dpi=400, format='pdf')
    pylab.savefig('OrientationQQ.jpg',dpi=400, format='jpg')
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
plt.ylim((-7,7))
plt.yticks([-7,0,7],[-7, 0,7])
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
plt.ylim((-7,7))
plt.yticks([-7,0,7],[-7, 0,7])
if Flag == 'LP':
    plt.ylabel('Mean Removed PSD 30 to 100 s Period (dB rel. 1 $(m/s^2)^2/Hz$)')
else:
    plt.ylabel('Mean Removed PSD 0.1 to 30 s Period (dB rel. 1 $(m/s^2)^2/Hz$)')
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
plt.ylim((-7,7))
plt.yticks([-7,0,7],[-7, 0,7])
plt.xlabel('Trial Number')
plt.legend(loc=4, frameon=True)





box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.3,
                 box.width, box.height * 0.8])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5),
          fancybox=False, shadow=False, ncol=3)







plt.savefig("PSDNoiseMeanRemoved" + Flag + ".pdf",dpi=400, format= 'pdf')
plt.savefig("PSDNoiseMeanRemoved" + Flag + ".jpg",dpi=400)
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
plt.ylim((-7,7))
plt.yticks([-7,0,7],[-7, 0,7])
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
plt.ylim((-7,7))
plt.yticks([-7,0,7],[-7, 0,7])
if Flag == 'LP':
    plt.ylabel('Mean Removed Self-Noise 30 to 100 s Period (dB rel. 1 $(m/s^2)^2/Hz$)',fontsize=10)
else:
    plt.ylabel('Mean Removed Self-Noise 0.1 to 30 s Period (dB rel. 1 $(m/s^2)^2/Hz$)',fontsize=10)
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
plt.ylim((-7,7))
plt.yticks([-7,0,7],[-7, 0,7])
plt.xlabel('Trial Number')
#plt.legend(loc=4, frameon=True)

box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.3,
                 box.width, box.height * 0.8])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5),
          fancybox=False, shadow=False, ncol=3)




plt.savefig("SelfNoiseMeanRemoved" + Flag + ".pdf",dpi=400, format='pdf')
plt.savefig("SelfNoiseMeanRemoved" + Flag + ".jpg",dpi=400)
plt.clf()

######################################################################################################################################
