#!/usr/bin/env python
import argparse
import sys
import math
import pickle
import numpy as np


from scipy.optimize import root



from obspy.core import UTCDateTime, read, Stream
from obspy.signal import PPSD, bandpass
from obspy.signal.invsim import paz_to_freq_resp
from obspy.signal.spectral_estimation import get_nlnm, get_nhnm
from matplotlib.mlab import csd
from math import pi
from matplotlib.pyplot import (figure,axes,plot,xlabel,ylabel,title,subplot,legend,savefig,show,xscale, xlim, clf, rcParams)
import matplotlib as mpl
mpl.rc('font',family='serif')
mpl.rc('font',serif='Times') 
mpl.rc('text', usetex=True)
mpl.rc('font',size=12)
chans = ['BH0', 'BH1', 'BH2']
stas = ['XX_TST5','XX_TST5','XX_TST6']
locs = ['00','10','00']


def computeresp(resp,delta,lenfft):
    respval = paz_to_freq_resp(resp['poles'],resp['zeros'],resp['sensitivity']*resp['gain'],t_samp = delta, 
        nfft=lenfft,freq = False)
    respval = np.absolute(respval*np.conjugate(respval))
    respval = respval[1:]
    return respval

def cp(tr1,tr2,lenfft,lenol,delta):
    sr = 1/delta
    cpval,fre = csd(tr1.data,tr2.data,NFFT=lenfft,Fs=sr,noverlap=lenol,scale_by_freq=True)
    fre = fre[1:]
    cpval = cpval[1:]
    return cpval, fre


paz = {'zeros': [0.j, 0.j, -392. + 0.j, -1960. + 0.j, -1490. + 1740.j, -1490. -1740.j],
    'poles': [-0.3691 + 0.03702j, -0.3691 - 0.03702j, -343. + 0.j, -370. + 467.j, -370. -467.j,
        -836. + 1522.j, -836. -1522.j, -4900. + 4700.j, -4900. - 4700.j, -6900. + 0.j, -15000. + 0.j],
        'gain': 4.344928*10**17, 'sensitivity': 754.3*2.**26/40.}
        
        
        
#stime = UTCDateTime('2016-196T01:00:00')
#stime = UTCDateTime('2016-197T05:30:00')
#stime = UTCDateTime('2016-198T02:00:00')
#stime = UTCDateTime('2016-201T22:00:00')
#stime = UTCDateTime('2016-203T04:00:00')
#stime = UTCDateTime('2016-206T01:00:00')
#stime = UTCDateTime('2016-208T03:40:00')
#stime = UTCDateTime('2016-213T11:50:00')
#stime = UTCDateTime('2016-215T02:00:00')
#stime = UTCDateTime('2016-216T05:00:00')
#stime = UTCDateTime('2016-217T07:00:00')
#stime = UTCDateTime('2016-218T07:00:00')
#stime = UTCDateTime('2016-220T04:00:00')
#stime = UTCDateTime('2016-222T06:00:00')
stime = UTCDateTime('2016-223T05:00:00')
etime = stime + 6.*60.*60.        
        

length=4*4096
overlap = 2*1024

f = open(str(stime.year) + '_' + str(stime.julday).zfill(3) + '_' + str(stime.hour).zfill(2) + '_' + str(stime.minute).zfill(2) + 'RESULTS','w')

for chan in chans:

  
    st = Stream()
    st += read('/tr1/telemetry_days/' + stas[0] + '/2016/2016_' + str(stime.julday).zfill(3) + '/' + locs[0] + '_' + chan + '.512.seed')
    st += read('/tr1/telemetry_days/' + stas[1] + '/2016/2016_' + str(stime.julday).zfill(3) + '/' + locs[1] + '_' + chan + '.512.seed')

    st += read('/tr1/telemetry_days/' + stas[2] + '/2016/2016_' + str(stime.julday).zfill(3) + '/' + locs[2] + '_' + chan + '.512.seed')

    st.trim(starttime=stime, endtime=etime)





    delta = st[0].stats.delta
    instresp = computeresp(paz, delta, length)

    (p11, fre1) = cp(st[0],st[0],length,overlap,delta)
    (p22, fre1) = cp(st[1],st[1],length,overlap,delta)
    (p33, fre1) = cp(st[2],st[2],length,overlap,delta)

    (p21, fre1) = cp(st[1],st[0],length,overlap,delta)
    (p13, fre1) = cp(st[0],st[2],length,overlap,delta)
    (p23, fre1) = cp(st[1],st[2],length,overlap,delta)

    n11 = ((2*pi*fre1)**2)*(p11 - p21*p13/p23)/instresp
    n22 = ((2*pi*fre1)**2)*(p22 - np.conjugate(p23)*p21/np.conjugate(p13))/instresp
    n33 = ((2*pi*fre1)**2)*(p33 - p23*np.conjugate(p13)/p21)/instresp

    psd1 = 10*np.log10(((2*pi*fre1)**2)*p11/instresp)
    psd2 = 10*np.log10(((2*pi*fre1)**2)*p22/instresp)
    psd3 = 10*np.log10(((2*pi*fre1)**2)*p33/instresp)

    per = 1/fre1






    NLNMper,NLNMpower = get_nlnm()
    NHNMper,NHNMpower = get_nhnm()


    titlelegend = st[0].stats.channel + ' Self-Noise Start Time: ' + str(st[0].stats.starttime.year) + ' ' + str(st[0].stats.starttime.julday) + ' ' + \
    str(st[0].stats.starttime.hour) + ':' + str(st[0].stats.starttime.minute) + ':' + str(st[0].stats.starttime.second) + \
    ' Duration: ' + str(int(st[0].stats.npts*delta/(60*60))) + ' Hours'
    noiseplot = figure(1)
    subplot(1,1,1)
    title(titlelegend,fontsize=12)
    plot(1/fre1,psd1,'r',label='PSD ' + st[0].stats.station + ' ' + \
    st[0].stats.location, linewidth=1.5)
    plot(1/fre1,psd2,'b',label='PSD ' + st[1].stats.station + ' ' + \
    st[1].stats.location, linewidth=1.5)
    plot(1/fre1,psd3,'g',label='PSD ' + st[2].stats.station + ' ' + \
    st[2].stats.location, linewidth=1.5)
    plot(1/fre1,10*np.log10(n11),'r:',label='Noise ' + st[0].stats.station + ' ' + st[0].stats.location, linewidth=1.5)
    plot(1/fre1,10*np.log10(n22),'b:',label='Noise ' + st[1].stats.station + ' ' + st[1].stats.location, linewidth=1.5)
    plot(1/fre1,10*np.log10(n33),'g:',label='Noise ' + st[2].stats.station + ' ' + st[2].stats.location, linewidth=1.5)
    plot(NLNMper,NLNMpower,'k', label='NHNM/NLNM', linewidth=2.0)
    plot(NHNMper,NHNMpower,'k', linewidth=2.0)
    legend(frameon=True)
    xlabel('Period (s)')
    ylabel('Power (dB rel. 1 $(m/s^2)^2/Hz)$')
    xscale('log')
    xlim((1/20., 200. ))
    savefig('NOISE' + str(st[0].stats.starttime.year) + str(st[0].stats.starttime.julday) + \
        str(st[0].stats.starttime.hour) + str(st[0].stats.starttime.minute) + \
    st[0].stats.station + st[0].stats.location + st[0].stats.channel + \
    st[1].stats.station + st[1].stats.location + st[1].stats.channel + \
    st[2].stats.station + st[2].stats.location + st[2].stats.channel + \
    '.jpg', format = 'jpeg', dpi=400)
    savefig('NOISE' + str(st[0].stats.starttime.year) + str(st[0].stats.starttime.julday) + \
        str(st[0].stats.starttime.hour) + str(st[0].stats.starttime.minute) + \
    st[0].stats.station + st[0].stats.location + st[0].stats.channel + \
    st[1].stats.station + st[1].stats.location + st[1].stats.channel + \
    st[2].stats.station + st[2].stats.location + st[2].stats.channel + \
    '.pdf', format = 'pdf', dpi=400)
    clf()

    st.detrend()

    titlelegend = 'Time Series: ' + str(st[0].stats.starttime.year) + ' ' + str(st[0].stats.starttime.julday) + ' ' + \
    str(st[0].stats.starttime.hour) + ':' + str(st[0].stats.starttime.minute) + ':' + str(st[0].stats.starttime.second) + \
    ' ' + str(st[0].stats.npts*delta) + ' seconds'
    tval=np.arange(0,st[0].stats.npts / st[0].stats.sampling_rate, st[0].stats.delta)
    tseriesplot = figure(2)
    title(titlelegend,fontsize=12)

    subplot(311)
    title(titlelegend,fontsize=12)
    plot(tval,st[0].data,'r',label='TSeries ' + st[0].stats.station + ' ' + \
    st[0].stats.location + ' ' + st[0].stats.channel )
    legend(prop={'size':12})
    xlim((0, np.amax(tval) ))
    subplot(312)
    plot(tval,st[1].data,'b',label='TSeries ' + st[1].stats.station + ' ' + \
    st[1].stats.location + ' ' + st[1].stats.channel )
    legend(prop={'size':12})
    xlim((0, np.amax(tval) ))
    subplot(313)
    plot(tval,st[2].data,'g',label='TSeries ' + st[2].stats.station + ' ' + \
    st[2].stats.location + ' ' + st[2].stats.channel  )
    xlabel('Time (s)')
    ylabel('Counts')
    legend(prop={'size':12})
    xlim((0, np.amax(tval) ))
    savefig('TSERIES' + str(st[0].stats.starttime.year) + str(st[0].stats.starttime.julday) + \
        str(st[0].stats.starttime.hour) + str(st[0].stats.starttime.minute) + \
    st[0].stats.station + st[0].stats.location + st[0].stats.channel + \
    st[1].stats.station + st[1].stats.location + st[1].stats.channel + \
    st[2].stats.station + st[2].stats.location + st[2].stats.channel + \
    '.jpg', format = 'jpeg', dpi=400)
    clf()

    ################################################
    # Time to calculate the relative gain
    mb1 = psd1[(4.0 <= per) & (per <= 8.0)]
    mb2 = psd2[(4.0 <= per) & (per <= 8.0)]
    mb3 = psd3[(4.0 <= per) & (per <= 8.0)]

    diff1 = np.average(mb1/mb2)
    diff2 = np.average(mb2/mb3)
    diff3 = np.average(mb1/mb3)
    std1 = np.std(mb1/mb2)
    std2 = np.std(mb2 /mb3)
    std3 = np.std(mb1 /mb3)
    f.write('Start Time:' + str(stime) + ' End Time:' + str(etime) + '\n')
    f.write('Sensor 1:' + str(st[0]) + '\n')
    f.write('Sensor 1:' + str(st[1]) + '\n')
    f.write('Sensor 1:' + str(st[2]) + '\n')
    f.write('Mean Ratio from 1 and 2: ' + chan + ' ' + str(diff1.real) + '+/-' + str(std1) + '\n')
    f.write('Mean Ratio from 2 and 3: ' + chan + ' ' + str(diff2.real) + '+/-' + str(std2) + '\n')
    f.write('Mean Ratio from 1 and 3: ' + chan + ' ' + str(diff3.real) + '+/-' + str(std3) + '\n')



    ####################################################
    # Time to calculate the noise 
    ns1 = 10.*np.log10(n11[(.1 <= per) & (per <= 30.0)])
    ns2 = 10.*np.log10(n22[(.1 <= per) & (per <= 30.0)])
    ns3 = 10.*np.log10(n33[(.1 <= per) & (per <= 30.0)])

    f.write('Self-Noise .1 to 30 s sensor 1: ' + chan + ' ' + str(np.average(ns1.real)) + '+/-' + str(np.std(ns1)) + '\n')
    f.write('Self-Noise .1 to 30 s sensor 2: ' + chan + ' ' + str(np.average(ns2.real)) + '+/-' + str(np.std(ns2)) + '\n')
    f.write('Self-Noise .1 to 30 s sensor 3: ' + chan + ' ' + str(np.average(ns3.real)) + '+/-' + str(np.std(ns3)) + '\n')

    ps1 = psd1[(.1 <= per) & (per <= 30.0)]
    ps2 = psd2[(.1 <= per) & (per <= 30.0)]
    ps3 = psd3[(.1 <= per) & (per <= 30.0)]

    f.write('PSD Noise .1 to 30 s sensor 1: ' + chan + ' ' + str(np.average(ps1.real)) + '+/-' + str(np.std(ps1)) + '\n')
    f.write('PSD Noise .1 to 30 s sensor 2: ' + chan + ' ' + str(np.average(ps2.real)) + '+/-' + str(np.std(ps2)) + '\n')
    f.write('PSD Noise .1 to 30 s sensor 3: ' + chan + ' ' + str(np.average(ps3.real)) + '+/-' + str(np.std(ps3)) + '\n')


#####################################################
# Time to estimate the relative orientation

sr = 'B'


st1 = Stream()
st1 += read('/tr1/telemetry_days/' + stas[0] + '/2016/2016_' + str(stime.julday).zfill(3) + '/' + locs[0]+ '_' + sr + 'H1.512.seed')
st1 += read('/tr1/telemetry_days/' + stas[1] + '/2016/2016_' + str(stime.julday).zfill(3) + '/' + locs[1]+ '_' + sr + 'H1.512.seed')

st1 += read('/tr1/telemetry_days/' + stas[2] + '/2016/2016_' + str(stime.julday).zfill(3) + '/' + locs[2]+ '_' + sr + 'H1.512.seed')

st1.trim(starttime=stime, endtime=etime)
st1.detrend('constant')
st2 = Stream()
st2 += read('/tr1/telemetry_days/' + stas[0] + '/2016/2016_' + str(stime.julday).zfill(3) + '/' + locs[0]+ '_' + sr + 'H2.512.seed')
st2 += read('/tr1/telemetry_days/' + stas[1] + '/2016/2016_' + str(stime.julday).zfill(3) + '/' + locs[1]+ '_' + sr + 'H2.512.seed')

st2 += read('/tr1/telemetry_days/' + stas[2] + '/2016/2016_' + str(stime.julday).zfill(3) + '/' + locs[2]+ '_' + sr + 'H2.512.seed')

st2.trim(starttime=stime, endtime=etime)
st2.detrend('constant')
for tr in st1:
    #tr.data /= paz['sensitivity']

    tr.data = bandpass(tr.data, 1./10., 1.,1./delta, 4, False)
    tr.taper(0.05)
for tr in st2:
    #tr.data /= paz['sensitivity']
    tr.data = bandpass(tr.data, 1./10., 1.,1./delta, 4, False)
    tr.taper(0.05)
    
    

#Sensor 1 and 2
def rot1(theta):
    theta = theta % 360.
    cosd=np.cos(np.deg2rad(theta))
    sind=np.sin(np.deg2rad(theta))
    data1 = cosd*st1[0].data -sind*st2[0].data
    data2 = sind*st1[0].data + cosd*st2[0].data
    resi = abs(sum(data1*st1[1].data)/np.sqrt(sum(data1**2)*sum(st1[1].data**2)) -1.)

    return resi
# Sensor 2 and 3
def rot2(theta):
    theta = theta % 360.
    cosd=np.cos(np.deg2rad(theta))
    sind=np.sin(np.deg2rad(theta))
    data1 = cosd*st1[1].data -sind*st2[1].data
    data2 = sind*st1[1].data + cosd*st2[1].data
    resi = abs(sum(data1*st1[2].data)/np.sqrt(sum(data1**2)*sum(st1[2].data**2)) -1.)
    return resi
    
 #Sensor 1 and 3
def rot3(theta):
    theta = theta % 360.
    cosd=np.cos(np.deg2rad(theta))
    sind=np.sin(np.deg2rad(theta))
    data1 = cosd*st1[0].data -sind*st2[0].data
    data2 = sind*st1[0].data + cosd*st2[0].data
    resi = abs(sum(data1*st1[2].data)/np.sqrt(sum(data1**2)*sum(st1[2].data**2)) -1.)
    return resi   


theta1 = root(rot1, 0., method = 'lm' ).x[0]
theta2 = root(rot2, 0., method = 'lm' ).x[0]
theta3 = root(rot3, 0., method = 'lm' ).x[0]


f.write('Orientation of 1 to 2:' + str(theta1) + ' residual ' + str(rot1(theta1)) + '\n')
f.write('Orientation of 2 to 3:' + str(theta2) + ' residual ' + str(rot2(theta2))+ '\n')
f.write('Orientation of 1 to 3:' + str(theta3) + ' residual ' + str(rot3(theta3))+ '\n')
f.close()


