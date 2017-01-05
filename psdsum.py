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
from matplotlib.pyplot import (figure,axes,plot,xlabel,ylabel,title,subplot,legend,savefig,show,xscale, xlim, clf, ylim, fill_between)


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
    'poles': [-0.03691 + 0.03702j, -0.03691 - 0.03702j, -343. + 0.j, -370. + 467.j, -370. -467.j,
        -836. + 1522.j, -836. -1522.j, -4900. + 4700.j, -4900. - 4700.j, -6900. + 0.j, -15000. + 0.j],
        'gain': 4.344928*10**17, 'sensitivity': 754.3*2.**26/40.}
 
paz2 = {'gain': 5.96806*10**7, 'zeros': [0, 0], 'poles': [-0.035647 - 0.036879j,
        -0.035647 + 0.036879j, -251.33, -131.04 - 467.29j, -131.04 + 467.29j],
        'sensitivity': 3.355500*10**10}       
        
        
stimes = []
stimes.append(UTCDateTime('2016-196T01:00:00'))
stimes.append(UTCDateTime('2016-197T05:30:00'))
stimes.append(UTCDateTime('2016-198T02:00:00'))
stimes.append(UTCDateTime('2016-201T22:00:00'))
stimes.append(UTCDateTime('2016-203T04:00:00'))
stimes.append(UTCDateTime('2016-206T01:00:00'))
stimes.append(UTCDateTime('2016-208T03:40:00'))
stimes.append(UTCDateTime('2016-213T11:50:00'))
stimes.append(UTCDateTime('2016-215T02:00:00'))
stimes.append(UTCDateTime('2016-216T05:00:00'))
stimes.append(UTCDateTime('2016-217T07:00:00'))
stimes.append(UTCDateTime('2016-218T07:00:00'))
stimes.append(UTCDateTime('2016-220T04:00:00'))
stimes.append(UTCDateTime('2016-222T06:00:00'))
stimes.append(UTCDateTime('2016-223T05:00:00'))


ps =[]
ns =[]
psts =[]

for stime in stimes:


    etime = stime + 6.*60.*60.        
            

    length=4*4096
    overlap = 2*1024

    f = open(str(stime.year) + '_' + str(stime.julday).zfill(3) + '_' + str(stime.hour).zfill(2) + '_' + str(stime.minute).zfill(2) + 'RESULTSLP','w')

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
        
        
        ps.append(psd1)
        ps.append(psd2)
        ps.append(psd3)
        
        
        ns.append(10.*np.log10(n11))
        ns.append(10.*np.log10(n22))
        ns.append(10.*np.log10(n33))

        per = 1/fre1
        if chan == 'BH0':
            st2 = Stream()
            st2 += read('/tr1/telemetry_days/XX_TST1/2016/2016_' + str(stime.julday).zfill(3) + '/00_' + chan + '.512.seed')
            (pref, fre1) = cp(st2[0], st2[0], length, overlap, delta)
            instref = computeresp(paz2, delta, length)
            pref = 10.*np.log10(((2*pi*fre1)**2)*pref/instref)
            psts.append(pref)

    #sys.exit()
    
    
nsm = np.mean(ns, axis=0)

psm = np.mean(ps, axis=0)
pss = np.std(ps, axis =0)
nss = np.std(ns, axis =0)

pstsm = np.mean(psts, axis=0)
pstss = np.std(psts, axis=0)
NLNMper,NLNMpower = get_nlnm()
NHNMper,NHNMpower = get_nhnm()
titlelegend = 'Self-Noise 15 Trials'
noiseplot = figure(1)
subplot(1,1,1)
title(titlelegend,fontsize=12)
plot(1/fre1, pstsm, color='0.75', label='STS-2 Vertical')
plot(1/fre1,psm,'r',label='Mean PSD')
plot(1/fre1,nsm,'b',label='Mean Self-Noise')
fill_between(1/fre1, nsm - 2.576*nss,nsm +2.576*nss,facecolor='b', label='Self-Noise 99th Percentile', alpha=0.3)
fill_between(1/fre1, psm - 2.576*pss, psm +2.576*pss, facecolor='r', label = 'PSD 99th Percentile', alpha =0.3)

plot(NLNMper,NLNMpower,'k')
plot(NHNMper,NHNMpower,'k')
legend(prop={'size':12})
xlabel('Period (s)')
ylabel('Power rel. 1 (m/s^2)^2/Hz')
xscale('log')
xlim((1/20., 150. ))
ylim((-200., -50.))
savefig('TEST.jpg', format='jpeg',dpi=400)
        #savefig('NOISE' + str(st[0].stats.starttime.year) + str(st[0].stats.starttime.julday) + \
            #str(st[0].stats.starttime.hour) + str(st[0].stats.starttime.minute) + \
        #st[0].stats.station + st[0].stats.location + st[0].stats.channel + \
        #st[1].stats.station + st[1].stats.location + st[1].stats.channel + \
        #st[2].stats.station + st[2].stats.location + st[2].stats.channel + \
        #'.jpg', format = 'jpeg', dpi=400)
        #clf()

        #st.detrend()

        #titlelegend = 'Time Series: ' + str(st[0].stats.starttime.year) + ' ' + str(st[0].stats.starttime.julday) + ' ' + \
        #str(st[0].stats.starttime.hour) + ':' + str(st[0].stats.starttime.minute) + ':' + str(st[0].stats.starttime.second) + \
        #' ' + str(st[0].stats.npts*delta) + ' seconds'
        #tval=np.arange(0,st[0].stats.npts / st[0].stats.sampling_rate, st[0].stats.delta)
        #tseriesplot = figure(2)
        #title(titlelegend,fontsize=12)

        #subplot(311)
        #title(titlelegend,fontsize=12)
        #plot(tval,st[0].data,'r',label='TSeries ' + st[0].stats.station + ' ' + \
        #st[0].stats.location + ' ' + st[0].stats.channel )
        #legend(prop={'size':12})
        #xlim((0, np.amax(tval) ))
        #subplot(312)
        #plot(tval,st[1].data,'b',label='TSeries ' + st[1].stats.station + ' ' + \
        #st[1].stats.location + ' ' + st[1].stats.channel )
        #legend(prop={'size':12})
        #xlim((0, np.amax(tval) ))
        #subplot(313)
        #plot(tval,st[2].data,'g',label='TSeries ' + st[2].stats.station + ' ' + \
        #st[2].stats.location + ' ' + st[2].stats.channel  )
        #xlabel('Time (s)')
        #ylabel('Counts')
        #legend(prop={'size':12})
        #xlim((0, np.amax(tval) ))
        #savefig('TSERIES' + str(st[0].stats.starttime.year) + str(st[0].stats.starttime.julday) + \
            #str(st[0].stats.starttime.hour) + str(st[0].stats.starttime.minute) + \
        #st[0].stats.station + st[0].stats.location + st[0].stats.channel + \
        #st[1].stats.station + st[1].stats.location + st[1].stats.channel + \
        #st[2].stats.station + st[2].stats.location + st[2].stats.channel + \
        #'.jpg', format = 'jpeg', dpi=400)
        #clf()

       
