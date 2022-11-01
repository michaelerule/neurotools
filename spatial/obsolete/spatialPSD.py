#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
2D spatial power spectral density
'''
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

def psd2d(data,spacing = 0.4):
    '''
    2D power spectral density
    accepts HxWx... data
    returns frequencies in 1/mm and power in uV^2

    # test code
    
        data = randn(6,8)
        clf()
        cut = 1
        subplot(221)
        f,x=psd2d(dctCutAA(data,cut))
        scatter(f,x)
        xlim(0,cut*2)
        subplot(222)
        f,x=psd2d(dctCut(data,cut))
        scatter(f,x)
        xlim(0,cut*2)
        subplot(223)
        imshow(getMaskAA(shape(data),3,0.4,cut),interpolation='nearest')
        subplot(224)
        imshow(getMask(shape(data),0.4,cut),interpolation='nearest')
    '''
    H,W = shape(data)[:2]
    therest = shape(data)[2:]
    data = reflect2D(data)
    FT = fft2(data,axes=(0,1))
    power = real(FT*conj(FT)) # units are uV^2/mm^2
    spacing = 0.4 #units are mm
    freqsW = fftfreq(W*2,spacing)[:W] # units are 1/mm
    freqsH = fftfreq(H*2,spacing)[:H] # units are 1/mm
    dfW = diff(freqsW)[0] # units are 1/mm
    dfH = diff(freqsH)[0] # units are 1/mm
    df2 = dfW*dfH # units are 1/mm^2
    absf = abs(add.outer(freqsH,1j*freqsW)) # units are 1/mm
    angf = angle(add.outer(freqsH,1j*freqsW)) # units are 1/mm
    df = cos(angf)*dfH + sin(angf)*dfW # units are 1/mm
    psd = (power[:H,:W,...].T * df.T).T # unit are uV^2 / mm
    return reshape(absf,(H*W,)),reshape(psd,(H*W,)+therest)

'''
# some test code on the CGID data
event      = 6
start      = -1000
stop       = 3000
epoch      = event,start,stop
lowf       = 1
highf      = 55
FS         = 1000.0
SS = 'SPK120918'
SS = 'SPK120924'
RS = 'RUS120518'
fa,fb = 15,30
session = SS
area = 'PMv'
trial = good_trials(session)[0]
lfp = get_all_analytic_lfp(session,area,trial,fa,fb,epoch,onlygood=False)
frames = packArrayDataInterpolate(session,area,lfp)
f,y = psd2d(frames)
clf()
X,Y = ravel(f),ravel(y[:,0])
a,b = power_law_regression(X,Y,1/X)
line = plot(sorted(X),b*arr(sorted(X))**a)[0]
sc = scatter(f,y[:,0])
semilogy()
ylim(1,10000000)
for i in range(stop-start):
    X,Y = ravel(f),ravel(y[:,i])
    a,b = power_law_regression(X,Y,1/X)
    line.set_data(sorted(X),b*arr(sorted(X))**a)
    sc.set_offsets(arr([X,Y]).T)
    title('Scale = %s '%(-a))
    draw()
    print(i,a,b)
AB = arr([power_law_regression(f,y[:,i],1/f) for i in range(shape(y)[-1])])
scale = -AB[:,0]
'''


def spatialPSD_length_scale_spectrum(session,area,trial,lowf,highf):
    print(trial,'computing wavelet transforms')
    freqs,cwt = get_wavelet_transforms(session,area,trial,lowf,highf,4)
    cwt = cwt[:,:,::25]
    nf        = len(freqs)
    print(trial,'reconstructing array spatial locations')
    frames    = arr([packArrayDataInterpolate(session,area,cwt[:,i,:]).T for i in range(nf)])
    print(trial,'extracting spatial power spectra')
    f,y       = psd2d(frames.T)
    reshaped  = reshape(y,(len(f),prod(shape(y)[1:])))
    print(trial,'performing power law regression')
    AB        = arr([power_law_regression(f,q,1/f) for q in reshaped.T])
    scale     = reshape(-AB[:,0], shape(y)[1:])
    return scale

def spatialPSD_parallel_wrapper(args):
    (session,area,trial,lowf,highf) = args #py 2.x->3 compatibility
    print(trial,'starting computations')
    scale = spatialPSD_length_scale_spectrum(session,area,trial,lowf,highf)
    return trial,scale

def all_spatial_PSD_parallel(session,area,lowf,highf):
    jobs = [(session,area,trial,lowf,highf) for trial in get_good_trials(session,area)]
    return squeeze(arr(parmap(spatialPSD_parallel_wrapper,jobs,fakeit=True)))

if __name__=='=__main__':
    execfile('./regressions.py')
    execfile('./cgid/cgidsetup.py')

    event      = 6
    start      = -1000
    stop       = 6000
    epoch      = event,start,stop
    lowf       = 1
    highf      = 55
    FS         = 1000.0
    SS = 'SPK120918'
    SS = 'SPK120924'
    RS = 'RUS120518'
    fa,fb = 15,30
    session = SS
    area = 'PMv'
    extent = (0,7,55,1)
    data = {}
    #reset_pool()
    for area in areas:
        for session in (SS,RS):
            name = 'course_distace_spectrogram_%s_%s_%s_%s_%s.mat'%(session,area,event,start,stop)
            if name in os.listdir('.'):
                test = loadmat(name)['test']
            else:
                print(get_good_trials(session)) # PRELOAD!
                print(get_good_channels(session,area)) # PRELOAD!
                nowarn()
                test = all_spatial_PSD_parallel(session,area,lowf,highf)
                show = median(test,0).T
                imshow(show,interpolation='nearest',extent=extent,aspect=1./100)
                savemat(name,{'test':test})
            data[session,area] = test
    cbsc = 15.0
    vmin=1
    vmax=2.5
    aspect = 1./100
    session = SS
    figure(1)
    clf()
    subplot(411)
    imshow(median(data[session,'M1'],0).T,interpolation='nearest',extent=extent,cmap=luminance,vmin=vmin,vmax=vmax,aspect=aspect)
    title('M1')
    ylabel('Hz')
    subplot(412)
    imshow(median(data[session,'PMd'],0).T,interpolation='nearest',extent=extent,cmap=luminance,vmin=vmin,vmax=vmax,aspect=aspect)
    title('PMd')
    ylabel('Hz')
    subplot(413)
    imshow(median(data[session,'PMv'],0).T,interpolation='nearest',extent=extent,cmap=luminance,vmin=vmin,vmax=vmax,aspect=aspect)
    logmap = int32(exp(linspace(log(1),log(55),55))+0.5)-1
    title('PMv')
    ylabel('Hz')
    xlabel('Time (s)')
    subplot(414)
    imshow([linspace(0,1,100)],aspect=(vmax-vmin)/cbsc,vmin=0,vmax=1,cmap=luminance,extent=(vmin,vmax,0,1))
    yticks([])
    xticks([vmin,vmax])
    subplots_adjust(top=0.85,bottom=0.05)
    suptitle('%s trial average spatial spectral scale (dB/decade?)'%(SS),fontsize=16)
    xlabel('spatial spectral scale (dB/decade?)')
    savefig('spatial_spectral_scale_%s.pdf'%SS)


    vmin=0
    vmax=3
    session = RS
    figure(2)
    clf()
    subplot(411)
    imshow(median(data[session,'M1'],0).T,interpolation='nearest',extent=extent,cmap=luminance,vmin=vmin,vmax=vmax,aspect=aspect)
    title('M1')
    ylabel('Hz')
    subplot(412)
    imshow(median(data[session,'PMd'],0).T,interpolation='nearest',extent=extent,cmap=luminance,vmin=vmin,vmax=vmax,aspect=aspect)
    title('PMd')
    ylabel('Hz')
    subplot(413)
    imshow(median(data[session,'PMv'],0).T,interpolation='nearest',extent=extent,cmap=luminance,vmin=vmin,vmax=vmax,aspect=aspect)
    logmap = int32(exp(linspace(log(1),log(55),55))+0.5)-1
    title('PMv')
    xlabel('Time (s)')
    ylabel('Hz')
    subplot(414)
    imshow([linspace(0,1,100)],aspect=(vmax-vmin)/cbsc,vmin=0,vmax=1,cmap=luminance,extent=(vmin,vmax,0,1))
    yticks([])
    xticks([vmin,vmax])
    tight_layout()
    subplots_adjust(top=0.85,bottom=0.05)
    suptitle('%s trial average spatial spectral scale (dB/decade?)'%(RS),fontsize=16)
    xlabel('spatial spectral scale (dB/decade?)')
    savefig('spatial_spectral_scale_%s.pdf'%RS)
