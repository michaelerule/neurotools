
from numpy import *

def cut_spikes(s,cut):
    '''
    downsampling spike raster by factor cut
    just sums up the bins (can generate counts >1)
    '''
    return array([sum(s[i:i+cut]) for i in arange(0,len(s),cut)])
    
def times_to_raster(spikes,duration=1000):
    result = zeros((1000,),dtype=float32)
    if len(spikes)>0: result[spikes]=1
    return result

def bin_spikes(train,binsize=5):
    bins = int(ceil(len(train)/float(binsize)))
    return histogram(find(train),bins,(0,bins*binsize))[0]

