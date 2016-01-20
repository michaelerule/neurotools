'''
Utility functions for handling Blackrock arrays
This is redundant to neurotools.array
Do not use it
'''
assert 0

def pack_array_data(session,area,data):
    '''
    Accepts a collection of signals from array channels, as well as
    an array map containing indecies (1-indexed for backwards compatibility
    with matlab) into that list of channel data. 
    
    This will interpolate missing channels as an average of nearest 
    neighbors.
    
    If any complete rows or columns are missing at the edge of the array,
    they will be trimmed off before packing the data into the array. 
    
    Args:
        data (ndarray): First dimension should be channels. This must match
            the array map, such that the first row corresponds to the 
            first column and so on. Every channel noted in the arry map 
            must have a corresponding row in the data array.
        M (ndarray): Nrows x Ncols map of electrode positions. By
            convention, channels are indexed starting at 1. Channel numbers
            0 and -1 may denote missing, disconnected, or invalid channels.
            All other negative values are unsupported / undefined behavior.
            
    Returns:
        ndarray: Data packed into an interpolated array. Empty rows and 
            columns are trimmed from the array map, so the packed data may 
            not have the same dimension as the array map.
    '''
    # first, trim off any empty rows or columns from the M
    M = trim_array(M)
    # prepare array into which to pack data
    L,K = shape(M)
    NCH = shape(data)[0]
    packed = zeros((L,K)+shape(data)[1:],dtype=complex64)
    for i,row in enumerate(M):
        for j,ch in enumerate(row):
            # convert channel to channel index
            if ch==-1 or ch==0:
                # this is a missing channel
                # interpolate from nearest neighbors
                ip = ()
                if i>0  :ip+=(M[i-1,j],)
                if j>0  :ip+=(M[i,j-1],)
                if i+1<L:ip+=(M[i+1,j],)
                if j+1<K:ip+=(M[i,j+1],)
                ip = [ch for ch in ip if ch>0]
                packed[i,j,:] = mean([data[ch2chi(session,area,k),:] for k in ip])
            else:
                packed[i,j,:] = data[ch2chi(session,area,ch),:]
    return packed





