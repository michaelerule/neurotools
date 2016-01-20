def ndgeentropy(sigma):
    n,k = shape(sigma)
    assert n==k
    return 0.5*k*(1+lin(2*pi)) + 0.5*log(det(sigma))
    
