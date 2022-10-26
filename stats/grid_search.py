#!/usr/bin/python
# -*- coding: UTF-8 -*-

from pylab import *


def grid_search(pargrid,evaluate):
    '''
    Grid search hyperparameter optimization    
    
    Parameters
    ----------
    pargrid: list of arrays
        A list; Each element is a list of values for a given 
        parameter to search over
    
    evaluate: function
        Arguments:
            Parameters: Tuple
                Parameters taken from the parameter search grid
            State: List of arrays
                Saves initial conditions (optional, default None)
        Returns:
            state: the inferred model fit, in the form of a list 
                of floating-point numpy arrays, to be re-used as 
                initial conditions for subsequent parameters.
            likelihood: float
                Scalar summary of fit quality, higher is better
            info: object
                Anything else you'd like to save
    
    Returns
    -------
    best: 
        best index into parameter grid
    pars: 
        values of best parameters
    results[best]: 
        (state, likelihood, info) at best parameters.
        `info` is determined by the third element in the
        3-tuple return-value of the `evaluate` function,
        passed by the user. `state` is also user-defined.
    allresults: 
        all other results
    '''
    
    # - Get shape of search grid
    # - Prepare an object array to save search results
    # - Start the search in the middle of this grid
    # - Get the initial parameters 
    # - Evalute the performance at these parameters    
    gridshape = [*map(len,pargrid)]
    NPARAMS   = len(gridshape)
    results   = np.full(gridshape,None,dtype='object')
    pari      = [l//2 for l in gridshape]
    pars      = [pr[i] for pr,i in zip(pargrid,pari)]
    result0   = evaluate(pars,state=None)
    state0, likelihood0, info0 = result0

    # Tell me which parameters were the best, so far
    def current_best():
        nonlocal results
        ll = array([-inf if r is None else r[1] for r in results.ravel()])
        return unravel_index(argmax(ll),results.shape), np.max(ll)

    # Bounds test for grid search
    ingrid = lambda ix:all([i>=0 and i<Ni for i,Ni in zip(ix,gridshape)])
    
    # Recursive grid search function
    def search(index,suggested_direction=None):
        nonlocal results
        index = tuple(index)
        # Do nothing if we're outside the grid or already evaluated this index
        if not ingrid(index) or results[index] is not None: return
        initial = [*map(array,state0)]
        
        # Compute result and save
        pars            = [pr[i] for pr,i in zip(pargrid,index)]
        results[index]  = evaluate(pars,state=None)
        state, ll, info = results[index]
        print('\r[%s](%s) loss=%e'%\
            (','.join(['%d'%i for i in index]),
             ','.join(['%0.2e'%p for p in pars]),ll),
              flush=True,end='')
        # Figure out where to go next
        # - Try continuing in current direction first
        # - Recurse along all other directions until better parameters found
        Δs = set()
        for i in range(NPARAMS):
            for d in [-1,1]:
                Δ = zeros(NPARAMS,'int32')
                Δ[i] += d
                Δs.add(tuple(Δ))
        if not suggested_direction is None:
            Δ = suggested_direction
            if current_best()[0]==index:
                search(int32(index)+Δ,Δ)
                Δs -= {tuple(Δ)}
        for Δ in Δs:
            if current_best()[0]!=index: break
            search(int32(index)+Δ,Δ)
        return
            
    search(pari)
    best = current_best()[0]
    pars = [pr[i] for pr,i in zip(pargrid,best)]
    print('(done)')
    return best,pars,results[best],results
