#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Helper functions for minimization and optimization
'''
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

import sys
import numpy as np
import scipy
import warnings
import traceback

from scipy.linalg import lstsq,pinv
from numpy.linalg.linalg import cholesky as chol
from numpy.linalg.linalg import LinAlgError

import neurotools.util.functions
from neurotools.util.functions import sexp,slog
from neurotools.util.time      import current_milli_time
from neurotools.util.string    import v2str_long
    
class FailureError(RuntimeError):
    '''
    Re-named `RuntimeError` to distinguish error 
    conditions arising from failure of optimization
    routines. 
    '''
    pass

def minimize_retry(objective,initial,jac=None,hess=None,
                   verbose=False,
                   printerrors=True,
                   failthrough=True,
                   tol=1e-5,
                   simplex_only=False,
                   show_progress=False,
                   dontuse = {},
                   maxfeval = None,
                   maxgeval = None,
                   **kwargs):
    '''
    Call `scipy.optimize.minimize`, retrying a few times in case
    one solver doesn't work.
    
    This addresses unresolved bugs that can cause exceptions in some of
    the gradient-based solvers in Scipy. If we happen upon these bugs, 
    we can continue optimization using slower but more robused methods. 
    
    Ultimately, this routine falls-back to the gradient-free Nelder-Mead
    simplex algorithm, although it will try to use faster routines if
    the hessian and gradient are providede. 

    Parameters
    ----------
    objective: objective, passed to scipy.minimize
    initial: initial parameter guess
    jac: (optional) jacobian, passed to scipy.minimize
    hess: (optional) Hessian, passed to scipy.minimize
    verbose: print extra information
    failthrough: return best params found so far, even if minimization fails
    tol: convergence tolerance, passed to scipy.minimize
    simple_only: Force it to use only the Nelder-Mead simplex optimizer
    show_progress: Print status updates during minimization
    dontuse: Set of methods *not* to try
    maxfeval: Maximum number of objective function evaluations to allow
    maxgeval: Maximum number of gradient function evaluations to allow
    '''
    # Store and track result so we can keep best value, even if it crashes
    result = None
    x0     = np.array(initial).ravel()
    g0     = np.zeros(x0.shape)*np.nan
    nfeval = 0
    ngeval = 0
    vg = objective(x0,*kwargs['args']) if 'args' in kwargs else objective(x0)
    v  = vg[0] if jac is True else vg
    best = v
    # Show progress of the optimization?
    if show_progress:
        sys.stdout.write('\n')
    last_shown = current_milli_time()
    nonlocals = {}
    nonlocals['best']=best
    nonlocals['recent']=best
    nonlocals['x0']=x0
    nonlocals['nfeval']=nfeval
    nonlocals['ngeval']=ngeval
    nonlocals['last_shown']=last_shown

    def progress_update(params=None):
        #nonlocal best, x0, nfeval, ngeval, last_shown
        if not (current_milli_time() - nonlocals['last_shown'] > 300): return
        if not params is None:
            # update from the wrapped jacobian function; let's check convergence
            vg = objective(params,*kwargs['args']) if 'args' in kwargs else objective(params)
            v  = vg[0] if jac is True else vg
            nonlocals['recent'] = v
            if np.isfinite(v) and v<nonlocals['best']:
                nonlocals['best'] = v
                nonlocals['x0']   = params
        if show_progress: 
            ss = '%0.9e'%nonlocals['best']#
            ss += ' '*(15-len(ss))
            s2 = '%0.9e'%nonlocals['recent']
            s2 += ' '*(15-len(ss))
            out = '\r#feval %6d \t#geval %6d \tBest %s \tPrev %s'%(nonlocals['nfeval'],nonlocals['ngeval'],ss,s2)
            print(out,end='',flush=True)
        nonlocals['last_shown'] = current_milli_time()

    def clear_progress():
        if show_progress: 
            progress_update()
            print('\n',end='',flush=True)
    
    # Wrap the provided gradient and objective functions, so that we can
    # capture the function values as they are being optimized. This way, 
    # if optimization throws an exception, we can still remember the best
    # value it obtained, and resume optimization from there using a slower
    # but more reliable method. These wrapper functions also act as 
    # callbacks and allow us to print the optimization progress on screen.
    #nonlocals['ocache']={}
    #nonlocals['jcache']={}
    if jac is True:
        def wrapped_objective(params):
            #nonlocal best, x0, nfeval, ngeval
            if not maxfeval is None and nonlocals['nfeval']>maxfeval:
                raise FailureError('Maximum # of function evaluations exceeded')
            v,g = objective(params,*kwargs['args']) if 'args' in kwargs else objective(params)
            nonlocals['recent'] = v
            if np.isfinite(v) and v<nonlocals['best']:
                nonlocals['best'] = v
                nonlocals['x0']   = params
            nonlocals['nfeval'] += 1
            nonlocals['ngeval'] += 1
            progress_update()
            return v,g
    else:
        def wrapped_objective(params):
            #nonlocal best, x0, nfeval
            if not maxfeval is None and nonlocals['nfeval']>maxfeval:
                raise FailureError('Maximum # of function evaluations exceeded')
            v = objective(params,*kwargs['args']) if 'args' in kwargs else objective(params)
            nonlocals['recent'] = v
            if np.isfinite(v) and v<nonlocals['best']:
                nonlocals['best'] = v
                nonlocals['x0']   = params
            nonlocals['nfeval'] += 1
            progress_update()
            return v 
    if hasattr(jac, '__call__'):
        # Jacobain is function
        original_jac = jac
        def wrapped_jacobian(params):
            #nonlocal best, x0, nfeval, ngeval
            if not maxgeval is None and nonlocals['ngeval']>maxgeval:
                raise FailureError('Maximum # of gradient evaluations exceeded')
            g = original_jac(params)
            nonlocals['ngeval'] += 1
            progress_update(params)
            return g
        jac = wrapped_jacobian

    # There are still some unresolved bugs in some of the optimizers that
    # can lead to exceptions and crashes! This routine catches these errors
    # and failes gracefully. Note that system interrupts are not caught, 
    # and other unexpected errors are caught but reported, in case they
    # reflect an exception arising from a user-provided gradient or 
    # objective function.
    def try_to_optimize(method,validoptions,jac_=None):
        if method in dontuse:
            if verbose:
                print('Skipping method %s\n'%method,end='',flush=True)
            return False
        if verbose:
            print('Trying method %s\n'%method,end='',flush=True)
        try:
            options = {k:v for (k,v) in kwargs.items() if k in validoptions.split()}
            others  = {k:v for (k,v) in kwargs.items() if not k in validoptions.split()}
            result = scipy.optimize.minimize(wrapped_objective,nonlocals['x0'].copy(),
                jac=jac_,hess=hess,method=method,tol=tol,options=options,**others)
            _ = wrapped_objective(result.x)
            clear_progress()
            if result.success: 
                return True
            if verbose or printerrors:
                print('%s reported "%s"\n'%(method,result.message),end='',flush=True)
        except (KeyboardInterrupt, SystemExit): 
            # Don't catch system interrupts
            raise
        except (TypeError,NameError):
            # Likely an internal bug in scipy; don't report it
            clear_progress()
            return False
        except FailureError:
            # Failed, but from our code
            raise
        except Exception:
            # Unexpected error, might be user error, report it
            traceback.print_exc()
            clear_progress()
            if verbose or printerrors:
                print('Error using minimize with %s:\n'%method,end='',flush=True)
                traceback.print_exc()
                sys.stderr.flush()
            return False
        return False

    # We try a few different optimization, in order
    # -- If Hessian is available, Newton-CG should be fast! try it
    # -- Otherwise, BFGS is a fast gradient-only optimizer
    # -- Fall back to Nelder-Mead simplex algorithm if all else fails
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",message='Method Nelder-Mead does not use')
            warnings.filterwarnings("ignore",message='Method BFGS does not use')
            # If gradient is provided....
            if not jac is None and not jac is False and not simplex_only:
                if try_to_optimize('Newton-CG','disp xtol maxiter eps',jac_=jac):
                    return nonlocals['x0']
                if try_to_optimize('CG','disp maxiter gtol norm eps return_all finite_diff_rel_step',jac_=jac):
                    return nonlocals['x0']
                if try_to_optimize('BFGS','disp gtol maxiter eps norm',jac_=jac):
                    return nonlocals['x0']
                #if try_to_optimize('L-BFGS-B','disp maxcor ftol gtol eps maxfun maxiter iprint callback maxls finite_diff_rel_step',jac_=jac):
                #    return nonlocals['x0']
            # Without gradient...
            if not simplex_only:
                if try_to_optimize('BFGS','disp gtol maxiter eps norm',\
                    jac_=True if jac is True else None):
                    return nonlocals['x0']
            # Simplex is last resort, slower but robust
            if try_to_optimize('Nelder-Mead',
                    'disp maxiter maxfev initial_simplex xatol fatol',
                    jac_=True if jac is True else None):
                return nonlocals['x0']
    except (KeyboardInterrupt, SystemExit):
        print('Best parameters are %s with value %s'%(v2str_long(nonlocals['x0']),nonlocals['best']))
        raise
    except FailureError as e:
        sys.stderr.write(str(e)+'\n')
        sys.stderr.flush()
        if not failthrough: raise
    except Exception:
        traceback.print_exc()
        if not failthrough: raise
    # If we've reached here, it means that all optimizers terminated with
    # an error, or reported a failure to converge. If `failthrough` is 
    # set, we can still return the best value found so far. 
    if failthrough:
        if verbose:
            sys.stderr.write('Minimization may not have converged\n')
            sys.stderr.flush()
        return nonlocals['x0'] # fail through
    sys.stderr.write('All minimization attempts failed\n')
    sys.stderr.flush()
    return None
