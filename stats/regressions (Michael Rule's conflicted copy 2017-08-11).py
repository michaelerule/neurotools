#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import print_function

'''
I need to regress on the following model for synchrony
synchrony(x) = cos(wx)*exp(-x/tau)+b

which means that the angular synchrony cos(theta_x1-theta_x2) should
decay as a damped cosine, with some constant offset b. Note that
a nonzero constant offset may not indicate uniform synchrony, for
example, the direction of constant phase in a plane wave will contribute
a DC component.

Uses L2 penaly

X: List of distances
W: List of weights
Y: List of average pairwise distances

Model is cos(wx)*exp(-x/L)+b
Generates predictions Z
error is \sum W*(Z-Y)^2

gradient of the error
dErr/dw sum(W*(cos(w*x)*exp(-x/L)+b-Y)**2)
dErr/dL sum(W*(cos(w*x)*exp(-x/L)+b-Y)**2)
dErr/db sum(W*(cos(w*x)*exp(-x/L)+b-Y)**2)

sum(W* dErr/dw (cos(w*x)*exp(-x/L)+b-Y)**2)
sum(W* dErr/dL (cos(w*x)*exp(-x/L)+b-Y)**2)
sum(W* dErr/db (cos(w*x)*exp(-x/L)+b-Y)**2)

sum(W* 2(cos(w*x)*exp(-x/L)+b-Y) dErr/dw (cos(w*x)*exp(-x/L)+b-Y))
sum(W* 2(cos(w*x)*exp(-x/L)+b-Y) dErr/dL (cos(w*x)*exp(-x/L)+b-Y))
sum(W* 2(cos(w*x)*exp(-x/L)+b-Y) dErr/db (cos(w*x)*exp(-x/L)+b-Y))

sum(W* 2(cos(w*x)*exp(-x/L)+b-Y) * -sin(w*x)*exp(-x/L) )
sum(W* 2(cos(w*x)*exp(-x/L)+b-Y) *  cos(w*x)*(-1/L)*exp(-x/L) )
sum(W* 2(cos(w*x)*exp(-x/L)+b-Y))


objective function is

def objective(w,L,b):
    z = cos(w*x)*exp(-x/L)+b
    error = sum( W*(z-Y)**2 )
    return error

def gradient(w,lambda,b):
    z = cos(w*x)*exp(-x/L)+b
    h = 2*(z-Y)
    dEdw = sum(W*h*-sin(w*x)*exp(-x/L))
    dEdL = sum(W*h* cos(w*x)*(-1/L)*exp(-x/L))
    dEdb = sum(W*H)
    return [dEdw,dEdL,dEdb]

We use the minimize function from scipy.optimize.

scipy.optimize.minimize(fun, x0, args=(), method=None, jac=None, hess=None,
    hessp=None, bounds=None, constraints=(), tol=None, callback=None,
    options=None)

    Minimization of scalar function of one or more variables.

    New in version 0.11.0.
    Parameters:
    fun : callable
        Objective function.
    x0 : ndarray
        Initial guess.
    args : tuple, optional
        Extra arguments passed to the objective function and its derivatives
        (Jacobian, Hessian).
    method : str or callable, optional
        Type of solver. Should be one of
            'Nelder-Mead'
            'Powell'
            'CG'
            'BFGS'
            'Newton-CG'
            'Anneal (deprecated as of scipy version 0.14.0)'
            'L-BFGS-B'
            'TNC'
            'COBYLA'
            'SLSQP'
            'dogleg'
            'trust-ncg'
            custom - a callable object (added in version 0.14.0)
        If not given, chosen to be one of BFGS, L-BFGS-B, SLSQP, d
        epending if the problem has constraints or bounds.
    jac : bool or callable, optional
        Jacobian (gradient) of objective function. Only for CG, BFGS,
        Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg. If jac is a Boolean
        and is True, fun is assumed to return the gradient along with the
        objective function. If False, the gradient will be estimated
        numerically. jac can also be a callable returning the gradient of the
        objective. In this case, it must accept the same arguments as fun.
    hess, hessp : callable, optional
        Hessian (matrix of second-order derivatives) of objective function or
        Hessian of objective function times an arbitrary vector p. Only for
        Newton-CG, dogleg, trust-ncg. Only one of hessp or hess needs to be
        given. If hess is provided, then hessp will be ignored. If neither hess
        nor hessp is provided, then the Hessian product will be approximated
        using finite differences on jac. hessp must compute the Hessian times
        an arbitrary vector.
    bounds : sequence, optional
        Bounds for variables (only for L-BFGS-B, TNC and SLSQP). (min, max)
        pairs for each element in x, defining the bounds on that parameter. Use
        None for one of min or max when there is no bound in that direction.
    constraints : dict or sequence of dict, optional
        Constraints definition (only for COBYLA and SLSQP). Each constraint is
        defined in a dictionary with fields:
            type : str
                Constraint type: 'eq' for equality, 'ineq' for inequality.
            fun : callable
                The function defining the constraint.
            jac : callable, optional
                The Jacobian of fun (only for SLSQP).
            args : sequence, optional
                Extra arguments to be passed to the function and Jacobian.
        Equality constraint means that the constraint function result is to be
        zero whereas inequality means that it is to be non-negative. Note that
        COBYLA only supports inequality constraints.
    tol : float, optional
        Tolerance for termination. For detailed control, use solver-specific
        options.
    options : dict, optional
        A dictionary of solver options. All methods accept the following
        generic options:
            maxiter : int
                Maximum number of iterations to perform.
            disp : bool
                Set to True to print convergence messages.
        For method-specific options, see show_options.
    callback : callable, optional
        Called after each iteration, as callback(xk), where xk is the current
        parameter vector.
    Returns:
    res : OptimizeResult
        The optimization result represented as a OptimizeResult object.
        Important attributes are: x the solution array, success a Boolean flag
        indicating if the optimizer exited successfully and message which
        describes the cause of the termination. See OptimizeResult for a
        description of other attributes.

'''

from scipy.optimize import minimize

def damped_cosine_regression(X,Y,W):
    '''
    X: List of distances
    W: List of weights
    Y: List of average pairwise distances

    todo: constrain b, L to be positive


    # simple test
    X = 0.4*arange(9)
    Y = exp(-X/4+1)*cos(X)
    Z = Y+randn(*shape(X))
    W = ones(shape(X))
    w,L,b = damped_cosine_regression(X,Z,W).x
    plot(X,Y)
    plot(X,Z)
    plot(X,cos(w*X)*exp(-X/L+b))
    '''
    def objective(wLb):
        (w,L,b) = wLb
        z = cos(w*X)*exp(-X/L+b)
        error = sum( W*(z-Y)**2 )
        return error
    def gradient(wLb):
        (w,L,b) = wLb
        # todo: gradient is wrong?
        z = cos(w*X)*exp(-X/L)+b
        h = 2*(z-Y)
        dEdw = sum(W*h*-sin(w*X)*exp(-X/L))
        dEdL = sum(W*h* cos(w*X)*(-1/L)*exp(-X/L))
        dEdb = sum(W*h)
        return arr([dEdw,dEdL,dEdb])
    result = minimize(objective,[1,1,0])#,jac=gradient)
    return result

from scipy.stats import linregress
def weighted_least_squares(X,Y,W):
    '''
    Was using this one to initialize power law fit
    EPS = 1e-10
    use = (X>EPS)&(Y>EPS)
    weighted_least_squares(log(X+EPS)[use],log(Y+EPS)[use],1/(EPS+X[use]))
    '''
    def objective(ab):
        a,b=(a,b)
        return sum( W*(Y-(X*a+b))**2)
    a,b,_,_,_ = linregress(X,Y)
    result = minimize(objective,[a,b])
    return result


def power_law_regression(X,Y,W):
    '''
    Fit a power law, but with error terms computed by r^2 in
    the original space.

    power law form is

    log(y)  = a*log(x)+b
    or
    y = b*x^a

    initial best guess using linear regression

    minimize failing, just stick to weighted log-log linear regress

    result = power_law_regression(X,Y,1/X**16)
    a,b = result.x

    EPS = 1e-10
    use = (X>EPS)&(Y>EPS)
    a,b = weighted_least_squares(log(X)[use],log(Y)[use],W[use]).x
    plot(sorted(X),b*arr(sorted(X))**a)

    from numpy.polynomial.polynomial import polyfit

    X,Y = ravel(f),ravel(y[:,i])
    a,b = power_law_regression(X,Y,1/X**2)
    plot(sorted(X),b*arr(sorted(X))**a)

    '''
    EPS = 1e-10
    use = (X>EPS)&(Y>EPS)
    a,b = polyfit(log(X)[use],log(Y)[use],1,w=W[use])
    '''
    def objective(ab):
        a,b = (a,b)
        z = exp(b+a*log(X))
        obj = sum((W*(Y-z)**2)[use])
        print(a,b,obj)
        return obj
    result = minimize(objective,[a,b])
    '''
    return a,exp(b)



def gaussian_function_regression(X,Y):
    '''
    X: List of distances
    Y: List of amplitudes
    '''
    def objective(theta):
        (mu,sigma,scale,dc) = theta
        z = npdf(mu,sigma,X)*scale+dc
        error = sum( (z-Y)**2 )
        return error
    result = minimize(objective,[0,1,1])
    return result



from neurotools.functions import npdf

def half_gaussian_function_regression(X,Y):
    '''
    X: List of distances
    Y: List of amplitudes
    '''
    def objective(theta):
        (sigma,scale,dc) = theta
        z = npdf(0,sigma,X)*scale+dc
        error = sum( (z-Y)**2 )
        return error
    result = minimize(objective,[1,1,0])
    sigma,scale,dc = result.x
    return sigma,scale,dc


def exponential_decay_regression(X,Y):
    '''
    X: List of distances
    Y: List of amplitudes
    '''
    def objective(theta):
        (lamb,scale,dc) = theta
        z = exp(-lamb*X)*scale+dc
        error = sum( (z-Y)**2 )
        return error
    result = minimize(objective,[1,1,1])
    lamb,scale,dc = result.x
    return lamb,scale,dc
