r'''
The 5-parameter Fisher-Bingham distribution or Kent distribution, named 
after Ronald Fisher, Christopher Bingham, and John T. Kent, is a probability
distribution on the two-dimensional unit sphere <math>S^{2}\,</math> in
<math>\Bbb{R}^{3}</math> .  It is the analogue on the two-dimensional unit 
sphere of the bivariate normal distribution with an unconstrained covariance
matrix. The distribution belongs to the field of directional statistics. 
The Kent distribution was proposed by John T. Kent in 1982, and is used in
geology, bioinformatics.

The probability density function <math>f(\mathbf{x})\,</math> of the Kent 
istribution is given by: 

f(\mathbf{x})=\frac{1}{\textrm{c}(\kappa,\beta)}\exp\{\kappa\boldsymbol{\gamma}_{1}\cdot\mathbf{x}+\beta[(\boldsymbol{\gamma}_{2}\cdot\mathbf{x})^{2}-(\boldsymbol{\gamma}_{3}\cdot\mathbf{x})^{2}]\} 

where  <math>\mathbf{x}\,</math>  is a three-dimensional unit vector and the normalizing constant  <math>\textrm{c}(\kappa,\beta)\,</math>  is:

<math>
c(\kappa,\beta)=2\pi\sum_{j=0}^\infty\frac{\Gamma(j+\frac{1}{2})}{\Gamma(j+1)}\beta^{2j}\left(\frac{1}{2}\kappa\right)^{-2j-\frac{1}{2}}{I}_{2j+\frac{1}{2}}(\kappa)
</math>

Where <math>{I}_v(\kappa)</math> is the modified Bessel function.  Note that <math>c(0,0) = 4\pi</math> and <math>c(\kappa,0)=4\pi\kappa^{-1}\sinh(\kappa)</math>, the normalizing constant of the Von Mises-Fisher distribution.

The parameter <math>\kappa\,</math>  (with <math>\kappa>0\,</math> ) determines the concentration or spread of the distribution, while  <math>\beta\,</math>  (with  <math>0\leq2\beta<\kappa</math> ) determines the ellipticity of the contours of equal probability. The higher the  <math>\kappa\,</math>  and  <math>\beta\,</math>  parameters, the more concentrated and elliptical the distribution will be, respectively. Vector  <math>\gamma_{1}\,</math>  is the mean direction, and vectors  <math>\gamma_{2},\gamma_{3}\,</math>  are the major and minor axes. The latter two vectors determine the orientation of the equal probability contours on the sphere, while the first vector determines the common center of the contours. The 3x3 matrix <math>(\gamma_{1},\gamma_{2},\gamma_{3})\,</math> must be orthogonal.
'''

from numpy import *
from numpy.linalg import *
from tools import *

def riemann2complex(x,y,z):
    x,y,z = normed([x,y,z])
    zeta = (x+1j*y)/(1-z)
    return zeta

def complex2riemann(z):
    X,Y = real(z),imag(z)
    scale = 1.0/(1+X**2+Y**2)
    x = 2*X*scale
    y = 2*Y*scale
    z = (X**2+Y**2-1)*scale
    return x,y,z

def kentPDF(x,k,b,(g1,g2,g3)):
    warn('this is not normalized')
    logpdf = k*dot(g1,x)+b*(dot(g2,x)**2-dot(g3,x)**2)
    return exp(logpdf)

def kentPDFRiemann(z,(k,b,G)):
    x = complex2riemann(z)
    return kentPDF(x,k,b,G)

def rotatex(p,theta):
    ct,st = cos(theta),sin(theta)
    rotation = array([[1, 0,  0],
                    [0,ct,-st],
                    [0,st, ct]])
    return dot(rotation,p)

def rotatey(p,theta):
    ct,st = cos(theta),sin(theta)
    rotation = array([[ct, 0,-st],
                    [0,  1, 0 ],
                    [st, 0, ct]])
    return dot(rotation,p)

def rotatez(p,theta):
    ct,st = cos(theta),sin(theta)
    rotation = array([[ct,-st, 0],
                    [st, ct, 0],
                    [ 0,  0, 1]])
    return dot(rotation,p)


"""
'''
We are going to do an experiment and try to use the
 kent distribution on the Reimann spehere.
First, an experiment to see if we can get the type of 
distributions of interest.
'''
g1 = [1,0,-10]
g2 = [0,1,0]
g3 = [1,1,0]
normed = lambda x:array(x)/sqrt(sum(array(x)**2))
G = array(map(normed,[g1,g2,g3]))
def orthogonalize(x):
    return dot(x,inv(cholesky(dot(x.T,x))))
G = orthogonalize(G)
G = array(map(normed,[g1,g2,g3]))




k = 10
b = -k/2
G = array([[0,0,-1],[1,0,0],[0,1,0]])

while 1:
    G = rotatey(G.T,pi/5).T
    G = rotatez(G.T,pi/7).T
    plane = array([[kentPDFRiemann(x+1j*y,(k,b,G)) for x in linspace(-2,2,50)] for y in linspace(-2,2,50)])
    imshow(plane)
    draw()




'''
Test riemann projection code
'''
close('all')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
N = 1000
data = (randn(N)+1j*randn(N))/5
points = [complex2riemann(z) for z in data]
x,y,z = zip(*points)
ax.scatter(x,y,z)

# check that riemann projection works
data = (randn(N)+1j*randn(N))/5
check = [riemann2complex(*complex2riemann(z)) for z in data]
print sum(abs(data-check)**2)
"""





