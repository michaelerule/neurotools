#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
This module contains utility for performing graph algorithms on the GPU.
'''
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

'''
Representations of Graph datastructures on GPU :

Sparse Matrix :
N nodes, [prologue][data]
prologue : N int2 onjects : [offset,neighbors]
Data for element i beings at offset, consists of neighbors entries of node
indecies.

Adjacency Matrix : row major square matricies 
integer matrix
float matrix
'''

def random_point_sheet(n,d):
    xpoints = gpuuniform(0,d)(n)
    ypoints = gpuuniform(0,d)(n)
    return (xpoints,ypoints)
    
def uniform_point_sheet(n,d,dt,iterations):
    x,y=random_point_sheet(n,d)
    nx=gpuarray.empty_like(x)
    ny=gpuarray.empty_like(y)
    repel = kernel('float *X, float *Y, float *NX, float *NY, int n, float d, float dt',
        ''' 
        float fx = 0.0f;
        float fy = 0.0f;
        float x = X[tid];
        float y = Y[tid];
        for (int i=0; i<n; i++) {
            float px = x-X[i];
            float py = y-Y[i];
            float rr = px*px+py*py;
            if (rr<0.1f) rr = 0.1f;
            rr = 1.0/rr;            
            fx += px*rr;
            fy += py*rr;
        }
        x+=dt*fx;
        y+=dt*fy;
        if (x<0.0) x=0.1f*d; else if (x>d) x=d*.9;
        if (y<0.0) y=0.1f*d; else if (y>d) y=d*.9; 
        NX[tid]=x;
        NY[tid]=y;
        ''',
        '''
        __device__ float randf(int *state,int tid) {
            int x = __mul24(state[tid],0xFD43FD)+0xC39EC3;
            state[tid] = x;
            return (x*.000000000465662+1.0000012)*0.5;
        }
        ''')(n)
    dt=np.float32(dt*0.5)
    d=np.float32(d)
    for i in xrange(iterations):
        repel(x,y,nx,ny,n,d,dt)
        repel(nx,ny,x,y,n,d,dt)
    return x,y
    
def connect_gaussian(n,k,s,p):
    datagraph = gpuarray.empty((int(n*n*k),),np.int32)
    rngstate  = gpuint(int32(np.random.random_integers(16777215,size=n*n)))   
    string = ('',r'if (nx==x&&ny==y) continue;')[p]
    print(string)
    connect = kernel('int *graph, int n, int k, float sigma, int *rng_state',
        '''
        int x = tid %% n;   
        int y = tid / n;
        int *neighbors = &graph[tid*k];
        for (int i=0; i<k; i++) {
            int done=0;
            while (!done) {
                //draw from a 2D gaussian without replacement
                float u1 = sigma*sqrt(-2.0f*__logf(randf(rng_state,tid)));
                float u2 = 6.28318531f*randf(rng_state,tid);
                float n1 = __sinf(u2)*u1;
                float n2 = __cosf(u2)*u1;
                int nx = (int)(n1+x+0.5F);
                int ny = (int)(n2+y+0.5F);
                %s
                while (nx<0) nx+=n;
                nx%%=n;
                while (ny<0) ny+=n;
                ny%%=n;
                int index = nx+ny*n;
                for (int j=0; j<i; j++)
                    if (index==neighbors[j])
                        continue;
                neighbors[i]=index;
                done=1;
            }
        }
        '''%string,
        '''
        __device__ float randf(int *state,int tid) {
            int x = __mul24(state[tid],0xFD43FD)+0xC39EC3;
            state[tid] = x;
            return (x*.000000000465662+1.0000012)*0.5;
        }
        ''')
    connect(n*n)(datagraph,np.int32(n),np.int32(k),np.float32(s),rngstate)
    return cut(cpu(datagraph),k)

"""    
def gaussian_smallworld(n,k,s,p=True):
    datagraph = gpuarray.empty((int(n*n*k),),np.int32)
    rngstate  = gpuint(int32(np.random.random_integers(16777215,size=n*n)))   
    string = ('',r'if (nx==x&&ny==y) continue;')[p]
    print(string)
    connect = kernel('int *graph, int n, int k, float sigma, int *rng_state',
        '''
        int x = tid %% n;   
        int y = tid / n;
        int *neighbors = &graph[tid*k];
        for (int i=0; i<k; i++) {
            int done=0;
            while (!done) {
                //draw from a 2D gaussian without replacement
                float u1 = sigma*sqrt(-2.0f*__logf(randf(rng_state,tid)));
                float u2 = 6.28318531f*randf(rng_state,tid);
                float n1 = __sinf(u2)*u1;
                float n2 = __cosf(u2)*u1;
                int nx = (int)(n1+x+0.5F);
                int ny = (int)(n2+y+0.5F);
                %s
                while (nx<0) nx+=n;
                nx%%=n;
                while (ny<0) ny+=n;
                ny%%=n;
                int index = nx+ny*n;
                for (int j=0; j<i; j++)
                    if (index==neighbors[j])
                        continue;
                neighbors[i]=index;
                done=1;
            }
        }
        '''%string,
        '''
        __device__ float randf(int *state,int tid) {
            int x = __mul24(state[tid],0xFD43FD)+0xC39EC3;
            state[tid] = x;
            return (x*.000000000465662+1.0000012)*0.5;
        }
        ''')
    connect(n*n)(datagraph,np.int32(n),np.int32(k),np.float32(s),rngstate)
    return cut(cpu(datagraph),k)
"""

'''
def suck v:
    R=gpuarray.zeros_like(v[0]);
    for x in v:
        R=r+x*x
    R=R**0.5*(1.0/r)
    return cmap(lambda x:x*R)(v)
'''

#cut = lambda n,m:lambda a:[gpuarray.GPUArray((n,),a.dtype,base=int(a.gpudata),gpudata=base+4*n*i) for i in xrange(m)]

def spherepoints(n,r,k):
    x=gpuuniform(-1,1)(n)
    y=gpuuniform(-1,1)(n)
    z=gpuuniform(-1,1)(n)
    X=x*x
    Z=y*y
    Y=z*z
    R=gpuarray.empty_like(x)
    gpusumeq(R,X)
    gpusumeq(R,Y)
    gpusumeq(R,Z)
    gpupow(-0.5)(R)
    gpumuleq(x,R)
    gpumuleq(y,R)
    gpumuleq(z,R)
    #sketchpad = [gpuarray.zeros(n*n,np.float32) for i in xrange(3)]
    #foregone  = cmap(cut(n,n,np.float32))(sketchpad)
        
    repelkernel = ElementwiseKernel(
        "float *x, float *y, float *z, float *X, float *Y, float *Z, float d",
        '''
        float dx=0.0f;
        float dy=0.0f;
        float dz=0.0f;
        for (int j=0; j<%d; j++) if (i!=j) {
            float xx=x[i]-x[j];
            float yy=y[i]-y[j];
            float zz=z[i]-z[j];
            float rr=xx*xx+yy*yy+zz*zz;
            if (rr>0.0000001f) {
                float force=pow((float)rr,-1.5f);
                dx+=force*xx;
                dy+=force*yy;
                dz+=force*zz;
            }
        }
        dx=x[i]+dx*d;
        dy=y[i]+dy*d;
        dz=z[i]+dz*d;
        float rr=pow((float)(dx*dx+dy*dy+dz*dz),-0.5f);
        X[i]=dx*rr;
        Y[i]=dy*rr;
        Z[i]=dz*rr;
        '''%(n),
        "repelkernel")
        
    d = np.float32(0.1/n)    
    for i in xrange((k+1)/2):
        repelkernel(x,y,z,X,Y,Z,d)
        repelkernel(X,Y,Z,x,y,z,d)
        
    return (x,y,z)
        

    
    
