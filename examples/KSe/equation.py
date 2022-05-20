from abc import ABC, abstractmethod
import numpy as np
import scipy as sp
import math

class function(ABC):
    @abstractmethod
    def f(self,t,state):
        pass


class KursivAS(function):
    def __init__(self,values = [2.*np.pi, 0.029910, 16]):
        self.L = values[0]
        self.nu = values[1]
        self.N = 16 # Number of Fourier modes k = 1,..., N
        
    def f(self, t, state):
        a = state # for readability
        N = self.N
        kk = np.arange(1,N+1) # wavevectors
        #indk = kk-1 # array index corresponding to k (first index is zero)
        # Commonly used numerical coefficients
        cf = 2*np.pi/self.L
        cf2 = np.power(cf,2)
        cfnu = cf*self.nu
        lin = cf2*(kk**2-cfnu*kk**4)*a
        nlin = np.zeros_like(lin)
        # compute nonlinear term for each k
        for k in kk: 
                sum1 = 0 # temporary variable
                for m in np.arange(1,k): # Go up to k since we want arange to contain k-1
                    sum1 = sum1+a[m-1]*a[k-m-1] # Shift m and k-m by -1 for zero-indexing
                for m in np.arange(k+1,N+1):  # Go up to N+1 since we want arange to contain N
                    sum1 = sum1-a[m-1]*a[m-k-1] # Shift m and m-k by -1 for zero-indexing
                for m in np.arange(1,N-k+1): # Go up to N-k+1 since we want arange to contain N-k
                    sum1 = sum1-a[m-1]*a[k+m-1] # Shift m and k+m by -1 for zero-indexing
                nlin[k-1] = -cf*k*sum1  # Shift k by -1 for zero-indexing (only for the index).      
        
        return lin+nlin

    def f_ps(self, t, state):
        # Antisymmetric KS equation in Fourier space with pseudospectral treatment of nonlinear term
        a = state # for readability
        N = self.N
        kk = np.arange(1,N+1) # wavevectors
        #indk = kk-1 # array index corresponding to k (first index is zero)
        # Commonly used numerical coefficients
        cf = 2*np.pi/self.L
        cf2 = np.power(cf,2)
        cfnu = cf*self.nu
        lin = cf2*(kk**2-cfnu*kk**4)*a
        # compute nonlinear term 
        utab = np.fft.irfft(np.hstack([0,1j*a, 0]), n = 2*(2*N+2) ) # Extend resolution for antialliasing
        norm = utab.shape[0] # normalization factor for inverse FFT
        nlin = norm*kk*cf*np.real(np.fft.rfft(utab*utab))[1:N+1]
        return lin+nlin


    def f_bvp(self, t, state, Tp):
        # KS derivatives in a form that can be used by solve_bvp
        # t is a vector and state is a matrix, so we need to loop over it
        # Equation are written in rescalled time tau = t/T_p
        ftmp = np.zeros_like(state)
        for i in range(0,t.shape[0]):
            ftmp[:,i]=Tp*self.f(t,state[:,i])
        return ftmp

    def f_bvp_ps(self, t, state, Tp):
        # KS derivatives in a form that can be used by solve_bvp
        # t is a vector and state is a matrix, so we need to loop over it
        # Equation are written in rescalled time tau = t/T_p
        # Calls pseudospectral version
        ftmp = np.zeros_like(state)
        for i in range(0,t.shape[0]):
            ftmp[:,i]=Tp*self.f_ps(t,state[:,i])
        return ftmp
        
    def J_bvp(self, t, state, Tp):
        # KS Jacobian in a form that can be used by solve_bvp
        # t is a vector and state is a matrix, so we need to loop over it
        # Equation are written in rescalled time tau = t/T_p
        df_dy = np.zeros((state.shape[0],state.shape[0],state.shape[1]))
        df_dp = np.zeros((state.shape[0], state.shape[1])) # Jacobian with respect to Tp
        for i in range(0,t.shape[0]):
            df_dy[:,:,i]=Tp*self.A(t,state[:,i])
            df_dp[:,i]=self.f(t,state[:,i])
        return [df_dy, df_dp]
 
    def f_Jac_ps(self, t, stateJac):
        # Return vector containing f and derivativ of the finite time Jacobian J^t.
        # Used to integrate dJdt=A.J alongside the equations of the flow
        state = stateJac[0:self.N] # State-space variable
        Jac = np.reshape(stateJac[self.N:], (self.N,self.N)) # matrix J
        return np.hstack([self.f_ps(t,state), np.dot(self.A(t,state), Jac).reshape(-1)]) 
    
    def A(self, t, state):
        # Jacobian (matrix of variations) for KSe in antisymmetric domain 
        a = state # for readability
        N = self.N
        kk = np.arange(1,N+1) # wavevectors
        #indk = kk-1 # array index corresponding to k (first index is zero)
        # Commonly used numerical coefficients
        cf = 2*np.pi/self.L
        cf_2 = 2*cf
        cf2 = np.power(cf,2)
        cfnu = cf*self.nu
        lin = np.diag(cf2*(kk**2-cfnu*kk**4))
        nlin = np.zeros_like(lin)
        # compute nonlinear term for each k
        for k in kk:
            for j in kk:
                sum1 = 0
                if k-j >= 1 : # Check bounds
                    sum1 = a[k-j-1] # Shift k-j by -1 for zero-indexing
                if j >= k+1 : # Check bounds
                    sum1 = sum1 - a[j-k-1] # Shift k-j by -1 for zero-indexing                    
                if k+j <= N: # Check bounds
                    sum1 = sum1 - a[k+j-1] # Shift k+j by -1 for zero-indexing
                nlin[k-1,j-1] = -cf_2*k*sum1  # Shift k and j by -1 for zero-indexing (only for the index).      
       
        return lin+nlin
