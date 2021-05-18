from abc import ABC, abstractmethod
import numpy as np
import scipy as sp

class function(ABC):
    @abstractmethod
    def f(self,t,state):
        pass

class Rossler(function):
    def __init__(self,values = [0.2,0.2,5.7]):
        self.a = values[0]
        self.b = values[1]
        self.c = values[2]

    def f(self,t,state):
      x, y, z = state  # unpack the state vector
      return -y-z, x+self.a*y,self.b+z*(x-self.c) # derivatives

    def f_bvp(self, t, state, Tp):
        # derivatives in a form that can be used by solve_bvp
        # t is a vector and state is a matrix, so we need to loop over it
        # Equations are written in rescalled time tau = t/T_p
        ftmp = np.zeros_like(state)
        for i in range(0,t.shape[0]):
            ftmp[:,i]=Tp*self.f(t,state[:,i])
        return ftmp
    


