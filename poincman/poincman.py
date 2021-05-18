import numpy as np
from math import sin
import math
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
import scipy as sp
from scipy import signal
from scipy.optimize import newton
from scipy.interpolate import splprep as spl
from scipy.interpolate import splev
from scipy.integrate import solve_ivp, solve_bvp
import copy
import itertools
import time
from colorama import Fore, Style

class PoincMap:
    # Finds intersections with Poincare surface of section
    # in the form of a hyperplane defined by
    # normVec: vector normal to the hyperplane
    # pointHyp: a sample point on the hyperplane
    # Simulation data are provided through a simulation instance
    # direction=1 can be used to specify the direction of intersection
    def __init__(self, normVec, pointHyp, simulator, direction=1):
        self.simulator = simulator
        self.equation = None # Setting this is required by orientCond
        self.embedding = None # Used to store and reuse embedding
        self.sign_norm_q = 1 # normalization choice; setting to -1 reverses the sign of q's returned from manifold learning
        self.qmin_0 = False # Whether to shift origin of variables to zero interval
        self.qmax_1 = False # Whether to map variables to unit interval (self.qmin_0 is set to true automatically)
        self.demb = 1 # Embedding dimension
        # Normalize vector (optional)
        self.normVec = np.array(normVec/np.linalg.norm(normVec))
        self.pointHyp = np.array(pointHyp) 
        self.direction = direction
        self.poincCrossTimes = None # Times at which Poincare section is crossed
        self.poincPoints = None # Points on Poincare section
        self.interpRM = None
        self.int_min_t = 0 # do not check Poincare condition if time is smaller than self.int_min_t; if set to finite time it can prevent early termination of integration by event locator in solve_ivp
        # The next attributes can be used to store indices of points needed to realize different maps
        self.idx_map_s_s = None
        self.idx_map_s_r = None
        self.idx_map_r_r = None
        self.idx_map_r_s = None
        self.idx_s_valid = None
        self.idx_s_valid = None
        # Global map parameters
        self.sc = None # Critical points
        self.invIrv = None # Invariant interval (endpoints)
        self.baseI = None # Base intervals
        # The next attributes can be used to store boolean values corresponding to the validity of indices for different maps
        self.r_valid = None
        self.s_valid = None
        self.split_case = 1 # Controls the behaviour of split_rs
        self.pol = None # Polynomial function produced by split_rs
        self.vc = None # Variable used to delimit domains in split_rc; needed by s2rs
        # The next attributes can be used to store interpolating function objects
        self.irpRM_r_s = None
        self.irpRM_s_r = None
        self.irpRM_r_r = None
        self.irpRM_s_s = None
        self.reduce_refl = False
        self.orient_preserv = None # List containing orientation-preserving branches (symbols) of return map   
        # Convenience functions
        self.flatten = lambda t: [item for sublist in t for item in sublist] # Flattens a list of lists
        self.str2list = lambda s: list(map(int, list(s))) # Convert string representation of symbol sequence to list representation with integer labels
        self.list2str = lambda s: ''.join(map(str, s)) # Convert list representation of symbol sequence to string representation
        # For periodic orbit extraction
        self.col = None # List with labels of columns for periodic orbit data
        self.nPoinc = None # number of intersections with Poincare section for single iterate of return map for each of the letters of the alphabet. If None, assume single intersection        
        self.seqChaotic = None # symbolic sequence for a chaotic orbit, used for shadowing searches for periodic orbits
        self.s = None # used to store single manifold coordinate
        
    def poincCond(self, x):
        # Function which returns zero if a point is on the Poincare section
        return np.matmul(self.normVec, x-self.pointHyp)

    def poincCondtx(self, t, x):
        # Function which returns zero if a point is on the Poincare section
        if t <= self.int_min_t: # do not check Poincare condition if time is smaller than self.int_min_t
            val = self.direction # use direction to avoid triggering change of sign as soon as t>self.int_min_t
        else: 
            val = np.matmul(self.normVec, x-self.pointHyp)
        return val


    def poincCondFun(self, t):
        # Function which returns zero if a point on a solution given
        # as an interpolating function in a simulator instance is on the Poincare section
        return np.inner(self.normVec, self.simulator.sol(t)-self.pointHyp)

    def orientCond(self, x, t=0):
        # Function which determines the orientation of a crossing of the Poincare section
        # Set t=0 by default, which works only for autonomous systems.
        # Has not been tested for non-autonomous systems
        return np.sign(np.inner(self.normVec, self.equation.f(t, x)))

    def detectCrossings(self):
        # Return indices corresponding to points just before a crossing of the Poincare section
        vecCond = np.apply_along_axis(self.poincCond, 0, self.simulator.data)
        # Check for sign change
        signs = np.array(np.sign(vecCond[0:-1]*vecCond[1:]))
        # Direction of crossing; for the time being this is computed also for non-crossing points
        # Instead of using the vector field velocity, we could approximate the flow direction
        # using the vector formed by the state space points before and after the crossing.
        dir_cross = np.apply_along_axis(
            self.orientCond, 0, self.simulator.data)
        # flatten serves to convert 1 x length(signs) array to a vector
        return np.asarray(((signs < 0) & (dir_cross[0:-1] == self.direction)).nonzero()).flatten()

    def detectCrossings_data(self, data):
        # Return indices corresponding to points just before a crossing of the Poincare section
        # Can be used with data for periodic orbit generated by solve_bvp or any other integrator
        vecCond = np.apply_along_axis(self.poincCond, 0, data)
        # Check for sign change
        signs = np.array(np.sign(vecCond[0:-1]*vecCond[1:]))
        # Direction of crossing; for the time being this is computed also for non-crossing points
        # Instead of using the vector field velocity, we could approximate the flow direction
        # using the vector formed by the state space points before and after the crossing.
        dir_cross = np.apply_along_axis(
            self.orientCond, 0, data)
        # flatten serves to convert 1 x length(signs) array to a vector
        return np.asarray(((signs < 0) & (dir_cross[0:-1] == self.direction)).nonzero()).flatten()

    def poincPoints_bvp(self,data,tmax):
        # Find points of intersection with Poincare section for data generated by solve_bvp 
        # (or any other solver for which the first and last point are on the Poincare section)
        # Uses a call to solve_ivp which should eventually provided with a more flexible interface
        # (replace with simulator object?)
        dat = np.roll(data[:,1:-1],-1, axis=1) # Assumes first and last point are on Poincare section
        crs = self.detectCrossings_data(dat)
        poinc = np.zeros_like(dat[:,crs])
        self.poincCondtx.__func__.terminal = True # terminate integration after intersection
        self.int_min_t = 0
        for idx1, idx2 in zip(crs,range(0,poinc.shape[1])):
            sol = solve_ivp(self.equation.f, [0,tmax], data[:,idx1], method='BDF', events = self.poincCondtx, jac=self.equation.A)
            poinc[:,idx2] = sol.y_events[0]
        return poinc.transpose()

    def pointPoincTime(self, idx):
        # Find time of intersection from interpolating function solution
        # given initial guess for the crossing
        return sp.optimize.newton(self.poincCondFun, self.simulator.timepoints()[idx])

    def poincTimes(self):
        crossIndices = self.detectCrossings()
        self.poincCrossTimes = [self.pointPoincTime(cr) for cr in crossIndices]
        return self.poincCrossTimes
    
    def getValues(self):
        ## Return points on the Poincare section
        if self.poincCrossTimes is None:
            self.poincTimes()
        self.poincPoints = np.asarray([self.simulator.sol(t) for t in self.poincCrossTimes])    
        return self.poincPoints
        
    def interp1dRM(self,s,sorted_map_label='None',sorted_label='None',step=1, kind = 'nearest'):
        # Given any data for 1D coordinate for a return map, construct an interpolating function
        # for the return map.
        # Useful mostly for data generated after manifold learning has been applied.
        # sorted_map_label: object attribute where the sorted data points for the map could be stored
        # sorted_label: object attribute where the sorted data points could be stored
        # step: used for downsampling (not recommended)
        # kind: interpolation kind 
        # Keep appropriate points for s_n, s_{n+1}:
        sn = s[:-1] 
        snp1 = s[1:]
        # Get indices that would sort s. Only consider positions 1 to -1 
        # in order to only deal with overlapping elements of sn and snp1.
        # Offset sorting indices by +1 in order to take into account that 
        # we start at position 1 instead of 0.
        sSortInd = np.argsort(s[1:-1])+1  # indices that sort s
        # Use the same indices in order to sort sn and snp1
        snSort = sn[sSortInd]
        snp1Sort = snp1[sSortInd]        
        if step != 1:
            #Downsample map but keep the same endpoints
            snSort = np.append(snSort[::step],snSort[-1])
            snp1Sort = np.append(snp1Sort[::step],snp1Sort[-1])
        if sorted_label is not 'None':
           setattr(self,sorted_label, snSort)        
        if sorted_map_label is not 'None':
           setattr(self,sorted_map_label, np.array([snSort,snp1Sort]).transpose())
        self.interpRM = sp.interpolate.interp1d(snSort,snp1Sort, kind = kind)
        # Return interpolating function
        return self.interpRM 

    def interp1d_map(self, S, sorted_map_label='None',sorted_label='None', kind = 'nearest'):
        # Given any data representing return map, construct an interpolating function
        # for the return map. Data could be for map from s to s or s to r or vice versa.
        # Useful mostly for data generated after manifold learning has been applied.
        # sorted_map_label: object attribute where the sorted data points could be stored
        # Keep appropriate points for S_n, S_{n+1}:
        sn = S[:,0] #s[:-1] 
        snp1 = S[:,1] # s[1:]
        # Get indices that would sort sn. 
        sSortInd = np.argsort(sn)  # indices that sort sn
        # Use the same indices in order to sort sn and snp1
        snSort = sn[sSortInd] 
        snp1Sort = snp1[sSortInd]
        if sorted_label is not 'None':
           setattr(self,sorted_label, snSort)            
        if sorted_map_label is not 'None':
            setattr(self,sorted_map_label,np.array([snSort,snp1Sort]).transpose())
        interpRM = sp.interpolate.interp1d(snSort,snp1Sort, kind = kind)
        # Return interpolating function
        return interpRM 
    
    def sParamSP(self,s, datSP=None, kind='nearest'):
        # Parametrize points on the Poincare section using s as parameter. State space points 
        # are provided by optional variable datSP or by self.poincPoints 
        # s should be a vector in 1 to 1 correspondance with self.poincPoints or datSP (if provided).
        # kind (string): option controling interpolation
        # Output is an interpolating function of the same dimension as the state-space
        if datSP is None: # If no data are passed, then use Poincare section points computed by previous call to getValues
            data = self.poincPoints.transpose()
        else:
            data = datSP.transpose()
        self.s2sp = [sp.interpolate.interp1d(s, dat, kind = kind) for dat in data]
        return self.s2sp

    def rParamSP(self,r, datSP=None, kind='nearest'):
        # Parametrize points on the Poincare section using r as parameter. State space points 
        # are provided by optional variable datSP or by self.poincPoints 
        # s should be a vector in 1 to 1 correspondance with self.poincPoints or datSP (if provided).
        # kind (string): option controling interpolation
        # Output is an interpolating function of the same dimension as the state-space
        if datSP is None: # If no data are passed, then use Poincare section points computed by previous call to getValues
            data = self.poincPoints.transpose()
        else:
            data = datSP.transpose()
        self.r2sp = [sp.interpolate.interp1d(r, dat, kind = kind) for dat in data]
        return self.r2sp


    def s2spV(self,sVal):
        # Vector valued function version of sParamSP.
        return np.array([self.s2sp[i](sVal) for i in range(0,len(self.s2sp))]).flatten()

    def r2spV(self,rVal):
        # Vector valued function version of rParamSP.
        return np.array([self.r2sp[i](rVal) for i in range(0,len(self.r2sp))]).flatten()

    def bcPO(self, ya, yb, Tp):
        # Boundary condition for periodic orbit with solve_bvp
        # ya and yb are initial and final points, respectively
        # Tp is the (unknown) period.
        # The last condition fixes time-translational invariance.
        return np.hstack([ya-yb, self.poincCond(yb)])
       
    def J_bc_bvp(self, t, state, Tp):
        # Jacobian for boundary conditions in a form that can be used by solve_bvp
        # Equations are written in rescalled time tau = t/T_p
        dbc_dya = np.zeros((state.shape[0],state.shape[0]))
        dbc_dyb = np.zeros((state.shape[0],state.shape[0]))
        dbc_dp = np.zeros(state.shape[0]) # Jacobian with respect to Tp
        dbc_dya = np.identity(state.shape[0])
        dbc_dyb = -np.identity(state.shape[0])
        dbc_dp = self.normVec
        return [df_dy, df_dp]

    def split_rs(self, s, r, sc, vc, case=1):
        # Split branches of a tree found by 2D manifold learning
        # s,r: the vectors defining the embedding
        # sc: list of length 2 determining limits of s to use for linear fit that determines the seperating line
        # vc: scalar that provides limits for s or r for which the separation is valid 
        # case: integer used to select different splitting strategies in a case-by-case basis
        self.vc = vc # Make this available to other functions, in particular sr2s
        self.split_case = case
        idx_choose_s=np.where(np.all( np.vstack( [sc[0]<s, s<sc[1] ]),axis=0) == True)
        ft = np.polyfit(s[idx_choose_s],r[idx_choose_s],1)
        self.pol = np.poly1d(ft)
        # Changes here need to propagate to sr2s !!!
        if case ==1:
            self.r_valid = np.all( np.vstack( [r > self.pol(s), r>vc] ), axis=0 )
        elif case ==2:    
            self.r_valid = np.all( np.vstack( [r < self.pol(s), r<vc] ), axis=0 )
        elif case ==3:
            self.r_valid = np.all( np.vstack( [r > self.pol(s), s<vc] ), axis=0 )
        elif case ==4:
            self.r_valid = np.all( np.vstack( [r < self.pol(s), s<vc] ), axis=0 )            
        self.s_valid = np.invert(self.r_valid)
        self.idx_s_valid = np.where(self.s_valid==True)
        self.idx_r_valid = np.where(self.r_valid==True)

    def sr2s(self, sr):
        # Given a trajectory parametrized by sr return parametrization by single variable s
        # In essence it skips points that would be parametrized by r.
        vc = self.vc # vc controls the splitting in self.split_rs.
        if self.split_case ==1:
            rvalid = np.all( np.vstack( [sr[:,1] > self.pol(sr[:,0]), sr[:,1]>vc] ), axis=0 )
        elif self.split_case ==2:
            rvalid = np.all( np.vstack( [sr[:,1] < self.pol(sr[:,0]), sr[:,1]<vc] ), axis=0 )
        elif self.split_case ==3:
            rvalid = np.all( np.vstack( [sr[:,1] > self.pol(sr[:,0]), sr[:,0]<vc] ), axis=0 )
        elif self.split_case ==4:
            rvalid = np.all( np.vstack( [sr[:,1] < self.pol(sr[:,0]), sr[:,0]<vc] ), axis=0 )            
        return sr[np.invert(rvalid),0]  

    def plotItinS(self, itin, cobweb=True, c='r', cc = None, linestyles='solid', s=10, periodic=False, zorder=2):
        # zorder can be used to control positioning on top or below other graphs in the figure and/or rasterization
        if s==0: # If s=0 we do not plot points, so we have to set cobweb =True
            cobweb = True
        if cc==None:
            cc=c # By default use same color for points and lines
        if periodic:
            itFirst = itin[0]
            itLast = itin[-1]
            itin = np.append(itin, itFirst)
            itin = np.insert(itin, 0, itLast)
        for i in range(0,itin.shape[0]-1):
                    if s != 0:
                        plt.scatter(itin[i],itin[i+1], c=c,s=s, zorder=zorder)
                    if cobweb and i<itin.shape[0]-2: 
                        plt.hlines(itin[i+1],itin[i],itin[i+1], color=cc, linestyles=linestyles, zorder=zorder)
                        plt.vlines(itin[i+1],itin[i+1],itin[i+2], color=cc, linestyles=linestyles, zorder=zorder)
                            
    def map_s_s(self, s, r):
        self.idx_map_s_s = self.idx_s_valid[0][np.where(np.diff(self.idx_s_valid).flatten() == 1)] # indices of points which map to s (they correspond to no gap in the data)  
        s_s = np.asarray([s[self.idx_map_s_s], s[(self.idx_map_s_s+np.ones_like(self.idx_map_s_s))]]).transpose() # define the map
        if s_s.shape[0] > 0:
            self.irpRM_s_s = self.interp1d_map(s_s) # Create interpolating function from map
        return s_s
        
    def map_s_r(self, s, r):
        self.idx_map_s_r = self.idx_s_valid[0][np.where(np.diff(self.idx_s_valid).flatten() > 1)] # indices of points which map to r
        s_r = np.asarray([s[self.idx_map_s_r],r[(self.idx_map_s_r+np.ones_like(self.idx_map_s_r))]]).transpose()
        if s_r.shape[0] > 0:
            self.irpRM_s_r = self.interp1d_map(s_r) # Create interpolating function from map
        return s_r

    def map_r_r(self, s, r):
        self.idx_map_r_r = self.idx_r_valid[0][np.where(np.diff(self.idx_r_valid).flatten() == 1)] # indices of points which map to r (they correspond to no gap in the data)
        r_r =  np.asarray([r[self.idx_map_r_r],r[(self.idx_map_r_r+np.ones_like(self.idx_map_r_r))]]).transpose()
        if r_r.shape[0] > 0:
            self.irpRM_r_r = self.interp1d_map(r_r) # Create interpolating function from map
        return r_r
    
    def map_r_s(self, s, r):
        self.idx_map_r_s = self.idx_r_valid[0][np.where(np.diff(self.idx_r_valid).flatten() > 1)] # indices of points which map to s 
        r_s = np.asarray([r[self.idx_map_r_s],s[(self.idx_map_r_s+np.ones_like(self.idx_map_r_s))]]).transpose()
        if r_s.shape[0] > 0:
            self.irpRM_r_s = self.interp1d_map(r_s) # Create interpolating function from map
        return r_s

    def map_S_S(self, s, sorted_map_label='None',sorted_label='None'):
        # Combined map in single coordinate S
        # Specific to the case self.idx_map_r_r.size == 0, add alternatives if needed
        if self.idx_map_r_r.size == 0:
            s_s_1 = np.asarray([s[self.idx_map_s_s], s[(self.idx_map_s_s+np.ones_like(self.idx_map_s_s))]]).transpose() # direct s to s branch
            s_s_2 = np.asarray([s[self.idx_map_s_r], s[(self.idx_map_s_r+2*np.ones_like(self.idx_map_s_r))]]).transpose() # s to r to s branch (skips a point in the s map)
            S_S = np.vstack([s_s_1, s_s_2]) # Combined map
            self.interpRM = self.interp1d_map(S_S, sorted_map_label=sorted_map_label,sorted_label=sorted_map_label)
        else:
            S_S=None
        return S_S


    def ssp2emb(self,ssp):
        # Compute re-embedding of state-space points to intrisic coordinates
        # ssp is 2D array of data points with shape (n_samples, d_space) 
        # or 1D array containing single data point.
        if ssp.ndim==1: # If single data point
            pts = ssp.reshape(1, -1)
        else:
            pts = ssp
        if self.reduce_refl:
            pts = self.reflRedTraj(pts)
        return self.normS(self.embedding.transform(pts))
  
    def findCycle(self, seq, TpoincMean, tol=1e-6, atol=1e-9, rtol=1e-6, jac_tol=1e-6, tminFrac=1e-2, timer=True, refine_RM = True, tol_RM=1e-10, method_RM='bisect', max_nodes=10000, jac_sample=1, full_output=True, stability = True, init_only = False, guess_from_RM_shadowing=False, repeat=1):
        # Find cycle with given symbolic sequence.
        # seq: symbolic sequence
        # TpoincMean: mean time of return to Poincare section.
        # tol: tolerance for boundary value solver
        # atol: atol parameter for solve_ivp for guess generation
        # rtol: rtol parameter for solve_ivp for guess generation
        # jac_tol: atol parameter for solve_ivp for Jacobian calculation 
        #           Since we decompose the Jacobian into parts, this need not be very low
        # timer: whether to output information on the time required for the solution of the boundary value problem
        # refine_RM: whether to refine initial guess using the return map
        # method_RM: method to use to find the cycle of the return map in the first step
        # tminFrac: option for guess generation (see guessMultiVarS)
        # max_nodes: maximum number of nodes allowed in solve_bvp
        # jac_sample: choose every jac_sample point on the solution as initial condition for the calculation of the Jacobian
        # init_only: if true, just initialize column labels for DataFrame and exit
        err_code = 0
        if self.col == None or init_only: # initialize the DataFrame column labels
                if stability and (not full_output):
                    # Specify columns of data structure
                    self.col = ['label', 'length', 'period','Poinc. points','s','multipliers', 'error code']                    
                if stability and full_output:
                    # Specify columns of data structure
                    self.col = ['label', 'length', 'period','Poinc. points','s','multipliers', 'time', 'data', 'error code']
                if (not stability) and full_output:
                    self.col = ['label', 'length', 'period','Poinc. points','s', 'time', 'data', 'error code']
                print('Initialized data labels. Exiting')
                if init_only: 
                    return err_code
        if guess_from_RM_shadowing:
            sPOrm = self.cycleICshad(seq, self.seqChaotic)                
        else:                
            if not np.all(np.product(self.fixedPointRM(self.cycleIrvs(seq),len(seq)),axis=1)<0):
                print(f"{Fore.MAGENTA}Generated intervals do not bound a solution. Check cycle admissibility for", ''.join(map(str, seq)) , f"{Style.RESET_ALL}")
                err_code = 11 # Issue warning and change error code but do not terminate
                refine_RM = False # Do not try to refine RM solution, just use IC from intervals
            if refine_RM:
                sPOrm = self.findCycleRM(seq, method=method_RM, tol=tol_RM) # First find PO of the return map
                seq1 = self.s2symb(sPOrm)
                if seq==seq1:
                    print("RM converged to requested PO ",''.join(map(str, seq1)))
                else:
                    print(f"{Fore.RED}Requested PO ", ''.join(map(str, seq)) , "RM converged to PO ", ''.join(map(str, seq1)),f"{Style.RESET_ALL}")    
                    err_code = 100
                    return [seq1, sPOrm, err_code]
            else:
                sPOrm=self.cycleIC(seq)
        if repeat != 1:
            sPOrm = sPOrm*repeat
            seq = seq*repeat
        POguess = self.guessMultiVarS(self.equation, sPOrm, seq, TpoincMean, atol=atol, rtol=rtol)
        TpGuess = POguess[0][-1]
        if timer:
            tic = time.perf_counter()
        # This will only work for KS (f_bvp_ps). Replace with simulator object
        solPO = solve_bvp(self.equation.f_bvp_ps, self.bcPO, POguess[0]/TpGuess, POguess[1], p=[TpGuess], tol=tol, max_nodes=max_nodes, verbose=2)
        Tp = solPO.p[0]
        if timer:
            toc = time.perf_counter()
            print(f"BVP solution time: {toc - tic:0.4f} seconds")
        if solPO.status == 0: # solve_bvp converged
            # Confirm symbolic sequence
            pPtsPO = self.poincPoints_bvp(solPO.y, 10*TpoincMean) # Find points on Poincare section from bvp solution
            sPO = self.ssp2emb(copy.deepcopy(pPtsPO)) # Re-embed points
            if self.demb == 2: # If two dimensional embedding
                sPO = self.sr2s(sPO) # Convert to single parameter (skips points parameterized by r)
            #print('sPO=',sPO)
            seq2 = self.s2symb(sPO)            
            # Check that we got the correct solution
            if seq2 in self.cycPerm(seq): # Poincare section points can be a cyclic permutation of requested sequence
                print("Converged to requested PO ",f"{Fore.GREEN}", ''.join(map(str, seq)),f"{Style.RESET_ALL}", 'with period Tp=',Tp)
                shift= self.cycPerm(seq).index(seq2) # Compute shift that would map Poincare section points to requested itinerary, for book-keeping reasons
                pPtsPO = np.roll(pPtsPO,shift,axis=0) # Shift Poincare section points accordingly
                sPO = np.roll(sPO,shift,axis=0) # Shift Poincare section points accordingly
                if stability:
                    jac = self.floquet_bvp(solPO,jac_sample=jac_sample,jac_tol=jac_tol, timer=timer)
                    mult = np.linalg.eigvals(jac) # Compute multipliers
                    print('Marginal direction error=',"{:.2e}".format(np.abs((mult-1)).min()))
                if stability and not full_output:
                    # Specify columns of data structure
                    return np.asarray([seq, len(seq), Tp, pPtsPO, sPO, mult, err_code],dtype=object)
                if stability and full_output:
                    # Specify columns of data structure
                    return np.asarray([seq, len(seq), Tp, pPtsPO, sPO, mult, solPO.x*solPO.p[0], solPO.y, err_code],dtype=object)
                if (not stability) and full_output:
                    return np.asarray([seq, len(seq), Tp, pPtsPO, sPO, solPO.x*solPO.p[0], solPO.y, err_code],dtype=object)
            else:
                print(f"{Fore.RED}Requested PO ", ''.join(map(str, seq)) , "but converged to PO ", ''.join(map(str, seq2)), 'with period Tp=',Tp, f"{Style.RESET_ALL}")
                err_code = 22
                return [seq, len(seq), Tp, pPtsPO, sPO, seq2, len(seq2), err_code]
        else: # solve_bvp did not converge
            print(f"{Fore.RED}Failed to converge to cycle", ''.join(map(str, seq)), f"{Style.RESET_ALL}")
            return [seq, solPO.status]


    def floquet_bvp(self,solPO, jac_tol=1e-6, timer=True, jac_sample=1):
        # Floquet eigenproblem of cycle found by solve_bvp
        # solPO is a solve_bvp solution object
        # jac_sample is sample rate for points on the solution
        # Return Jacobian
        if timer:
            tic = time.perf_counter()
        # For each point on the variational solution compute the Jacobian to the next point
        ictab0 = [np.hstack([solPO.y[:,i], np.identity(self.equation.N).reshape(-1)]) for i in range(solPO.x.shape[0]-1)]
        if jac_sample !=1: # Downsample the variational solution for faster calculation of the Jacobian
            isample = list(range(0,np.shape(solPO.x)[0],jac_sample))
            if isample[-1] != np.shape(solPO.x)[0]-1:
                isample.append(np.shape(solPO.x)[0]-1)
        else: 
            isample = list(range(0,np.shape(solPO.x)[0]))
        ictab = np.asarray(ictab0)[isample[0:-1]]
        tsample = solPO.x[isample]*solPO.p[0] # time in solve_bvp has been normalized to the period
        # BFD has too much overhead so we use RK45 here. Rewrite to allow flexibility
        Jpart = np.asarray([np.reshape(solve_ivp(self.equation.f_Jac_ps, [tsample[i], tsample[i+1]],ictab[i], atol=jac_tol, method='RK45').y[self.equation.N:,-1],(self.equation.N,self.equation.N)) for i in range(0,ictab.shape[0])])
        ## If using odeint (not recommended)
        #Jpart = np.asarray([np.reshape(odeint(self.equation.f_Jac_ps, ictab[i], [tsample[i], tsample[i+1]], tfirst=True, atol=atol)[-1,self.equation.N:],(self.equation.N,self.equation.N)) for i in range(0,ictab.shape[0])])                
        # Then multiply the partial Jacobians
        jac=np.identity(self.equation.N)
        for i in range(0,Jpart.shape[0]):
            jac=np.matmul(Jpart[i],jac)
        if timer:
            toc = time.perf_counter()
            print(f"Jacobian calculation time: {toc - tic:0.4f} seconds using ", len(ictab), "segments")
        #print('mult=',mult)
        return jac

            
    #def map_s_s(self, s, r, rc):
        #idx_low_r = np.where(r < rc)
        #self.idx_map_s_s = idx_low_r[0][np.where(np.diff(idx_low_r).flatten() == 1)] # indices of points which map to s (they correspond to no gap in the data)  
        #s_s = np.asarray([s[self.idx_map_s_s], s[(self.idx_map_s_s+np.ones_like(self.idx_map_s_s))]]).transpose() # define the map
        #if s_s.shape[0] > 0:
            #self.irpRM_s_s = self.interp1d_map(s_s) # Create interpolating function from map
        #return s_s
    
    #def map_s_r(self, s, r, rc):
        #idx_low_r = np.where(r < rc)
        #self.idx_map_s_r = idx_low_r[0][np.where(np.diff(idx_low_r).flatten() > 1)] # indices of points which map to r
        #s_r = np.asarray([s[self.idx_map_s_r],r[(self.idx_map_s_r+np.ones_like(self.idx_map_s_r))]]).transpose()
        #if s_r.shape[0] > 0:
            #self.irpRM_s_r = self.interp1d_map(s_r) # Create interpolating function from map
        #return s_r

    #def map_r_r(self, s, r, rc):
        #idx_high_r = np.where(r > rc)
        #self.idx_map_r_r = idx_high_r[0][np.where(np.diff(idx_high_r).flatten() == 1)] # indices of points which map to r (they correspond to no gap in the data)
        #r_r =  np.asarray([r[self.idx_map_r_r],r[(self.idx_map_r_r+np.ones_like(self.idx_map_r_r))]]).transpose()
        #if r_r.shape[0] > 0:
            #self.irpRM_r_r = self.interp1d_map(r_r) # Create interpolating function from map
        #return r_r
    
    #def map_r_s(self, s, r, rc):
        #idx_high_r = np.where(r > rc)
        #self.idx_map_r_s = idx_high_r[0][np.where(np.diff(idx_high_r).flatten() > 1)] # indices of points which map to s 
        #r_s = np.asarray([r[self.idx_map_r_s],s[(self.idx_map_r_s+np.ones_like(self.idx_map_r_s))]]).transpose()
        #if r_s.shape[0] > 0:
            #self.irpRM_r_s = self.interp1d_map(r_s) # Create interpolating function from map
        #return r_s

    def guessMultiVar(self, ds, c0, seq, TpoincReturn,  eps=1e-9):
        # Converts discrete map guess for periodic orbit to a 'loop' for solve_bvp
        # ds: dynamical system to be integrated (this should be an instance initialized in main)
        # c0: initial value for s or r
        # seq: a string of the form 'srssr...ssrrs' describing which sub-manifold the successive points belong to
        # TpoincReturn: average return time to poincare section
        # eps: integrate for at least eps before checking Poincare condition to avoid early termination
        # Returns:
        #   [t, y]: length two sequence containing solution times and array with dependent variables 
        # Add option for rtol atol !!!!
        if seq[0] == 's':
            icPOg = self.s2spV(c0)
        elif seq[0] == 'r':
            icPOg = self.r2spV(c0)
        self.poincCondtx.__func__.terminal = True # Terminate integration after first intersection
        self.int_min_t = eps #
        # Add check for ds.A
        solPOg = solve_ivp(ds.f_ps, [0,10*TpoincReturn], icPOg, method='BDF', events = self.poincCondtx, jac=ds.A, rtol=1e-6, atol=1e-9) # Integrate until next intersection with Poincare section.
        POseg = solPOg.y    
        POtseg = solPOg.t    
        cOld = c0
        
        for i in np.arange(0, len(seq)-1):
            if seq[i]=='s' and seq[i+1]=='r':
                cNew = self.irpRM_s_r(cOld)
                icPOg = self.r2spV(cNew)
                cOld = cNew
            elif seq[i]=='r' and seq[i+1]=='s':
                cNew = self.irpRM_r_s(cOld)
                icPOg = self.s2spV(cNew)
                cOld = cNew
            elif seq[i]=='s' and seq[i+1]=='s':
                cNew = self.irpRM_s_s(cOld)
                icPOg = self.s2spV(cNew)
                cOld = cNew
            elif seq[i]=='r' and seq[i+1]=='r':
                cNew = self.irpRM_r_r(cOld)
                icPOg = self.r2spV(cNew)
                cOld = cNew            
            # If we use reflection symmetry reduction and the previous solution segment ended out of the fundamental domain
            # then map present segment initial condition out of the fundamental domain
            if self.reduce_refl and not self.isInReflFD(POseg[:,-1]): 
                icPOg = self.refl(icPOg)                
            self.int_min_t = solPOg.t[-1]+eps 
            # It would be better to define a new simulator object and pass this to this function than having a hardcoded integrator; this would probably need to be part of PoincareMapper.py
            solPOg = solve_ivp(ds.f_ps, [solPOg.t[-1],solPOg.t[-1]+10*TpoincReturn], icPOg, method='BDF', events = self.poincCondtx, jac=ds.A, rtol=1e-6, atol=1e-9) # Integrate until next intersection with Poincare section.
            POseg = np.hstack([POseg,solPOg.y[:,1:]])    
            POtseg = np.hstack([POtseg,solPOg.t[1:]])
            
        return [POtseg, POseg]


    def itinerary(self, n, s=None, r=None):
        itin = np.full((n+1,2),None)
        if r is None: #we start with s
            itin[0,0] = np.asarray(s)
        else: # we start with r
            itin[0,1] = np.asarray(r)
        for i in range(1, itin.shape[0]):
            if itin[i-1,1] is None: # Previous symbol was s
                if self.irpRM_s_s.x[0] <= itin[i-1,0] <= self.irpRM_s_s.x[-1]: # s lies within range of validity of s to s map
                    itin[i,0] = self.irpRM_s_s(itin[i-1,0])
                else: 
                    itin[i,1] = self.irpRM_s_r(itin[i-1,0])
            else: # previous symbol was r
                if self.irpRM_r_s.x[0] <= itin[i-1,1] <= self.irpRM_r_s.x[-1]: # r lies within range of validity of r to s map
                    itin[i,0] = self.irpRM_r_s(itin[i-1,1])
                else:
                    itin[i,1] = self.irpRM_r_r(itin[i-1,1])
        return itin

    def itineraryS(self, s, n):
        itin = np.zeros(n+1)
        itin[0] = s
        for i in range(1, itin.shape[0]):
            if self.irpRM_s_s.x[0] <= itin[i-1] <= self.irpRM_s_s.x[-1]: # s lies within range of validity of s to s map
                itin[i] = self.irpRM_s_s(itin[i-1])
            else:
                itin[i] = self.irpRM_r_s(self.irpRM_s_r(itin[i-1]))
        return itin

    def itinSingle(self, s, n):
        # Itinerary for single variable case
        itin = np.zeros(n+1)
        itin[0] = s
        for i in range(1, itin.shape[0]):
                itin[i] = self.interpRM(itin[i-1])
        return itin

#    def itineraryS(self, n, s):
#        itin = np.full((n+1),None)
#        itin[0] = np.asarray(s)
#        for i in range(1, itin.shape[0]):
#                if self.irpRM_s_s.x[0] <= itin[i-1] <= self.irpRM_s_s.x[-1]: # s lies within range of validity of s to s map
#                    itin[i] = self.irpRM_s_s(itin[i])
#                else: 
#                    itin[i] = self.irpRM_r_s(self.irpRM_s_r(itin[i-1]))
#        return itin
    def refl(self,a):
        # Reflection for KS in antisymmetric domain
        aa = copy.deepcopy(a)
        aa[::2]=-aa[::2]
        return aa
    
    def isInReflFD(self,a):
        # Checks if solution is in fundamendal domain (FD) for reflections (for KS in antisymmetric domain)
        return a[2] >= 0
    
    def reflRed(self,a):
        # Maps a point to fundamendal domain
        if self.isInReflFD(a):
            return a 
        else:
            return self.refl(a)
    
    def reflRedTraj(self,a):
        # Maps all point in a trajectory (list) to fundamental domain
        return np.asarray([self.reflRed(aa) for aa in a])
    
    def reflTraj(self,a):
        # Applies reflection operation to all points in a trajectory
        return np.asarray([self.refl(aa) for aa in a])

    def computeNormS(self,q):
        # Computes normalization factors and normalizes the data.
        # Sets global variables self.norm_shift, self.norm.
        # Create dummy variable to avoid changing value of input variable
        # Reconsider this step if memory becomes an issue
        dum = copy.deepcopy(q)
        dum = self.sign_norm_q*dum # First perform any reflections with respect to the origin
        if self.qmax_1:        
            self.qmin_0 = True # Do this automatically since self.qmax_1 makes sense only if we are mapping variables to the unit interval
        if self.qmin_0:
            self.norm_shift = []
            if self.demb>1:
                for i in range(q.shape[1]):
                    self.norm_shift.append(- dum[:,i].min())
                    dum[:,i] = dum[:,i]  + self.norm_shift[i]
            else:
                self.norm_shift = - dum.min()
                dum = dum + self.norm_shift
        if self.qmax_1:
            self.norm = []
            if self.demb>1:            
                for i in range(q.shape[1]):
                    self.norm.append(dum[:,i].max())
                    dum[:,i] = dum[:,i]/self.norm[i]                
            else:
                self.norm = dum.max() 
                dum = dum/self.norm
        return dum
    

    
    def normS(self,q):
        # Normalization for 2d manifold learning
        # Create dummy variable to avoid changing value of input variable
        # Reconsider this step if memory becomes an issue
        dum = copy.deepcopy(q)
        dum = self.sign_norm_q*dum # First perform any reflections with respect to the origin
        if self.qmax_1:        
            self.qmin_0 = True # Do this automatically since self.qmax_1 makes sense only if we are mapping variables to the unit interval
        if self.qmin_0:
            if self.demb>1:
                for i in range(q.shape[1]):
                    dum[:,i] = dum[:,i]  + self.norm_shift[i]
            else:
                dum = dum  + + self.norm_shift
        if self.qmax_1:
            if self.demb>1:
                for i in range(q.shape[1]):
                    dum[:,i] = dum[:,i]/self.norm[i]                
            else:
                 dum = dum/self.norm                    
        return dum
    
    def symb(self,s):
        # Convert single value of s to symbol
        smb = self.sc.size
        for i in range(0,self.sc.size):
            if s < self.sc[i]:
                return i
        return smb

    def s2symb(self,itin):
        # Converts itinerary to symbol
        return [self.symb(s) for s in itin]

    def eps_parity_single(self,symb):
        # Parity of single symbol. Requires having defined self.orient_preserv which should store 
        # orientation preserving branches
        if symb in self.orient_preserv:
            return 1
        else:
            return -1
    
    def eps_parity(self,seq):
        # Parity of symbolic sequence. Requires having defined self.orient_preserv which should store 
        # orientation preserving branches
        orient = [self.eps_parity_single(symb) for symb in seq]
        return np.prod(orient)
        
    def is_spatially_ordered(self,seq1,seq2):
        # Returns true if seq1, seq2 are spatially ordered.
        # Based on Gilmore and Lefrance "Topology of chaos" book.
        # First check if we are dealing with the trivial case 
        # that the first symbol of the two sequences differs
        if seq1[0] < seq2[0]:
            return True
        # First decompose seq_i to Lambda s_i, where Lambda is the common part and s_i is the first symbol in which 
        # the two sequences differ
        isplit = np.where(np.asarray(seq1)-np.asarray(seq2) != 0)[0][0]
        Lambda = seq1[0:isplit]
        s1 = seq1[isplit]
        s2 = seq2[isplit]
        if ((s1<s2) and (self.eps_parity(Lambda)==1)) or ((s1>s2) and (self.eps_parity(Lambda)==-1)):
            return True
        else:
            return False
    
    def admis_crit_unimodal(self,seq, knead):
        # Admissibility criterion for periodic orbit point of bimodal map for a given symbolic sequence seq.
        # It is called by is_admis_unimodal
        seq1=copy.deepcopy(seq)
        if len(knead)<len(seq1):
            raise Exception("kneading sequence need to be longer than PO")
        len0 = len(seq1) # initial length of sequence
        while seq1 == knead[0:len(seq1)]: # Extend seq periodically, until seq and knead differ
            seq1.append(seq1[-len0])
        return self.is_spatially_ordered(seq1,knead[0:len(seq1)]) # orbit is admissible if seq<knead
    
    def is_admis_unimodal(self,po,knead):
        # Returns true if periodic orbit of a bimodal map is admissible
        # It is now obsolete since is_admis handles n-modal map case, 
        # but can be used for verification purposes
        N = len(self.alphabet) # Number of letters in the alphabet
        if po == [0]:
            return self.zero_admis
        else:
            poPts = self.cycPerm(po) # Generate all points on the PO
            self.sort_seq_spatial(poPts,cp=10) # Sort the points spatially
            if not self.list_is_spatially_ordered(poPts):
                raise Exception("Ordering failed, increase number of copies of basic sequence by setting cp argument")
            return self.admis_crit_unimodal(poPts[-1],knead) # Determine if right-most orbit point is admissible
        
    def is_admis(self,po, knead):
        # Check admissibility of periodic orbit for n-modal map (n is the number of critical points, N=n+1 is the number of letters in the alphabet)
        # knead is a list of kneading sequences of length n
        # Algorithm is based on K.T. Hansen's PhD thesis
        N = len(self.alphabet)
        if po == [0]:
            return self.zero_admis
        else:        
            poPts = self.cycPerm(po) # Generate all points on the PO
            for i in range(len(knead)): # repeat for all critical points, index starts at 0
                kval = self.theta_inv(knead[i],cp=1)
                pBoolean = [p[0]==i or p[0]==i+1 for p in poPts] # Only consider points with x_0 in intervals to the left and right of the critical point
                plist = [poPts[np.mod(k+1,len(poPts))] for k, x in enumerate(pBoolean) if x] # Use index k+1 because we want to use s_1s_2... (we based our check on first point which is s_0)        
                if len(plist)>0: # Found points within the given intervals
                    thlist=[self.theta_inv(p, cp=10) for p in plist] # Compute topological coordinate            
                    if (i+1)%2 !=0: # Case corresponding to odd index of critical point
                        if np.max(thlist)>kval: # Non-admissibility
                            return False
                    else: # Case corresponding to even index of critical point
                        if np.min(thlist)<kval: # Non-admissibility
                            return False
        return True # If we reached this point without exiting then the orbit is admissible
    
    def theta_inv(self,seq,cp=1):
        # invariant coordinate theta for a symbolic sequence
        # cp is number of copies of the sequence (used to ensure convergence for periodic orbits)
        N = len(self.alphabet) # Number of letters in the alphabet
        seq1 = seq*cp # Replicate basic block
        return np.sum([self.ti(seq1[0:i+1])/N**(i+1) for i in range(0,len(seq1)+1)])

    def thetaApplyPoints(self,pts):
        # Convenience function that applies theta_inv on a list of symbolic sequences (usually list of periodic points)
        return [self.theta_inv(cycPoint) for cycPoint in pts]

    def ti(self,seq):
        # Function returning partial weights used to find theta_inv
        N = len(self.alphabet) # Number of letters in the alphabet
        if self.eps_parity(seq[0:-1]) == 1: 
            return seq[-1]
        else:
            return (N-1)-seq[-1]
        
    def sort_seq_spatial(self,seqs,cp=1):
        N = len(self.alphabet) # Number of letters in the alphabet
        # Sorts symbolic sequences by invariant coordinate theta.
        seqs.sort(key= lambda seq : self.theta_inv(seq,cp=cp))
        
    def duval(self,n):
        # Duval's algorithm to generate all necklashes of up to length n (with N = len(self.alphabet) symbols)
        # These are equivalent to all prime cycles up to length n, in lexicographical order.
        N = len(self.alphabet) # Number of letters in the alphabet
        lw = [] # List of Lyndon words
        w = [None]*n # Create empty list of length n
        i=1
        w[0]=0
        while i != 0:
            for j in range(1,n-i+1):
                w[i+j-1] = w[j-1]
            lw.append(w[0:i])
            i=n
            while i>0 and w[i-1]==N-1:
                i = i-1
            if i>0: 
                w[i-1]= w[i-1]+1 
        return lw
    
    def primeCycles(self,n):
        # Compute all prime cycles of length n (lexicographical ordering).
        # Uses Duval's algorithm to compute all cycles up to length n and then 
        # select cycles of length exactly n, so it is somewhat inneficient
        # but runs very fast up to length 10
        N = len(self.alphabet) # Number of letters in the alphabet
        cycl = self.duval(n) # Generate all cycles up to n
        return [cyc for cyc in cycl if len(cyc)==n]

    def primeCyclesAdmis(self,n,knead):
        # Returns all admissible cycles of length n
        N = len(self.alphabet) # Number of letters in the alphabet
        cycl = self.primeCycles(n)
        return [cyc for cyc in cycl if self.is_admis(cyc,knead)]
    
    def primeCyclesNonAdmis(self,n,knead):
        # Returns all non-admissible cycles of length n
        N = len(self.alphabet) # Number of letters in the alphabet
        cycl = self.primeCycles(n)
        return [cyc for cyc in cycl if not self.is_admis(cyc,knead)]
    
    def cycPerm(self,seq):
        # Returns all cyclic permutations of a given sequence.
        # Can be used to generate all points on a cycle.
        s=np.asarray(seq)
        lst=[seq]
        for i in range(1,len(seq)):
            lst.append(np.roll(s,-i).tolist())
        return lst

    def periodicPointsLength(self,n):
        # Return symbolic sequences corresponding to all points on all prime cycles of length n
        # Sort the resulting least spatially
        lst = [cycPoint for cyc in self.primeCycles(n) for cycPoint in self.cycPerm(cyc)]
        self.sort_seq_spatial(lst, cp=10) # Sort spatially
        return lst

    def kneadValues(self,knead):
        # Return the kneading values for a list o kneading sequences
        return [self.theta_inv(kn,cp=1) for kn in knead]


    def partDepth(self,n):
        # Computes all sub-intervals at length n. Not useful as these are not in 1-1 correspondance with admissible periodic orbits of length n.
        if n>1:
                result = np.asarray([self.inv_map[i](q) for i in self.alphabet for q in self.partDepth(n-1)])
                result = result[~np.isnan(result)] # Useful if self.inv_map has fill_value='NaN'
                result = np.insert(result,0,self.sc)
                result = np.sort(result)            
        elif n == 1:
            result = self.sc
        else:
            result = None
        return result

    def list_is_spatially_ordered(self,lst):
        # Determines if a list of symbolic sequences is spatially ordered
        return np.all([self.is_spatially_ordered(lst[i],lst[i+1]) for i in range(len(lst)-1)])

    def partSymbDepth(self, n, cp=1):
        # Return all possible symbolic sequences of length n, spatially ordered
        # First compute all combinations of letters in the alphabet of length n:
        combs = list(itertools.product(self.alphabet, repeat=n))
        combs = [list(s) for s in combs] # Convert inner level to list
        self.sort_seq_spatial(combs, cp=cp)
        if not self.list_is_spatially_ordered(combs):
            raise Exception("Ordering failed, increase number of copies of basic sequence by setting cp argument")
        return combs

    def finiteIrv(self,n, order='spat', cp=1):
        # Returns all intervals at depth n (including non-prime ones). 
        # order:    'spat' for spatial ordering, 'lex' for lexicographical
        lst = [[self.list2str(i), np.diff(self.seq2irv(i))[0]>0] for i in self.partSymbDepth(n,cp)]
        return lst

    def seq2irvUnim(self, seq):
        # Returns interval associated with symbolic sequence
        # Uses interval arithmetic from Gillmore and Lefrance book, Chapter 2.
        if seq==[0]:
            q = np.asarray([self.interpRM(self.interpRM(self.sc[0])), self.sc[0]]).flatten()
            #print(''.join(map(str, seq)), "Length",q[1]-q[0]) 
            return q
        elif seq == [1]:
            q = np.asarray([self.sc[0],self.interpRM(self.sc[0])]).flatten()
            #print(''.join(map(str, seq)), "Length",q[1]-q[0]) 
            return q
        else:
            # Leading symbol will determine map to use
            s = seq[0] 
            # Compute parent interval pirv
            pirv = self.seq2irv(seq[1:])
            # Apply inverse map to interval        
            gpirv = self.fInvIrv(s,pirv)
            #print(gpirv)
            if s == 0: # Need to generalize for multi-modal map.
                # Take intersection with base interval
                q= np.asarray([gpirv[0], min(gpirv[1],self.sc[0])])
                #print(''.join(map(str, seq)), "Length",q[1]-q[0]) 
                return q
            else:
                # Take intersection with base interval                                       
                q= np.asarray([max(gpirv[0],self.sc[0]), gpirv[1]])
                #print(''.join(map(str, seq)), "Length",q[1]-q[0])
                return q

    def irv_inters(self,i1,i2):
        if i2[0]>i1[1] or i1[0]>i2[1]: # intersection is empty
            return np.asarray([])
        else:
            iS = np.max([i1[0],i2[0]])
            iE = np.min([i1[1],i1[1]])
            return np.asarray([iS,iE])

    def setBaseIrv(self):
        # Computes the base intervals
        self.baseI = np.asarray(self.flatten([[self.invIrv[0]], self.sc, [self.invIrv[1]]])).flatten()

    def seq2irv(self,seq):
        # Returns interval associated with symbolic sequence
        # Uses interval arithmetic from Gillmore and Lefrance book, Chapter 2.
        if len(seq) == 1: # We are dealing with one of the base intervals
            s = seq[0]
            q = self.baseI[s:s+2]
            return q
        else:
            # Leading symbol will determine map to use
            s = seq[0] 
            # Compute parent interval pirv
            pirv = self.seq2irv(seq[1:])
            # Apply inverse map to interval        
            gpirv = self.fInvIrv(s,pirv)
            #print(gpirv)
            return self.irv_inters(gpirv,self.seq2irv([s])) # return intersection of daughter interval with base interval

 
    def inverse_map(self,pk=None, kind= 'nearest'):
        # Computes inverse map(s) of self.sorted_s_s. One map is returned for each interval of monotonicity.
        # pk are indices of maxima or minima (or discontinuities) which split self.sorted_s_s to monotonic segments
        # If not provided, compute them using self.sc
        if pk==None:
            pk = [np.argmin(np.abs(self.sorted_s_s[:,0]-si)) for si in self.sc]
        irv = copy.deepcopy(pk) # Break invariant interval to intervals of monotonicity
        # add endpoints (0 and -1 are indices)
        irv.insert(0,0)  
        irv.append(-1)
        fill_vals = self.sorted_s_s[irv,0] # For fill_value in interpolation map
        imap = []
        for i in range(0,len(irv)-1):
            if i in self.orient_preserv: # Order of fill_value depends on this
                imap.append(sp.interpolate.interp1d(self.sorted_s_s[irv[i]:irv[i+1], 1], self.sorted_s_s[irv[i]:irv[i+1], 0], kind = kind, fill_value = (fill_vals[i],fill_vals[i+1]), bounds_error=False))
            else:
                imap.append(sp.interpolate.interp1d(self.sorted_s_s[irv[i]:irv[i+1], 1], self.sorted_s_s[irv[i]:irv[i+1], 0], kind = kind, fill_value = (fill_vals[i+1],fill_vals[i]), bounds_error=False))
        return imap
 
    def fInvIrv(self, s, irv):
        # Applies inverse map corresponding to letter s on interval irv
        i1 = [self.inv_map[s](irv[0]), self.inv_map[s](irv[1])]
        if not (s in self.orient_preserv): # If s does not correspond to orientation preserving branch we need to reverse the interval
            i1.reverse()
        return i1
        
        
    def seq2ic(self, seq):
        # For a given symbolic sequence seq compute initial condition corresponding to the midpoint of the interval associated with seq.
        irv = self.seq2irv(seq)
        return 0.5*(irv[0]+irv[1]) # return the midpoint

    def cycleIC(self, seq):
        cycSymb = self.cycPerm(seq) # We need all points on the cycle, in the order they appear by forward iteration
        return np.asarray([self.seq2ic(seq) for seq in cycSymb]).flatten()

    def cycleICshad(self,seq, seqChaotic, cp=2, threshold = 0, repeat=1):
        # Find guess IC for RM cycle from shadowing of chaotic trajectory seqChaotic in symbol space, rather than kneading theory
        POlabel= self.str2list(seq)*cp
        gs = POlabel[:len(POlabel)-threshold] # We usually would like shadowing to persist for (number of intersections) > (topological length of orbit) for a reliable guess
        gt= np.asarray([self.list2str(np.roll(gs,i)) in self.list2str(seqChaotic) for i in range(len(POlabel))])# Shift the string and detect its appearance in the chaotic sequence
        fg = np.arange(gt.shape[0])[gt == True][0] # We just need to work with the first succesfully detected shadowing event
        gs0 =  self.list2str(np.roll(gs,fg)) #                     
        idxg = self.list2str(seqChaotic).index(gs0) # Find the index of the shadowed string in the chaotic trajectory
        sg = list(self.s[idxg:idxg+len(seq)]) # Convert to s values
        return sg*repeat
       

    def cycleIrvs(self, seq):
        cycSymb = self.cycPerm(seq) # We need all points on the cycle, in the order they appear by forward iteration
        return np.asarray([self.seq2irv(seq) for seq in cycSymb])

    def fMultiShootRM(self,x):
        # Function which returns zero for multishooting with the return map
        xroll = np.roll(x,-1)
        return np.asarray([self.interpRM(x[i])-xroll[i] for i in range(x.shape[0])]).flatten()
 
    def fixedPointRM(self,s,n):
        # Returns condition for fixed point of RM applied n times 
        val = s
        for i in range(0,n):
            val= self.interpRM(val)
        return val-s

    def findCycleRM(self,seq, method='bisect', tol=1e-10):
        # For any symbol sequence find corresponding cycle from return map
        # bisection is the fail-safe method, but newton method in multipoint-shooting 
        # setup could be used
        if method == 'newton':
            return sp.optimize.newton(self.fMultiShootRM,self.cycleIC(seq), maxiter=100, tol=tol) 
        else: # bisection is the fail-safe method
            irv = self.seq2irv(seq)
            root = sp.optimize.bisect(self.fixedPointRM,irv[0],irv[1], args=(len(seq),))
            return self.itinSingle(root,len(seq)-1) # Is this stabel for very long cycles?
            #irvs = self.cycleIrvs(seq)
            #return [sp.optimize.bisect(self.fixedPointRM,irv[0],irv[1], args=(len(seq),), xtol=tol) for irv in irvs]
                                                        
    def guessMultiVarS(self, ds, c0, seq, TpoincReturn, tminFrac=1e-2, atol=1e-10, rtol=1e-7, plt_guess=False):
        # Converts discrete map guess for periodic orbit to a 'loop' for solve_bvp
        # ds: dynamical system to be integrated (this should be an instance initialized in main)
        # c0: vector of initial values for s 
        # seq: symbolic sequence associated with the orbit 
        # TpoincReturn: average return time to poincare section
        # tminFrac: integrate for at least tminFrac*TpoincReturn before checking Poincare condition to avoid early termination
        # Returns:
        #   [t, y]: length two sequence containing solution times and array with dependent variables 
        if plt_guess:
            plt.figure()
        icPOg = self.s2spV(c0[0])
        self.poincCondtx.__func__.terminal = True # Terminate integration after first intersection
        self.int_min_t = tminFrac*TpoincReturn # Minimum time to integrate in order to avoid early termination of integration
        # Add check for ds.A
        solPOg = solve_ivp(ds.f_ps, [0,10*TpoincReturn], icPOg, method='BDF', events = self.poincCondtx, jac=ds.A, atol=atol,rtol=rtol) # Integrate until next intersection with Poincare section.
        # Drop endpoint since we don't want segments to overlap
        POseg = solPOg.y[:,:-1]    
        POtseg = solPOg.t[:-1] 
        if plt_guess:
            plt.plot(solPOg.y[1],solPOg.y[2],'.-')
        for repeat in range(self.nPoinc[seq[0]]-1): # For symbols with nPoinc>1
            icPOg = copy.deepcopy(solPOg.y[:,-1])
            self.int_min_t = solPOg.t[-1]+tminFrac*TpoincReturn                    
            # It would be better to define a new simulator object and pass this to this function than having a hardcoded integrator; this would probably need to be part of PoincareMapper.py
            solPOg = solve_ivp(ds.f_ps, [solPOg.t[-1],solPOg.t[-1]+10*TpoincReturn], icPOg, method='BDF', events = self.poincCondtx, jac=ds.A, atol=atol, rtol=rtol) # Integrate until next intersection with Poincare section.
            # Drop start and endpoint since we don't want segments to overlap
            POseg = np.hstack([POseg,solPOg.y[:,1:-1]])    
            POtseg = np.hstack([POtseg,solPOg.t[1:-1]])
            if plt_guess:
                plt.plot(solPOg.y[1],solPOg.y[2],'.-')
        for i in np.arange(1, len(seq)):
            #cNew = self.interpRM(cOld) # Use the return map to advance solution one step forward
            icPOg = self.s2spV(c0[i])
            #cOld = cNew
            # If we use reflection symmetry reduction and the previous solution segment ended out of the fundamental domain
            # then map present segment initial condition out of the fundamental domain
            if self.reduce_refl and not self.isInReflFD(POseg[:,-1]): 
                icPOg = self.refl(icPOg)
            for repeat in range(self.nPoinc[seq[i]]): # For symbols with nPoinc>1 we integrate again after first return Poincare section. This eliminates the need to use r2spV.
                if repeat > 0: # If not the first integration (for nPoinc>1) compute new initial condition
                    icPOg = copy.deepcopy(solPOg.y[:,-1])
                self.int_min_t = solPOg.t[-1]+tminFrac*TpoincReturn                    
                # It would be better to define a new simulator object and pass this to this function than having a hardcoded integrator; this would probably need to be part of PoincareMapper.py
                solPOg = solve_ivp(ds.f_ps, [solPOg.t[-1],solPOg.t[-1]+10*TpoincReturn], icPOg, method='BDF', events = self.poincCondtx, jac=ds.A, atol=atol,rtol=rtol) # Integrate until next intersection with Poincare section.
                # Drop start and endpoint since we don't want segments to overlap
                POseg = np.hstack([POseg,solPOg.y[:,1:-1]])    
                POtseg = np.hstack([POtseg,solPOg.t[1:-1]])
                if plt_guess:
                    plt.plot(solPOg.y[1],solPOg.y[2],'.-')                    
        #if True: #fft_smooth:
                    #tgrid = np.linspace(POtseg[0],POtseg[-1], 1000)
                    #irpPOguess = [sp.interpolate.interp1d(POtseg, dat, kind='nearest') for dat in POseg]
                    #yguess = np.asarray([irp(tgrid) for irp in irpPOguess]) # interpolate in regular grid
                    #yguess = np.asarray([np.fft.irfft(np.fft.rfft(irp(tgrid))[:200],n=tgrid.shape[0]) for irp in irpPOguess]) # interpolate in regular grid and chop high-fourier modes 
        return [POtseg, POseg]
           
    def unstManif1d(self, POlabel, eps0, Npoints, Npoinc, TpoincMean, atol=1e-9, rtol=1e-6):
        # Compute part of unstable manifold of periodic orbit with symbolic sequence POlabel. 
        # It first recomputes the periodic orbit, computes its Jacobian, and then initializes trajectories on the local linear unstable manifold.
        # eps0: minimum distance of initial conditions from periodic point (on the unstable manifold
        # Npoints: number of points to use for the parameterization
        # Npoinc: number of intersection with Poincare section (estimate)
        # TpoincMean: mean time between intersections with Poincare section
        POguess = self.guessMultiVarS(self.equation, self.cycleIC(POlabel),POlabel , TpoincMean, atol=atol, rtol=rtol)
        TpGuess = POguess[0][-1]
        solPO = solve_bvp(self.equation.f_bvp_ps, self.bcPO, POguess[0]/TpGuess, POguess[1], p=[TpGuess], tol=1e-6, verbose=2, max_nodes=30000) # Remove adhoc parameters?
        #Tp = solPO.p[0]
        jac = self.floquet_bvp(solPO, jac_tol=1e-6, timer=True, jac_sample=1)
        eig = np.linalg.eig(jac)
        eil = eig[0]
        ev = np.real(eig[1][:,0])
        self.poincCondtx.__func__.terminal = False
        UMdataTab = []
        epstab = np.linspace(eps0, np.abs(np.real(eil[0]))*eps0, Npoints,endpoint=False)
        for eps in epstab:
            solUM = solve_ivp(self.equation.f_ps, [0,TpoincMean*Npoinc], solPO.y[:,0]+eps*ev, method='BDF', jac=self.equation.A,  events = self.poincCondtx, atol=atol, rtol=rtol) # integrate and record intersections with Poincare section
            UMdataOrig = np.asarray(solUM.y_events)[0] # compute the points on the Poincare section
            if self.reduce_refl:
                UMdata = self.reflRedTraj(UMdataOrig)
            UMdataTab.append(UMdata)
        return UMdataTab

    def sParamUM(self,UMpos, UMneg=np.asarray([])):
        # Parametrize 1d unstable manifold data with manifold learning coordinate s through a call to self.ssp2emb().
        # UMpos: 3d array containing unstable manifold data returned by unstManif1d
        # UMneg: optionally also include data obtained by initializing the manifold with negative sign of epsilon
        # Returns:
        #   sUM: s coordinates of data
        #   sParam: parametrization of state-space points on the manifolds by s.
        UMvisPos = np.reshape(UMpos, [UMpos.shape[0]*UMpos.shape[1],UMpos.shape[2]])
        if UMneg.shape[0] != 0:
            UMvisNeg = np.reshape(UMneg, [UMneg.shape[0]*UMneg.shape[1],UMneg.shape[2]])
            UMvis = np.concatenate((UMvisPos,UMvisNeg))
        else:
            UMvis = UMpos
        sUM = self.ssp2emb(UMvis)
        if sUM.ndim>1: # Check dimensionality of embedding
            sParamUM = self.sParamSP(sUM[:,0], datSP=UMvis, kind='nearest')
        else:
            sParamUM = self.sParamSP(sUM, datSP=UMvis, kind='nearest')
        return [sUM, sParamUM]
