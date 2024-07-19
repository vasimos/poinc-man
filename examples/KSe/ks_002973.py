import  equation 
from  poincman  import  poincman
import  numpy  as np
from  sklearn  import  manifold
import  matplotlib.pyplot  as plt
import time
import scipy as sp
from scipy.integrate import solve_ivp, solve_bvp
from sklearn.neighbors import NearestNeighbors
import  matplotlib  as mpl
import copy
from pickle import load, dump
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition, mark_inset)
from matplotlib.patches import Ellipse
from pathlib import Path

fig_dir = './ks_002973_symRed_LLE_figs/'
data_dir = './ks_002973_symRed_LLE_data/'

Path(fig_dir).mkdir(parents=True, exist_ok=True)
Path(data_dir).mkdir(parents=True, exist_ok=True)

load_int = True
load_transform = False # Loading transform does not work
save_transform = False

LLE = True # use LLE or isomap

save_cycles = True
load_cycles = False
comp_stab = True # Compute stability of cycles (takes substantial time)
viz_long_cycles = False # Vizualize some of the longer cycles (we would need to compute them first)

raster = True # Whether to rasterize heavy plots 
if raster:
    zord = 0 # I use set_rasterization_zorder(1) for axes that I want to rasterize, everything plotted with zorder=0 keyword will be rasterized
else:
    zord = 2

s0 = 3 # Pointsize for scatter plots

plt.ion()

fontsize = 9
fontsize_dual_axis=int(fontsize*0.95)
fontsize_inset=9

linewidth = 1
mpl.rcParams.update({'font.size': fontsize})
plt.rcParams.update({'axes.titlesize': int(1.2*fontsize), 'axes.labelsize': int(1.2*fontsize), 'legend.fontsize': int(0.8*fontsize), 'xtick.labelsize': fontsize,  'ytick.labelsize':fontsize}) # Fine tuning
mpl.rcParams.update({'text.usetex' : True})

inches_per_pt = 1.0/72.27               # Convert pt to inch
fwidth=510*inches_per_pt #510
fheight=0.4*fwidth #190

ks = equation.KursivAS() # Kuramoto-Sivashinsky in antisymmetric subspace

ks.N=16 # Choose number of Fourier modes. 

ks.nu = 0.02973 # 

# Define Poincare section
normalVector = np.zeros(ks.N) # vector normal to Poincare Section
normalVector[1] = 1

pPoinc = np.zeros(ks.N) # any point on the Poincare section

dirPoinc = 1 # direction of crossing Poincare section

pmap= poincman.PoincMap(normalVector, pPoinc, None, direction=dirPoinc) # initialize calculation of points on Poincare section
                                                    # Using none for simulator, since we will use solve_ivp instead of a simulator object

pmap.equation=ks # Setting equation attribute to Kuramoto Sivashinsky is needed by some methods

pmap.poincCondtx.__func__.direction = dirPoinc    # Setting this attribute is needed by solve_ivp and should be done in addition to setting direction in PoincareMapper. 

pmap.reduce_refl = True

x=np.linspace(0,2*np.pi,2*ks.N+2, endpoint=False) # Set up the grid from 0 to 2pi. Number of points is 2*ks.N+2 because we do not count a_0=0, a_{N+1}=0 in our expansion 

u0 = 0.01*np.sin(x) # Start with a simple wave with zero average

ic0 = np.imag(np.fft.rfft(u0))[1:-1] # Drop the zero frequency and highest mode of real FFT (take care that these are zero)

atol=1e-9
rtol=1e-6

# Use solve_ivp event detection to compute Poincare section intersections
sol1 = solve_ivp(ks.f, [0,10], ic0, method='BDF', jac=ks.A, atol=atol, rtol=rtol) # use to obtain initial condition on the attractor after transient

ic = sol1.y[:,-1]

if load_int:

    poincareSectionDataOrig = np.load(data_dir+'/ks_002973_t40000_a2_dir1_tol3.npy')

else:    

    tic = time.perf_counter()

    sol = solve_ivp(ks.f_ps, [0,40000], ic, method='BDF', jac=ks.A,  events = pmap.poincCondtx, atol=atol, rtol=rtol) # integrate and record intersections with Poincare section

    toc = time.perf_counter()

    print(f"Integration time: {toc - tic:0.4f} seconds")

    data = sol.y

    poincareSectionDataOrig = np.asarray(sol.y_events)[0] # compute the points on the Poincare section
    
    np.save(data_dir+'/ks_002973_t40000_a2_dir1_tol3.npy', poincareSectionDataOrig)



plt.figure(figsize=(0.5*fwidth, fheight)) 

plt.plot(poincareSectionDataOrig[:,0], poincareSectionDataOrig[:,2],'.')
plt.xlabel('$a_{1}$')
plt.ylabel('$a_{3}$')

plt.figure(figsize=(0.5*fwidth, fheight)) 

if pmap.reduce_refl:
    poincareSectionData = pmap.reflRedTraj(poincareSectionDataOrig)
else:
    poincareSectionData = poincareSectionDataOrig

plt.plot(poincareSectionData[:,0], poincareSectionData[:,2],'.')
plt.xlabel('$a_{1}$')
plt.ylabel('$a_{3}$')
plt.tight_layout()

plt.savefig(fig_dir+'ks_002973_PS.pdf')

# Plot the return map using the x coordinate
plt.figure()
plt.plot(poincareSectionData[:-1,2], poincareSectionData[1:,2],'.')
plt.xlabel('$x_n$')
plt.ylabel('$x_{n+1}$')

fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(fwidth, 0.75*fheight))

plt.sca(axes[0])
plt.scatter(poincareSectionData[:,1], poincareSectionData[:,2], s=0.05, marker='.')
plt.xlabel('$a_{2}$')
plt.ylabel('$a_{3}$')

plt.sca(axes[1])
plt.scatter(poincareSectionData[:,3], poincareSectionData[:,4], s=0.05, marker='.')
plt.xlabel('$a_{4}$')
plt.ylabel('$a_{5}$')

# Plot the return map using a_i coordinate
plt.sca(axes[2])
plt.scatter(poincareSectionData[:-1,2], poincareSectionData[1:,2], s=0.05, marker='.')
plt.xlabel('$a_{3,n}$')
plt.ylabel('$a_{3,n+1}$')
plt.tight_layout()

plt.savefig(fig_dir+'ks_002973_RM_a3.pdf')


# Manifold learning

# Use locally linear embedding to project Poincare section data to 1-dimension
pmap.demb = 2 # Dimension of embedding

if load_transform: 
    pmap.embedding = load(open(data_dir+'/publ_ks_002973_symRed_LLE_transform_tol3.pkl', 'rb'))
    poincIntr = load(open(data_dir+'/publ_ks_002973_symRed_LLE_fit_tol3.pkl', 'rb'))
    pmap.sign_norm_q, pmap.qmin_0, pmap.qmax_1 = load(open(data_dir+'/publ_ks_002973_symRed_LLE_norm_tol3.pkl', 'rb')) #
else:    
    # Workaround that allows to use nearest neighbors based on radius with LLE; however we are forced to limit n_neighbors to the minimum number of neighbors within certain radius and we have to recompute the connectivity matrix.
    #neigh = NearestNeighbors(radius=0.009)
    # Workable parameters:
    # between eps_neigh=0.0403 and 0.0436
    # around eps_neigh=0.023 to 0.0243
    #for eps_neigh in np.linspace(0.0226,0.0243,10):
        eps_neigh = 0.014
        #eps_neigh = 0.009 # works with modified LLE
        neigh = NearestNeighbors(radius=eps_neigh) # For the decreased tolerance case I had to increase the radius

        neigh.fit(poincareSectionData)
        A = neigh.radius_neighbors_graph(poincareSectionData)

        B=np.sum(A,axis=0)

        knmin = int(B.min()-1)
        knmax = int(B.max()-1)

        print('knmin=',knmin)


        if LLE:
            embedding = manifold.LocallyLinearEmbedding(n_neighbors=knmin, n_components = pmap.demb, random_state=1, reg=0.5e-3 ) #, method='modified', eigen_solver='arpack') #reg=0.3e-3)
        else: # Use isomap
            embedding = manifold.Isomap(n_neighbors=knmin, n_components=pmap.demb)
        if save_transform:
            dump(embedding, open(data_dir+'/publ_ks_002973_symRed_LLE_transform_tol3.pkl', 'wb'), protocol=4) # protocol=4 allows to save files larger than 4GB

        pmap.embedding = embedding # Set embedding attribute in mapper

        poincIntr = pmap.embedding.fit_transform(poincareSectionData)
        pmap.sign_norm_q = [1, 1] # Choose sign for normalization of transform
        pmap.qmin_0 = True
        pmap.qmax_1 = True
        
        if save_transform:
            dump(poincIntr, open(data_dir+'/publ_ks_002973_symRed_LLE_fit_tol3.pkl', 'wb'), protocol=4) # protocol=4 allows to save files larger than 4GB
            dump([pmap.sign_norm_q, pmap.qmin_0, pmap.qmax_1], open(data_dir+'/publ_ks_002973_symRed_LLE_norm_tol3.pkl', 'wb')) # Save transform normalization parameters (transform is saved unormalized)
        
s,r = pmap.computeNormS(poincIntr).transpose()

np.save(data_dir+'/ks_002973_LLE_q1q2_tol3.npy',np.asarray([s,r]))

plt.figure(); plt.plot(s[:-1],s[1:],'.',markersize=0.5)

fig, axes = plt.subplots(nrows=4, ncols=1,figsize=(0.5*fwidth, 4*fheight))

plt.sca(axes[2])
plt.scatter(s,r, c = s-s.min(), cmap='inferno', s=0.2)
plt.xlabel(r'$q_1$')
plt.ylabel(r'$q_2$')

clb=plt.colorbar()
clb.set_label(r'$q_1$')

clb.set_ticks(clb.get_ticks()[::2])
ticklabels = ['{:.2f}'.format(i) for i in clb.get_ticks()+s.min()]
clb.set_ticklabels(ticklabels)

plt.sca(axes[3])
plt.scatter(s,r, c = r-r.min(), cmap='plasma', s=0.2)
plt.xlabel(r'$q_1$')
plt.ylabel(r'$q_2$')

clb=plt.colorbar()
clb.set_label(r'$q_2$')

clb.set_ticks(clb.get_ticks()[::2])
ticklabels = ['{:.2f}'.format(i) for i in clb.get_ticks()+r.min()]
clb.set_ticklabels(ticklabels)

plt.sca(axes[0])
plt.scatter(poincareSectionData[:,0], poincareSectionData[:,2], c = s-s.min(), cmap='inferno', s=0.05, marker='.')
plt.xlabel('$a_{4}$')
plt.ylabel('$a_{5}$')

plt.sca(axes[1])
plt.scatter(poincareSectionData[:,0], poincareSectionData[:,2], c = r-r.min(), cmap='plasma', s=0.05, marker='.')
plt.xlabel('$a_{4}$')
plt.ylabel('$a_{5}$')

plt.tight_layout()

plt.savefig(fig_dir+'ks_002973_rs_LLE.pdf')


# Plot Poincare section, q1,q2 embedding, naive return map on q1
fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(fwidth, 0.75*fheight))
axes[0].set_rasterization_zorder(1)
axes[1].set_rasterization_zorder(1)
axes[2].set_rasterization_zorder(1)

plt.sca(axes[1])
plt.scatter(s,r, c = s-s.min(), cmap='inferno',  edgecolor=None, s=s0, zorder=zord, marker='.')
plt.xlabel(r'$q_1$')
plt.ylabel(r'$q_2$')
plt.text(-0.4,1,r'(b)')

plt.sca(axes[0])
plt.scatter(poincareSectionData[:,0], poincareSectionData[:,2], c = s-s.min(), cmap='inferno', edgecolor=None, s=s0, marker='.', zorder=zord)
plt.xlabel('$a_{1}$')
plt.ylabel('$a_{3}$')
plt.text(-0.3,.71,r'(a)')

plt.sca(axes[2])
#plt.plot(s[:-1],s[1:],'.',markersize=0.5)
plt.scatter(s[:-1], s[1:], c = s[:-1]-s.min(), cmap='inferno',edgecolor=None, s=s0, zorder=zord, marker='.')
plt.xlabel(r'$q_{1,n}$')
plt.ylabel(r'$q_{1,n+1}$')
plt.text(-0.4,1,r'(c)')

plt.subplots_adjust(right=0.8)

cbar_ax = fig.add_axes([0.91, 0.22, 0.025, 0.7])

clb=plt.colorbar(cax=cbar_ax)
clb.set_label(r'$q_1$')

fig.subplots_adjust(top=0.98, bottom=0.2, left=0.08, right=0.88, hspace=.2,wspace=0.4)

plt.savefig(fig_dir+'ks_002973_rs_RM_LLE.pdf', dpi=600)

vc = 0.5145 #0.175 # Cutoff for s, found by inspection
sc = [0.495, vc]
pmap.split_rs(s, r, sc, vc, case=4) 

s_s = pmap.map_s_s(s,r)
s_r = pmap.map_s_r(s,r)
r_s = pmap.map_r_s(s,r)
r_r = pmap.map_r_r(s,r)
print(len(r_r))
S_S = pmap.map_S_S(s, sorted_map_label='sorted_s_s', sorted_label='sorted_s') # Combined map

plt.figure() #(figsize=(0.5*fwidth, fheight))    

plt.plot(pmap.sorted_s_s[:,0], pmap.sorted_s_s[:,1], '.', markersize=2)

sTab = np.linspace(s_r[:,0].min(),s_s[:,0].max(),200)

plt.plot(sTab,sTab)

plt.xlabel(r'$q_{1,n}$')
plt.ylabel(r'$q_{1,n+1}$')

s2sp = pmap.sParamSP(s, poincareSectionData)#, kind='linear') # Generate mapping from s to state-space
r2sp = pmap.rParamSP(r, poincareSectionData)#, kind='linear') # Generate mapping from r to state-space

# Find peaks (critical points of map)
pk = sp.signal.find_peaks(pmap.sorted_s_s[:,1], width=200)

sc = pmap.sorted_s_s[pk[0],0]

# Beginning of domain of s_s is an additional critical point, insert it and set pmap.attribute 
# This works well for admissibility criterio even if there is a finite gap in the map, by 
# contrast to the choice of s_r[:,0].max() as critical point.
pmap.sc = np.insert(sc,-1,s_s[:,0].min())

pmap.invIrv = [s_r[:,0].min(), s_s[:,0].max()] # Invariant interval
pmap.setBaseIrv() # Call this to set value of pmap.baseI

pmap.orient_preserv = [0,2]
pmap.alphabet = range(0,4)
pmap.nPoinc = [2,2,1,1] # number of intersections with Poincare section for single iterate of return map for each of the letters of the alphabet


fig, axes = plt.subplots(nrows=1, ncols=4,figsize=(fwidth, 0.6*fheight))

axes[0].set_rasterization_zorder(1)
axes[1].set_rasterization_zorder(1)
axes[2].set_rasterization_zorder(1)
axes[3].set_rasterization_zorder(1)

plt.sca(axes[3])
plt.scatter(pmap.sorted_s_s[:,0], pmap.sorted_s_s[:,1], marker='.',edgecolor=None, s=s0, zorder=zord)


plt.vlines(pmap.sc,0,1, color='0.5',linestyles='--')

[plt.text(np.mean(pmap.baseI[i:i+2]),0.02,i, horizontalalignment='center') for i in range(len(pmap.baseI)-1)]

plt.xlabel(r'$q_{1,n}$')
plt.ylabel(r'$q_{1,n+1}$')
plt.title(r'$f$')
plt.xticks([0,0.5,1])
plt.text(-0.5,1.1,r'(d)')

plt.sca(axes[0])
plt.scatter(s_s[:,0], s_s[:,1],marker='.', edgecolor=None, s=s0, zorder=zord)

plt.xlabel(r'$q_{1,n}$')
plt.ylabel(r'$q_{1,n+1}$')
plt.title(r'$g$')
plt.xlim(0,1); plt.xticks([0,0.5,1])
plt.text(-0.45,1.1,r'(a)')

plt.sca(axes[1])
plt.scatter(s_r[:,0], s_r[:,1],marker='.', edgecolor=None, s=s0, zorder=zord)

plt.xlabel(r'$q_{1,n}$')
plt.ylabel(r'$q_{2,n+1}$')
plt.title(r'$h$')
plt.text(-0.1,.2,r'(b)')

plt.sca(axes[2])
plt.scatter(r_s[:,0], r_s[:,1],marker='.', edgecolor=None, s=s0, zorder=zord)

plt.xlabel(r'$q_{2,n}$')
plt.ylabel(r'$q_{1,n+1}$')
plt.title(r'$w$')
plt.text(-0.09,.78,r'(c)')

fig.subplots_adjust(top=0.89, bottom=0.23, left=0.07, right=0.97, hspace=.2, wspace=0.6)

plt.savefig(fig_dir+'ks_002973_tree_maps_interval_map.pdf', dpi=600)
# 

plt.figure(figsize=(0.5*fwidth, fheight))    
plt.plot(r_r[:,0], r_r[:,1],'.',markersize=1)

plt.xlabel(r'$q_{2,n}$')
plt.ylabel(r'$q_{2,n+1}$')

# Figure of spliting

fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(0.5*fwidth, 2.*fheight))

plt.sca(axes[0])
axes[0].set_rasterization_zorder(1)

plt.scatter(s[pmap.idx_s_valid],r[pmap.idx_s_valid], c = s[pmap.idx_s_valid]-s[pmap.idx_s_valid].min(), cmap='inferno', edgecolor=None, s=2*s0, marker='.', zorder=zord)

clb1 = plt.colorbar()
clb1.set_label(r'$q_1$')


plt.scatter(s[pmap.idx_r_valid],r[pmap.idx_r_valid], c = r[pmap.idx_r_valid]-r[pmap.idx_r_valid].min(), cmap='winter', edgecolor=None, s=2*s0, marker='.', zorder=zord)

plt.ylim(ymax=1.2)

plt.xlabel(r'$q_1$')
plt.ylabel(r'$q_2$')

plt.text(-0.38,1.2,r'(a)')

# Mark the edges
plt.text(0.19,0.39,r'$E_1$')
plt.text(0.89,0.55,r'$E_2$')
plt.text(0.29,0.05,r'$E_3$')

plt.sca(axes[1])

axes[1].set_rasterization_zorder(1)

plt.scatter(poincareSectionData[pmap.idx_s_valid,0][0], poincareSectionData[pmap.idx_s_valid,2][0], c = s[pmap.idx_s_valid]-s[pmap.idx_s_valid].min(), cmap='inferno', edgecolor=None, s=2*s0, marker='.', zorder=zord)

plt.scatter(poincareSectionData[pmap.idx_r_valid,0][0], poincareSectionData[pmap.idx_r_valid,2][0], c = r[pmap.idx_r_valid]-r[pmap.idx_r_valid].min(), cmap='winter', edgecolor=None, s=2*s0, marker='.', zorder=zord)

clb2 = plt.colorbar()
clb2.set_label(r'$q_2$')


plt.ylim(ymin=0.53)

plt.xlabel(r'$a_1$')
plt.ylabel(r'$a_3$')

plt.text(-0.28,0.71,r'(b)')

fig.subplots_adjust(top=0.98, bottom=0.07, left=0.2, right=0.88, hspace=.2,wspace=0.7)

el=Ellipse(xy=[0.5*s_r[:,0].max(), 0.5*(1+r[np.argmin(np.abs(s-s_r[:,0].max()))])], width=np.sqrt((1-r[np.argmin(np.abs(s-s_r[:,0].max()))])**2+s_r[:,0].max()**2)+0.02, height=0.05, angle=-np.arctan((1-r[np.argmin(np.abs(s-s_r[:,0].max()))])/s_r[:,0].max())*360./2./np.pi, facecolor='none', edgecolor='r', linestyle='dashed')

axes[0].add_artist(el)

# Draw inset
plt.sca(axes[0])
ax1 = plt.axes([0,0,1,1],label='ax1')
ip = InsetPosition(axes[0],[0.23,0.59,0.54,0.31])
ax1.set_axes_locator(ip)
ax1.set_rasterization_zorder(1)
plt.scatter(s[pmap.idx_s_valid],r[pmap.idx_s_valid], c = s[pmap.idx_s_valid]-s[pmap.idx_s_valid].min(), cmap='inferno', marker='.', edgecolor=None, s=2*s0, zorder=zord)
plt.scatter(s[pmap.idx_r_valid],r[pmap.idx_r_valid], c = r[pmap.idx_r_valid]-r[pmap.idx_r_valid].min(), cmap='winter', edgecolor=None, s=2*s0, marker='.', zorder=zord)
ax1.set_xlim(0.45,0.53)
ax1.set_ylim(0.1,0.21)
ax1.yaxis.tick_right()
ax1.xaxis.tick_top()
ax1.tick_params(pad=2, axis='y')
ax1.tick_params(pad=0, axis='x')
mark_inset(axes[0], ax1, loc1=3, loc2=4, fc="none", ec='0.5')
# Mark the vertex
arrowprops = dict(
    arrowstyle="->",
    connectionstyle="angle,angleA=0,angleB=90,rad=10")
ax1.annotate(r'$V$', xy=(vc, 0.178), xytext=(0.485, 0.115), arrowprops=arrowprops)

# Draw inset
plt.sca(axes[1])
ax2 = plt.axes([0,0,1,1],label='ax2')
ip = InsetPosition(axes[1],[0.2,0.12,0.6,0.35])
ax2.set_axes_locator(ip)
ax2.set_rasterization_zorder(1)
ax2.scatter(poincareSectionData[pmap.idx_s_valid,0][0], poincareSectionData[pmap.idx_s_valid,2][0], c = s[pmap.idx_s_valid]-s[pmap.idx_s_valid].min(), cmap='inferno', edgecolor=None, s=2*s0, marker='.', zorder=zord)
ax2.scatter(poincareSectionData[pmap.idx_r_valid,0][0], poincareSectionData[pmap.idx_r_valid,2][0], c = r[pmap.idx_r_valid]-r[pmap.idx_r_valid].min(), facecolor = r[pmap.idx_r_valid]-r[pmap.idx_r_valid].min(), cmap='winter', edgecolor=None, s=2*s0, marker='.', zorder=zord)
ax2.set_ylim(0.647,0.658)
ax2.set_xlim(-0.105,-0.083)
ax2.yaxis.tick_right()
ax2.tick_params(pad=2, axis='y')
ax2.tick_params(pad=0, axis='x')
mark_inset(axes[1], ax2, loc1=1, loc2=2, fc="none", ec='0.5')


plt.savefig(fig_dir+'ks_002973_rs_split.pdf', dpi=600)

            
sTab = np.linspace(s_r[:,0].min(),s_s[:,0].max(),50)

# Compute inverse map
pmap.inv_map = pmap.inverse_map()

itinSc = list(map(pmap.itinSingle, pmap.sc, np.full_like(pmap.sc,30,dtype=int)))

knead = [pmap.s2symb(it[1:]) for it in itinSc]

TpoincMean = 0.44 # Mean time of return to Poincare section. This should be determined by data, but an upper bound of return times works well enough


if load_cycles:
    cyclesDF = pd.read_pickle(data_dir+'/ks_cycles_nu_002973.pkl')
else:
    #
    
    POlabel = [3,3] # 
    
    cyc = pmap.findCycle(POlabel,1, tol=1e-6, atol=atol, rtol=rtol, jac_sample=1, init_only=True, stability = comp_stab, tminFrac=1e-3) # 
    cyclesDF = pd.DataFrame(columns= pmap.col) # Initialize DataFrame
    
    pmap.zero_admis=False
    
    rejectedCyc=[]
    
    Ncyc = 2
    
    for i in range (1,Ncyc+1):
        admCyc = pmap.primeCyclesAdmis(i,knead)
        for j in range(0,len(admCyc)):
            print ('Cycle ',j+1,'/',len(admCyc),' of length ', i)
            rep=1 # By default look for asymmetric cycles
            guess = pmap.guessMultiVarS(pmap.equation, pmap.cycleIC(admCyc[j]), admCyc[j], TpoincMean, plt_guess=True)
            if np.linalg.norm(np.abs(guess[1][:,0]-guess[1][:,-1])) > 10*np.linalg.norm(np.abs(guess[1][:,-1]-guess[1][:,-2])): # If there is a large jump between the last point and the first, the cycle is probably self-dual
                print('Try r=2')
                rep = 2 # try a second repeat. This should take care of self-dual orbits but also helps refine the initial guess by restricting the length of intervals generated by cycleIrvs.
                guess = pmap.guessMultiVarS(pmap.equation, pmap.cycleIC(admCyc[j]*rep), admCyc[j]*rep, TpoincMean, plt_guess=True)
            cyc = pmap.findCycle(admCyc[j]*rep, TpoincMean, tol=1e-6, atol=atol, rtol = rtol, jac_tol=1e-6, max_nodes=30000,jac_sample=2, stability = comp_stab)
            if cyc[-1] in [0,11]: # Check that no error occured (0), also accept error code 11 (warning about sub-interval not bounding solution).
                cycArr = np.asarray([cyc],dtype=object)
                cycDF = pd.DataFrame(cycArr,columns=pmap.col, index=[pmap.list2str(cyc[0])])
                cyclesDF = pd.concat([cyclesDF, cycDF])#, ignore_index=True)
                if save_cycles:
                    cyclesDF.to_pickle(data_dir+'/ks_cycles_nu_002973.pkl') # Save after each cycle computation
            else:
                rejectedCyc.append(pmap.list2str(cyc[0]))
                if save_cycles:
                    np.save(data_dir+'/rejectedCyc.npy')

# Position of marginal eigenvalue for each cycle
margLambdaPos = [np.argmin(np.abs(cyclesDF['multipliers'][i]-1)) for i in range(cyclesDF.shape[0])]

print('Maximum deviation from unity in marginal eigenvalue=',np.max(np.abs([cyclesDF['multipliers'][i][margLambdaPos[i]]-1 for i in range(cyclesDF.shape[0])])) )

print('Max number of expanding directions is:',  np.max(margLambdaPos)) # This is based on the fact that eigenvalues are ordered, but are they order by magnitude for complex eigenvalue case?

if np.max(margLambdaPos)>1:
    print('Position of max number of expanding directions:', np.where(margLambdaPos == max(margLambdaPos)))

def cyclesFoundLen(n):
   return cyclesDF[cyclesDF.length==n].index

if viz_long_cycles:
    fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(fwidth, 0.75*fheight))

    plt.sca(axes[0])
    axes[0].set_rasterization_zorder(3)

    plt.plot(pmap.sorted_s_s[:,0], pmap.sorted_s_s[:,1], markersize=0.03, marker='.', zorder=1)

    sTab = np.linspace(s_r[:,0].min(),s_s[:,0].max(),200)

    plt.plot(sTab,sTab, zorder=0, c='k')

    plt.xlabel(r'$q_{1,n}$')
    plt.ylabel(r'$q_{1,n+1}$')

    plt.text(-0.35,1,r'(a)')

    pmap.plotItinS(pmap.findCycleRM(pmap.str2list('13231323')), c='r',cc='r', s=0, periodic=True, zorder=2)

    pmap.plotItinS(pmap.findCycleRM(pmap.str2list('03231323')), c='g',cc='g', s=0, linestyles = 'dashed', periodic=True, zorder=2)

    plt.sca(axes[1])

    plt.plot(cyclesDF.loc['13231323']['data'][2], cyclesDF.loc['13231323']['data'][3],'r-')
    plt.xlabel(r'$a_3$')
    plt.ylabel(r'$a_4$')
    plt.text(-1.9,-0.7,r'(b)')

    plt.sca(axes[2])

    plt.plot(cyclesDF.loc['03231323']['data'][2], cyclesDF.loc['03231323']['data'][3],'g--')
    plt.xlabel(r'$a_3$')
    plt.ylabel(r'$a_4$')
    plt.text(-1.9,-0.7,r'(c)')

    fig.subplots_adjust(top=0.98, bottom=0.2, left=0.07, right=0.97, hspace=.2, wspace=0.4)

    plt.savefig(fig_dir+'ks_002973_cycles.pdf', dpi=600)


# Find unstable manifold

eps0=1e-4
Npoinc=12
 

UMdataTab3pos = pmap.unstManif1d(pmap.str2list('33'), 1e-5, 50, 14, TpoincMean)
UMdataTab3neg = pmap.unstManif1d(pmap.str2list('33'), -1e-5, 50, 14, TpoincMean)

UMdataTab23pos = pmap.unstManif1d(pmap.str2list('23'), 1e-5, 50, 22, TpoincMean)
UMdataTab23neg = pmap.unstManif1d(pmap.str2list('23'), -1e-5, 50, 22, TpoincMean)


sUM23, sParamUM23 = pmap.sParamUM(np.asarray(UMdataTab23neg)[:,1:16:2], UMneg = np.asarray(UMdataTab23pos)[:,1:18:2])
sUM3, sParamUM3 = pmap.sParamUM(np.asarray(UMdataTab3neg)[:,1:12:2], UMneg = np.asarray(UMdataTab3pos)[:,1:12:2])

fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(0.5*fwidth, 0.75*fheight))

axes.set_rasterization_zorder(3)

plt.scatter(poincareSectionData[:,0], poincareSectionData[:,2], c = '0.6', edgecolor=None, s=4*s0, marker='.', zorder=zord)
plt.xlabel('$a_{1}$')
plt.ylabel('$a_{3}$')


sTab=np.linspace(sUM23[:,0].min(),sUM23[:,0].max(),200)
plt.plot(sParamUM23[0](sTab),sParamUM23[2](sTab),'-', linewidth=1, zorder=zord+1)

sTab=np.linspace(sUM3[:,0].min(),sUM3[:,0].max(),200)
plt.plot(sParamUM3[0](sTab),sParamUM3[2](sTab),'-', linewidth=1, zorder=zord+1)

plt.scatter(pmap.reflRedTraj(cyclesDF.loc['23']['Poinc. points'])[:,0],pmap.reflRedTraj(cyclesDF.loc['23']['Poinc. points'])[:,2],c='tab:blue', s=10, marker='v', zorder=zord+2)
plt.scatter(pmap.reflRedTraj(cyclesDF.loc['33']['Poinc. points'])[:,0],pmap.reflRedTraj(cyclesDF.loc['33']['Poinc. points'])[:,2],c='tab:orange', s=10, marker='s', zorder=zord+2) 

plt.xlabel(r'$a_1$')
plt.ylabel(r'$a_3$')

plt.title(r'$\nu=0.02973$')

fig.subplots_adjust(top=0.91, bottom=0.18, left=0.17, right=0.98, hspace=.2,wspace=0.4)

plt.text(-0.25,0.72,r'(b)')

plt.text(pmap.reflRedTraj(cyclesDF.loc['33']['Poinc. points'])[0,0],pmap.reflRedTraj(cyclesDF.loc['33']['Poinc. points'])[0,2]+0.006,r'$\overline{3}$',ha='center')
plt.text(pmap.reflRedTraj(cyclesDF.loc['23']['Poinc. points'])[1,0],pmap.reflRedTraj(cyclesDF.loc['23']['Poinc. points'])[1,2]+0.006,r'$\overline{23}$',ha='center')

plt.savefig(fig_dir+'ks_002973_UM.pdf', dpi=600)


