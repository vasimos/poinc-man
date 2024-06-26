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
from pathlib import Path

fig_dir = 'ks_00299_figs/'
data_dir = 'ks_00299_data/'
Path(fig_dir).mkdir(parents=True, exist_ok=True)
Path(data_dir).mkdir(parents=True, exist_ok=True)

load_int = True
load_transform = False
save_transform = False

LLE = True # use LLE or isomap

save_cycles = True
load_cycles = False

raster = True # Whether to rasterize heavy plots 
if raster:
    zord = 0 # I use set_rasterization_zorder(1) for axes that I want to rasterize, everything plotted with zorder=0 keyword will be rasterized
else:
    zord = 2

plt.ion()

fontsize = 9
fontsize_dual_axis=int(fontsize*0.95)
fontsize_inset=8

linewidth = 1
mpl.rcParams.update({'font.size': fontsize})
plt.rcParams.update({'axes.titlesize': int(1.2*fontsize), 'axes.labelsize': int(1.2*fontsize), 'legend.fontsize': int(0.8*fontsize), 'xtick.labelsize': fontsize,  'ytick.labelsize':fontsize}) # Fine tuning
mpl.rcParams.update({'text.usetex' : True})

inches_per_pt = 1.0/72.27               # Convert pt to inch
fwidth=510*inches_per_pt #510
fheight=0.4*fwidth #190

ks = equation.KursivAS()

ks.N=16 # Choose number of Fourier modes. 

ks.nu = 0.0299 # Bimodal map with 3 POs

# Define Poincare section
normalVector = np.zeros(ks.N) # vector normal to Poincare Section
normalVector[1] = 1

pPoinc = np.zeros(ks.N) # any point on the Poincare section

dirPoinc = 1 # direction of crossing Poincare section

pmap = poincman.PoincMap(normalVector, pPoinc, None, direction=dirPoinc) # initialize calculation of points on Poincare section
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
sol1 = solve_ivp(ks.f, [0,10], ic0, method='BDF', jac=ks.A) # use to obtain initial condition on the attractor after transient

ic = sol1.y[:,-1]

if load_int:

    poincareSectionDataOrig = np.load(data_dir+'t4000_tol3.npy')

else:    

    tic = time.perf_counter()

    sol = solve_ivp(ks.f_ps, [0,4000], ic, method='BDF', jac=ks.A,  events = pmap.poincCondtx,atol=atol, rtol=rtol) # integrate and record intersections with Poincare section
        
    toc = time.perf_counter()

    print(f"Integration time: {toc - tic:0.4f} seconds")

    data = sol.y

    Et = np.sum(0.5*sol.y**2,axis=0) # Energy as a function of time
    E = np.trapz(Et,x=sol.t)# integrated observable for energy
    Eave = E/(sol.t[-1]-sol.t[0])

    
    # Study convergence of mean energy
    Etab = [np.trapz(Et[0:i],x=sol.t[0:i]) for i in range(500,Et.shape[0],100)]

    EaveTab = Etab/sol.t[500:Et.shape[0]:100]

    plt.figure(); plt.plot(sol.t[500:Et.shape[0]:100], EaveTab)

    poincareSectionDataOrig = np.asarray(sol.y_events)[0] # compute the points on the Poincare section

    np.save(data_dir+'t4000_tol3.npy',poincareSectionDataOrig)

#utab = np.apply_along_axis(np.fft.irfft, 0, np.vstack([np.zeros(sol.t.shape),1j*sol.y, np.zeros(sol.t.shape)])) # Go back to u(x,t)

#plt.figure(); plt.plot(sol.t,np.sum(utab,axis=0))
#plt.figure(); plt.plot(sol.t,sol.y[-1])

#simulator.tredimplot() # Plot the attractor

#plt.plot(poincareSectionData[:,0], poincareSectionData[:,1], poincareSectionData[:,2],'.') # Plot the points on the section together with the attractor

plt.figure(figsize=(0.5*fwidth, fheight)) 

plt.plot(poincareSectionDataOrig[:,2], poincareSectionDataOrig[:,3],'.', markersize=1)

plt.xlabel(r'$a_3$')
plt.ylabel(r'$a_4$')

plt.tight_layout()

plt.savefig(fig_dir+'ks_00299_full_proj.pdf')

plt.figure(figsize=(0.5*fwidth, fheight)) 

if pmap.reduce_refl:
    poincareSectionData = pmap.reflRedTraj(poincareSectionDataOrig)
else:
    poincareSectionData = poincareSectionDataOrig

plt.plot(poincareSectionData[:,2], poincareSectionData[:,3],'.', markersize=1)

plt.xlabel(r'$a_3$')
plt.ylabel(r'$a_4$')

plt.tight_layout()

plt.savefig(fig_dir+'ks_00299_FD_proj.pdf')

# Plot the return map using the x coordinate
plt.figure()
plt.plot(poincareSectionData[:-1,1], poincareSectionData[1:,1],'.')
plt.xlabel('$x_n$')
plt.ylabel('$x_{n+1}$')


# Manifold learning

# Use locally linear embedding to project Poincare section data to 1-dimension

# Workaround that allows to use nearest neighbors based on radius with LLE; however we are forced to limit n_neighbors to the minimum number of neighbors within certain radius and we have to recompute the connectivity matrix.
neigh = NearestNeighbors(radius=0.014)
neigh.fit(poincareSectionData)
A = neigh.radius_neighbors_graph(poincareSectionData)

B=np.sum(A,axis=0)

knmin = int(B.min()-1)
knmax = int(B.max()-1)

print(knmin)

if LLE:
    embedding = manifold.LocallyLinearEmbedding(n_neighbors=knmin, n_components =1, random_state=1)
else: # Use isomap
    embedding = manifold.Isomap(n_neighbors=knmin, n_components=1)

pmap.embedding = embedding # Set embedding attribute in mapper

s = pmap.embedding.fit_transform(poincareSectionData)[:,0]#.transpose()

pmap.sign_norm_q = 1 # Choose sign for normalization of transform
pmap.qmin_0 = True
pmap.qmax_1 = True

s = pmap.computeNormS(s)

np.save('./ks_00299_data/ks_00299_LLE.npy',np.asarray([s]))

pmap.s = s

#pmap.embedding = poincIso # Set embedding attribute in mapper

#poincIntr = poincIso.fit_transform(poincareSectionData)

#s=poincIntr[:,0]

# Construct interpolating function for return map
irpRM = pmap.interp1dRM(s, sorted_map_label='sorted_s_s', sorted_label='sorted_s')

pk = sp.signal.find_peaks(pmap.sorted_s_s[:,1], width=100) # p.sorted_s_s is generated as a byproduct of calling p.interp1dRM

#if pk[0].shape[0]==0: # check for maximum, if none is found invert the map (not required, for uniform visualization only) and redo the interpolating function
    #s=-s
    #irpRM = pmap.interp1dRM(s, sorted_map_label='sorted_s_s', sorted_label='sorted_s') # Recompute
    #pk = sp.signal.find_peaks(pmap.sorted_s_s[:,1], width=50) # recompute
    #pmap.sign_norm_q = -1
    #print('inverted s')

s2sp = pmap.sParamSP(s, poincareSectionData, kind='linear') # Generate mapping from s to state-space

pmap.sc = pmap.sorted_s_s[:,0][pk[0]]

itinSc = list(map(pmap.itinSingle, pmap.sc, np.full_like(pmap.sc,30,dtype=int)))
knead = [pmap.s2symb(it[1:]) for it in itinSc]

#icC = pmap.s2spV(pmap.sc[0]) # Initial condition for integration of critical point
#icC = poincareSectionData[pk[0]][0]
#solC = solve_ivp(ks.f_ps, [0,30], icC, method='BDF', jac=ks.A,  events = pmap.poincCondtx) # integrate and record intersections with Poincare section

#itinScInt = pmap.normS(poincIso.transform(pmap.reflRedTraj(solC.y_events[0])))[:,0]
#itinScInt = np.insert(itinScInt,0,pmap.sc[0])
#kneadInt = pmap.s2symb(itinScInt[1:21])

pmap.invIrv = [pmap.sorted_s_s[0,0], pmap.sorted_s_s[-1,0]] # Invariant interval
pmap.setBaseIrv() # Call this to set value of pmap.baseI

pmap.orient_preserv = [0]
pmap.alphabet = range(0,2)
pmap.nPoinc = [1 for i in pmap.alphabet]

# Compute inverse map
pmap.inv_map = pmap.inverse_map()

#fill_value=(pmap.interpRM(pmap.sc),pmap.interpRM(pmap.interpRM(pmap.sc)))

#partitionSymb = partSymbDepth(n,cp) # Generate all possible symbols to length n
#partitionSymbStr = np.asarray([''.join(map(str, xx)) for xx in partitionSymb]) # Convert to array of strings
    



pmap.zero_admis = False

#poincS , err = manifold.locally_linear_embedding(poincareSectionData, n_neighbors = 36,  n_components =1, random_state=1, method='modified')

#s = poincS[:,0] # Choose first coordinate (this flattens array if we already computed 1D manifold)


# Plot the new return map
plt.figure()
#ax = fig.add_subplot(111)
#plt.plot(s[:-1], s[1:],'.') # s_n, s_{n+1}
plt.scatter(s[:-1], s[1:], c = s[:-1]-s.min(), cmap='viridis', s=0.2)
plt.xlabel('$s_n$')
plt.ylabel('$s_{n+1}$')

sTab = np.linspace(s.min(),s.max(),50)

#plt.plot(sTab, pmap.interpRM(sTab))

# Look for fixed points of the map

mvals = [pmap.itinSingle(sval,10)[-1]-sval for sval in pmap.sorted_s]

plt.figure(); plt.plot(pmap.sorted_s, mvals, '-', markersize=2)

plt.hlines(0,s.min(),s.max())

nFixed = 1

fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(fwidth, fheight))

plt.sca(axes[0])
plt.scatter(poincareSectionData[:,2], poincareSectionData[:,3], c = s-s.min(), cmap='viridis', s=0.2)
plt.xlabel('$a_{3}$')
plt.ylabel('$a_{4}$')

# Plot the return map using a_i coordinate
plt.sca(axes[1])
plt.scatter(poincareSectionData[:-1,3], poincareSectionData[1:,3],  s=0.2)
plt.xlabel('$a_{4,n}$')
plt.ylabel('$a_{4,n+1}$')

# Plot the new return map
plt.sca(axes[2])
plt.scatter(s[:-1], s[1:], c = s[:-1]-s.min(), cmap='viridis', s=0.2)

plt.plot(sTab,sTab)

plt.xlabel('$s_n$')
plt.ylabel('$s_{n+1}$')

plt.tight_layout()

plt.savefig(fig_dir+'ks_00299_s.pdf')


# Plot Poincare section, q1,q2 embedding, naive return map on q1
fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(fwidth, 0.75*fheight))
axes[0].set_rasterization_zorder(1)
axes[1].set_rasterization_zorder(1)
axes[2].set_rasterization_zorder(1)

#plt.sca(axes[1])
#plt.scatter(s,r, c = s-s.min(), cmap='inferno', s=0.03, zorder=zord)
#plt.xlabel(r'$q_1$')
#plt.ylabel(r'$q_2$')
#plt.text(-0.4,1,r'(b)')

plt.sca(axes[0])
plt.scatter(poincareSectionDataOrig[:,0], poincareSectionDataOrig[:,2], c = s-s.min(), cmap='inferno', s=0.3, marker='o', edgecolor=None, zorder=zord)
plt.xlabel('$a_{1}$')
plt.ylabel('$a_{3}$')
plt.text(-0.28,.685,r'(a)')

plt.sca(axes[1])
plt.scatter(poincareSectionData[:,0], poincareSectionData[:,2], c = s-s.min(), cmap='inferno', marker='o', edgecolor=None, s=0.3, zorder=zord)
plt.xlabel('$a_{1}$')
plt.ylabel('$a_{3}$')
plt.xlim(xmax=0.07)
plt.text(-0.2,.685,r'(b)')

plt.sca(axes[2])
#plt.plot(s[:-1],s[1:],'.',markersize=0.5)
plt.scatter(s[:-1], s[1:], c = s[:-1]-s.min(), cmap='inferno', marker='o', edgecolor=None, s=0.3, zorder=zord)
plt.xlabel(r'$q_{1,n}$')
plt.ylabel(r'$q_{1,n+1}$')
plt.text(-0.4,1,r'(c)')

plt.text(0.02,0.7,r'0')
plt.text(0.85,0.15,r'1')

plt.subplots_adjust(right=0.8)

cbar_ax = fig.add_axes([0.91, 0.22, 0.025, 0.7])

clb=plt.colorbar(cax=cbar_ax)
clb.set_label(r'$q_1$')

##newticks = np.linspace(clb.get_clim()[0], clb.get_clim()[-1],5)
##clb.set_ticks(newticks)
#clb.set_ticks(clb.get_ticks()[::2])
#ticklabels = ['{:.2f}'.format(i) for i in clb.get_ticks()+s.min()]
#clb.set_ticklabels(ticklabels)

fig.subplots_adjust(top=0.98, bottom=0.2, left=0.08, right=0.88, hspace=.2,wspace=0.4)

# Insets
axIns = plt.axes([0,0,1,1],label='axIns')
ip = InsetPosition(axes[2],[0.38,0.6,0.4,0.33])
axIns.set_axes_locator(ip)
axIns.set_rasterization_zorder(1)

axIns.scatter(s[:-1], s[1:], c = s[:-1]-s.min(), marker='o', edgecolor=None, s=0.3,  zorder=zord, cmap='inferno')
axIns.set_xlim(pmap.sc-0.017, pmap.sc+0.017)
axIns.set_ylim(0.95, 1.02)
axIns.tick_params(left=False,labelleft=False, right=True,labelright=True,labelsize=fontsize_inset)

mark_inset(axes[2], axIns, loc1=1, loc2=3, fc="none", ec='0.5')


#
axIns2 = plt.axes([0,0,1,1],label='axIns2')
ip = InsetPosition(axes[1],[0.38,0.13,0.35,0.31])
axIns2.set_axes_locator(ip)
axIns2.set_rasterization_zorder(1)

axIns2.scatter(poincareSectionData[:,0], poincareSectionData[:,2], c = s-s.min(), cmap='inferno', marker='o', edgecolor=None, s=0.3, zorder=zord)
axIns2.set_xlim(-0.151, -0.135)
axIns2.set_ylim(0.632, 0.64)
axIns2.tick_params(left=False,labelleft=False, right=True,labelright=True,labelsize=fontsize_inset)


mark_inset(axes[1], axIns2, loc1=1, loc2=2, fc="none", ec='0.5')



plt.savefig(fig_dir+'ks_00299_Sect_RM.pdf', dpi=600)

#def partition_init(sc,):
    

#idum=0

#sGuess = [np.mean(pmap.itinSingle(pmap.sc[-1],1))] # This should give a guess for cycle 1

#sPOtab = []

#for s0 in sGuess:

    #sPOtab = np.append(sPOtab, sp.optimize.newton(pmap.fixedPointRM,s0, args=(nFixed,)) )# Find PO of the return map, using a guess
    

#POlabel = pmap.primeCyclesAdmis(2,knead)[0]

TpoincMean = 0.44 # Mean time of return to Poincare section. This should be determined by data, but an upper bound works well enough


if load_cycles:
    cyclesDF = pd.read_pickle(data_dir+'/ks_cycles_nu_00299_symRed_tol3_dirPoinc1.pkl')
else:


    POlabel = pmap.str2list('0101111111')

    print(POlabel, 'admiss', pmap.is_admis(POlabel,knead))

    comp_stab = False
    cyc = pmap.findCycle(POlabel,1, tol=1e-6, atol=1e-3,jac_sample=1, init_only=True, stability = comp_stab) # 
    cyclesDF = pd.DataFrame(columns= pmap.col) # Initialize DataFrame

    #cycArr = np.asarray([cyc],dtype=object)
    #cycDF = pd.DataFrame(cycArr,columns=pmap.col, index=[pmap.list2str(cyc[0])])
    #cyclesDF = cyclesDF.append(cycDF)

    rejectedCyc=[]

    Ncyc = 12
    pmap.seqChaotic = pmap.s2symb(s)

    for i in range (1,Ncyc+1):
        admCyc = pmap.primeCyclesAdmis(i,knead)
        for j in range(0,len(admCyc)):
            print ('Cycle ',j+1,'/',len(admCyc),' of length ', i)
            rep=1 # By default look for asymmetric cycles
            #sPOrm=pmap.findCycleRM(admCyc[j])        
            guess = pmap.guessMultiVarS(pmap.equation, pmap.cycleIC(admCyc[j]), admCyc[j], TpoincMean)
            #resid1 = np.linalg.norm(np.diff(guess[1], axis=1, append = np.asarray([guess[1][:,0]]).transpose()))/guess[1].shape[1]# The residual looks for continuity of the loop with periodic boundary condition
            ## Test if the residual for self-dual guess is significantly reduced
            ## Do not generate new guess from return map, just repeat the list of points list(sPOrm)*2
            #guess = pmap.guessMultiVarS(pmap.equation, list(sPOrm)*2, admCyc[j]*2, TpoincMean)
            #resid2 = np.linalg.norm(np.diff(guess[1], axis=1, append = np.asarray([guess[1][:,0]]).transpose()))/guess[1].shape[1]# The residual looks for continuity of the loop with periodic boundary condition
            #print('Residual=', resid1/resid2)
            #if resid1/resid2 > 3.: # If guess for self-dual is much better than guess for asymmetric
            if np.linalg.norm(np.abs(guess[1][:,0]-guess[1][:,-1])) > 10*np.linalg.norm(np.abs(guess[1][:,-1]-guess[1][:,-2])): # If there is a large jump between the last point and the first, the cycle is probably self-dual
                print('Try r=2')
                rep = 2 # try a second repeat. This should take care of self-dual orbits but also helps refine the initial guess by restricting the length of intervals generated by cycleIrvs.
            cyc = pmap.findCycle(admCyc[j]*rep, TpoincMean, tol=1e-6, atol=atol, rtol = rtol, jac_tol=1e-6, max_nodes=30000,jac_sample=2, stability = comp_stab, guess_from_RM_shadowing=False)
            if cyc[-1] in [0,11]: # Check that no error occured (0), also accept error code 11 (warning about sub-interval not bounding solution).
                cycArr = np.asarray([cyc],dtype=object)
                cycDF = pd.DataFrame(cycArr,columns=pmap.col, index=[pmap.list2str(cyc[0])])
                cyclesDF = pd.concat([cyclesDF, cycDF])#, ignore_index=True)
                if save_cycles:
                    cyclesDF.to_pickle(data_dir+'ks_cycles_nu_00299_symRed_tol3_dirPoinc1.pkl') # Save after each cycle computation
            else:
                rejectedCyc.append(pmap.list2str(cyc[0]))


brk

# Find unstable manifold

POlabel = [3,3]
eps0=1e-4
Npoinc=12


UMdataTab3pos = pmap.unstManif1d(pmap.str2list('11'), 1e-3, 50, 14, TpoincMean)
UMdataTab3neg = pmap.unstManif1d(pmap.str2list('11'), -1e-3, 50, 14, TpoincMean)

UMdataTab23pos = pmap.unstManif1d(pmap.str2list('01'), 1e-3, 50, 22, TpoincMean)
UMdataTab23neg = pmap.unstManif1d(pmap.str2list('01'), -1e-3, 50, 22, TpoincMean)


sUM23, sParamUM23 = pmap.sParamUM(np.asarray(UMdataTab23neg)[:,1:12:2], UMneg = np.asarray(UMdataTab23pos)[:,1:12:2])
sUM3, sParamUM3 = pmap.sParamUM(np.asarray(UMdataTab3neg)[:,1:8:2], UMneg = np.asarray(UMdataTab3pos)[:,1:8:2])

fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(0.5*fwidth, 0.75*fheight))

#plt.sca(axes[0])
axes.set_rasterization_zorder(3)

plt.scatter(poincareSectionData[:,0], poincareSectionData[:,2], c = '0.6', edgecolor=None, s=12, marker='.', zorder=zord)
plt.xlabel('$a_{1}$')
plt.ylabel('$a_{3}$')


sTab23=np.linspace(sUM23[:,0].min(),sUM23[:,0].max(),200)
plt.plot(sParamUM23[0](sTab23),sParamUM23[2](sTab23),'-', linewidth=1, zorder=zord+1)

sTab3=np.linspace(sUM3[:,0].min(),sUM3[:,0].max(),200)
plt.plot(sParamUM3[0](sTab3),sParamUM3[2](sTab3),'-', linewidth=1, zorder=zord+1)

plt.scatter(pmap.reflRedTraj(cyclesDF.loc['01']['Poinc. points'])[:,0],pmap.reflRedTraj(cyclesDF.loc['01']['Poinc. points'])[:,2],c='tab:blue', s=10, marker='v', zorder=zord+2)
plt.scatter(pmap.reflRedTraj(cyclesDF.loc['11']['Poinc. points'])[:,0],pmap.reflRedTraj(cyclesDF.loc['11']['Poinc. points'])[:,2],c='tab:orange', s=10, marker='s', zorder=zord+2) 

plt.xlabel(r'$a_1$')
plt.ylabel(r'$a_3$')

plt.xlim(xmax=0.04)

plt.title(r'$\nu=0.0299$')

fig.subplots_adjust(top=0.91, bottom=0.18, left=0.17, right=0.98, hspace=.2,wspace=0.4)

plt.text(-0.2,0.69,r'(a)')

plt.text(pmap.reflRedTraj(cyclesDF.loc['11']['Poinc. points'])[0,0],pmap.reflRedTraj(cyclesDF.loc['11']['Poinc. points'])[0,2]+0.002,r'$\overline{1}$',ha='center')
plt.text(pmap.reflRedTraj(cyclesDF.loc['01']['Poinc. points'])[1,0],pmap.reflRedTraj(cyclesDF.loc['01']['Poinc. points'])[1,2]+0.003,r'$\overline{01}$',ha='center')

# Inset
axIns = plt.axes([0,0,1,1],label='axIns')
ip = InsetPosition(axes,[0.59,0.14,0.4,0.33])
axIns.set_axes_locator(ip)
axIns.set_rasterization_zorder(1)

axIns.scatter(poincareSectionData[:,0], poincareSectionData[:,2], c = '0.6', edgecolor=None, s=30, marker='.', zorder=zord)
axIns.plot(sParamUM23[0](sTab23),sParamUM23[2](sTab23),'-', linewidth=1, zorder=zord+1)
axIns.plot(sParamUM3[0](sTab3),sParamUM3[2](sTab3),'-', linewidth=1, zorder=zord+1)
plt.scatter(pmap.reflRedTraj(cyclesDF.loc['01']['Poinc. points'])[:,0],pmap.reflRedTraj(cyclesDF.loc['01']['Poinc. points'])[:,2],c='tab:blue', s=14, marker='v', zorder=zord+2)
axIns.set_xlim(-0.012, -0.002)
axIns.set_ylim(0.672, 0.6745)


axIns.tick_params(left=True,labelleft=True, right=False,labelright=False,labelsize=fontsize_inset)

mark_inset(axes, axIns, loc1=1, loc2=2, fc="none", ec='0.5')



plt.savefig(fig_dir+'ks_00299_UM.pdf', dpi=600)


