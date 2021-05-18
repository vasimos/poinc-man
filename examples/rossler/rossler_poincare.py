import  equation 
from  poincman import  poincman
import  numpy  as np
from  sklearn  import  manifold
import  matplotlib  as mpl
import  matplotlib.pyplot  as plt
import time
import scipy as sp
from scipy.integrate import solve_ivp, solve_bvp
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition, mark_inset)

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

inset = False # Plot inset (True/False). It takes a lot time to generate the data.

clist = ['tab:blue', 'tab:green']

raster = True # Whether to rasterize heavy plots 
if raster:
    zord = 0 # I use set_rasterization_zorder(1) for axes that I want to rasterize, everything plotted with zorder=0 keyword will be rasterized
else:
    zord = 2


rossler = equation.Rossler(values=[0.2,0.2,5.7])


# Define Poincare section
normalVectors = [np.array([1,1,0]), np.array([np.sin(-0.3*np.pi),np.cos(-0.3*np.pi),0]) ]# vector normal to Poincare Section
pPoinc = [0,0,0] # any point on the Poincare section
dirPoinc = 1 # direction of crossing Poincare section

poinc = []
idum=0 # Use this as auxiliary counter for number of data sets plotted


for normalVector in normalVectors:
    
    poinc.append(poincman.PoincMap(normalVector, pPoinc, None, direction=dirPoinc)) # initialize calculation of points on Poincare section
                                                    # Using none for simulator, since we will use solve_ivp instead of a simulator object
                                                    
for p in poinc:
    
    p.poincCondtx.__func__.direction = dirPoinc    # Setting this attribute is needed by solve_ivp and should be done in addition to setting direction in poincmaps. 


ric = [0.1,0.2, 0.3] # ic for short time integration until transients die

atol=1e-9
rtol=1e-6

sol1 = solve_ivp(rossler.f, [0,100], ric, method='RK45', atol=atol, rtol=rtol) # use to obtain initial condition on the attractor after transient

ic = sol1.y[:,-1]

# Use solve_ivp event detection to compute Poincare section intersections
tic = time.perf_counter()

tmult = 16
tbasic = 1600

sol = solve_ivp(rossler.f, [0,tmult*tbasic], ic, method='RK45',  events = [p.poincCondtx for p in poinc],dense_output=True, atol=atol, rtol=rtol) # integrate and record intersections with Poincare section

toc = time.perf_counter()

print(f"Integration time: {toc - tic:0.4f} seconds")

data = sol.y

pData = sol.y_events # compute the points on the Poincare section

t = np.linspace(0,sol.t[-1]/(4*tmult), int(2e5)) # Drop most points for clarity?
a = sol.sol(t)


# First figure
nrows=2
ncols=5
fig1 = plt.figure(figsize=(fwidth,fheight))
ax1 = plt.subplot2grid((nrows, ncols), (0,0), rowspan=2, colspan=2, projection='3d')
ax1.set_rasterization_zorder(1)
#ax3d = plt.gca(projection ='3d')
ax1.plot(a[0], a[1], a[2],linewidth=0.5,alpha = 0.9, color='0.7', zorder=zord)

ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$y$')
ax1.set_zlabel(r'$z$')

# make the panes transparent
ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# make the grid lines transparent
ax1.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax1.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax1.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

ax1.text(-10,10,20,'(a)')

for pd in pData:
    
    iKeep = int(pd.shape[0]/(4*tmult))
    ax1.plot(pd[:iKeep,0], pd[:iKeep,1], pd[:iKeep,2],'.',markersize=1, color=clist[idum], zorder=zord) # Plot the points on the section together with the attractor

    #if idum==1:
        #break

    ax2=plt.subplot2grid((nrows, ncols),(idum,2))
    ax2.set_rasterization_zorder(1)
    
    r = np.sqrt(pd[:,0]**2+pd[:,1]**2) # radial coordinate

    plt.scatter(r, pd[:,2], marker='o', edgecolor='', facecolors=clist[idum], s=1, c=clist[idum], zorder=zord) # Plot a the section in (r,z) 

    if idum==0:
        plt.text(3.5,0.18,'(b)')
    else:    
        plt.xlim(xmax=8)
        plt.ylim(ymax=15)
        plt.text(6.7,12,'(e)')
       
    plt.xlabel(r'$r$')
    plt.ylabel(r'$z$')

    # Plot the return map using the radial coordinate
    ax0 = plt.subplot2grid((nrows, ncols),(idum,3))
    ax0.set_rasterization_zorder(1)
    plt.scatter(r[:-1], r[1:], marker='o', edgecolor='', facecolors=clist[idum], s=1, c=clist[idum], zorder=zord)
    #plt.plot(r[:-1], r[1:], '.',  markersize=0.1, c=clist[idum], zorder=zord)

    if idum==0: # find kneading sequence
            pk = np.argmax(r[1:])
            sc = r[:-1][pk]
            itinSc = r[1:][pk:pk+30]
            rknead = np.asarray([it>sc for it in itinSc])*1
            print('r knead=', p.list2str(rknead))
            

    
    if idum==0:
        plt.text(3.5,11,'(c)')
    else:    
        plt.xlim(xmax=8)
        plt.text(2.4,6.5,'(f)')
    
    plt.xlabel('$r_n$')
    plt.ylabel('$r_{n+1}$')

    if inset:
        
        if idum==0:
    
            ax0 = plt.gca()
           
            if load_inset:
                 pDataIns = np.load('roessler_inset.npy')
            else:
                 ## Generate data for the inset
                 poinc[0].poincCondtx.__func__.terminal = False # Do not terminate integration after first intersection
                 sol = solve_ivp(rossler.f, [0,800*tmult*tbasic], ic, method='RK45',  events = poinc[0].poincCondtx, atol=atol, rtol=rtol) # integrate and record intersections with Poincare section
                 pDataIns = sol.y_events[0] # compute the points on the Poincare section

                 np.save('roessler_inset.npy')

            axIns = plt.axes([0,0,1,1],label='axIns')
            ip = InsetPosition(ax0,[0.15,0.07,0.59,0.33])
            axIns.set_axes_locator(ip)
            axIns.set_rasterization_zorder(1)

           
            r = np.sqrt(pDataIns[:,0]**2+pDataIns[:,1]**2) # radial coordinate
            #axIns.plot(r[:-1], r[1:],'.',markersize=1, color=clist[0])
            axIns.scatter(r[:-1], r[1:], c=clist[0], marker='o', edgecolor='', facecolors=clist[idum], s=1,  zorder=zord)
            axIns.set_xlim(7.7105, 7.713)
            axIns.set_ylim(12.10019, 12.1001917)
            axIns.tick_params(bottom=False, labelbottom=False,left=False,labelleft=False)
            axIns.set_yticks([])
            
            mark_inset(ax0, axIns, loc1=1, loc2=2, fc="none", ec='0.5')
        

            plt.figure() # This is just to create a new figure and then get back to the old. Otherwise the inset is not visible

            plt.sca(fig1.gca()) # Switch to first figure
                
    idum = idum+1


# Second figure
fig2, axes = plt.subplots(nrows=1, ncols=2,figsize=(0.5*fwidth, 0.2*fwidth))

idum = 0 # reset counter

for p, pd in zip(poinc, pData):
    print('LLE with Np=',pd.shape[0], 'K=',10*tmult)
    poincS , err = manifold.locally_linear_embedding(pd, n_neighbors = 10*tmult,  n_components =1,  random_state=1)

    s = poincS[:,0] # Choose first coordinate (this flattens array if we already computed 1D manifold)

    # Construct interpolating function for return map
    irpRM = p.interp1dRM(s, sorted_map_label='sorted_s_s', sorted_label='sorted_s')

    pk = sp.signal.find_peaks(p.sorted_s_s[:,1], width=50) # p.sorted_s_s is generated as a byproduct of calling p.interp1dRM

    if pk[0].shape[0]==0: # check for maximum, if none is found invert the map (not required, for uniform visualization only) and redo the interpolating function
        s=-s
        #p.sign_norm_q = -1
        print('inverted s')


    p.sign_norm_q = 1 # Choose sign for normalization of transform
    p.qmin_0 = True
    p.qmax_1 = True

    s = p.computeNormS(s)

    irpRM = p.interp1dRM(s, sorted_map_label='sorted_s_s', sorted_label='sorted_s') # Recompute
    pk = sp.signal.find_peaks(p.sorted_s_s[:,1], width=50) # recompute


    p.sc = p.sorted_s_s[:,0][pk[0]]

    s2sp = p.sParamSP(s, pd, kind='linear') # Generate mapping from s to state-space
 
    itinSc = list(map(p.itinSingle, p.sc, np.full_like(p.sc,30,dtype=int)))
    knead = [p.s2symb(it[1:]) for it in itinSc]
    
    print('knead=',p.list2str(knead[0]))

    p.invIrv=[s.min(), s.max()] # Set the invariant interval
    p.setBaseIrv() # Set base intervals
    p.orient_preserv = [0]
    p.alphabet = range(0,2)
    p.nPoinc = [1 for i in p.alphabet]

    # Compute inverse map
    p.inv_map = p.inverse_map()
    
    # Plot the new return map
    plt.sca(fig1.gca()) # Switch to first figure
    ax3=plt.subplot2grid((nrows, ncols),(idum,4))
    ax3.set_rasterization_zorder(1)
    plt.scatter(s[:-1], s[1:], marker='o', edgecolor='', facecolors=clist[idum], s=1, c=clist[idum], zorder=zord) # s_n, s_{n+1}
    slim = 1.1*np.abs(s).max()

    plt.xlim(-0.1,slim)
    plt.ylim(-0.1,slim)
    
    if idum==0:
        plt.text(-0.05,.9,'(d)')
    else:    
        plt.text(-0.05,.9,'(g)')    
    
    plt.xlabel('$q_{1,n}$')
    plt.ylabel('$q_{1,n+1}$')#, labelpad=-2)
    
    sTab = np.linspace(s.min(),s.max(),50)


    plt.plot(sTab,sTab,'--',color='0.5')


    nFixed = 1

    sPO = p.findCycleRM([1]) # Find PO of the return map, using a guess

    plt.plot([sPO],[sPO], c='k', marker='.', markersize=6)
    
 
    if idum==1:
        
        plt.sca(axes[0])
        axes[0].set_rasterization_zorder(1)
        
        r = np.sqrt(pd[:,0]**2+pd[:,1]**2) # radial coordinate

        plt.scatter(r, pd[:,2], c = s-s.min(), cmap='viridis', s=0.2, zorder=zord)
        
        plt.xlim(xmax=8)
        plt.ylim(ymax=15)
        
        clb = plt.colorbar()
        clb.set_label(r'$q_1$', labelpad=-2)
        
        clb.set_ticks(clb.get_ticks())
        
        ticklabels = ['{:.2f}'.format(i) for i in clb.get_ticks()+s.min()]
        
        clb.set_ticklabels(ticklabels)
        
        plt.xlabel(r'$r$')
        plt.ylabel(r'$z$')
        
        plt.text(6.5,12.,'(a)')
        
     
        plt.sca(axes[1])
        axes[1].set_rasterization_zorder(1)
        
        
        sSorted = p.sorted_s_s[:,0] # Use this in order to plot continuous curve
        
        plt.plot(sSorted, np.sqrt(s2sp[0](sSorted)**2+s2sp[1](sSorted)**2), 'k' , zorder=zord)
        plt.plot(sSorted, s2sp[2](sSorted),'k--', zorder=zord)
        plt.ylim(ymax=15)
        plt.xlim(-0.05,slim)
        
        plt.text(0.7,12.6,r'$z$')
        plt.text(0.75,5,r'$r$')              

        plt.text(0., 12.,'(b)')

        
        plt.xlabel(r'$q_1$')
        plt.ylabel(r'$r, z$')

    icPOg = p.s2spV(sPO) # Generate initial condition for guess PO in state-space

    # Integrate i.c. for guess PO

    p.poincCondtx.__func__.terminal = True # Terminate integration after first intersection

    TpoincReturn = np.mean(np.diff(sol.t_events[idum])) # Average time for return to Poincare section

    p.int_min_t = (nFixed-1)*TpoincReturn +  1e-9 # Do not terminate integration before required number of intersections. For nFixed=1, we add a small time interval, since the initial point is on Poincare section.

    solPOg = solve_ivp(rossler.f, [0,10*TpoincReturn], icPOg, method='RK45', events = p.poincCondtx) # Integrate until next intersection with Poincare section.

    TpGuess = solPOg.t_events[-1]

    dataPOg = np.asarray(solPOg.y)

    pDataPOg = np.asarray(solPOg.y_events)[0] # points on the Poincare section


    print('Guess error=', np.linalg.norm(solPOg.y[:,-1]-icPOg))

    # Find PO

    solPO = solve_bvp(rossler.f_bvp, p.bcPO, solPOg.t/TpGuess, solPOg.y, p=TpGuess, tol=1e-6, max_nodes=10000, verbose=2)

    
    if idum==1:
        ax1.plot(solPO.y[0],solPO.y[1],solPO.y[2], 'k',linewidth=1.)

    ax1.view_init(azim=280, elev=30) # Adjust viewpoint
        
    idum=idum+1

plt.show()

fig1.subplots_adjust(top=0.98, bottom=0.15, left=0., right=0.99, hspace=.6,wspace=0.7)
fig1.savefig('rossler_maps.pdf', dpi=600)

fig2.subplots_adjust(top=0.95, bottom=0.3, left=0.12, right=0.97, hspace=.7,wspace=0.8)
fig2.savefig('rossler_s.pdf', dpi=600)


