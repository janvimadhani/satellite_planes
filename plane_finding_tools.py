import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch
from matplotlib import cm
import scipy.stats as ss
import scipy.signal as sig
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle as Rectangle
import mpl_toolkits.mplot3d.art3d as art3d
import random
import pickle
import os


def read_systems(systems_file):
    """
    Input: systems_file,str: path to systems file

    Returns: systems,dict: dictionary object of MW type systems
    """

    with open(systems_file, 'rb') as handle:
        systems = pickle.loads(handle.read())
        
    return systems


def dist(x,y,z,normal_vect,d):
    """
    distance between a point and a plane defined by a normal vector 
    
    """
    
    
    nx, ny, nz = normal_vect[0],normal_vect[1],normal_vect[2]
    num = np.abs(nx*x + ny*y + nz*z + d)
    den = np.sqrt(nx**2 + ny**2 + nz**2)
    
    dist = num/den
    
    return dist


def best_plane(systems,system,level=1,n=10,mock=False,rand=False,verbose=False):
    """
    Input: systems,dict: all MW sytems
           specifiy level = 1 for all true satellites
                    level = 2 for galactic plane
                    level = 0 *dummy 
           n,int: number of iterations
    Returns: best_u1,best_u2,best_u3,nx,ny,nz,rms,best_rms,delta_s,best_cos_theta
    """
    if verbose:
        if not mock:
            print(f'Fitting plane to level {level} satellites... ')
 
    niter = n

    plane_finder = {}
    plane_finder['u1'] = []
    plane_finder['u2'] = []
    plane_finder['u3'] = []
    plane_finder['delta_s'] =[]
    plane_finder['rms_dist'] = []
    plane_finder['nx'] = []
    plane_finder['ny'] = []
    plane_finder['nz'] = []
    plane_finder['cos_theta'] = []
    plane_finder['phi'] = []

    for i in range(niter):


        u1 = random.uniform(0,1) #[0,1]  
        u2 = random.uniform(0,1) #[0,1]
        
        u3 = random.uniform(0,1) #sign

        plane_finder['u1'].append(u1)
        plane_finder['u2'].append(u2)
        plane_finder['u3'].append(u3)
            



        cos_theta = 2*u1 - 1   #makes sure cos_theta is bw -1,1

        sin_theta = np.sqrt(1-cos_theta**2)
        #randomly select sign of arccos 
        
        if u3 <= 0.5:
            sin_theta = -1*sin_theta

        
        phi = 2*np.pi*u2  #[0,2*pi] 
        
        plane_finder['cos_theta'].append(cos_theta)
        plane_finder['phi'].append(phi)




        nx = np.cos(phi)*sin_theta
        ny = np.sin(phi)*sin_theta
        nz = cos_theta
        n = np.array([nx,ny,nz])
        mag_n = np.linalg.norm(n)
        unit_n = n/mag_n
        #print(unit_n)
        
        plane_finder['nx'].append(unit_n[0])
        plane_finder['ny'].append(unit_n[1])
        plane_finder['nz'].append(unit_n[2])

        if mock:
            x0 = system['MW_px']
            y0 = system['MW_py']
            z0 = system['MW_pz']
        elif rand:
            x0 = system['MW_px']
            y0 = system['MW_py']
            z0 = system['MW_pz']
            
        else:
            x0 = systems[system]['MW_px'][0]
            y0 = systems[system]['MW_py'][0]
            z0 = systems[system]['MW_pz'][0]

        gal_center = np.array([x0,y0,z0])

        d = np.dot(-gal_center,unit_n)
        #print('A,B,C,D:',unit_n[0],unit_n[1],unit_n[2],d)

        #equation of plane (Ax + By + Cz + D = 0): unit_n[0] * x + unit_n[1] * y  + unit_n[2]*z + d = 0 

        #calculate distances

        distances = []
        #nsat = len(system['sat_pxs'])
        if mock:
            nsats = len(system['sat_pxs'])
            for k in range(len(system['sat_pxs'])):
                x,y,z = system['sat_pxs'][k],system['sat_pys'][k],system['sat_pzs'][k]
                s = dist(x,y,z,unit_n,d)
                distances.append(s)
                
        elif rand:
            nsats = len(system['sat_px'])
            for k in range(len(system['sat_px'])):
                x,y,z = system['sat_px'][k],system['sat_py'][k],system['sat_pz'][k]
                s = dist(x,y,z,unit_n,d)
                distances.append(s)
            
                
        else:
            

            if level == 1:
                level_one_sats = np.where(systems[system]['sat_levels'] == 1)
                nsats = len(level_one_sats[0]) 
                for k in range(nsats):
                    x,y,z = systems[system]['sat_pxs'][level_one_sats][k],systems[system]['sat_pys'][level_one_sats][k],systems[system]['sat_pzs'][level_one_sats][k]
                    s = dist(x,y,z,unit_n,d)
                    distances.append(s)
            elif level == 2:
                level_two_sats = np.where(systems[system]['sat_levels'] == 2)
                nsats = len(level_two_sats[0])
                for k in range(nsats):
                    x,y,z = systems[system]['sat_pxs'][level_two_sats][k],systems[system]['sat_pys'][level_two_sats][k],systems[system]['sat_pzs'][level_two_sats][k]
                    s = dist(x,y,z,unit_n,d)
                    distances.append(s)
           
            elif level == 3:
                level_three_sats = np.where(systems[system]['sat_levels'] == 3)
                nsats = len(level_three_sats[0])
                for k in range(nsats):
                    x,y,z = systems[system]['sat_pxs'][level_three_sats][k],systems[system]['sat_pys'][level_three_sats][k],systems[system]['sat_pzs'][level_three_sats][k]
                    s = dist(x,y,z,unit_n,d)
                    distances.append(s)
            else:
                for k in range(len(systems[system]['sat_pxs'])):
                    x,y,z = systems[system]['sat_pxs'][k],systems[system]['sat_pys'][k],systems[system]['sat_pzs'][k]
                    s = dist(x,y,z,unit_n,d)
                    distances.append(s)
                    
                
              
        distances = np.asarray(distances)

        rms = np.sqrt(np.mean(distances**2))
        #print(rms)
        plane_finder['delta_s'].append(distances)
        plane_finder['rms_dist'].append(rms)
        
        
    #find minimum rms and corresponding u1, u2 
        
    u1_a = np.asarray(plane_finder['u1'])
    #print(len(u1_a))
    u2_a = np.asarray(plane_finder['u2'])
    u3_a = np.asarray(plane_finder['u3'])
    rms_a = plane_finder['rms_dist']

    #print(rms_a)
    
    cos_theta_a = np.asarray(plane_finder['cos_theta'])
    phi_a = np.asarray(plane_finder['phi'])
    #print(rms_a)

    best_plane = np.argmin(rms_a)
        
    
    best_rms = plane_finder['rms_dist'][best_plane]
        
    if verbose:
        print(f'Fitting to {nsats} satellites...')  
        print('best plane index:',best_plane)
        print('Best plane has:')
        print('Cos(theta):', cos_theta_a[best_plane])
        print('Phi',phi_a[best_plane])
        print(f'u1 = {u1_a[best_plane]}; u2 = {u2_a[best_plane]}')
        print(f'Best rms = {best_rms}')
    
    return u1_a[best_plane],u2_a[best_plane],u3_a[best_plane],plane_finder['nx'],plane_finder['ny'],plane_finder['nz'],plane_finder['rms_dist'],best_rms,plane_finder['delta_s'],cos_theta_a[best_plane]
        


def get_plane(u1,u2,u3,systems,system,mock=False):



    cos_theta = 2*u1 - 1   #makes sure cos_phi is bw 0,1

    sin_theta = np.sqrt(1-cos_theta**2)
    #randomly select sign of arccos 

    if u3 <= 0.5:
        sin_theta = -1*sin_theta


    phi = 2*np.pi*u2  #[-pi,pi]  

    nx = np.cos(phi)*sin_theta
    ny = np.sin(phi)*sin_theta
    nz = cos_theta
    n = np.array([nx,ny,nz])
    mag_n = np.linalg.norm(n)
    unit_n = n/mag_n
    
    if mock:
        x0 = system['MW_px']
        y0 = system['MW_py']
        z0 = system['MW_pz']
    else: #both random dist and actual dist are centered at same central galaxy
        x0 = systems[system]['MW_px'][0]
        y0 = systems[system]['MW_py'][0]
        z0 = systems[system]['MW_pz'][0]

    gal_center = np.array([x0,y0,z0])

    d = np.dot(-gal_center,unit_n)

    if mock:
        xx,yy = np.meshgrid(np.linspace(-1,1,100),np.linspace(-1,1,100))
        
    else:
        #"""
        xx, yy = np.meshgrid(np.linspace(systems[system]['halo_px']-systems[system]['halo_rvir']*8e-1,systems[system]['halo_px']+systems[system]['halo_rvir']*8e-1,100),
        np.linspace(systems[system]['halo_py']-systems[system]['halo_rvir']*8e-1,systems[system]['halo_py']+systems[system]['halo_rvir']*8e-1,100))
        #"""
        
        """        
        min_x,max_x= np.min(systems[system]['sat_pxs']),np.max(systems[system]['sat_pxs'])
        min_y,max_y= np.min(systems[system]['sat_pys']),np.max(systems[system]['sat_pys'])
        xx, yy = np.meshgrid(np.linspace(min_x,max_x,100),np.linspace(min_y,max_y,100))
        """

    # calculate corresponding z
    z = (-unit_n[0] * xx - unit_n[1] * yy - d) * 1. /unit_n[2]
    #z = (- unit_n[1]*xx)/unit_n[2] - (unit_n[0]*yy)/unit_n[2]
    
    return z,xx,yy,unit_n

def save_3Dplot(name_of_plot,systems,syst,snapshot,xx,yy,z_best):
    ## Figure for presentation

    fig = plt.figure(figsize=[8,6])
    ax = plt.axes(projection='3d')

    M_to_k = 1000
    MW_x,MW_y,MW_z = systems[syst]['MW_px'],systems[syst]['MW_py'],systems[syst]['MW_pz']
    nsats = len(np.where(systems[syst]['sat_levels'] == 1)[0])

    scaleby = 3


    for i in range(nsats):
        sat = Circle(((systems[syst]['sat_pxs'][i]-MW_x)*M_to_k, (systems[syst]['sat_pys'][i] - MW_y)*M_to_k), radius=systems[syst]['sat_rvirs'][i]*M_to_k,color='black',alpha=0.4)
        ax.add_patch(sat)
        art3d.pathpatch_2d_to_3d(sat, (systems[syst]['sat_pzs'][i]-MW_z)*M_to_k, zdir="z")
    imsats = ax.scatter3D((systems[syst]['sat_pxs'] - MW_x)*M_to_k,(systems[syst]['sat_pys']- MW_y)*M_to_k,(systems[syst]['sat_pzs'] - MW_z)*M_to_k,
                        s=systems[syst]['sat_rvirs']*M_to_k*scaleby**2,c=20*(systems[syst]['sat_vxs']-systems[syst]['MW_vx']),cmap='seismic',vmin=-300,vmax=300,alpha=0.8,label='Satellites')
    imcentral = ax.scatter3D((systems[syst]['MW_px']- MW_x)*M_to_k,(systems[syst]['MW_py']- MW_y)*M_to_k ,(systems[syst]['MW_pz'] - MW_z)*M_to_k,
                        s=systems[syst]['MW_rvir']*M_to_k*scaleby**2,c='slateblue',edgecolors='darkblue',alpha=0.4,label='Central')
    #spin = ax.quiver((systems[syst]['MW_px']-MW_x)*M_to_k,(systems[syst]['MW_py']-MW_y)*M_to_k,(systems[syst]['MW_pz']-MW_z)*M_to_k,
                        # systems[syst]['MW_lx'],systems[syst]['MW_ly'],systems[syst]['MW_lz'],color='black', length= 30, normalize=True,label='Spin')

    central = Circle(((systems[syst]['MW_px'] - MW_x)*M_to_k, (systems[syst]['MW_py'] - MW_y)*M_to_k), radius=systems[syst]['MW_rvir']*M_to_k,color='slateblue',alpha=0.4,label='Rvir of MW')
    ax.add_patch(central)
    art3d.pathpatch_2d_to_3d(central, (systems[syst]['MW_pz']-MW_z)*M_to_k, zdir="z")

    #plot the plane
    plane = ax.plot_surface((xx-MW_x)*M_to_k,(yy-MW_y)*M_to_k, (z_best-MW_z)*M_to_k,color='k' ,alpha=0.4)


    ax.set_title(r'MW type Satellite System, $N_{nsats}$ =' + f'{nsats}',y=1.15)
    plt.colorbar(imsats,label=r'Velocity of Satellites [km/s]')
    ax.autoscale('False')
    ax.set_xlabel('X [kpc]')
    ax.set_ylabel('Y [kpc]')
    ax.set_zlabel('Z [kpc]')
    plt.legend(loc="upper left", markerscale=.4)

    #"""
    extent = 500 #kpc
    ax.set_xlim(-extent,extent)
    ax.set_ylim(-extent,extent)
    ax.set_zlim(-extent,extent)
    #"""

    # Save plot
    #check if directory exists

    #script_dir = os.path.dirname(__file__)
    
    #results_dir = os.path.join(script_dir, '3Dplots/'+str(snapshot) +'/')
    results_dir = '/data78/welker/madhani/3Dplots/' + str(snapshot) + '/'
    

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    print(f'Saving 3D Plot to:  {results_dir + name_of_plot}')
    plt.savefig(results_dir + name_of_plot)




######################################
# Isotropy Analysis
######################################

def rand_angle():
    u1 = random.uniform(0,1) #[0,1]  
    u2 = random.uniform(0,1) #[0,1]

    u3 = random.uniform(0,1) #sign


    cos_theta = 2*u1 - 1   #makes sure cos_theta is bw -1,1


    sin_theta = np.sqrt(1-cos_theta**2)

    #randomly select sign of arccos 

    if u3 <= 0.5:
        sin_theta = -1*sin_theta


    phi = 2*np.pi*u2  #[0,2*pi] 
    
    return cos_theta, sin_theta, phi

    

def rand_spherical_dist(systems,system,level=1):
 
   
    spherical_isotropy = {}

    spherical_isotropy['sat_x'] = []
    spherical_isotropy['sat_y'] = []
    spherical_isotropy['sat_z'] = []
    x0 = systems[system]['MW_px'][0]
    y0 = systems[system]['MW_py'][0]
    z0 = systems[system]['MW_pz'][0]

    #redistribute satellites, preserving their separation vector, but new angles

    if level == 1:
        level_one_sats = np.where(systems[system]['sat_levels'] == 1)
        nsats = len(level_one_sats[0]) 
        for k in range(nsats):
            x,y,z = systems[system]['sat_pxs'][level_one_sats][k],systems[system]['sat_pys'][level_one_sats][k],systems[system]['sat_pzs'][level_one_sats][k]
            rx,ry,rz = x-x0,y-y0,z-z0
            r = np.sqrt(rx**2 + ry**2 + rz**2)
            
            cos_theta,sin_theta, phi = rand_angle()
            
            xp = r*np.cos(phi)*sin_theta + x0
            yp = r*np.sin(phi)*sin_theta + y0
            zp = r*cos_theta + z0

            spherical_isotropy['sat_x'].append(xp)
            spherical_isotropy['sat_y'].append(yp)
            spherical_isotropy['sat_z'].append(zp)


    elif level == 2:
        level_two_sats = np.where(systems[system]['sat_levels'] == 2)
        nsats = len(level_two_sats[0])

        for k in range(nsats):
            x,y,z = systems[system]['sat_pxs'][level_two_sats][k],systems[system]['sat_pys'][level_two_sats][k],systems[system]['sat_pzs'][level_two_sats][k]
            rx,ry,rz = x-x0,y-y0,z-z0
            r = np.sqrt(rx**2 + ry**2 + rz**2)
            
            cos_theta,sin_theta, phi = rand_angle()
            
            xp = rx*np.cos(phi)*sin_theta + x0
            yp = ry*np.sin(phi)*sin_theta + y0 
            zp = rz*cos_theta + z0

            spherical_isotropy['sat_x'].append(xp)
            spherical_isotropy['sat_y'].append(yp)
            spherical_isotropy['sat_z'].append(zp)


    elif level == 3:
        level_three_sats = np.where(systems[system]['sat_levels'] == 3)
        nsats = len(level_three_sats[0])

        for k in range(nsats):
            x,y,z = systems[system]['sat_pxs'][level_three_sats][k],systems[system]['sat_pys'][level_three_sats][k],systems[system]['sat_pzs'][level_three_sats][k]
            rx,ry,rz = x-x0,y-y0,z-z0
            r = np.sqrt(rx**2 + ry**2 + rz**2)
            
            cos_theta,sin_theta, phi = rand_angle()
            
            xp = rx*np.cos(phi)*sin_theta + x0
            yp = ry*np.sin(phi)*sin_theta + y0
            zp = rz*cos_theta + z0

            spherical_isotropy['sat_x'].append(xp)
            spherical_isotropy['sat_y'].append(yp)
            spherical_isotropy['sat_z'].append(zp)

    else:

        for k in range(len(systems[system]['sat_pxs'])):
            x,y,z = systems[system]['sat_pxs'][k],systems[system]['sat_pys'][k],systems[system]['sat_pzs'][k]
            rx,ry,rz = x-x0,y-y0,z-z0
            r = np.sqrt(rx**2 + ry**2 + rz**2)
            
            cos_theta,sin_theta, phi = rand_angle()
            
            xp = rx*np.cos(phi)*sin_theta + x0 
            yp = ry*np.sin(phi)*sin_theta + y0 
            zp = rz*cos_theta + z0

            spherical_isotropy['sat_x'].append(xp)
            spherical_isotropy['sat_y'].append(yp)
            spherical_isotropy['sat_z'].append(zp)


    return spherical_isotropy['sat_x'],spherical_isotropy['sat_y'],spherical_isotropy['sat_z']


def check_isotropy(systems,syst,n=2000):
## check that it's uniformly dist by running n times
    t0 = time.time()

    n = 2000
    rand_systems = {}
    rand_systems['systems'] = []


    #make n random rystems
    for i in range(n):
        rand_system = {}
        #rand_system['sat_px'] = []
        #rand_system['sat_py'] = []
        #rand_system['sat_pz'] = []
        rand_system['MW_px'] = systems[syst]['MW_px'][0]
        rand_system['MW_py'] = systems[syst]['MW_py'][0]
        rand_system['MW_pz'] = systems[syst]['MW_pz'][0]
        
        sx,sy,sz = rand_spherical_dist(systems,syst,level=1)

        
        rand_system['sat_px'] = sx
        rand_system['sat_py'] = sy
        rand_system['sat_pz'] = sz
        rand_systems['systems'].append(rand_system)

    #find best fit plane of n random systems
    print(f'Finding best fit plane of {n} random, isotropically distributed systems...')
    mean_rms = []

    for rand_syst in range(n):
        
        best_u1,best_u2,best_u3,nx,ny,nz,rms,rand_rms,delta_s,best_rand_cos_theta = best_plane(systems=systems,system=rand_systems['systems'][rand_syst],n=n,rand=True)
        mean_rms.append(rand_rms)

    t1 = time.time()

    print(f'Took {t1-t0} seconds.')


    return mean_rms

def save_hist(name_of_plot,best_rms,mean_rms,histbins=70):
    n = len(mean_rms)

    fig, ax = plt.subplots(1, 1,
                            figsize =(8,5), 
                            tight_layout = True)

    histbins = 70
    
    ncounts,dense_bins,patches = ax.hist(mean_rms, density=True,bins =histbins,ec='purple',fc='thistle')
    ax.axvline(x=best_rms,c='black',label='mean rms of actual distribution')

    mu, sigma = ss.norm.fit(mean_rms)
    best_fit_line = ss.norm.pdf(dense_bins, mu, sigma)
    ax.plot(dense_bins,best_fit_line,c='purple',label='PDF')

    ax.set_title(f'Distribution of {n} Isotropically Distributed Planes')
    ax.set_xlabel(r'Mean RMS')
    ax.legend()
    
    # Save plot
    #check if directory exists

    #script_dir = os.path.dirname(__file__)
    #results_dir = os.path.join(script_dir, 'iso_histograms/')

    results_dir = '/data78/welker/madhani/iso_histograms/' + str(snapshot) + '/'
    

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    print(f'Saving histogram to:  {results_dir + name_of_plot}')
    plt.savefig(results_dir + name_of_plot)

   
        


