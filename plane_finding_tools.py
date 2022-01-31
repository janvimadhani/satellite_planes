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
from pathlib import Path
import os
import json
import csv

def read_systems(systems_file):
    """
    Input: systems_file,str: path to systems file

    Returns: systems,dict: dictionary object of MW type systems
    """
    print('Reading file from:', systems_file)

    with open(systems_file, 'rb') as handle:
        systems = pickle.loads(handle.read())
        
    return systems

def write_to_pickle(dictionary,snapshot,name_to_save,rewrite=True):
    """
    writes any dictionary to a pickle file, easy to read later 
    Inputs: dictionary, dict: python dictionary to be saved 
            snapshot of file, int: snapshot to create directory under
            name,string: name of file to be written
    Output: path,pickle: pickle file written at specified location
    """
    script_dir = os.path.dirname(__file__)
    #results_dir = os.path.join(script_dir, 'systems/')
    results_dir = '/data78/welker/madhani/analysis_data/' + str(snapshot) + '/'
    
    #local results dir
    #results_dir = '/Users/JanviMadhani/satellite_planes/systems/'


    if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

    path_of_file = results_dir + name_to_save + '.pickle'

    path = Path(path_of_file)


    if path.is_file():
        if rewrite:
            print(f'File already exists, rewriting anyway to {path} ...')
            with open(path_of_file,'wb') as handle:
                pickle.dump(dictionary,handle,protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print('File already exists.')

    else:
        print(f'Writing file to {path} ...')
        with open(path_of_file,'wb') as handle:
            pickle.dump(dictionary,handle,protocol=pickle.HIGHEST_PROTOCOL)

def dist(x,y,z,normal_vect,d):
    """
    distance between a point and a plane defined by a normal vector 
    
    """
    
    
    nx, ny, nz = normal_vect[0],normal_vect[1],normal_vect[2]
    num = np.abs(nx*x + ny*y + nz*z + d)
    den = np.sqrt(nx**2 + ny**2 + nz**2)
    
    dist = num/den
    
    return dist

### Write evolutionary algorithm

def evolutionary_plane_finder(systems,system,n_iter,n_start,n_erase,n_avg_mutants,level=1,rand=False,verbose=False):
    """
    Input: systems,dict: all MW systems
           specifiy level = 1 for all true satellites
                    level = 2 for galactic plane
                    level = 0 *dummy 
           n,int: number of iterations
    Returns: best_u1,best_u2,best_u3,
    """

    if rand:
        nsats = len(system['sat_pxs'])
    else:
        level_sats = np.where(systems[system]['sat_levels'] == level)
        nsats = len(level_sats[0])
    
    def fitness(system,level,unit_n,random=rand):
        if random:
            x0 = system['MW_px']
            y0 = system['MW_py']
            z0 = system['MW_pz']
        else:
            x0 = systems[system]['MW_px'][0]
            y0 = systems[system]['MW_py'][0]
            z0 = systems[system]['MW_pz'][0]

        gal_center = np.array([x0,y0,z0])

        d = np.dot(-gal_center,unit_n)
        
        distances = []
        if random:
            nsats = len(system['sat_pxs'])
            for k in range(len(system['sat_pxs'])):
                x,y,z = system['sat_pxs'][k],system['sat_pys'][k],system['sat_pzs'][k]
                s = dist(x,y,z,unit_n,d)
                distances.append(s)
        else:
            level_sats = np.where(systems[system]['sat_levels'] == level)
            nsats = len(level_sats[0]) 
            for k in range(nsats):
                x,y,z = systems[system]['sat_pxs'][level_sats][k],systems[system]['sat_pys'][level_sats][k],systems[system]['sat_pzs'][level_sats][k]
                s = dist(x,y,z,unit_n,d)
                distances.append(s)
            
        distances = np.asarray(distances)

        rms = np.sqrt(np.mean(distances**2))
        return rms
        
    
    plane_finder = {}
    plane_finder['rms_dist'] = []
    plane_finder['nx'] = []
    plane_finder['ny'] = []
    plane_finder['nz'] = []
    plane_finder['u1'] = []
    plane_finder['u2'] = []
    plane_finder['u3'] = []
    
    #level_sats = np.where(systems[system]['sat_levels'] == level)
    #nsats = len(level_sats[0]) 
    #start with creating an initial population 
    for k in range(n_start):

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

        nx = np.cos(phi)*sin_theta
        ny = np.sin(phi)*sin_theta
        nz = cos_theta
        n = np.array([nx,ny,nz])
        mag_n = np.linalg.norm(n)
        unit_n = n/mag_n
        
        plane_finder['nx'].append(unit_n[0])
        plane_finder['ny'].append(unit_n[1])
        plane_finder['nz'].append(unit_n[2])
        
        rms = fitness(system,level,unit_n,rand)
        #plane_finder['delta_s'].append(distances)
        plane_finder['rms_dist'].append(rms)
        
    #now, evolve n_iter times
    i = 0 
    n_mutants = n_erase - n_avg_mutants
    while i <= n_iter:
        
        #eliminate n_erase
        #first sort list of rms to be from min to max so you can eliminate n_erase
        
        sorted_pop_rms = np.argsort(plane_finder['rms_dist'])
        #print(f'Index {i}, sorted rms {sorted_pop_rms}')
        to_erase = []
        for e in range(n_erase):
            i_to_erase = sorted_pop_rms[-(e+1)]
            to_erase.append(i_to_erase)

        for m in range(n_avg_mutants):

            #add the average of all the erased
            avg_u1 = np.mean([plane_finder['u1'][k] for i,k in enumerate(to_erase)])
            avg_u2 = np.mean([plane_finder['u2'][k] for i,k in enumerate(to_erase)])
            avg_u3 = np.mean([plane_finder['u3'][k] for i,k in enumerate(to_erase)])


            cos_theta_avg = 2*avg_u1 - 1   #makes sure cos_theta is bw -1,1

            sin_theta_avg = np.sqrt(1-cos_theta_avg**2)
            #randomly select sign of arccos 

            if avg_u3 <= 0.5:
                sin_theta_avg = -1*sin_theta_avg


            phi_avg = 2*np.pi*avg_u2  #[0,2*pi] 

            nx_avg = np.cos(phi_avg)*sin_theta_avg
            ny_avg = np.sin(phi_avg)*sin_theta_avg
            nz_avg = cos_theta_avg
            n_avg = np.array([nx_avg,ny_avg,nz_avg])
            mag_n_avg = np.linalg.norm(n_avg)
            unit_n_avg = n_avg/mag_n_avg

            #now that you're done using to be erased data, erase it
            plane_finder['rms_dist'].pop(i_to_erase)
            plane_finder['u1'].pop(i_to_erase)
            plane_finder['u2'].pop(i_to_erase)
            plane_finder['u3'].pop(i_to_erase)
            plane_finder['nx'].pop(i_to_erase)
            plane_finder['ny'].pop(i_to_erase)
            plane_finder['nz'].pop(i_to_erase)


            #add the avg to the population
            avg_rms = fitness(system,level,unit_n_avg,rand)
            plane_finder['rms_dist'].append(avg_rms)
            plane_finder['u1'].append(avg_u1)
            plane_finder['u2'].append(avg_u2)
            plane_finder['u3'].append(avg_u3)
            plane_finder['nx'].append(nx_avg)
            plane_finder['ny'].append(ny_avg)
            plane_finder['nz'].append(nz_avg)


        #add true mutants to the population
        for t in range(n_mutants):
            u1 = random.uniform(0,1) #[0,1]  
            u2 = random.uniform(0,1) #[0,1]
            u3 = random.uniform(0,1) #sign

            cos_theta = 2*u1 - 1   #makes sure cos_theta is bw -1,1
            sin_theta = np.sqrt(1-cos_theta**2)

            #randomly select sign of arccos 
            if u3 <= 0.5:
                sin_theta = -1*sin_theta


            phi = 2*np.pi*u2  #[0,2*pi] 

            nx = np.cos(phi)*sin_theta
            ny = np.sin(phi)*sin_theta
            nz = cos_theta
            n = np.array([nx,ny,nz])
            mag_n = np.linalg.norm(n)
            unit_n = n/mag_n


            #find the distance/rms of these new guys and add them to the rms list
            mut_rms = fitness(system,level,unit_n,rand)

            #add them all to the existing population
            plane_finder['rms_dist'].append(mut_rms)
            plane_finder['u1'].append(u1)
            plane_finder['u2'].append(u2)
            plane_finder['u3'].append(u3)
            plane_finder['nx'].append(nx)
            plane_finder['ny'].append(ny)
            plane_finder['nz'].append(nz)






        i+=1

    #return the best plane
    
    u1_a = np.asarray(plane_finder['u1'])
    u2_a = np.asarray(plane_finder['u2'])
    u3_a = np.asarray(plane_finder['u3'])
    rms_a = plane_finder['rms_dist']

    
    best_plane = np.argmin(rms_a)
    best_rms = plane_finder['rms_dist'][best_plane]
    
    if verbose:
        print(f'Fitting to {nsats} satellites...')  
        print('best plane index:',best_plane)
        print('Best plane has:')
        print('Cos(theta):', 2*u1_a[best_plane] - 1 )
        print('Phi',2*np.pi*u2)
        print(f'u1 = {u1_a[best_plane]}; u2 = {u2_a[best_plane]}')
        print(f'Best rms = {best_rms}')
    
    return u1_a[best_plane],u2_a[best_plane],u3_a[best_plane],best_rms

##### ARCHIVED PLANE FINDER -- USES BRUTE FORCE
"""
def best_plane(systems,system,level=1,n=10,mock=False,rand=False,verbose=False):
    
    Input: systems,dict: all MW sytems
           specifiy level = 1 for all true satellites
                    level = 2 for galactic plane
                    level = 0 *dummy 
           n,int: number of iterations
    Returns: best_u1,best_u2,best_u3,nx,ny,nz,rms,best_rms,delta_s,best_cos_theta
    
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
"""      


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

    #line of sight,theta,phi  
    los = [np.arccos(cos_theta),phi]
    
    return z,xx,yy,unit_n,los

def project_on_los(sat_velocity,plos):
    """
    Input: sat_velocity, array: [vx,vy,vz]
           line of sight, array: [nx,0,0] for example
    Returns: component of sat_velocity along line of sight
    
    Make sure satellite velocity is in frame of ref of central, i.e. subtract off central velociy first
    """
    
    dot = np.dot(sat_velocity,plos)
    
    mag_los = np.linalg.norm(plos)
    unit_los = plos/mag_los
    
    projection = (dot/mag_los) * unit_los
    
    return projection


def corotating_frac(systems,syst,unit_n,actual_rms,rand=False,nrms=1,level=1):
    """
    find the ratio of corotation for all satellites that are within rms of plane
    Input: systems, dict: systems dictionary
           syst, int: index of relevant system
           rms, int: within how many rms you want to inspect corotation
           level, int: what level satellites you want to consider

    """



    Lx = systems[syst]['MW_lx']
    Ly = systems[syst]['MW_ly']
    Lz = systems[syst]['MW_lz']
    L = np.array([Lx,Ly,Lz])
    L_mag = np.linalg.norm(L)
    unit_ez = L/L_mag
    unit_ez = unit_ez.reshape(3,)


    def get_vrot(satvx,satvy,satvz,r,unit_ez):
        rz = r[2]
        b = (rz*unit_ez)
        b = b.flatten()
        a = (r - b)
        unit_er = a/np.linalg.norm(a)

        unit_etheta = np.cross(unit_ez,unit_er)
        vx = satvx
        vy = satvy
        vz = satvz
        v = np.array([vx,vy,vz])
        
        vrot = np.dot(v,unit_etheta)

        return vrot
    if rand:
        x0 = systems[syst]['MW_px']
        y0 = systems[syst]['MW_py']
        z0 = systems[syst]['MW_pz']
    else:
        x0 = systems[syst]['MW_px'][0]
        y0 = systems[syst]['MW_py'][0]
        z0 = systems[syst]['MW_pz'][0]

    gal_center = np.array([x0,y0,z0])

    d = np.dot(-gal_center,unit_n)



    #level_sats = np.where(systems[syst]['sat_levels'] == level)
    #nsats = len(level_sats[0]) 

    distances = []
    sep_vect = []

    #calculate the distance of ALL the sats (needed for shape)
    for k in range(len(systems[syst]['sat_pxs'])):
        x,y,z = systems[syst]['sat_pxs'][k],systems[syst]['sat_pys'][k],systems[syst]['sat_pzs'][k]
        rx,ry,rz = x-x0,y-y0,z-z0
        r = np.sqrt(rx**2 + ry**2 + rz**2)
        sep_vect.append(r)
        s = dist(x,y,z,unit_n,d)
        distances.append(s) 

           
    distances = np.asarray(distances)
    sep_vect = np.asarray(sep_vect)
    
    #find satellites within n * rms AND right level
    #nrms = the number of rms within which you want to consider satellites as "on-plane"
    if rand:
        win_rms = np.where(distances <= nrms * actual_rms)
    
    else:
        win_rms = np.where((distances <= nrms * actual_rms) & (systems[syst]['sat_levels'] == level))
    
    nsats = len(win_rms[0])
    

    vrots = []
    for k in range(nsats):
        r = sep_vect[win_rms][k]
        vx,vy,vz = systems[syst]['sat_vxs'][win_rms][k],systems[syst]['sat_vys'][win_rms][k],systems[syst]['sat_vzs'][win_rms][k]
        vrot = get_vrot(vx,vy,vz,r,unit_ez)
        vrots.append(vrot)

    pos = 0
    neg = 0

    for v in vrots:
        if v > 0:
            pos +=1
        elif v < 0:
            neg += 1
        else:
            print('Check velocities, something off!')

    ntot = len(vrots)
    Rpos = pos/ntot
    Rneg = neg/ntot

    if Rpos > Rneg:
        corot_frac = Rpos
    elif Rneg > Rpos:
        corot_frac = Rneg
    
    else:
        corot_frac = 0

    return corot_frac




    

def save_3Dplot(name_of_plot,systems,syst,snapshot,xx,yy,z_best,los,unit_n,phys_ext,inertia=None):

    """
    Input:
    Returns: saves a plot
    """
    ## Figure for presentation
    p_a,p_b,p_c,p_c_to_a = phys_ext[0],phys_ext[1],phys_ext[2],phys_ext[3]

    #"""
    #project sat velocities onto los vector of the plane 
    #project_onto = np.array([unit_n[0],0,0]) #change this vector if you want to project onto a different vector
    project_onto = los 

    projected_v = []
    mv = np.array([systems[syst]['MW_vx'],systems[syst]['MW_vy'],systems[syst]['MW_vz']])

    for i in range(len(systems[syst]['sat_vxs'])):
        sv = np.array([systems[syst]['sat_vxs'][i],systems[syst]['sat_vys'][i],systems[syst]['sat_vzs'][i] ])
        v = sv-mv

        
        vx = project_on_los(v,project_onto)
        
        projected_v.append(vx[0]) #change this if you're projecting onto some component other than x
    projected_v = np.asarray(projected_v)
    #"""


    fig = plt.figure(figsize=[8,6])
    ax = plt.axes(projection='3d')

    #initialize the plane edge on and add some random offset so it's not completely edge on
    randomize_theta = random.randint(-5,5)
    randomize_phi = random.randint(-5,5)
    ax.view_init(np.degrees(los[0])+randomize_theta,np.degrees(los[1])+randomize_phi)

    M_to_k = 1000
    MW_x,MW_y,MW_z = systems[syst]['MW_px'],systems[syst]['MW_py'],systems[syst]['MW_pz']
    nsats = len(np.where(systems[syst]['sat_levels'] == 1)[0])

    scaleby = 3


    for i in range(nsats):
        sat = Circle(((systems[syst]['sat_pxs'][i]-MW_x)*M_to_k, (systems[syst]['sat_pys'][i] - MW_y)*M_to_k), radius=systems[syst]['sat_rvirs'][i]*M_to_k,color='black',alpha=0.4)
        ax.add_patch(sat)
        art3d.pathpatch_2d_to_3d(sat, (systems[syst]['sat_pzs'][i]-MW_z)*M_to_k, zdir="z")
    #colored by x line of sight velocity
    """
    imsats = ax.scatter3D((systems[syst]['sat_pxs'] - MW_x)*M_to_k,(systems[syst]['sat_pys']- MW_y)*M_to_k,(systems[syst]['sat_pzs'] - MW_z)*M_to_k,
                        s=systems[syst]['sat_rvirs']*M_to_k*scaleby**2,c=20*(systems[syst]['sat_vxs']-systems[syst]['MW_vx']),cmap='seismic',vmin=-300,vmax=300,alpha=0.8,label='Satellites')
    """
    
    #colored by los along nx vector of plane
    #"""
    for i in range(len(systems[syst]['sat_pxs'] )):
        color = projected_v[i]
        if color > 0:
            ax.scatter3D((systems[syst]['sat_pxs'][i] - MW_x)*M_to_k,(systems[syst]['sat_pys'][i]- MW_y)*M_to_k,(systems[syst]['sat_pzs'][i] - MW_z)*M_to_k,
                        s=systems[syst]['sat_rvirs'][i]*M_to_k*scaleby**2,c='blue',alpha=0.8)

        else:
            ax.scatter3D((systems[syst]['sat_pxs'][i] - MW_x)*M_to_k,(systems[syst]['sat_pys'][i]- MW_y)*M_to_k,(systems[syst]['sat_pzs'][i] - MW_z)*M_to_k,
                        s=systems[syst]['sat_rvirs'][i]*M_to_k*scaleby**2,c='red',alpha=0.8)

    
    #imsats = ax.scatter3D((systems[syst]['sat_pxs'] - MW_x)*M_to_k,(systems[syst]['sat_pys']- MW_y)*M_to_k,(systems[syst]['sat_pzs'] - MW_z)*M_to_k,
                        #s=systems[syst]['sat_rvirs']*M_to_k*scaleby**2,c=projected_v,cmap='seismic',vmin=-300,vmax=300,alpha=0.8,label='Satellites')

    #"""
    
    
    imcentral = ax.scatter3D((systems[syst]['MW_px']- MW_x)*M_to_k,(systems[syst]['MW_py']- MW_y)*M_to_k ,(systems[syst]['MW_pz'] - MW_z)*M_to_k,
                        s=systems[syst]['MW_rvir']*M_to_k*scaleby**2,c='slateblue',edgecolors='darkblue',alpha=0.4,label='Central')
    #spin = ax.quiver((systems[syst]['MW_px']-MW_x)*M_to_k,(systems[syst]['MW_py']-MW_y)*M_to_k,(systems[syst]['MW_pz']-MW_z)*M_to_k,
                        # systems[syst]['MW_lx'],systems[syst]['MW_ly'],systems[syst]['MW_lz'],color='black', length= 30, normalize=True,label='Spin')

    central = Circle(((systems[syst]['MW_px'] - MW_x)*M_to_k, (systems[syst]['MW_py'] - MW_y)*M_to_k), radius=systems[syst]['MW_rvir']*M_to_k,color='slateblue',alpha=0.4,label='Rvir of MW')
    ax.add_patch(central)
    art3d.pathpatch_2d_to_3d(central, (systems[syst]['MW_pz']-MW_z)*M_to_k, zdir="z")

    #plot the plane
    plane = ax.plot_surface((xx-MW_x)*M_to_k,(yy-MW_y)*M_to_k, (z_best-MW_z)*M_to_k,color='k' ,alpha=0.4)
    #plane_extent = np.max((xx-MW_x)*M_to_k) - np.min((xx-MW_x)*M_to_k)
    #plane_extent = "{0:.2f}".format(plane_extent)

    #plot the vector of projected velocities
    proj_vec = ax.quiver((systems[syst]['MW_px']-MW_x)*M_to_k,(systems[syst]['MW_py']-MW_y)*M_to_k,(systems[syst]['MW_pz']-MW_z)*M_to_k,
            project_onto[0],project_onto[1],project_onto[2],color='red', length= 200, normalize=True,label='Line of Sight')


    if inertia:
        v1,v2,v3,i_c_to_a = inertia[0],inertia[1],inertia[2],inertia[3]
        vec1 = ax.quiver((systems[syst]['MW_px']-MW_x)*M_to_k,(systems[syst]['MW_py']-MW_y)*M_to_k,(systems[syst]['MW_pz']-MW_z)*M_to_k,
                 v1[0],v1[1],v1[2],color='black', length= 200, normalize=True,label='Axes of Rotation')
        vec2 = ax.quiver((systems[syst]['MW_px']-MW_x)*M_to_k,(systems[syst]['MW_py']-MW_y)*M_to_k,(systems[syst]['MW_pz']-MW_z)*M_to_k,
                 v2[0],v2[1],v2[2],color='black', length= 200, normalize=True)
        vec3 = ax.quiver((systems[syst]['MW_px']-MW_x)*M_to_k,(systems[syst]['MW_py']-MW_y)*M_to_k,(systems[syst]['MW_pz']-MW_z)*M_to_k,
                 v3[0],v3[1],v3[2],color='black', length= 200, normalize=True)

        #i_c_to_a = "{0:.2f}".format(i_c_to_a)
        ax.set_title(r'MW type Satellite System, $N_{nsats}$ =' + f'{nsats}\n Physical extent, c/a:{p_c_to_a}, Inertial extent, c/a:{i_c_to_a}',y=1.15)
    else:
        ax.set_title(r'MW type Satellite System, $N_{nsats}$ =' + f'{nsats}',y=1.15)
    #plt.colorbar(imsats,label=r'Velocity of Satellites [km/s]')
    ax.autoscale('False')
    ax.set_xlabel('X [kpc]')
    ax.set_ylabel('Y [kpc]')
    ax.set_zlabel('Z [kpc]')
    plt.legend(loc="upper left", markerscale=.4)

    #"""
    extent = 600 #kpc
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

    

def rand_spherical_dist(systems,system,vel=False,level=1):

    """
    Input: systems, dict
           system, dict
           level, int
           vel, bool: if True, will assign a random velocity to your satellite
    Returns: x,y,z, float: ONE spherical redistribution of all level sats in system
    
    """

 
   
    spherical_isotropy = {}

    spherical_isotropy['sat_x'] = []
    spherical_isotropy['sat_y'] = []
    spherical_isotropy['sat_z'] = []

    spherical_isotropy['sat_vxs'] = []
    spherical_isotropy['sat_vys'] = []
    spherical_isotropy['sat_vzs'] = []

    x0 = systems[system]['MW_px'][0]
    y0 = systems[system]['MW_py'][0]
    z0 = systems[system]['MW_pz'][0]

    #redistribute satellites, preserving their separation vector, but new angles

    level_sats = np.where(systems[system]['sat_levels'] == level)
    nsats = len(level_sats[0]) 

    for k in range(nsats):
        x,y,z = systems[system]['sat_pxs'][level_sats][k],systems[system]['sat_pys'][level_sats][k],systems[system]['sat_pzs'][level_sats][k]
        rx,ry,rz = x-x0,y-y0,z-z0
        r = np.sqrt(rx**2 + ry**2 + rz**2)
 
        
        cos_theta,sin_theta, phi = rand_angle()
        
        xp = r*np.cos(phi)*sin_theta + x0
        yp = r*np.sin(phi)*sin_theta + y0
        zp = r*cos_theta + z0


        spherical_isotropy['sat_x'].append(xp)
        spherical_isotropy['sat_y'].append(yp)
        spherical_isotropy['sat_z'].append(zp)

        if vel:
            #either +100 km/s, -100 km/s
            vx = random.choice([-100,100])
            vy = random.choice([-100,100])
            vz = random.choice([-100,100])

            spherical_isotropy['sat_vxs'].append(vx)
            spherical_isotropy['sat_vys'].append(vy)
            spherical_isotropy['sat_vzs'].append(vz)



    if vel:
        return spherical_isotropy['sat_x'],spherical_isotropy['sat_y'],spherical_isotropy['sat_z'],spherical_isotropy['sat_vxs'],spherical_isotropy['sat_vys'],spherical_isotropy['sat_vzs']
    else:
        return spherical_isotropy['sat_x'],spherical_isotropy['sat_y'],spherical_isotropy['sat_z']

def rand_elliptical_dist(systems,system,vel=False,level=1,niter=1000):
    """
    Input: systems, dict
           system, dict
           level, int
           niter, int: default 1000, how many MC points to define ellipsoid of DM halo
    Returns: x,y,z, float: ONE elliptical redistribution of all level sats in system
    
    """

    

    elliptical_isotropy = {}
    elliptical_isotropy['sat_x'] = []
    elliptical_isotropy['sat_y'] = []
    elliptical_isotropy['sat_z'] = []
    elliptical_isotropy['sat_vxs']= []
    elliptical_isotropy['sat_vys'] = []
    elliptical_isotropy['sat_vzs'] = []
    x0 = systems[system]['MW_px'][0]
    y0 = systems[system]['MW_py'][0]
    z0 = systems[system]['MW_pz'][0]
    a = systems[system]['halo_a'][0]
    b = systems[system]['halo_b'][0]
    c = systems[system]['halo_c'][0]
    

    #redistribute satellites, preserving their separation vector, but new angles

    level_sats = np.where(systems[system]['sat_levels'] == level)

    nsats = len(level_sats[0]) 
    
    #generate an elliptical cloud of niter scatter points to pull nsat angles out of 
  
    xrand = [np.random.normal(0,scale=a/2) for n in range(niter)]
    yrand = [np.random.normal(0,scale=b/2) for n in range(niter)]
    zrand = [np.random.normal(0,scale=c/2) for n in range(niter)]

    for k in range(nsats):
        #separation vector of actual satellite
        x,y,z = systems[system]['sat_pxs'][level_sats][k],systems[system]['sat_pys'][level_sats][k],systems[system]['sat_pzs'][level_sats][k]
        rx,ry,rz = x-x0,y-y0,z-z0
        r = np.sqrt(rx**2 + ry**2 + rz**2)

        
        #pick a random angle from distribution
        rand_idx = random.randrange(niter)
        
        randx = xrand[rand_idx]
        randy = yrand[rand_idx]
        randz = zrand[rand_idx]
        
        
        #find the angles
        
        theta = np.arccos(randz/ (np.sqrt(randx**2 + randy**2 + randz**2)))
        if randx > 0:
            phi = np.arctan(randy/randx)
        elif randx < 0:
            phi = np.arctan(randy/randx) + np.pi 
        else:
            phi = np.pi/2
            
            
        #back into cartesian, using the original radius of the satellite
        
        xp = r*np.cos(phi)*np.sin(theta) + x0
        yp = r*np.sin(phi)*np.sin(theta) + y0
        zp = r*np.cos(theta) + z0
        
        
        
        elliptical_isotropy['sat_x'].append(xp)
        elliptical_isotropy['sat_y'].append(yp)
        elliptical_isotropy['sat_z'].append(zp)

        if vel:
            #either +100 km/s, -100 km/s
            vx = random.choice([-100,100])
            vy = random.choice([-100,100])
            vz = random.choice([-100,100])

            elliptical_isotropy['sat_vxs'].append(vx)
            elliptical_isotropy['sat_vys'].append(vy)
            elliptical_isotropy['sat_vzs'].append(vz)


    if vel:
        return elliptical_isotropy['sat_x'],elliptical_isotropy['sat_y'],elliptical_isotropy['sat_z'],elliptical_isotropy['sat_vxs'],elliptical_isotropy['sat_vys'],elliptical_isotropy['sat_vzs']

    else:
        return elliptical_isotropy['sat_x'],elliptical_isotropy['sat_y'],elliptical_isotropy['sat_z']



def check_isotropy(systems,syst,unit_n,actual_rms,n=2000,corot=False):
## check that it's uniformly dist by running n times
    t0 = time.time()

    
    rand_s_systems = {}
    rand_s_systems['systems'] = []

    rand_e_systems = {}
    rand_e_systems['systems'] = []


    #make n random systems
    for i in range(n):
        rand_s_system = {}
        rand_e_system = {}
        
        rand_s_system['MW_px'] = systems[syst]['MW_px'][0]
        rand_s_system['MW_py'] = systems[syst]['MW_py'][0]
        rand_s_system['MW_pz'] = systems[syst]['MW_pz'][0]

        rand_s_system['MW_lx'] = systems[syst]['MW_lx'][0]
        rand_s_system['MW_ly'] = systems[syst]['MW_ly'][0]
        rand_s_system['MW_lz'] = systems[syst]['MW_lz'][0]
    
    
        
        rand_e_system['MW_px'] = systems[syst]['MW_px'][0]
        rand_e_system['MW_py'] = systems[syst]['MW_py'][0]
        rand_e_system['MW_pz'] = systems[syst]['MW_pz'][0]

        rand_e_system['MW_lx'] = systems[syst]['MW_lx'][0]
        rand_e_system['MW_ly'] = systems[syst]['MW_ly'][0]
        rand_e_system['MW_lz'] = systems[syst]['MW_lz'][0]

        #define things for spherical 
        
        sx,sy,sz,svx,svy,svz= rand_spherical_dist(systems,syst,vel=True,level=1)
        
        rand_s_system['sat_pxs'] = np.asarray(sx)
        rand_s_system['sat_pys'] = np.asarray(sy)
        rand_s_system['sat_pzs'] = np.asarray(sz)


        #assign random velocities

        rand_s_system['sat_vxs'] = np.asarray(svx)
        rand_s_system['sat_vys'] = np.asarray(svy)
        rand_s_system['sat_vzs'] = np.asarray(svz)

        rand_s_systems['systems'].append(rand_s_system)
        
        #do the same for elliptical
        ex,ey,ez,evx,evy,evz = rand_elliptical_dist(systems,syst,vel=True,level=1,niter=2000)

        rand_e_system['sat_pxs'] = np.asarray(ex)
        rand_e_system['sat_pys'] = np.asarray(ey)
        rand_e_system['sat_pzs'] = np.asarray(ez)

        #assign random velocities

        rand_e_system['sat_vxs'] = np.asarray(evx)
        rand_e_system['sat_vys'] = np.asarray(evy)
        rand_e_system['sat_vzs'] = np.asarray(evz)


        rand_e_systems['systems'].append(rand_e_system)


    
    sph_mean_rms = []
    sph_corot_frac = []
    sph_c_to_a = []



    ell_mean_rms = []
    ell_corot_frac = []
    ell_c_to_a = []

    #corot_frac = corotating_frac(systems=systems,syst=syst,unit_n=unit_n,actual_rms=actual_rms,level=1)
    if corot:
        print(f'Finding best fit plane of {n} random, isotropically distributed systems...')
    for rand_syst in range(n):

        #print('SAT VXS',rand_s_systems['systems'][rand_syst]['sat_vxs'])
        
        s_best_u1,s_best_u2,s_best_u3,sph_rand_rms = evolutionary_plane_finder(systems=systems,system=rand_s_systems['systems'][rand_syst],n_iter = 200,n_start=25,n_erase=10,n_avg_mutants=5,level=1,rand=True,verbose=False)
        sph_mean_rms.append(sph_rand_rms)

        e_best_u1,e_best_u2,e_best_u3,ell_rand_rms = evolutionary_plane_finder(systems=systems,system=rand_e_systems['systems'][rand_syst],n_iter = 200,n_start=25,n_erase=10,n_avg_mutants=5,level=1,rand=True,verbose=False) 
        ell_mean_rms.append(ell_rand_rms)
    

        if corot:
            #find best fit plane of n random systems
            
            a,b,c,s_phys_c_to_a = find_physical_extent(u1=s_best_u1,u2=s_best_u2,u3=s_best_u3,systems=rand_s_systems['systems'],system = rand_syst,actual_rms=sph_rand_rms,rand = True,nrms = 2,level=1)
            sph_c_to_a.append(s_phys_c_to_a)

            s_corot_frac = corotating_frac(systems=rand_s_systems['systems'],syst=rand_syst,unit_n=unit_n,actual_rms=sph_rand_rms,rand=True,nrms=1,level=1)
            sph_corot_frac.append(s_corot_frac)

            a,b,c,e_phys_c_to_a = find_physical_extent(u1=e_best_u1,u2=e_best_u2,u3=e_best_u3,systems=rand_e_systems['systems'],system=rand_syst,actual_rms=ell_rand_rms,rand = True,nrms = 2,level=1)
            ell_c_to_a.append(e_phys_c_to_a)

            e_corot_frac = corotating_frac(systems=rand_e_systems['systems'],syst=rand_syst,unit_n=unit_n,actual_rms=ell_rand_rms,rand=True,nrms=1,level=1)
            ell_corot_frac.append(corot_frac)

    t1 = time.time()

    print(f'Took {t1-t0} seconds.')

    if corot:
        return sph_mean_rms,ell_mean_rms,sph_corot_frac,sph_c_to_a,ell_corot_frac,ell_c_to_a
    else:
        return sph_mean_rms,ell_mean_rms

def save_hist(name_of_plot,best_rms,mean_rms,snapshot,type='spherical',histbins=70):
    """
    Input:

    Output:
        saves figure
        returns significance 
    """

    n = len(mean_rms)

    fig, ax = plt.subplots(1, 1,
                            figsize =(8,5), 
                            tight_layout = True)

    histbins = 70
    
    #change color according to which type of distribution you're pulling from
    if type=='spherical':
        ncounts,dense_bins,patches = ax.hist(mean_rms, density=True,bins =histbins,ec='purple',fc='thistle')

    elif type=='elliptical':
        ncounts,dense_bins,patches = ax.hist(mean_rms, density=True,bins =histbins,ec='royalblue',fc='cornflowerblue')
    ax.axvline(x=best_rms,c='black',label='mean rms of actual distribution')

    mu, sigma = ss.norm.fit(mean_rms)
    best_fit_line = ss.norm.pdf(dense_bins, mu, sigma)
    significance = (mu-best_rms)/sigma
    
    if type=='spherical':
        ax.plot(dense_bins,best_fit_line,c='purple',label='PDF')
        ax.set_title(f'Spherically Isotropic Distribution of {n} Planes')

    elif type=='elliptical':
        ax.plot(dense_bins,best_fit_line,c='midnightblue',label='PDF')
        ax.set_title(f'Elliptical (Tracing DM Halo) Distribution of {n} Planes')

    ax.axvline(x=mu,c='red',label='$\mu$, mean rms of random distributions')
    ax.axvline(x=best_rms,c='black',label=f'mean rms of actual distribution, {significance:.2f} $\sigma$ from $\mu$')

    ax.set_xlabel(r'Mean RMS')
    ax.legend()
    
    # Save plot
    #check if directory exists

    #script_dir = os.path.dirname(__file__)
    #results_dir = os.path.join(script_dir, 'iso_histograms/')

    #where it will be saved to 
    if type=='spherical':
        results_dir = '/data78/welker/madhani/iso_histograms/spherical/' + str(snapshot) + '/'
    elif type=='elliptical':
        results_dir = '/data78/welker/madhani/iso_histograms/elliptical/' + str(snapshot) + '/'
    

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    print(f'Saving histogram to:  {results_dir + name_of_plot}')
    plt.savefig(results_dir + name_of_plot)

    return significance 



######################################
# Line of Sight Analysis and Extent of Plane
######################################


def find_inertia_tensor(syst,level=1):
    """
    Input: dictionary, syst
           integer, level: what level sats you're looking at the inertia tensor of
    Returns: 3x3 array, inertia tensor 
    """
    level_sats = np.where(syst['sat_levels'] == level)
    nsats = len(level_sats[0]) 
    sat_ms = syst['sat_mvirs'][level_sats]
    sat_ms /= 10e8
    sat_xs = syst['sat_pxs'][level_sats]
    sat_ys = syst['sat_pys'][level_sats]
    sat_zs = syst['sat_pzs'][level_sats]

    Ixx = np.sum([(sat_ys[i]**2 + sat_zs[i]**2) * sat_ms[i] for i in range(nsats)])
    Iyy = np.sum([(sat_xs[i]**2 + sat_zs[i]**2) * sat_ms[i] for i in range(nsats)])
    Izz = np.sum([(sat_xs[i]**2 + sat_ys[i]**2) * sat_ms[i] for i in range(nsats)])
    Ixy = np.sum([(sat_xs[i] * sat_ys[i]) * sat_ms[i] for i in range(nsats)])
    Ixz = np.sum([(sat_xs[i] * sat_zs[i]) * sat_ms[i] for i in range(nsats)])
    Iyz = np.sum([(sat_ys[i] * sat_zs[i]) * sat_ms[i] for i in range(nsats)])
    
    Iyx = Ixy
    Izx = Ixz
    Izy = Iyz
    
    I = np.array(((Ixx,-Ixy,-Ixz),(-Iyx,Iyy,-Iyz),(-Izx,-Izy,Izz)))

    return I 

def find_axes_of_rot(I):
    evals,evec = np.linalg.eig(I)
    vec1,vec2,vec3 = evec[:,0],evec[:,1],evec[:,2]
    return vec1,vec2,vec3


def find_axes_ratios(I):
    evals,evec = np.linalg.eig(I)
    lam1, lam2, lam3 = evals[0],evals[1],evals[2]

    lam3,lam2,lam1 = np.sort([lam1,lam2,lam3])
    
    c_to_a = np.sqrt(lam3 + lam2 - lam1) / np.sqrt(lam1 + lam2 - lam3)
    
    return float(c_to_a)

def find_physical_extent(u1,u2,u3,systems,system,actual_rms,rand=False,nrms = 2,level=1):
    if rand:
        nsats = len(systems[system]['sat_vxs'])
    
    else:
        level_sats = np.where(systems[system]['sat_levels'] == level)
        nsats = len(level_sats[0]) 
    
    #calc relevant angles 
    
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
    
    
    unx, uny, unz = unit_n[0],unit_n[1],unit_n[2]

    if rand:
        #calculate distances to best plane, then the extent
        x0 = systems[system]['MW_px']
        y0 = systems[system]['MW_py']
        z0 = systems[system]['MW_pz']
    
    else:
        #calculate distances to best plane, then the extent
        x0 = systems[system]['MW_px'][0]
        y0 = systems[system]['MW_py'][0]
        z0 = systems[system]['MW_pz'][0]
    
    

    gal_center = np.array([x0,y0,z0])


    d = np.dot(-gal_center,unit_n)
    
    distances = []
    sep_vect = []
    #calculate the distance of ALL the sats (needed for shape)
    for k in range(len(systems[system]['sat_pxs'])):
        x,y,z = systems[system]['sat_pxs'][k],systems[system]['sat_pys'][k],systems[system]['sat_pzs'][k]
        rx,ry,rz = x-x0,y-y0,z-z0
        r = np.sqrt(rx**2 + ry**2 + rz**2)
        s = dist(x,y,z,unit_n,d)
        distances.append(s) 
        sep_vect.append(r)
        
        
    distances = np.asarray(distances)
    sep_vect = np.asarray(sep_vect)
    
    #find satellites within n * rms AND right level
    #nrms = the number of rms within which you want to consider satellites as "on-plane"
    if rand:
        win_rms = np.where(distances <= nrms * actual_rms) 

    else:
        win_rms = np.where((distances <= nrms * actual_rms) & (systems[system]['sat_levels'] == level))
    
    
    distances = distances[win_rms]
    #print('sat_pxs',systems[system]['sat_pxs'])
    #print('type',type(systems[system]['sat_pxs']))
    x_win_rms = systems[system]['sat_pxs'][win_rms]
    y_win_rms = systems[system]['sat_pys'][win_rms]
    z_win_rms = systems[system]['sat_pzs'][win_rms]
    sep_vect = sep_vect[win_rms]
    
    xdists = []
    ydists = []
    
    
    #Now use similar triangles, using the separation vector's projection on the plane (the satellite's hypotenuse)
    #and the hypotenuse of the normal vector's projection on the plane, ie. the unit nx and ny vector's hyp
    
    u_n_hyp = np.sqrt(unx**2 + uny**2)
    
    #projection of sep vectors on plane
    a1 = np.sqrt(sep_vect**2 - distances**2)
    
    #x extents
    a2 = unx * (a1/u_n_hyp)  
    
    #y extents 
    
    a3 = uny * (a1/u_n_hyp) 

    #put in a clause to catch instances where there is no array
    
    if len(a2) > 1:
        xmin, xmax = np.min(a2), np.max(a2)
        ymin, ymax = np.min(a3), np.max(a3)
        zmin, zmax = np.min(distances), np.max(distances)  
    
    

        """
        xmin, xmax = np.min(x_win_rms), np.max(x_win_rms)
        ymin, ymax = np.min(y_win_rms), np.max(y_win_rms)
        zmin, zmax = np.min(z_win_rms), np.max(z_win_rms)
        """
    
        M_to_k = 1000
        x_extent = np.abs(xmax-xmin) * M_to_k
        y_extent = np.abs(ymax-ymin) * M_to_k
        z_extent = np.abs(zmax-zmin) * M_to_k

        

        #print(xmin,xmax,ymin,ymax,zmin,zmax)
        #print(x_extent,y_extent,z_extent)
        
        extents = [x_extent,y_extent,z_extent]
        extents = sorted(extents)
        c,b,a = extents[0], extents[1], extents[2]
        c_to_a = c/a
        return(a,b,c,c_to_a)
    
    #can mask out these erroneous vals later
    else:
        return(0,0,0,0)

    

def save_outputs(name_of_file,snapshot,systems,syst,inertial,physical,sig_spherical=2,sig_elliptical=2):
    """
    Input: systems_file,str: path to systems file

    Returns: system_[x]_data, .json file : where all the relevant information is stored 
    """

    #extract information you want to include in new file 
    halo_id = systems[syst]['halo_ID']
    location_of_central = [ systems[syst]['MW_px'], systems[syst]['MW_py'], systems[syst]['MW_pz'] ]
    halo_axes = [ systems[syst]['halo_a'], systems[syst]['halo_b'], systems[syst]['halo_c'] ]
    a = systems[syst]['aexp']
    z = 1/(1+a)

    syst_analysis = {}
    syst_analysis['aexp'] = a
    syst_analysis['z'] = z
    syst_analysis['halo_id'] = halo_id
    syst_analysis['location_of_central'] = location_of_central
    syst_analysis['halo_axes'] = halo_axes
    syst_analysis['physical_extent: a,b,c,c_to_a'] = physical
    syst_analysis['inertial_extent: a,b,c,c_to_a'] = inertial

    #ONCE YOU ACTUALLY CALCULATE THIS, THIS WILL CHANGE FROM RANDO DEFAULT VAL
    syst_analysis['spherical_significance'] = sig_spherical
    syst_analysis['elliptical_significance'] = sig_elliptical 

   
    results_dir = '/data78/welker/madhani/analysis_data/' + str(snapshot) + '/'
    

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    print(f'Saving output data to:  {results_dir + name_of_file}')
    #file = open(results_dir + name_of_file, "w")
    #json.dump(syst_analysis, file)
    #file.close()

    file = results_dir + name_of_file
    w = csv.writer(open(file, "w"))

    # loop over dictionary keys and values
    for key, val in syst_analysis.items():

        # write every key and value to file
        w.writerow([key, val])



