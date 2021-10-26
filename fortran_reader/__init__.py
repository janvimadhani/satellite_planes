import os
from os.path import join
import re
import numpy as np
import time as time
from scipy.io import FortranFile
import ctypes as c
import struct





class ReadTreebrick_lowp:
    #low precision Treebricks (mostly HAGN)
    def __init__(self,file_path=None):
        
        """
        Make a treebricks dictionary out of file
        """
        self.file_path = file_path
        self.treebricks_dict = None
        self.read_data()
   
        
    def read_data(self):
        t0 = time.time()
        f = FortranFile(self.file_path, 'r')
        
        nbodies = f.read_record('i')
        mpart = f.read_record('f')
        aexp = f.read_record('f')
        omega_t = f.read_record('f')
        age_univ = f.read_record('f')
        nsub = f.read_record('i')
        
        
        self.treebricks_dict = {}
        self.treebricks_dict['nbodies'] = nbodies
        self.treebricks_dict['mpart'] = mpart
        self.treebricks_dict['aexp'] = aexp
        self.treebricks_dict['omega_t'] = omega_t
        self.treebricks_dict['age_univ'] = age_univ
        self.treebricks_dict['nh_old'] = nsub[0]
        self.treebricks_dict['nsub_old'] = nsub[1]
        nhaloes = nsub[0]+nsub[1]
        self.treebricks_dict['nhaloes'] = nhaloes
        #initialize empty list to hold all halos
        self.treebricks_dict['haloes'] = []
        
        print('nbodies:',nbodies,
              'mpart:', mpart,
              'aexp:', aexp,
              'omega_t:', omega_t,
              'age:', age_univ,
              'nsub:',nsub,
              'nhaloes:', nhaloes)
        
        
        
        nb_of_haloes = self.treebricks_dict['nh_old']+self.treebricks_dict['nsub_old']
        
        #define the length of the box in Mpc for New Horizon
        lbox_NH=2.0*0.07*(142.8571428)*aexp 
        lbox_HAGN = (142.8571428)*aexp
        halfbox=lbox_NH/2.0
        self.treebricks_dict['lbox_NH'] = lbox_NH
        self.treebricks_dict['lbox_HAGN'] = lbox_HAGN
        
        
        #def read_halo():
        t1 = time.time()
        for i in range(nhaloes):
            halo_dict = {}

            npart = f.read_record('i')
            halo_dict['npart'] = npart.tolist()

            #members = f.read_record('i').reshape(npart, order='F')
            members = f.read_record('i')
            halo_dict['members'] = members.tolist()

            my_number = f.read_record('i')
            halo_dict['my_number'] = my_number.tolist()

            my_timestep = f.read_record('i')
            halo_dict['my_timestep'] = my_timestep.tolist()


            level_ids = f.read_record('i')
            level_ids = level_ids.tolist()

            level,host_halo,host_sub,nchild,nextsub = level_ids[0],level_ids[1],level_ids[2],level_ids[3],level_ids[4]
            halo_dict['level'] = level
            halo_dict['host_halo']  = host_halo
            halo_dict['host_sub'] = host_sub
            halo_dict['nchild'] = nchild
            halo_dict['nextsub'] = nextsub

            mass = f.read_record('f') #d for NH
            halo_dict['mass'] = mass.tolist()

            p = f.read_record('f')
            p = p.tolist()
            py,px,pz = p[0],p[1],p[2]
            halo_dict['px'] = px
            halo_dict['py'] = py
            halo_dict['pz'] = pz

            v = f.read_record('f')
            v = v.tolist()
            vx,vy,vz = v[0],v[1],v[2]  
            halo_dict['vx'] = vx
            halo_dict['vy'] = vy
            halo_dict['vz'] = vz

            L = f.read_record('f')
            L = L.tolist()
            Lx,Ly,Lz = L[0],L[1],L[2]
            halo_dict['Lx'] = Lx
            halo_dict['Ly'] = Ly
            halo_dict['Lz'] = Lz

            shape = f.read_record('f')
            shape = shape.tolist()
            rmax,a,b,c = shape[0],shape[1],shape[2],shape[3]
            halo_dict['rmax'] = rmax
            halo_dict['a'] = a 
            halo_dict['b'] = b
            halo_dict['c'] = c

            energy = f.read_record('f')
            energy = energy.tolist()
            ek,ep,et = energy[0],energy[1],energy[2]
            halo_dict['ek'] = ek
            halo_dict['ep'] = ep
            halo_dict['et'] = et

            spin = f.read_record('f')
            halo_dict['spin'] = spin.tolist()

            virial = f.read_record('f')
            virial = virial.tolist()
            rvir,mvir,tvir,cvel = virial[0],virial[1],virial[2],virial[3]
            halo_dict['rvir'] = rvir
            halo_dict['mvir'] = mvir
            halo_dict['tvir'] = tvir
            halo_dict['cvel'] = cvel 

            halo_profile = f.read_record('f')
            halo_profile = halo_profile.tolist()
            rho_0, r_c = halo_profile[0],halo_profile[1]
            halo_dict['rho_0'] = rho_0
            halo_dict['r_c'] = r_c
            
            #Positions are in Mpc, we now put them back in "code units", that is assuming the length of the simulation is 1.
            #for New Horizon scale
            """
            
            px_NH = (halo_dict['px']/self.treebricks_dict['lbox_NH']) + 0.5
            halo_dict['px_NH'] = px_NH
            
            py_NH = (halo_dict['py']/self.treebricks_dict['lbox_NH']) + 0.5
            halo_dict['py_NH'] = py_NH
           
            pz_NH = (halo_dict['pz']/self.treebricks_dict['lbox_NH']) + 0.5
            halo_dict['pz_NH'] = pz_NH
            
            rvir_NH = (halo_dict['rvir']/self.treebricks_dict['lbox_NH'])
            halo_dict['rvir_NH'] = rvir_NH

            
            #for Horizon AGN scale
            px_HAGN = (halo_dict['px']/self.treebricks_dict['lbox_HAGN']) + 0.5
            halo_dict['px_HAGN'] = px_HAGN
            
            py_HAGN = (halo_dict['py']/self.treebricks_dict['lbox_HAGN']) + 0.5
            halo_dict['py_HAGN'] = py_HAGN
            
            pz_HAGN = (halo_dict['pz']/self.treebricks_dict['lbox_HAGN']) + 0.5
            halo_dict['pz_HAGN'] = pz_HAGN
            
            rvir_HAGN = (halo_dict['rvir']/self.treebricks_dict['lbox_HAGN'])
            halo_dict['rvir_HAGN'] = rvir_HAGN
            """

            #add halo dict to main dict under key of my_number
            
            #return halo_dict
            self.treebricks_dict['haloes'].append(halo_dict)
            
            
        #t1 = time.time()
        #for i in range(nhaloes):
            #halo_dict = read_halo()
            #self.treebricks_dict['haloes'].append(halo_dict)
            
        t2 = time.time()
        print('Reading haloes took {:0.2f} secs.'.format(t2-t1))
        print('Total time was {:0.2f} secs.'.format(t2-t0))
            
        return self.treebricks_dict
    
class GalaxyCatalog:
    #High precision Treebricks (NH)
    def __init__(self,file_path=None):
        
        """
        Make a treebricks dictionary out of file
        """
        self.file_path = file_path
        self.treebricks_dict = None
        self.read_data()
   
        
    def read_data(self):
        t0 = time.time()
        f = FortranFile(self.file_path, 'r')
        
        nbodies = f.read_record('i')
        mpart = f.read_record('d')
        aexp = f.read_record('d')
        omega_t = f.read_record('d')
        age_univ = f.read_record('d')
        nsub = f.read_record('i')
        
        
        self.treebricks_dict = {}
        self.treebricks_dict['nbodies'] = nbodies
        self.treebricks_dict['mpart'] = mpart
        self.treebricks_dict['aexp'] = aexp
        self.treebricks_dict['omega_t'] = omega_t
        self.treebricks_dict['age_univ'] = age_univ
        self.treebricks_dict['nb_of_galaxies'] = nsub[0]
        self.treebricks_dict['nb_of_subgals'] = nsub[1]
        nmax = nsub[0]+nsub[1]
        self.treebricks_dict['nmax'] = nmax
        #initialize empty list to hold all halos
        self.treebricks_dict['galaxies'] = []
        
        print('nbodies:',nbodies,
              'mpart:', mpart,
              'aexp:', aexp,
              'omega_t:', omega_t,
              'age:', age_univ,
              'nsub:',nsub,
              'nmax:', nmax)
        
        
        
                
        #define the length of the box in Mpc for New Horizon
        lbox_NH=2.0*0.07*(142.8571428)*aexp 
        lbox_HAGN = (142.8571428)*aexp
        halfbox=lbox_NH/2.0
        self.treebricks_dict['lbox_NH'] = lbox_NH
        self.treebricks_dict['lbox_HAGN'] = lbox_HAGN
        
        
        #def read_halo():
        t1 = time.time()
        for i in range(nmax):
            gal_dict = {}

            npart = f.read_record('i')
            gal_dict['npart'] = npart.tolist()

            #members = f.read_record('i').reshape(npart, order='F')
            members = f.read_record('i')
            gal_dict['members'] = members.tolist()

            my_number = f.read_record('i')
            gal_dict['my_number'] = my_number.tolist()

            my_timestep = f.read_record('i')
            gal_dict['my_timestep'] = my_timestep.tolist()


            level_ids = f.read_record('i')
            level_ids = level_ids.tolist()
            #print('level ids',level_ids)
            level,host_gal,host_subgal,nchild,nextsub = level_ids[0],level_ids[1],level_ids[2],level_ids[3],level_ids[4]
            gal_dict['level'] = level
            gal_dict['host_gal']  = host_gal
            gal_dict['host_subgal'] = host_subgal
            gal_dict['nchild'] = nchild
            gal_dict['nextsub'] = nextsub

            mass = f.read_record('d') #d for NH
            #print('mass',mass)
            gal_dict['mass'] = mass.tolist()

            p = f.read_record('d')
            #print('p',p)
            p = p.tolist()
            py,px,pz = p[0],p[1],p[2]
            gal_dict['px'] = px
            gal_dict['py'] = py
            gal_dict['pz'] = pz


            v = f.read_record('d')
            #print('v',v)
            v = v.tolist()
            vx,vy,vz = v[0],v[1],v[2]  
            gal_dict['vx'] = vx
            gal_dict['vy'] = vy
            gal_dict['vz'] = vz

            L = f.read_record('d')
            #print('L',L)
            L = L.tolist()
            Lx,Ly,Lz = L[0],L[1],L[2]
            gal_dict['Lx'] = Lx
            gal_dict['Ly'] = Ly
            gal_dict['Lz'] = Lz

            shape = f.read_record('d')
            #print('shape',shape)
            shape = shape.tolist()
            rmax,a,b,c = shape[0],shape[1],shape[2],shape[3]
            gal_dict['rmax'] = rmax
            gal_dict['a'] = a 
            gal_dict['b'] = b
            gal_dict['c'] = c

            energy = f.read_record('d')
            #print('energy',energy)
            energy = energy.tolist()
            ek,ep,et = energy[0],energy[1],energy[2]
            gal_dict['ek'] = ek
            gal_dict['ep'] = ep
            gal_dict['et'] = et

            spin = f.read_record('d')
            #print('spin',spin)
            gal_dict['spin'] = spin.tolist()

            sigma = f.read_record('d')
            #print('sigma',sigma)
            sigma = sigma.tolist()
            sig,sigma_bulge,m_bulge = sigma[0],sigma[1],sigma[2]
            gal_dict['sigma'] = sig
            gal_dict['sigma_bulge'] = sigma_bulge
            gal_dict['m_bulge'] = m_bulge
            
            virial = f.read_record('d')
            virial = virial.tolist()
            rvir,mvir,tvir,cvel = virial[0],virial[1],virial[2],virial[3]
            gal_dict['rvir'] = rvir
            gal_dict['mvir'] = mvir
            gal_dict['tvir'] = tvir
            gal_dict['cvel'] = cvel 

            
            halo_profile = f.read_record('d')
            #print('halo profile',halo_profile)
            halo_profile = halo_profile.tolist()
            rho_0, r_c = halo_profile[0],halo_profile[1]
            gal_dict['rho_0'] = rho_0
            gal_dict['r_c'] = r_c
            
            #stellar density profiles
            
            nbins = f.read_record('i')
            #print('nbins:',nbins)
            
            rr = f.read_record('d') #array (nbins)
            gal_dict['rr'] = rr
            rho = f.read_record('d')
            gal_dict['rr'] = rho
            
            #print('shape of density profile',np.shape(rho))
            


            #add gal dict to main dict 
            
            #return halo_dict
            self.treebricks_dict['galaxies'].append(gal_dict)
            
            
        #t1 = time.time()
        #for i in range(nhaloes):
            #halo_dict = read_halo()
            #self.treebricks_dict['haloes'].append(halo_dict)
            
        t2 = time.time()
        print('Reading galaxies took {:0.2f} secs.'.format(t2-t1))
        print('Total time was {:0.2f} secs.'.format(t2-t0))
            
        return self.treebricks_dict
    
    

class ReadGalaxy:
    def __init__(self,file_path=None):
        
        """
        Make a galaxies dictionary out of binary file
        """
        self.file_path = file_path
        self.galaxies_dict = None
        self.read_data()
   
        
    def read_data(self):
        t0 = time.time()
        f = FortranFile(self.file_path, 'r')
        
        self.galaxies_dict = {}
        
        
        my_number = f.read_record('i')
        self.galaxies_dict['my_number'] = my_number
        
        #gal or sub-gal => merger detection?
        level = f.read_record('i')
        self.galaxies_dict['level'] = level
        
        #galactic mass
        mass = f.read_record('d')
        self.galaxies_dict['mass'] = mass
        
        #galactic position
        p = f.read_record('d')
        p = p.tolist()
        py,px,pz = p[0],p[1],p[2]
        self.galaxies_dict['px'] = px
        self.galaxies_dict['py'] = py
        self.galaxies_dict['pz'] = pz
        
        #galactic velocity
        v = f.read_record('d')
        v = v.tolist()
        vx,vy,vz = v[0],v[1],v[2]  
        self.galaxies_dict['vx'] = vx
        self.galaxies_dict['vy'] = vy
        self.galaxies_dict['vz'] = vz
        
        #galactic ang mom
        L = f.read_record('d')
        L = L.tolist()
        Lx,Ly,Lz = L[0],L[1],L[2]
        self.galaxies_dict['Lx'] = Lx
        self.galaxies_dict['Ly'] = Ly
        self.galaxies_dict['Lz'] = Lz
        
        #number of stars
        nstars = f.read_record('i')
        self.galaxies_dict['nstars'] = nstars     
        
        #stellar positions
        x_stars = f.read_record('d')
        self.galaxies_dict['x_stars'] = x_stars.tolist()
        
        y_stars = f.read_record('d')
        self.galaxies_dict['y_stars'] = y_stars.tolist()
        
        z_stars = f.read_record('d')
        self.galaxies_dict['z_stars'] = z_stars.tolist()
        
        
        #stellar velocity
        vx_stars = f.read_record('d')
        self.galaxies_dict['vx_stars'] = vx_stars.tolist()
        
        vy_stars = f.read_record('d')
        self.galaxies_dict['vy_stars'] = vy_stars.tolist()
        
        vz_stars = f.read_record('d')
        self.galaxies_dict['vz_stars'] = vz_stars.tolist()
        
        #stellar mass
        mass_stars = f.read_record('d')
        self.galaxies_dict['mass_stars'] = mass_stars.tolist()
        
        #star id
        id_stars = f.read_record('i')
        self.galaxies_dict['id_stars'] = id_stars.tolist()
        
        #stellar age
        age_stars = f.read_record('d')
        self.galaxies_dict['age_stars'] = age_stars.tolist()
        
        #metallicity
        zz_stars = f.read_record('d')
        self.galaxies_dict['zz_stars'] = zz_stars.tolist()
        
        t1 = time.time()
        
        print('Reading galaxy took {:0.2f} secs.'.format(t1-t0))
        return self.galaxies_dict
    
    
    
    
class ReadDat:
    def __init__(self,file_path=None):
        
        """
        Return a cube array
        """
        self.file_path = file_path
        self.cube = None
        self.read_cube_data()
        
        
    def read_cube_data(self):
        t0 = time.time()
        f = FortranFile(self.file_path, 'r')
        n = f.read_record('i')
        nx,ny,nz = n[0],n[1],n[2]
        
        self.cube = f.read_record('f4').reshape(nx,ny,nz, order='F')
        t1 = time.time()
        
        print('Reading cube took {:0.2f} secs.'.format(t1-t0))
        return self.cube
        
        
   
    
class ReadFilament:
    def __init__(self,file_path=None):
        
        """
        Make a treebricks dictionary out of file
        """
        self.file_path = file_path
        self.filament_dict = None
        self.read_data()
   

        
    def read_data(self):
        t0 = time.time()
        f = FortranFile(self.file_path, 'r')
        
        self.filament_dict = {}
        
        header1 = f.read_record('i')
        print('header1',header1)
        
        tag = f.read_record()
        self.filament_dict['tag'] = tag
        print('tag:',tag)
        
        header2 = f.read_record('i')
        header3 = f.read_record('i')
        
        #comment
        comment,ndims2,dims,x0,delta,nsegs,nnodes,nsegdata,nnodedata= f.read_record('str','i','i','d','d','i','i','i','i')
        print(comment)
        
        self.filament_dict['ndims2'] = ndims2
        self.filament_dict['dims'] = dims
        self.filament_dict['x0'] = x0
        self.filament_dict['delta'] = delta
        self.filament_dict['nsegs'] = nsegs #number of filament segments
        self.filament_dict['nnodes'] = nnodes #number of filament segments, nnodes is the number of nodes the 'data' tags indicate how many variables are stored for filaments and nodes (such as density of the filament segment for instance"
        self.filament_dict['nsegdata'] = nsegdata
        self.filament_dict['nnodedata'] = nnodedata
        
        print(ndims2)
        
        dum = f.read_record('i')
        
        if nsegdata != 0:
            dum = f.read_record('i')
            seg_variable_names = f.read_record('str') #these Characters actually give the names of the nsegdata variables (such as "density") 
            self.filament_dict['seg variable names'] = seg_variable_names
            dum = f.read_record('i')
        
        if nnodedata != 0:
            dum = f.read_record('i')
            node_variable_names = f.read_record('str') #these Characters actually give the names of the nnodedata  variables 
            self.filament_dict['node variable names'] = node_variable_names
            dum = f.read_record('i')
            
        dum = f.read_record('i')
        
        #read filament segments (positions of each extremity)
        segpos = f.read_record('f')
        self.filament_dict['seg pos'] = segpos
        
        dum = f.read_record('i')
        dum = f.read_record('i')
        
        #read node positions
        nodepos = f.read_record('f')
        self.filament_dict['node pos'] = nodepos
        
        dum = f.read_record('i')
        dum = f.read_record('i')      
        
        #segment data
        segdata = f.read_record('d')
        self.filament_dict['seg data'] = segdata
        
        dum = f.read_record('i')
        dum = f.read_record('i')  
        
        #node data
        nodedata = f.read_record('d')
        self.filament_dict['node data'] = nodedata
        
        dum = f.read_record('i') 
        
        self.filament_dict['nodes'] = []
        
        for i in range(nnodes):
            node_dict = {}
            dum = f.read_record('i')
            indexn,dum,dum,narcs,ttype = f.read_record('i','i','i','i','i')
            dum,dumi,dumi = f.read_record('i','i','i')
            
            node_dict['type'] = ttype
            node_dict['indexn'] = indexn
            node_dict['narcs'] = narcs
            self.filament_dict['nodes'].append(node_dict)
    
    
        t1 = time.time()
        print('Reading filament took {:0.2f} secs.'.format(t1 - t0))
            
        return self.filament_dict
    
    