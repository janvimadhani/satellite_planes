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
        Make a filaments dictionary out of ASCII NDSKL file
        """
        self.file_path = file_path
        self.filament_dict = None
        self.read_data()
   

        
    def read_data(self):
        t0 = time.time()
        self.filament_dict = {}

        #read the file first and write each line into data
        data = []
        f = open(self.file_path,'r')
        for line in f:
            data.append(line)
        f.close()
        
        def convert_to_list(ascii_chars,type=float):
            #strip whitespace from ends
            ascii_chars = str(ascii_chars)

            char_list = list(ascii_chars.split(" "))
            char_list = ' '.join(char_list).split()
            

            py_list = list(map(type, char_list))
            
            return py_list
        
        header1 = data[0]
        print('header1,',header1)
        
        ndims = data[1]
        print('ndims,', ndims)

        comments = data[2]
        print('Comments,',comments)

        extent = data[3]
        print('Bounding box,', extent)

        #data[4] is str(Critical Points)

        ncrit = int(data[5])
        print('ncrit,', ncrit)
        self.filament_dict['ncrit'] = ncrit

        #store all data for critical points in here
        self.filament_dict['critical_points'] = []

        #CPs
        
        add_to_idx = 6 
        for i in range(ncrit):
            cp_dict = {}
            i = 0
            
            i += add_to_idx #make sure you are at the right line in the data list 
            critical_vals = data[i]
            
            c_idx, px, py, pz, value, pairID, boundary = convert_to_list(critical_vals)
            #next line in data

            cp_dict['cp_idx']  = c_idx
            cp_dict['px'] = px 
            cp_dict['py'] = py
            cp_dict['pz'] = pz
            cp_dict['pair_ID'] = pairID
            cp_dict['boundary'] = boundary 


            i += 1
            nfil = int(data[i])
            cp_dict['nfil'] = nfil
            cp_dict['destID,filID'] = []
            
            for k in range(nfil):

                i += 1
                cp_on_fil = data[i]
                destID, filID = convert_to_list(cp_on_fil,int)
        
                cp_dict['destID,filID'].append([destID,filID])

            #make this to fill out later
            cp_dict['Field Vals'] = []
            #add all info to cp dict
            self.filament_dict['critical_points'].append(cp_dict)
            
            add_to_idx = i + 1


        #Filaments
        #store all data for filaments in here
        self.filament_dict['filaments'] = []

        fil_idx = i + 1
        nfils = int(data[fil_idx+1])
        self.filament_dict['nfils'] = nfils
        print('nfils,', nfils)

        fil_add = fil_idx+2
        for i in range(nfils):
            i = 0
            fil_dict = {}
            
            i += fil_add #make sure you are at the right line in the data list 
            fil_info = data[i]
            
            cp1_idx, cp2_idx, nsamp = convert_to_list(fil_info)
            nsamp = int(nsamp)
            
            fil_dict['cp1_idx'] = cp1_idx
            fil_dict['cp2_idx'] = cp2_idx
            fil_dict['nsamp'] = nsamp
            fil_dict['px,py,pz'] = []

            
            for k in range(nsamp):

                i += 1
                positions = data[i]
                px,py,pz = convert_to_list(positions)
                #print('px,py,pz:',px,py,pz)
                fil_dict['px,py,pz'].append([px,py,pz])
            
            #make this to fill out later
            fil_dict['Field Vals'] = []

            #add filament info to dict
            self.filament_dict['filaments'].append(fil_dict)
            fil_add = i + 1

        cp_dat_idx = i + 1


        #Field Data
        print('Reading data fields:')
        nb_cp_dat_fields = int(data[cp_dat_idx+1])
        cp_dat_add = cp_dat_idx+2
        self.filament_dict['nb_CP_fields'] = nb_cp_dat_fields
        self.filament_dict['CP_fields'] = []

        for i in range(nb_cp_dat_fields):
            i = 0
            i += cp_dat_add #make sure you are at the right line in the data list 
            cp_field_info = data[i]
            print('CP field:',cp_field_info)
            self.filament_dict['CP_fields'].append(cp_field_info)
            
            cp_dat_add = i + 1

        cp_field_val_idx = i + 1 

        cp_val_add = cp_field_val_idx
        cp_field_vals = []
        for i in range(ncrit):
            i = 0
            i += cp_val_add #make sure you are at the right line in the data list 
            cp_field_val_info = data[i]
            list_of_cp_vals = convert_to_list(cp_field_val_info)
            cp_field_vals.append(list_of_cp_vals)
            
            cp_val_add = i + 1
            
        fil_dat_idx = i + 1  

        #put the field vals in the right place in the dictionary
        for j in range(ncrit):
            self.filament_dict['critical_points'][j]['Field Vals'] = cp_field_vals[j]
        
        nb_fil_dat_fields = int(data[fil_dat_idx+1])
        self.filament_dict['nb_fil_fields'] = nb_fil_dat_fields
        self.filament_dict['fil_fields'] = []
        fil_dat_add = fil_dat_idx+2
        for i in range(nb_fil_dat_fields):
            i = 0
            i += fil_dat_add #make sure you are at the right line in the data list 
            fil_field_info = data[i]
            print('Filament field:',fil_field_info)
            self.filament_dict['fil_fields'].append(fil_field_info)
            
            fil_dat_add = i + 1

        fil_field_val_idx = i + 1  

        fil_val_add = fil_field_val_idx

        fil_field_vals = []
        for i in range(nfils):
            i = 0
            i += fil_val_add #make sure you are at the right line in the data list 
            fil_field_val_info = data[i]
            list_of_fil_vals = convert_to_list(fil_field_val_info)
            fil_field_vals.append(list_of_fil_vals)
            
            fil_val_add = i + 1
        #put the field vals in the right place in the dictionary
        for j in range(nfils):
            self.filament_dict['filaments'][j]['Field Vals'] = fil_field_vals[j]
    

    
        t1 = time.time()
        print('Reading filaments took {:0.2f} secs.'.format(t1 - t0))
            
        return self.filament_dict
    
    