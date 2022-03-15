import os
from os.path import join
import re
import numpy as np
import time as time
from scipy.io import FortranFile
import ctypes as c
import struct





class ReadHaloMergerTree:
    #low precision Treebricks (mostly HAGN)
    def __init__(self,file_path=None):
        
        """
        Make a treebricks dictionary out of file
        """
        self.file_path = file_path
        self.merger_tree = None
        self.read_data()
   
        
    def read_data(self):
        t0 = time.time()

        """
        header stuff
        """

        f = FortranFile(self.file_path, 'r')
        
        nsteps = f.read_record('i')
        nsteps.tolist()
        halos = f.read_record('i')
        aexp = f.read_record('f')
        omega_t = f.read_record('f')
        age_univ = f.read_record('f')
        nh_old = halos[::2]
        nsubh_old = halos[1::2]
        
        
        self.merger_tree = {} 
        self.merger_tree['nsteps'] = nsteps
        self.merger_tree['aexp'] = aexp
        self.merger_tree['omega_t'] = omega_t
        self.merger_tree['age_univ'] = age_univ
        self.merger_tree['nh_old'] = nh_old
        self.merger_tree['nsub_old'] = nsubh_old
        nhaloes = nh_old + nsubh_old
        self.merger_tree['nhaloes'] = nhaloes


        print('nsteps:',nsteps,
              'aexp:', aexp,
              'omega_t:', omega_t,
              'age:', age_univ,
              'nsub, nhost:',halos,
              'nhaloes:', nhaloes)
        
        
        #def read_halo():
        t1 = time.time()
        for i in range(int(nsteps)):
            ts = str(i)
            halo_key = 'halos_ts' + ts
            
    
            #this will hold dictionaries for a all halo information at this timestep 
            self.merger_tree[halo_key] = []
            nhalos = nh_old[i] + nsubh_old[i]

            for i in range(nhalos):
                halo_dict = {}
                
                #read halo stuff
                my_number = f.read_record('i')
                halo_dict['my_number'] = my_number.tolist()
                
                bush_id = f.read_record('i')
                halo_dict['bush_ID'] = bush_id
                
                st = f.read_record('i')
                halo_dict['st'] = st
                
                halo_stuff = f.read_record('i')
                halo_stuff.tolist()
                #print(halo_stuff)
                
                level, hosthalo, hostsub, nbsub, nextsub = halo_stuff[0],halo_stuff[1],halo_stuff[2],halo_stuff[3],halo_stuff[4] 
                halo_dict['level'] = level
                halo_dict['host_halo'] = hosthalo
                halo_dict['host_sub'] = hostsub
                halo_dict['nbsub'] = nbsub
                halo_dict['nextsub'] = nextsub
                
                mass = f.read_record('f') #d for NH
                halo_dict['mass'] = mass.tolist()
                #print(mass)
                        
                macc = f.read_record('f') #d for NH
                halo_dict['macc'] = macc.tolist()
                #print(macc)
                
                p = f.read_record('f')
                p = p.tolist()
                py,px,pz = p[0],p[1],p[2]
                halo_dict['px'] = px
                halo_dict['py'] = py
                halo_dict['pz'] = pz
                #print(p)

                v = f.read_record('f')
                v = v.tolist()
                vx,vy,vz = v[0],v[1],v[2]  
                halo_dict['vx'] = vx
                halo_dict['vy'] = vy
                halo_dict['vz'] = vz
                #print(v)

                L = f.read_record('f')
                L = L.tolist()
                Lx,Ly,Lz = L[0],L[1],L[2]
                halo_dict['Lx'] = Lx
                halo_dict['Ly'] = Ly
                halo_dict['Lz'] = Lz
                #print(L)
                
                shape = f.read_record('f')
                shape = shape.tolist()
                rmax,a,b,c = shape[0],shape[1],shape[2],shape[3]
                halo_dict['rmax'] = rmax
                halo_dict['a'] = a 
                halo_dict['b'] = b
                halo_dict['c'] = c
                #print(shape)

                energy = f.read_record('f')
                energy = energy.tolist()
                ek,ep,et = energy[0],energy[1],energy[2]
                halo_dict['ek'] = ek
                halo_dict['ep'] = ep
                halo_dict['et'] = et
                #print(energy)

                spin = f.read_record('f')
                halo_dict['spin'] = spin.tolist()
                #print('Spin',spin)
                

                
                nb_fathers = f.read_record('i')
                #print('nb_fathers',nb_fathers)
                halo_dict['nb_fathers'] = nb_fathers
                
                if nb_fathers != 0:
                    
                    list_fathers = f.read_record('i')
                    #print('list_fathers', list_fathers)
                    halo_dict['list_fathers'] = list_fathers
                
                    mass_fathers = f.read_record('f')
                    #print('mass fathers', mass_fathers)
                    halo_dict['mass_fathers'] =  mass_fathers
                
                        
                nb_sons = f.read_record('i')
                #print('nb sons',nb_sons)
                halo_dict['nb_sons'] = nb_sons
                
                if nb_sons != 0:
                
                    list_sons = f.read_record('i')
                    #print('list_sons', list_sons)
                    halo_dict['list_sons'] = list_sons
                
                virial = f.read_record('f')
                virial = virial.tolist()
                rvir,mvir,tvir,cvel = virial[0],virial[1],virial[2],virial[3]
                #print('Virial',virial)
                halo_dict['rvir'] = rvir
                halo_dict['mvir'] = mvir
                halo_dict['tvir'] = tvir
                halo_dict['cvel'] = cvel 

                halo_profile = f.read_record('f')
                halo_profile = halo_profile.tolist()
                rho_0, r_c = halo_profile[0],halo_profile[1]
                #print('halo profile',halo_profile)
                halo_dict['rho_0'] = rho_0
                halo_dict['r_c'] = r_c
                
                self.merger_tree[halo_key].append(halo_dict) 
            print(f'Done reading halos in time step {ts}.')
                
                
            t2 = time.time()
        print('Reading haloes took {:0.2f} secs.'.format(t2-t1))
        print('Total time was {:0.2f} secs.'.format(t2-t0))
                
        return self.merger_tree
        
class ReadGalMergerTree:
    #low precision Treebricks (mostly HAGN)
    def __init__(self,file_path=None):
        
        """
        Make a treebricks dictionary out of file
        """
        self.file_path = file_path
        self.merger_tree = None
        self.read_data()
   
        
    def read_data(self):
        t0 = time.time()

        """
        header stuff
        """

        f = FortranFile(self.file_path, 'r')
        
        nsteps = f.read_record('i')
        nsteps.tolist()
        nbodies = f.read_record('i')
        aexp = f.read_reals('f')
        omega_t = f.read_record('f')
        age_univ = f.read_record('f')
        ngal_old = nbodies[::2]
        nsubgal_old = nbodies[1::2]
        
        
        self.merger_tree = {} 
        self.merger_tree['nsteps'] = nsteps
        self.merger_tree['aexp'] = aexp
        self.merger_tree['omega_t'] = omega_t
        self.merger_tree['age_univ'] = age_univ
        self.merger_tree['ngal_old'] = ngal_old
        self.merger_tree['nsubgal_old'] = nsubgal_old
        ngals = ngal_old + nsubgal_old
        self.merger_tree['ngals'] = ngals


        print('nsteps:',nsteps,
              'aexp:', aexp,
              'omega_t:', omega_t,
              'age:', age_univ,
              'nsubgals, ngals:',nbodies,
              'ngals:', ngals)
        
        
        #def read_gal():
        t1 = time.time()
        for i in range(int(nsteps)):
            ts = str(i)
            gal_key = 'gals_ts' + ts
            
    
            #this will hold dictionaries for a all gal information at this timestep 
            self.merger_tree[gal_key] = []
            ngals = ngal_old[i] + nsubgal_old[i]

            for i in range(ngals):
                gal_dict = {}
                
                #read gal stuff
                my_number = f.read_record('i')
                gal_dict['my_number'] = my_number.tolist()
                
                bush_id = f.read_record('i')
                gal_dict['bush_ID'] = bush_id
                
                st = f.read_record('i')
                gal_dict['st'] = st
                
                gal_stuff = f.read_record('i')
                gal_stuff.tolist()
                #print(halo_stuff)
                
                level,host_gal,host_subgal,nchild,nextsub = gal_stuff[0],gal_stuff[1],gal_stuff[2],gal_stuff[3],gal_stuff[4] 
                gal_dict['level'] = level
                gal_dict['host_gal']  = host_gal
                gal_dict['host_subgal'] = host_subgal
                gal_dict['nchild'] = nchild
                gal_dict['nextsub'] = nextsub
                
                mass = f.read_record('d') #d for NH
                gal_dict['mass'] = mass.tolist()
   
                        
                macc = f.read_record('d') #d for NH
                gal_dict['macc'] = macc.tolist()
                #print(macc)
                
                p = f.read_record('d')
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
                

                
                nb_fathers = f.read_record('i')
                #print('nb_fathers',nb_fathers)
                gal_dict['nb_fathers'] = nb_fathers
                
                if nb_fathers != 0:
                    
                    list_fathers = f.read_record('i')
                    #print('list_fathers', list_fathers)
                    gal_dict['list_fathers'] = list_fathers
                
                    mass_fathers = f.read_record('d')
                    #print('mass fathers', mass_fathers)
                    gal_dict['mass_fathers'] =  mass_fathers
                
                        
                nb_sons = f.read_record('i')
                #print('nb sons',nb_sons)
                gal_dict['nb_sons'] = nb_sons
                
                if nb_sons != 0:
                
                    list_sons = f.read_record('i')
                    #print('list_sons', list_sons)
                    gal_dict['list_sons'] = list_sons
                
                virial = f.read_record('d')
                virial = virial.tolist()
                rvir,mvir,tvir,cvel = virial[0],virial[1],virial[2],virial[3]
                gal_dict['rvir'] = rvir
                gal_dict['mvir'] = mvir
                gal_dict['tvir'] = tvir
                gal_dict['cvel'] = cvel 

                halo_profile = f.read_record('d')
                halo_profile = halo_profile.tolist()
                rho_0, r_c = halo_profile[0],halo_profile[1]
                #print('halo profile',halo_profile)
                gal_dict['rho_0'] = rho_0
                gal_dict['r_c'] = r_c
                
                self.merger_tree[gal_key].append(gal_dict) 
            print(f'Done reading galaxies in time step {ts}.')
                
                
            t2 = time.time()
        print('Reading galaxies took {:0.2f} secs.'.format(t2-t1))
        print('Total time was {:0.2f} secs.'.format(t2-t0))
                
        return self.merger_tree
        
