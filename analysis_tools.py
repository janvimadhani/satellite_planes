import fortran_reader as fr
import numpy as np
import time as time
import pickle as pickle
from pathlib import Path
import os


################################################################################
"""
This is where all of the functions to analyze the catalogs will live.
They will all (for the most part), take a dictionary object (i.e. a "system")
as their input parameters
"""
################################################################################



class MWsystems:

    """
    Takes the raw halo and galaxy catalogs from NH 
    and finds MW type analog systems and some analytical functions
    to add information to these systems

    Inputs: haloes_dict   [dict]
            galaxies_dict [dict]
    """
    
    def __init__(self,haloes_dict,galaxies_dict,haloIDs=None):
        self.haloes = None
        self.galaxies = None
        self.find_MWsystems(haloes_dict,galaxies_dict)
        self.find_systs_by_halo_id(haloIDs,haloes_dict,galaxies_dict)


        

    
    def get_vrot(self):

        for MWsystem in self.MWsystems:
            Lx = MWsystem['MW_lx']
            Ly = MWsystem['MW_ly']
            Lz = MWsystem['MW_lz']
            L = np.array([Lx,Ly,Lz])
            L_mag = np.linalg.norm(L)
            unit_ez = L/L_mag
            unit_ez = unit_ez.reshape(3,)
            #print('Unit_ez',unit_ez)
            er = []
            etheta = []
            vrots = []
            
            for i in range(len(MWsystem['sat_pxs'])):
                r = MWsystem['r_sep'][i]
                rz = r[2]
                b = (rz*unit_ez)
                b = b.flatten()
                a = (r - b)
                unit_er = a/np.linalg.norm(a)
                #print('Unit_er',unit_er)
                #print('shape',shape(unit_er))
                #print('shape',shape(unit_ez))
                unit_etheta = np.cross(unit_ez,unit_er)
                
                er.append(unit_er)
                etheta.append(unit_etheta)
                #print('unit_etheta',unit_etheta)
                
                vx = MWsystem['sat_vxs'][i]
                vy = MWsystem['sat_vys'][i]
                vz = MWsystem['sat_vzs'][i]
                v = np.array([vx,vy,vz])
                
                vrot = np.dot(v,unit_etheta)
                vrots.append(vrot)
            
                
            MWsystem['unit_ez'] = unit_ez
            MWsystem['unit_er'] = er
            MWsystem['unit_etheta'] = etheta
            MWsystem['v_rot'] = vrots



    def find_MWsystems(self,haloes_dict,galaxies_dict,mw_halo_mass=10**11.5,mw_mass=10**10.5,rvir_search_thresh=2):
        """
        Input:  haloes_dict,dict: dictionary of halo catalog
                galaxies_dict,dict: dictionary of galaxy catalog
                mw_halo_mass,float: minimum mass of mw type halo, 
                mw_mass,float: minimum mass of mw type gal
                rvir_search_thresh,int: search for satellites within [x]*rvir of DM halo

        Returns: systems,dictionary: dictionary of MW type systems
                

        """
        def get_sep_vector(MWsystem):
            angles = []
            dots = []
            rseps = []
            for i in range(len(MWsystem['sat_pxs'])):
                sx = MWsystem['sat_pxs'][i]
                sy = MWsystem['sat_pys'][i]
                sz = MWsystem['sat_pzs'][i]
                cx = MWsystem['MW_px']
                cy = MWsystem['MW_py']
                cz = MWsystem['MW_pz']

                spin = MWsystem['MW_spin']
                Lx = MWsystem['MW_lx']
                Ly = MWsystem['MW_ly']
                Lz = MWsystem['MW_lz']
                L = np.array([Lx,Ly,Lz])
                #print(L)
                L_mag = np.sqrt(Lx**2 + Ly**2 + Lz**2)

                #separation vector = satellite vector - central vector
                x = sx - cx
                y = sy - cy
                z = sz - cz 
                r = np.array([x,y,z])
                #print(r)
                r_mag = np.sqrt(x**2 + y**2 + z**2)
                L_dot_r = (Lx*x) + (Ly*y) +(Lz*z)
                
                rseps.append(r)
                dots.append(L_dot_r)
                cos = L_dot_r/(L_mag*r_mag)
                angles.append(cos)
            MWsystem['r_sep'] = np.squeeze(np.asarray(rseps))
            MWsystem['cos'] = np.squeeze(np.asarray(angles))
            MWsystem['dot'] = np.squeeze(np.asarray(dots))

        self.haloes = haloes_dict
        self.galaxies = galaxies_dict

        #HALOES
        nhaloes = self.haloes['nhaloes']
        aexp = self.haloes['aexp']


        halo_px = [self.haloes['haloes'][i]['px'] for i in range(nhaloes)]
        halo_py = [self.haloes['haloes'][i]['py'] for i in range(nhaloes)]
        halo_pz = [self.haloes['haloes'][i]['pz'] for i in range(nhaloes)]

        halo_px = np.asarray(halo_px)
        halo_py = np.asarray(halo_py)
        halo_pz = np.asarray(halo_pz)

        halo_a = [self.haloes['haloes'][i]['a'] for i in range(nhaloes)]
        halo_b = [self.haloes['haloes'][i]['b'] for i in range(nhaloes)]
        halo_c = [self.haloes['haloes'][i]['c'] for i in range(nhaloes)]

        halo_a = np.asarray(halo_a)
        halo_b = np.asarray(halo_b)
        halo_c = np.asarray(halo_c)

        #get IDs
        halo_ID = [self.haloes['haloes'][i]['my_number'] for i in range(nhaloes)]
        halo_ID = np.asarray(halo_ID)
        halo_ID = np.squeeze(halo_ID)
         

        #then, constrain to 'zoom portion', which is radius of 20 Mpc
        r = 10
        halo_zoom = np.where(((-r < halo_px) & (halo_px < r )) & ((-r < halo_py) & (halo_py < r )) &
                      ((-r < halo_pz) & (halo_pz < r )))



        halo_rvir = [self.haloes['haloes'][i]['rvir'] for i in range(nhaloes)]
        halo_rvir = np.asarray(halo_rvir)


        #apply zoom masks
        halo_px = halo_px[halo_zoom]
        halo_py = halo_py[halo_zoom]
        halo_pz = halo_pz[halo_zoom]
        halo_a = halo_a[halo_zoom]
        halo_b = halo_b[halo_zoom]
        halo_c = halo_c[halo_zoom]
        halo_rvir = halo_rvir[halo_zoom]
        halo_ID = halo_ID[halo_zoom]   

        halo_lx = [self.haloes['haloes'][i]['Lx'] for i in range(nhaloes)]
        halo_ly = [self.haloes['haloes'][i]['Ly'] for i in range(nhaloes)]
        halo_lz = [self.haloes['haloes'][i]['Lz'] for i in range(nhaloes)]

        halo_lx = np.asarray(halo_lx)
        halo_ly = np.asarray(halo_ly)
        halo_lz = np.asarray(halo_lz)

        halo_lx = halo_lx[halo_zoom]
        halo_ly = halo_ly[halo_zoom]
        halo_lz = halo_lz[halo_zoom]    

        halo_level = [self.haloes['haloes'][i]['level'] for i in range(nhaloes)]
        halo_level = np.asarray(halo_level)
        halo_level = halo_level[halo_zoom]
  
        halo_masses = [self.haloes['haloes'][i]['mass'] for i in range(nhaloes)]
        halo_masses = np.asarray(halo_masses)*10e10 
        halo_masses = np.squeeze(halo_masses)   
        halo_masses = halo_masses[halo_zoom]
        #MW_type_haloes = np.where(mw_halo_mass < halo_masses ) 


        

        #GALAXIES

        ngalaxies = self.galaxies['nmax']

        gal_mvir = [self.galaxies['galaxies'][i]['mvir'] for i in range(ngalaxies)]
        gal_mvir = np.asarray(gal_mvir)*10e10 
        gal_mvir = np.squeeze(gal_mvir)
        MW_type_gals = np.where((gal_mvir > mw_mass))

        gal_px = [self.galaxies['galaxies'][i]['px'] for i in range(ngalaxies)]
        gal_py = [self.galaxies['galaxies'][i]['py'] for i in range(ngalaxies)]
        gal_pz = [self.galaxies['galaxies'][i]['pz'] for i in range(ngalaxies)]

        gal_px = np.asarray(gal_px)
        gal_py = np.asarray(gal_py)
        gal_pz = np.asarray(gal_pz)

        gal_rvir = [self.galaxies['galaxies'][i]['rvir'] for i in range(ngalaxies)]
        gal_rvir = np.asarray(gal_rvir)

        gal_vx = [self.galaxies['galaxies'][i]['vx'] for i in range(ngalaxies)]
        gal_vy = [self.galaxies['galaxies'][i]['vy'] for i in range(ngalaxies)]
        gal_vz = [self.galaxies['galaxies'][i]['vz'] for i in range(ngalaxies)]

        gal_vx = np.asarray(gal_vx)
        gal_vy = np.asarray(gal_vy)
        gal_vz = np.asarray(gal_vz)

        gal_lz = [self.galaxies['galaxies'][i]['Lz'] for i in range(ngalaxies)]
        gal_lx = [self.galaxies['galaxies'][i]['Lx'] for i in range(ngalaxies)]
        gal_ly = [self.galaxies['galaxies'][i]['Ly'] for i in range(ngalaxies)]
        gal_spins = [self.galaxies['galaxies'][i]['spin'] for i in range(ngalaxies)]

        gal_lz = np.asarray(gal_lz)
        gal_lx = np.asarray(gal_lx)
        gal_ly = np.asarray(gal_ly)
        gal_spins = np.asarray(gal_spins)

        #get level
        gal_level = [self.galaxies['galaxies'][i]['level'] for i in range(ngalaxies)]
        gal_level = np.asarray(gal_level)       

        systems = []
        for i,val in enumerate(MW_type_gals[0]):
            gal_search_thresh = rvir_search_thresh #number of virial radii
            h_pxs = halo_px
            h_pys = halo_py
            h_pzs = halo_pz
            gvir = gal_rvir[val]
            g_px = gal_px[val]
            g_py = gal_py[val]
            g_pz = gal_pz[val]
            

            
            #within search radius AND >10e12 Msun
            possible_haloes = np.where(((g_px - gal_search_thresh*gvir  < h_pxs) & (h_pxs < gal_search_thresh*gvir + g_px))
                                    & ((g_py - gal_search_thresh*gvir  < h_pys) & (h_pys < gal_search_thresh*gvir + g_py))
                                    & ((g_pz - gal_search_thresh*gvir  < h_pzs) & (h_pzs < gal_search_thresh*gvir + g_pz))
                                    & (halo_masses>=mw_halo_mass))

            #print(possible_haloes[0])
            if len(halo_masses[possible_haloes]) < 1:
                print(f'No halo satisfies conditions of MW type system {i}')
            
            else:
                system = {}
                host_halo_mask = np.where(halo_masses == np.max(halo_masses[possible_haloes]))
                host_halo_mass = halo_masses[host_halo_mask]
                #print(host_halo_mass)
                h_px = halo_px[host_halo_mask]
                h_py = halo_py[host_halo_mask]
                h_pz = halo_pz[host_halo_mask]
                hrvir = halo_rvir[host_halo_mask]
                h_lx = halo_lx[host_halo_mask]
                h_ly = halo_ly[host_halo_mask]
                h_lz = halo_lz[host_halo_mask]
                h_lev = halo_level[host_halo_mask]
                h_id = halo_ID[host_halo_mask]
                h_a = halo_a[host_halo_mask]
                h_b = halo_b[host_halo_mask]
                h_c = halo_c[host_halo_mask]
                
                h_angmom = np.sqrt(h_lz**2 + h_ly**2 + h_lx**2)
                h_iz = np.degrees(np.arccos(h_lz/h_angmom))
                system['halo_iz'] = h_iz
                #print('hrvir = ',hrvir)
                
                
                
                system['halo_ID'] = h_id[0]
                system['halo_px'] = h_px
                system['halo_py'] = h_py
                system['halo_pz'] = h_pz
                system['halo_a'] = h_a
                system['halo_b'] = h_b
                system['halo_c'] = h_c
                system['halo_rvir'] = hrvir
                system['halo_mass'] = host_halo_mass
                system['halo_level'] = h_lev
                g_pxs = gal_px
                g_pys = gal_py
                g_pzs = gal_pz

                
                #find all galaxies within this 1 virial radii halo, and identify the central 
                sat_thresh = 2
                #"""        
                within_rad = np.where(((h_px - sat_thresh*hrvir  < g_pxs) & (g_pxs < sat_thresh*hrvir + h_px)) &
                                    ((h_py - sat_thresh*hrvir  < g_pys) & (g_pys < sat_thresh*hrvir + h_py)) &
                                    ((h_pz - sat_thresh*hrvir  < g_pzs) & (g_pzs < sat_thresh*hrvir + h_pz)))
                                    
                #"""
                """
                
                distance = 0.3 #look for satellites within 0.3 Mpc *1000 kpc = 300 kpc
                within_rad = np.where(((h_px - distance  < g_pxs) & (g_pxs < distance + h_px)) &
                                    ((h_py - distance  < g_pys) & (g_pys < distance + h_py)) &
                                    ((h_pz - distance  < g_pzs) & (g_pzs < distance + h_pz)))
                """
                
                
                #assign sattelite galaxies parameters
                sat_pxs = g_pxs[within_rad]
                sat_pys = g_pys[within_rad]
                sat_pzs = g_pzs[within_rad]
                sat_rvirs = gal_rvir[within_rad]
                sat_mvirs = gal_mvir[within_rad]
                sat_lx = gal_lx[within_rad]
                sat_ly = gal_ly[within_rad]
                sat_lz = gal_lz[within_rad]
                sat_vx = gal_vx[within_rad]
                sat_vy = gal_vy[within_rad]
                sat_vz = gal_vz[within_rad]
                sat_spins = gal_spins[within_rad]
                sat_levs = gal_level[within_rad]
                

                
                #find the most massive galaxy that is within 0.2 of the host halo -- this is MW analog
                vir_thresh = 0.2

                within_vir = np.where(((h_px - vir_thresh*hrvir  < sat_pxs) & (sat_pxs < vir_thresh*hrvir + h_px)) &
                                    ((h_py - vir_thresh*hrvir  < sat_pys) & (sat_pys < vir_thresh*hrvir + h_py)) &
                                    ((h_pz - vir_thresh*hrvir  < sat_pzs) & (sat_pzs < vir_thresh*hrvir + h_pz)))
                if len(within_vir) < 1:
                    print('No satisfactory system found within this halo. Please try searching a different halo.')
                else:
                    MW_analog_mask = np.where(sat_mvirs == np.max(sat_mvirs[within_vir]))
                    #print('MW MASK',MW_analog_mask)
                    
                    MW_px = sat_pxs[MW_analog_mask]
                    MW_py = sat_pys[MW_analog_mask]
                    MW_pz = sat_pzs[MW_analog_mask]
                    MW_mvir = sat_mvirs[MW_analog_mask]
                    MW_rvir = sat_rvirs[MW_analog_mask]
                    MW_lx = sat_lx[MW_analog_mask]
                    MW_ly = sat_ly[MW_analog_mask]
                    MW_lz = sat_lz[MW_analog_mask]
                    MW_vx = sat_vx[MW_analog_mask]
                    MW_vy = sat_vy[MW_analog_mask]
                    MW_vz = sat_vz[MW_analog_mask]
                    MW_spin = sat_spins[MW_analog_mask]
                    MW_level = sat_levs[MW_analog_mask]
                    
                    MW_angmom = np.sqrt(MW_lz**2 + MW_ly**2 + MW_lx**2)
                    MW_iz = np.degrees(np.arccos(MW_lz/MW_angmom))
                    system['MW_iz'] = MW_iz
                    #print('hrvir = ',hrvir)
                    
                    
                    system['MW_px'] = MW_px
                    system['MW_py'] = MW_py
                    system['MW_pz'] = MW_pz
                    system['MW_mvir'] = MW_mvir
                    system['MW_rvir'] = MW_rvir
                    system['MW_spin'] = MW_spin
                    system['MW_lx'] = MW_lx
                    system['MW_ly'] = MW_ly
                    system['MW_lz'] = MW_lz
                    system['MW_vx'] = MW_vx
                    system['MW_vy'] = MW_vy
                    system['MW_vz'] = MW_vz
                    system['MW_level'] = MW_level
                    
                    #remove central galaxy from satellite list
                    sat_pxs = np.delete(sat_pxs,MW_analog_mask)
                    sat_pys = np.delete(sat_pys,MW_analog_mask)
                    sat_pzs = np.delete(sat_pzs,MW_analog_mask)
                    sat_rvirs = np.delete(sat_rvirs,MW_analog_mask)
                    sat_mvirs = np.delete(sat_mvirs,MW_analog_mask)
                    sat_lx = np.delete(sat_lx,MW_analog_mask)
                    sat_ly = np.delete(sat_ly,MW_analog_mask)
                    sat_lz = np.delete(sat_lz,MW_analog_mask)
                    sat_vx = np.delete(sat_vx,MW_analog_mask)
                    sat_vy = np.delete(sat_vy,MW_analog_mask)
                    sat_vz = np.delete(sat_vz,MW_analog_mask)
                    sat_spins = np.delete(sat_spins,MW_analog_mask)
                    sat_levs = np.delete(sat_levs,MW_analog_mask)
                    
                    
                    
                    
                    sat_angmom = np.sqrt(sat_lz**2 + sat_ly**2 + sat_lx**2)
                    sat_iz = np.degrees(np.arccos(sat_lz/sat_angmom))
                    system['sat_iz'] = sat_iz
                    
                    system['sat_pxs'] = sat_pxs
                    system['sat_pys'] = sat_pys
                    system['sat_pzs'] = sat_pzs
                    system['sat_vxs'] = sat_vx
                    system['sat_vys'] = sat_vy
                    system['sat_vzs'] = sat_vz
                    system['sat_rvirs'] = sat_rvirs
                    system['sat_mvirs'] = sat_mvirs
                    system['sat_levels'] = sat_levs
                    

                    #add global info
                    system['aexp'] = aexp

                    systems.append(system)
        for system in systems:
            get_sep_vector(system)
        self.MWsystems = systems


                
        return self.MWsystems

    

    def find_systs_by_halo_id(self,haloIDs,haloes_dict,galaxies_dict,rvir_search_thresh=1):
        """
        Input:  haloIDs,list: list of haloIDs, ints
                haloes_dict,dict: dictionary of halo catalog
                galaxies_dict,dict: dictionary of galaxy catalog
                mw_halo_mass,float: minimum mass of mw type halo, 
                mw_mass,float: minimum mass of mw type gal
                rvir_search_thresh,int: search for satellites within [x]*rvir of DM halo

        Returns: systems,dictionary: dictionary of MW type systems
                

        """
        def get_sep_vector(MWsystem):
            angles = []
            dots = []
            rseps = []
            for i in range(len(MWsystem['sat_pxs'])):
                sx = MWsystem['sat_pxs'][i]
                sy = MWsystem['sat_pys'][i]
                sz = MWsystem['sat_pzs'][i]
                cx = MWsystem['MW_px']
                cy = MWsystem['MW_py']
                cz = MWsystem['MW_pz']

                spin = MWsystem['MW_spin']
                Lx = MWsystem['MW_lx']
                Ly = MWsystem['MW_ly']
                Lz = MWsystem['MW_lz']
                L = np.array([Lx,Ly,Lz])
                #print(L)
                L_mag = np.sqrt(Lx**2 + Ly**2 + Lz**2)

                #separation vector = satellite vector - central vector
                x = sx - cx
                y = sy - cy
                z = sz - cz 
                r = np.array([x,y,z])
                #print(r)
                r_mag = np.sqrt(x**2 + y**2 + z**2)
                L_dot_r = (Lx*x) + (Ly*y) +(Lz*z)
                
                rseps.append(r)
                dots.append(L_dot_r)
                cos = L_dot_r/(L_mag*r_mag)
                angles.append(cos)
            MWsystem['r_sep'] = np.squeeze(np.asarray(rseps))
            MWsystem['cos'] = np.squeeze(np.asarray(angles))
            MWsystem['dot'] = np.squeeze(np.asarray(dots))

        self.haloes = haloes_dict
        self.galaxies = galaxies_dict

        #HALOES
        nhaloes = self.haloes['nhaloes']
        aexp = self.haloes['aexp']


        halo_px = [self.haloes['haloes'][i]['px'] for i in range(nhaloes)]
        halo_py = [self.haloes['haloes'][i]['py'] for i in range(nhaloes)]
        halo_pz = [self.haloes['haloes'][i]['pz'] for i in range(nhaloes)]

        halo_px = np.asarray(halo_px)
        halo_py = np.asarray(halo_py)
        halo_pz = np.asarray(halo_pz)

        #get IDs
        halo_ID = [self.haloes['haloes'][i]['my_number'] for i in range(nhaloes)]
        halo_ID = np.asarray(halo_ID)
        halo_ID = np.squeeze(halo_ID)
         

        #then, constrain to 'zoom portion', which is radius of 10 Mpc
        r = 10
        halo_zoom = np.where(((-r < halo_px) & (halo_px < r )) & ((-r < halo_py) & (halo_py < r )) &
                      ((-r < halo_pz) & (halo_pz < r )))



        halo_rvir = [self.haloes['haloes'][i]['rvir'] for i in range(nhaloes)]
        halo_rvir = np.asarray(halo_rvir)


        #apply zoom masks
        halo_px = halo_px[halo_zoom]
        halo_py = halo_py[halo_zoom]
        halo_pz = halo_pz[halo_zoom]
        halo_rvir = halo_rvir[halo_zoom]
        halo_ID = halo_ID[halo_zoom]   

        halo_lx = [self.haloes['haloes'][i]['Lx'] for i in range(nhaloes)]
        halo_ly = [self.haloes['haloes'][i]['Ly'] for i in range(nhaloes)]
        halo_lz = [self.haloes['haloes'][i]['Lz'] for i in range(nhaloes)]

        halo_lx = np.asarray(halo_lx)
        halo_ly = np.asarray(halo_ly)
        halo_lz = np.asarray(halo_lz)

        halo_lx = halo_lx[halo_zoom]
        halo_ly = halo_ly[halo_zoom]
        halo_lz = halo_lz[halo_zoom]    

        halo_level = [self.haloes['haloes'][i]['level'] for i in range(nhaloes)]
        halo_level = np.asarray(halo_level)
        halo_level = halo_level[halo_zoom]
  
        halo_masses = [self.haloes['haloes'][i]['mass'] for i in range(nhaloes)]
        halo_masses = np.asarray(halo_masses)*10e10
        halo_masses = np.squeeze(halo_masses)   
        halo_masses = halo_masses[halo_zoom]
        #MW_type_haloes = np.where(mw_halo_mass < halo_masses ) 


        

        #GALAXIES

        ngalaxies = self.galaxies['nmax']

        gal_mvir = [self.galaxies['galaxies'][i]['mvir'] for i in range(ngalaxies)]
        gal_mvir = np.asarray(gal_mvir)*10e10 
        gal_mvir = np.squeeze(gal_mvir)
        #MW_type_gals = np.where((gal_mvir > mw_mass))

        gal_px = [self.galaxies['galaxies'][i]['px'] for i in range(ngalaxies)]
        gal_py = [self.galaxies['galaxies'][i]['py'] for i in range(ngalaxies)]
        gal_pz = [self.galaxies['galaxies'][i]['pz'] for i in range(ngalaxies)]

        gal_px = np.asarray(gal_px)
        gal_py = np.asarray(gal_py)
        gal_pz = np.asarray(gal_pz)

        gal_rvir = [self.galaxies['galaxies'][i]['rvir'] for i in range(ngalaxies)]
        gal_rvir = np.asarray(gal_rvir)

        gal_vx = [self.galaxies['galaxies'][i]['vx'] for i in range(ngalaxies)]
        gal_vy = [self.galaxies['galaxies'][i]['vy'] for i in range(ngalaxies)]
        gal_vz = [self.galaxies['galaxies'][i]['vz'] for i in range(ngalaxies)]

        gal_vx = np.asarray(gal_vx)
        gal_vy = np.asarray(gal_vy)
        gal_vz = np.asarray(gal_vz)

        gal_lz = [self.galaxies['galaxies'][i]['Lz'] for i in range(ngalaxies)]
        gal_lx = [self.galaxies['galaxies'][i]['Lx'] for i in range(ngalaxies)]
        gal_ly = [self.galaxies['galaxies'][i]['Ly'] for i in range(ngalaxies)]
        gal_spins = [self.galaxies['galaxies'][i]['spin'] for i in range(ngalaxies)]

        gal_lz = np.asarray(gal_lz)
        gal_lx = np.asarray(gal_lx)
        gal_ly = np.asarray(gal_ly)
        gal_spins = np.asarray(gal_spins)

        #get level
        gal_level = [self.galaxies['galaxies'][i]['level'] for i in range(ngalaxies)]
        gal_level = np.asarray(gal_level)       

        systems = []

        for haloID in haloIDs:
            #print('HALO ID ', haloID)

            system = {}

            host_halo_mask = np.where(halo_ID == haloID)
            #print('DId mask work?',halo_ID[host_halo_mask])
            host_halo_mass = halo_masses[host_halo_mask]
            #print(host_halo_mass)
            h_px = halo_px[host_halo_mask]
            h_py = halo_py[host_halo_mask]
            h_pz = halo_pz[host_halo_mask]
            hrvir = halo_rvir[host_halo_mask]
            h_lx = halo_lx[host_halo_mask]
            h_ly = halo_ly[host_halo_mask]
            h_lz = halo_lz[host_halo_mask]
            h_lev = halo_level[host_halo_mask]
            h_id = halo_ID[host_halo_mask]
            
            h_angmom = np.sqrt(h_lz**2 + h_ly**2 + h_lx**2)
            h_iz = np.degrees(np.arccos(h_lz/h_angmom))
            system['halo_iz'] = h_iz
            #print('hrvir = ',hrvir)
            
            
            
            system['halo_ID'] = h_id[0]
            system['halo_px'] = h_px
            system['halo_py'] = h_py
            system['halo_pz'] = h_pz
            system['halo_rvir'] = hrvir
            system['halo_mass'] = host_halo_mass
            system['halo_level'] = h_lev
            g_pxs = gal_px
            g_pys = gal_py
            g_pzs = gal_pz

            
            #find all galaxies within this 1 virial radii halo, and identify the central 
            sat_thresh = 2
            #"""        
            within_rad = np.where(((h_px - sat_thresh*hrvir  < g_pxs) & (g_pxs < sat_thresh*hrvir + h_px)) &
                                ((h_py - sat_thresh*hrvir  < g_pys) & (g_pys < sat_thresh*hrvir + h_py)) &
                                ((h_pz - sat_thresh*hrvir  < g_pzs) & (g_pzs < sat_thresh*hrvir + h_pz)))
                                
            #"""
            """
            
            distance = 0.3 #look for satellites within 0.3 Mpc *1000 kpc = 300 kpc
            within_rad = np.where(((h_px - distance  < g_pxs) & (g_pxs < distance + h_px)) &
                                ((h_py - distance  < g_pys) & (g_pys < distance + h_py)) &
                                ((h_pz - distance  < g_pzs) & (g_pzs < distance + h_pz)))
            """
            
            print('Did it find satellites within 2 rvir of halo?',within_rad)
            #assign sattelite galaxies parameters
            sat_pxs = g_pxs[within_rad]
            sat_pys = g_pys[within_rad]
            sat_pzs = g_pzs[within_rad]
            sat_rvirs = gal_rvir[within_rad]
            sat_mvirs = gal_mvir[within_rad]
            sat_lx = gal_lx[within_rad]
            sat_ly = gal_ly[within_rad]
            sat_lz = gal_lz[within_rad]
            sat_vx = gal_vx[within_rad]
            sat_vy = gal_vy[within_rad]
            sat_vz = gal_vz[within_rad]
            sat_spins = gal_spins[within_rad]
            sat_levs = gal_level[within_rad]
            

            
            #find the most massive galaxy that is within 0.2 of the host halo -- this is MW analog
            vir_thresh = 0.5

            within_vir = np.where(((h_px - vir_thresh*hrvir  < sat_pxs) & (sat_pxs < vir_thresh*hrvir + h_px)) &
                                ((h_py - vir_thresh*hrvir  < sat_pys) & (sat_pys < vir_thresh*hrvir + h_py)) &
                                ((h_pz - vir_thresh*hrvir  < sat_pzs) & (sat_pzs < vir_thresh*hrvir + h_pz)))

            print('WITHIN_VIR',within_vir)
            ### Check that there is a system of satellites that satisfies these conditions for a halo
            if len(within_vir) < 1:
                print('No satisfactory system found within this halo. Please try searching a different halo.')
            else:


                MW_analog_mask = np.where(sat_mvirs == np.max(sat_mvirs[within_vir]))
                #print('MW MASK',MW_analog_mask)
                
                MW_px = sat_pxs[MW_analog_mask]
                MW_py = sat_pys[MW_analog_mask]
                MW_pz = sat_pzs[MW_analog_mask]
                MW_mvir = sat_mvirs[MW_analog_mask]
                MW_rvir = sat_rvirs[MW_analog_mask]
                MW_lx = sat_lx[MW_analog_mask]
                MW_ly = sat_ly[MW_analog_mask]
                MW_lz = sat_lz[MW_analog_mask]
                MW_vx = sat_vx[MW_analog_mask]
                MW_vy = sat_vy[MW_analog_mask]
                MW_vz = sat_vz[MW_analog_mask]
                MW_spin = sat_spins[MW_analog_mask]
                MW_level = sat_levs[MW_analog_mask]
                
                MW_angmom = np.sqrt(MW_lz**2 + MW_ly**2 + MW_lx**2)
                MW_iz = np.degrees(np.arccos(MW_lz/MW_angmom))
                system['MW_iz'] = MW_iz
                #print('hrvir = ',hrvir)
                
                
                system['MW_px'] = MW_px
                system['MW_py'] = MW_py
                system['MW_pz'] = MW_pz
                system['MW_mvir'] = MW_mvir
                system['MW_rvir'] = MW_rvir
                system['MW_spin'] = MW_spin
                system['MW_lx'] = MW_lx
                system['MW_ly'] = MW_ly
                system['MW_lz'] = MW_lz
                system['MW_vx'] = MW_vx
                system['MW_vy'] = MW_vy
                system['MW_vz'] = MW_vz
                system['MW_level'] = MW_level
                
                #remove central galaxy from satellite list
                sat_pxs = np.delete(sat_pxs,MW_analog_mask)
                sat_pys = np.delete(sat_pys,MW_analog_mask)
                sat_pzs = np.delete(sat_pzs,MW_analog_mask)
                sat_rvirs = np.delete(sat_rvirs,MW_analog_mask)
                sat_mvirs = np.delete(sat_mvirs,MW_analog_mask)
                sat_lx = np.delete(sat_lx,MW_analog_mask)
                sat_ly = np.delete(sat_ly,MW_analog_mask)
                sat_lz = np.delete(sat_lz,MW_analog_mask)
                sat_vx = np.delete(sat_vx,MW_analog_mask)
                sat_vy = np.delete(sat_vy,MW_analog_mask)
                sat_vz = np.delete(sat_vz,MW_analog_mask)
                sat_spins = np.delete(sat_spins,MW_analog_mask)
                sat_levs = np.delete(sat_levs,MW_analog_mask)
                
                
                
                
                sat_angmom = np.sqrt(sat_lz**2 + sat_ly**2 + sat_lx**2)
                sat_iz = np.degrees(np.arccos(sat_lz/sat_angmom))
                system['sat_iz'] = sat_iz
                
                system['sat_pxs'] = sat_pxs
                system['sat_pys'] = sat_pys
                system['sat_pzs'] = sat_pzs
                system['sat_vxs'] = sat_vx
                system['sat_vys'] = sat_vy
                system['sat_vzs'] = sat_vz
                system['sat_rvirs'] = sat_rvirs
                system['sat_mvirs'] = sat_mvirs
                system['sat_levels'] = sat_levs

                #add global info
                system['aexp'] = aexp
                #add global info

                systems.append(system)
        
        for system in systems:
            get_sep_vector(system)
        self.halosystems = systems
                
        return self.halosystem

    




                


    def write_to_pickle(self,name_to_save,rewrite=True):
        """
        writes MWsystems dictionary to a pickle file, easy to read later 
        Inputs: name,string: name of file to be written
        Output: path,pickle: pickle file written at specified location
        """
        script_dir = os.path.dirname(__file__)
        #results_dir = os.path.join(script_dir, 'systems/')
        results_dir = '/data78/welker/madhani/systems/'
        
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
                    pickle.dump(self.MWsystems,handle,protocol=pickle.HIGHEST_PROTOCOL)
            else:
                print('File already exists.')

        else:
            print(f'Writing file to {path} ...')
            with open(path_of_file,'wb') as handle:
                pickle.dump(self.MWsystems,handle,protocol=pickle.HIGHEST_PROTOCOL)
                
                    
                                    
                                    
                            
                        
                    