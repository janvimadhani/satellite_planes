import numpy as np
import time as time
import plane_finding_tools as pf 
import sys

snapshot = sys.argv[1]


systems_file='/data78/welker/madhani/systems/systems_' + str(snapshot) + '.pickle'
#systems_file = '/Users/JanviMadhani/satellite_planes/systems/MWsystems.pickle'

systems = pf.read_systems(systems_file)

corotation_dict = {}
corotation_dict['syst_ID'] = []
corotation_dict['best_rms'] = []
corotation_dict['phys_c_to_a'] = []
corotation_dict['corotating_frac'] = []
corotation_dict['sph_corotating_frac'] = []
corotation_dict['ell_corotating_frac'] = []
corotation_dict['sph_c_to_a'] = []
corotation_dict['ell_c_to_a'] = []




# JUST CHECK WITH ONE RIGHT NOW
#syst = 46
for syst in range(len(systems)):
 
    print('System with Halo ID:', systems[syst]['halo_ID'])
    name_of_syst = systems[syst]['halo_ID']
    corotation_dict['syst_ID'].append(name_of_syst)
    

    best_u1,best_u2,best_u3,best_rms = pf.evolutionary_plane_finder(systems=systems,system=syst,n_iter=200,n_start=25,n_erase=10,n_avg_mutants=5,level=1,rand=False,verbose=True)
    z_best,xx,yy,unit_n,los = pf.get_plane(u1=best_u1,u2=best_u2,u3=best_u3,systems=systems,system=syst)
    corotation_dict['best_rms'].append(best_rms)

    ## get physical extent, c_to_a:
    a,b,c,phys_c_to_a = pf.find_physical_extent(u1=best_u1,u2=best_u2,u3=best_u3,systems=systems,system=syst,actual_rms=best_rms,nrms = 2,level=1)
    phys_ext = [a,b,c,phys_c_to_a]
    corotation_dict['phys_c_to_a'].append(phys_c_to_a)
    
    corot_frac = pf.corotating_frac(systems=systems,syst=syst,unit_n,rms=1,level=1)
    
    corotation_dict['corotating_frac'].append(corot_frac)

    ## check for isotropy n times and find n rms dists
    iso_sph_systs_rms,iso_ell_systs_rms,sph_corot_frac,sph_c_to_a,ell_corot_frac,ell_c_to_a = pf.check_isotropy(systems=systems,syst=syst,n=2000,corot=True)

    corotation_dict['sph_corotating_frac'].append(sph_corot_frac)
    corotation_dict['ell_corotating_frac'].append(ell_corot_frac)
    corotation_dict['sph_c_to_a'].append(sph_c_to_a)
    corotation_dict['ell_c_to_a'].append(ell_c_to_a)

#save corotation dictionary to pickle for later analysis

name_of_corot_dict = 'corotation_analysis_' + str(snapshot)
pf.write_to_pickle(corotation_dict,snapshot,name_of_corot_dict)
