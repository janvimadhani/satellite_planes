import numpy as np
import time as time
import plane_finding_tools as pf 
import sys

snapshot = sys.argv[1]


systems_file='/data80/madhani/systems/systems_' + str(snapshot) + '.pickle'
#systems_file = '/Users/JanviMadhani/satellite_planes/systems/MWsystems.pickle'

systems = pf.read_systems(systems_file)

corotation_dict = {}
corotation_dict['syst_ID'] = []
corotation_dict['best_rms'] = []
corotation_dict['phys_c_to_a'] = []
corotation_dict['inertia_mw_c_to_a'] = []
corotation_dict['inertia_phys_c_to_a'] = []
corotation_dict['corotating_frac'] = []
#corotation_dict['sph_corotating_frac'] = []
corotation_dict['ell_corotating_frac'] = []
#corotation_dict['sph_c_to_a'] = []
corotation_dict['ell_c_to_a'] = []
corotation_dict['ell_rms'] = []




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
    phys_c_to_a = pf.find_physical_extent(u1=best_u1,u2=best_u2,u3=best_u3,systems=systems,system=syst,actual_rms=best_rms,nrms = 2,level=1)
    corotation_dict['phys_c_to_a'].append(phys_c_to_a)

    ## get inertia tensor
    # mass weighted 
    inertia_tensor_massw = pf.find_inertia_tensor(systems= systems, syst=syst,mass=True)
    axes_ratios_massw = pf.find_axes_ratios(inertia_tensor_massw)
    corotation_dict['inertia_mw_c_to_a'].append(axes_ratios_massw)

    # just physical 
    inertia_tensor_phys = pf.find_inertia_tensor(systems = systems, syst=syst,mass=False)
    axes_ratios_phys = pf.find_axes_ratios(inertia_tensor_phys)
    corotation_dict['inertia_phys_c_to_a'].append(axes_ratios_phys)

    #reduced inertia tensor, Chisari+ 15 


    vrot,corot_frac = pf.corotating_frac(systems=systems,syst=syst, actual_rms=best_rms,nrms=1,level=1)
    
    corotation_dict['corotating_frac'].append(corot_frac)

    ## check for isotropy n times and find n rms dists
    iso_ell_systs_rms,ell_corot_frac,ell_c_to_a = pf.create_corot_background(systems=systems,syst=syst,n=5000)

    corotation_dict['ell_rms'].append(iso_ell_systs_rms)
    corotation_dict['ell_corotating_frac'].append(ell_corot_frac)
    corotation_dict['ell_c_to_a'].append(ell_c_to_a)


#save corotation dictionary to pickle for later analysis

name_of_corot_dict = 'corotation_analysis_' + str(snapshot)
pf.write_to_pickle(corotation_dict,snapshot,name_of_corot_dict)
