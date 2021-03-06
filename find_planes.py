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
    a,b,c,phys_c_to_a,ulos= pf.find_physical_extent(u1=best_u1,u2=best_u2,u3=best_u3,systems=systems,system=syst,actual_rms=best_rms,nrms = 2,level=1)
    phys_ext = [a,b,c,phys_c_to_a]
    corotation_dict['phys_c_to_a'].append(phys_c_to_a)


    ## find inertia tensor
    I = pf.find_inertia_tensor(systems[syst])
    v1,v2,v3 = pf.find_axes_of_rot(I)
    i_c_to_a = pf.find_axes_ratios(I)

    inertia = [v1,v2,v3,i_c_to_a]
    
    vrots,corot_frac = pf.corotating_frac(systems=systems,syst=syst,plos =ulos,actual_rms =best_rms,nrms=1,level=1)
    
    corotation_dict['corotating_frac'].append(corot_frac)

    name_of_3dplot = 'system_' + str(name_of_syst) +'.png'
    pf.save_3Dplot(name_of_3dplot,systems=systems,syst=syst,snapshot=snapshot,xx=xx,yy=yy,z_best=z_best,los=v2,unit_n=unit_n,vrots = vrot,phys_ext = phys_ext, inertia=inertia)

    #####################
    # ISOTROPY ANALYSIS #
    #####################

    ## check for isotropy n times and find n rms dists
    iso_sph_systs_rms,iso_ell_systs_rms = pf.check_isotropy(systems=systems,syst=syst,unit_n=unit_n,actual_rms=best_rms,n=2000,corot=False)


    name_of_hist = 'system_' + str(name_of_syst) +'_hist.png'
    ## save spherical and get significance 
    sph_sig = pf.save_hist(name_of_hist,best_rms,iso_sph_systs_rms,snapshot=snapshot,type='spherical',savedat=True)

    ## save elliptical and get significance 
    ell_sig = pf.save_hist(name_of_hist,best_rms,iso_ell_systs_rms,snapshot=snapshot,type='elliptical',savedat=True)


    ## find significance of rms then change below file to include this info


    #save all information to a .csv file
    #name_of_file = 'system_' + str(name_of_syst) + '.csv'
    #pf.save_outputs(name_of_file,snapshot=snapshot,systems=systems,syst=syst,inertial=inertia,physical=phys_ext,sig_spherical=sph_sig,sig_elliptical=ell_sig)

#save corotation dictionary to pickle for later analysis

name_of_corot_dict = 'corotation_analysis_' + str(snapshot)
pf.write_to_pickle(corotation_dict,snapshot,name_of_corot_dict)