import numpy as np
import time as time
import plane_finding_tools as pf 
import sys

snapshot = sys.argv[1]


systems_file='/data78/welker/madhani/systems/systems_' + str(snapshot) + '.pickle'
#systems_file = '/Users/JanviMadhani/satellite_planes/systems/MWsystems.pickle'

systems = pf.read_systems(systems_file)


# JUST CHECK WITH ONE RIGHT NOW
#syst = 46
for syst in range(len(systems)):
 
    print('System with Halo ID:', systems[syst]['halo_ID'])
    name_of_syst = systems[syst]['halo_ID']

    best_u1,best_u2,best_u3,best_rms = pf.evolutionary_plane_finder(systems=systems,system=syst,n_iter=200,n_start=25,n_erase=10,n_avg_mutants=5,level=1,rand=False,verbose=True)
    z_best,xx,yy,unit_n,los = pf.get_plane(u1=best_u1,u2=best_u2,u3=best_u3,systems=systems,system=syst)

    #get physical extent, c_to_a:
    a,b,c,phys_c_to_a = pf.find_physical_extent(u1=best_u1,u2=best_u2,u3=best_u3,systems=systems,system=syst,actual_rms=best_rms,nrms = 2,level=1)
    phys_ext = [a,b,c,phys_c_to_a]

    #find inertia tensor
    I = pf.find_inertia_tensor(systems[syst])
    v1,v2,v3 = pf.find_axes_of_rot(I)
    i_c_to_a = pf.find_axes_ratios(I)

    inertia = [v1,v2,v3,i_c_to_a]

    name_of_3dplot = 'system_' + str(name_of_syst) +'.png'
    pf.save_3Dplot(name_of_3dplot,systems=systems,syst=syst,snapshot=snapshot,xx=xx,yy=yy,z_best=z_best,los=los,unit_n=unit_n,phys_ext = phys_ext, inertia=inertia)



    #check for isotropy n times and find n rms dists
    #iso_systs_rms = pf.check_isotropy(systems=systems,syst=syst,n=2000)


    #name_of_hist = 'system_' + str(syst) +'_hist.png'
    #pf.save_hist(name_of_hist,best_rms,iso_systs_rms,snapshot=snapshot)


    #save all information to a .json file
    name_of_file = 'system_' + str(name_of_syst) + '.json'
    pf.save_outputs(name_of_file,snapshot,systems,syst,inertia,physical,sig_spherical=2,sig_elliptical=2)

