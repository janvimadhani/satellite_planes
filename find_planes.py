import numpy as np
import time as time
import plane_finding_tools as pf 
import sys

snapshot = sys.argv[1]


systems_file='/home/madhani/satellite_planes/systems/systems_' + str(snapshot) + '.pickle'
#systems_file = '/Users/JanviMadhani/satellite_planes/systems/MWsystems.pickle'

systems = pf.read_systems(systems_file)


# JUST CHECK WITH ONE RIGHT NOW
#syst = 46
for i in range(len(systems)):
    syst = i 

    best_u1,best_u2,best_u3,nx,ny,nz,rms,best_rms,delta_s,best_cos_theta = pf.best_plane(systems=systems,system=syst,level=1,n=5000,verbose=True)
    z_best,xx,yy,unit_n = pf.get_plane(u1=best_u1,u2=best_u2,u3=best_u3,systems=systems,system=syst)

    name_of_3dplot = 'system_' + str(syst) +'.png'
    pf.save_3Dplot(name_of_3dplot,systems=systems,syst=syst,snapshot=snapshot,xx=xx,yy=yy,z_best=z_best)



    #check for isotropy n times and find n rms dists
    iso_systs_rms = pf.check_isotropy(systems=systems,syst=syst,n=2000)

    name_of_hist = 'system_' + str(syst) +'_hist.png'
    pf.save_hist(name_of_hist,best_rms,iso_systs_rms)