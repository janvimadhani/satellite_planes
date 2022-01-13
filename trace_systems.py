import numpy as np
import fortran_reader as fr
import analysis_tools as analyze
import time as time
import sys
#from analysis_tools import MWsystems as MWS

t0 = time.time()

#CHANGE THIS TO BE THE SNAPSHOT YOU ARE ANALYZING -- 970 IS THE ONE I'VE WORKED WITH MOST
#when you run this file, run it as python3 find_systems.py 970 
snapshot = sys.argv[1]

#LOCAL PATHS
#filepath_haloes = '/Users/JanviMadhani/satellite_planes/catalogs/tree_bricks'+str(snapshot)
#filepath_galaxies = '/Users/JanviMadhani/satellite_planes/catalogs/tree_bricks' +str(snapshot) +'_stars_NH'

#INFINITY PATHS
filepath_haloes = '/home/madhani/satellite_planes/catalogs/Halos/TREE_DM/tree_bricks'+str(snapshot)
filepath_galaxies = '/home/madhani/satellite_planes/catalogs/Stars/TREE_STARS_HOP_dp_SCnew_gross/tree_brick_'+str(snapshot)


haloes = fr.ReadTreebrick_lowp(filepath_haloes)
galaxies = fr.GalaxyCatalog(filepath_galaxies)

haloes_dict = haloes.treebricks_dict
galaxies_dict = galaxies.treebricks_dict


#the halos you are trying to trace
haloIDs = [1301,10889,1421,563,1388,563,3331,5485,675,5498,10886,8292,467]

systs = analyze.halosystems(haloes_dict,galaxies_dict,haloIDs) #first make a class object of all systems
halosystems = systs.find_systs_by_halo_ids(haloIDs,haloes_dict,galaxies_dict) #then find systems corresponding to these halos

#write file 
print('Writing Halo Systems to file...')
systs.write_to_pickle('traced_systems_'+str(snapshot),rewrite=True)
t1 = time.time()


print('Total time taken:', t1-t0, 's')
