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

filepath_haloes = '/home/madhani/satellite_planes/catalogs/Halos/TREE_DM/tree_bricks'+str(snapshot)
#filepath_haloes = '/Users/JanviMadhani/Desktop/Satellite Galaxies/Analysis/New_Horizon/tree_bricks970'
filepath_galaxies = '/home/madhani/satellite_planes/catalogs/Stars/TREE_STARS_HOP_dp_SCnew_gross/tree_brick_'+str(snapshot)
#filepath_galaxies = '/Users/JanviMadhani/Desktop/Satellite Galaxies/Analysis/New_Horizon/tree_bricks970_stars_NH'

haloes = fr.ReadTreebrick_lowp(filepath_haloes)
galaxies = fr.GalaxyCatalog(filepath_galaxies)

haloes_dict = haloes.treebricks_dict
galaxies_dict = galaxies.treebricks_dict

systs = analyze.MWsystems(haloes_dict,galaxies_dict) #first make a class object of all systems
MWsystems = systs.find_MWsystems(haloes_dict,galaxies_dict) #then identify MW types



#write file 
print('Writing MW Systems to file...')
systs.write_to_pickle('systems_'+str(snapshot),rewrite=True)
t1 = time.time()


print('Total time taken:', t1-t0, 's')
