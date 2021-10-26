import numpy as np
import fortran_reader as fr
import analysis_tools as analyze
import time as time
#from analysis_tools import MWsystems as MWS

t0 = time.time()
filepath_haloes = '/Users/JanviMadhani/Desktop/Satellite Galaxies/Analysis/New_Horizon/tree_bricks970'
filepath_galaxies = '/Users/JanviMadhani/Desktop/Satellite Galaxies/Analysis/New_Horizon/tree_bricks970_stars_NH'
haloes = fr.ReadTreebrick_lowp(filepath_haloes)
galaxies = fr.GalaxyCatalog(filepath_galaxies)

haloes_dict = haloes.treebricks_dict
galaxies_dict = galaxies.treebricks_dict

systs = analyze.MWsystems(haloes_dict,galaxies_dict) #first make a class object of all systems
MWsystems = systs.find_MWsystems(haloes_dict,galaxies_dict) #then identify MW types



#write file 
print('Writing MW Systems to file...')
systs.write_to_pickle('MWsystems','systems',rewrite=True)
t1 = time.time()


print('Total time taken:', t1-t0, 's')