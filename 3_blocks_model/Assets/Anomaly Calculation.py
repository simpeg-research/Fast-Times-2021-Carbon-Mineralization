from SimPEG import utils
import numpy as np

dpred_grav_29NV2021 = utils.io_utils.read_grav3d_ubc('grav_data_29NV2021.obs')
mass_true = -0.2*1e3*35*1e9 + 0.1*1e3*15*1e9 
mass_pred = -250*250*dpred_grav_29NV2021.dobs.sum()*1e-5/(2*np.pi*6.67408*1e-11)