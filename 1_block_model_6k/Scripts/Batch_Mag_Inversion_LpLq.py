#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
#from mpi4py import MPI
import sys
sys.path.insert(0,'/tera_raid/mmitchel/Programs/simpeg/simpeg-main-0152')
from SimPEG import dask
import SimPEG
print(SimPEG.__path__)
print(SimPEG.__version__)

import discretize as ds
print(ds.__path__)
print(ds.__version__)

#comm = MPI.COMM_WORLD
#size = comm.Get_size()
#rank = comm.Get_rank()


import SimPEG.potential_fields as pf
from SimPEG import (
    maps, utils, simulation, inverse_problem, inversion, optimization, regularization, data_misfit, directives
)
from SimPEG.utils import io_utils
import numpy as np

#Reproducible science
np.random.seed(518936)

mesh = ds.TreeMesh.read_UBC('mesh_CaMP.ubc')

data_mag = io_utils.read_mag3d_ubc('1block_magnetic_data.obs')
print("maximum mag data {} nT".format(data_mag.dobs.max()))
data_grav = io_utils.read_grav3d_ubc('1block_gravity_data.obs')

actvMap = maps.IdentityMap(mesh)

# mag problem
simulation_mag = pf.magnetics.simulation.Simulation3DIntegral(
        survey=data_mag.survey,
        mesh=mesh,
        chiMap=actvMap,
)

# Grav problem
simulation_grav = pf.gravity.simulation.Simulation3DIntegral(
        survey=data_grav.survey,
        mesh=mesh,
        rhoMap=actvMap,
)

# ## Create simulations and data misfits
def run_grav_inversion(directory='',dw=True, alpha_s=1., alphasmoothdefault=True, name='CaMP_gravity_synthetic_inversion_model'):
    
    # Grav problem
    dmis_grav = data_misfit.L2DataMisfit(data=data_grav, simulation=simulation_grav)

    # Initial Model
    m0 = np.zeros(mesh.n_cells)

    if alphasmoothdefault:
        alpha_x, alpha_y, alpha_z = 1.0, 1.0, 1.0
    else:
        alpha_x = mesh.hx.min()**2
        alpha_y = mesh.hy.min()**2
        alpha_z = mesh.hz.min()**2
    
    # Define the regularization (model objective function).
    reg = regularization.Sparse(
        mesh,
        indActive=np.ones(mesh.n_cells, dtype=bool),
        mapping=actvMap,
        alpha_s=alpha_s,
        alpha_x=alpha_x,
        alpha_y=alpha_y,
        alpha_z=alpha_z,
        gradientType='total'
    )
    reg.norms = [[0, 1, 1, 1]]
    
    if dw:
        wr = utils.depth_weighting(
            mesh, data_grav.survey.receiver_locations, 
            indActive=np.ones(mesh.n_cells, dtype=bool), 
            exponent=2
        )
        reg.cell_weights = wr
        directives_list = []
    else:
        # Add sensitivity weights
        sensitivity_weights = directives.UpdateSensitivityWeights(everyIter=False)
        directives_list = [sensitivity_weights]

    # Define how the optimization problem is solved. Here we will use a projected
    # Gauss-Newton approach that employs the conjugate gradient solver.
    opt = optimization.ProjectedGNCG(
        maxIter=100, lower=-1.0, upper=1.0, maxIterLS=20, maxIterCG=100, tolCG=1e-4
    )

    # Here we define the inverse problem that is to be solved
    inv_prob = inverse_problem.BaseInvProblem(dmis_grav, reg, opt)

    # Defining a starting value for the trade-off parameter (beta) between the data
    # misfit and the regularization.
    starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=1e0)
    beta_schedule = directives.BetaSchedule(coolingFactor=5, coolingRate=1)
    update_jacobi = directives.UpdatePreconditioner()
    IRLS = directives.Update_IRLS(f_min_change=1e-4, max_irls_iterations=50, beta_tol=1e-2,)

    # save every iteration
    save_dict = directives.SaveOutputDictEveryIteration(saveOnDisk=True)
    every_iter_folder = directory + os.path.sep + 'EveryIteration_'+ name + os.path.sep
    os.makedirs(every_iter_folder, exist_ok=True)
    save_dict.directory = every_iter_folder
    

    # The directives are defined as a list.
    directives_list = directives_list + [
        starting_beta,
        beta_schedule,
        IRLS,
        update_jacobi,
        save_dict,
    ]

    inv3 = inversion.BaseInversion(inv_prob, directives_list)

    recovered_model_grav = inv3.run(m0)
    os.makedirs(directory, exist_ok=True)
    mesh.write_model_UBC(directory + os.path.sep + name + ".den", recovered_model_grav)


def run_mag_inversion(directory='',dw=True, alpha_s=1., alphasmoothdefault=True, name='CaMP_magnetic_synthetic_inversion_model'):
    
    dmis_mag = data_misfit.L2DataMisfit(data=data_mag, simulation=simulation_mag)

    # Initial Model
    m0 = 1e-4 * np.ones(mesh.nC)

    if alphasmoothdefault:
        alpha_x, alpha_y, alpha_z = 1.0, 1.0, 1.0
    else:
        alpha_x = mesh.hx.min()**2
        alpha_y = mesh.hy.min()**2
        alpha_z = mesh.hz.min()**2
    
    # Define the regularization (model objective function).
    reg = regularization.Sparse(
        mesh,
        indActive=np.ones(mesh.n_cells, dtype=bool),
        mapping=actvMap,
        alpha_s=alpha_s,
        alpha_x=alpha_x,
        alpha_y=alpha_y,
        alpha_z=alpha_z,
        gradientType='total'
    )
    reg.norms = [[0, 1, 1, 1]]
    
    if dw:
        wr = utils.depth_weighting(
            mesh, data_mag.survey.receiver_locations, 
            indActive=np.ones(mesh.n_cells, dtype=bool), 
            exponent=3
        )
        reg.cell_weights = wr
        directives_list = []
    else:
        # Add sensitivity weights
        sensitivity_weights = directives.UpdateSensitivityWeights(everyIter=False)
        directives_list = [sensitivity_weights]

    opt = optimization.ProjectedGNCG(
        maxIter=100, lower=0.0, upper=1.0, maxIterLS=20, maxIterCG=100, tolCG=1e-4
    )

    inv_prob = inverse_problem.BaseInvProblem(dmis_mag, reg, opt)

    # Defining a starting value for the trade-off parameter (beta) between the data
    # misfit and the regularization.
    starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=1e-2)
    beta_schedule = directives.BetaSchedule(coolingFactor=5, coolingRate=1)
    update_jacobi = directives.UpdatePreconditioner()
    IRLS = directives.Update_IRLS(f_min_change=1e-4, max_irls_iterations=50, beta_tol=1e-2,)

    # save every iteration
    save_dict = directives.SaveOutputDictEveryIteration(saveOnDisk=True)
    every_iter_folder = directory + os.path.sep + 'EveryIteration_'+ name + os.path.sep
    os.makedirs(every_iter_folder, exist_ok=True)
    save_dict.directory = every_iter_folder
    
    # The directives are defined as a list.
    directives_list = directives_list + [
        starting_beta,
        beta_schedule,
        IRLS,
        update_jacobi,
        save_dict,
    ]

    inv = inversion.BaseInversion(inv_prob, directives_list)

    # Run inversion
    recovered_model_mag = inv.run(m0)
    os.makedirs(directory, exist_ok=True)
    mesh.write_model_UBC(directory + os.path.sep + name + ".sus", recovered_model_mag)


################################################################    
alphaslist = np.r_[1]
#alphasmoothdefault = [True, False]
#dw = [True,False]

#namealpha = ['as{}'.format(alphas) for alphas in alphaslist]
#namereg = ['defaultSmooth1_','defaultSmooth0_']
#namew = ['dw_','sw_']

#d1s, d2s = np.meshgrid(alphaslist, alphasmoothdefault, dw)
#d1s = d1s.reshape(-1)
#d2s = d2s.reshape(-1)

#for i in range(rank, n_params, size):
    
directory_mag='magnetic_inversions_LpLq'
directory_grav='gravity_inversions_LpLq'

for alphas in alphaslist:
    for alphasmoothdefault,namereg in zip([True], ['defaultSmooth1_']):
        for dw, namew in zip([True], ['dw_']):
            
            name = 'minv_'+namereg+namew+'as{}'.format(alphas)
            print("magnetic: ", name)
            run_mag_inversion(directory=directory_mag, dw=dw, alphasmoothdefault=alphasmoothdefault, alpha_s=alphas, name=name)

#for alphas in alphaslist:
#    for alphasmoothdefault,namereg in zip([False, True], ['defaultSmooth0_','defaultSmooth1_']):
#        for dw, namew in zip([True,False], ['dw_','sw_']):
#
#            name = 'minv_'+namereg+namew+'as{}'.format(alphas)
#            print("gravity: ", name)
#            run_grav_inversion(directory=directory_grav, dw=dw, alphasmoothdefault=alphasmoothdefault, alpha_s=alphas, name=name)
