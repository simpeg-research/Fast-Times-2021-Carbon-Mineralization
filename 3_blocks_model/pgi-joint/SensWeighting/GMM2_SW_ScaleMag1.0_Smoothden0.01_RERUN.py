#!/usr/bin/env python
# coding: utf-8

# # Import

# In[ ]:


import os

import sys
sys.path.insert(0,'/tera_raid/mmitchel/Programs/simpeg/simpeg-main-0152')
from SimPEG import dask
import SimPEG
print(SimPEG.__path__)
print(SimPEG.__version__)

import discretize as ds
print(ds.__path__)
print(ds.__version__)


# In[2]:


import discretize as ds
import SimPEG.potential_fields as pf
from SimPEG import (
    maps, utils, simulation, inverse_problem, inversion, optimization, regularization, data_misfit, directives
)
from SimPEG.utils import io_utils
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

# In[3]:

#Reproducible science
np.random.seed(518936)


# # Setup

# ## Load Mesh

# In[4]:


mesh = ds.TreeMesh.read_UBC('mesh_CaMP.ubc')


# ## Load True geological model for comparison with inversion result

# In[9]:

true_geology = mesh.read_model_UBC('CaMP_magnetic_synthetic_model.ubc')
true_geology[true_geology==0.15] = 2
true_geology[true_geology==0.05] = 1


# ## Load geophysical data

# In[12]:

data_grav = io_utils.read_grav3d_ubc('grav_data.obs')
data_mag = io_utils.read_mag3d_ubc('magnetic_data.obs')


# In[15]:


actvMap = maps.IdentityMap(mesh)
actv = np.ones(mesh.nC, dtype='bool')
nactv = mesh.nC


# ## Create simulations and data misfits

# In[16]:


# Wires mapping
wires = maps.Wires(('den', actvMap.nP), ('sus', actvMap.nP))
gravmap = actvMap * wires.den
magmap = actvMap * wires.sus
idenMap = maps.IdentityMap(nP=nactv)


# In[17]:


# Grav problem
simulation_grav = pf.gravity.simulation.Simulation3DIntegral(
    survey=data_grav.survey,
    mesh=mesh,
    rhoMap=wires.den,
    actInd=actv,
)
dmis_grav = data_misfit.L2DataMisfit(data=data_grav, simulation=simulation_grav)


# In[18]:


# Mag problem
simulation_mag = pf.magnetics.simulation.Simulation3DIntegral(
    survey=data_mag.survey,
    mesh=mesh,
    chiMap=wires.sus,
    actInd=actv,
)
dmis_mag = data_misfit.L2DataMisfit(data=data_mag, simulation=simulation_mag)


# In[19]:

# Data Misfit
dmis = 0.5 * dmis_grav + 0.5 * dmis_mag

# In[21]:

# Initial Model
m0 = np.r_[0. * np.ones(actvMap.nP), 1e-4 * np.ones(actvMap.nP)]

# # Inversion with full petrophysical information

# ## Create petrophysical GMM

# In[43]:

covariances_list = [
    #0.5 * np.array([[2.5e-05, 2.5e-05],[3e-04, 1e-04],[2.4e-03, 4e-04]]),
    #np.array([[1e-05, 1e-06],[1e-04, 1e-04],[4e-04, 4e-04]]),
    np.array([[5e-05, 5e-05],[2.5e-04, 5e-04],[5e-04, 1e-03]]),
    #np.array([[5e-05, 5e-06],[1e-04, 5e-04],[5e-04, 1e-03]])
]

inversion_list = []
for scale_mag in [1.]:
    for alpha_den in [1e-2]:
        for i,cov in enumerate(covariances_list):
            for dwname in ['SW']:
    
                folder_name = 'GMM{}_'.format(i+2) + dwname + '_ScaleMag{}'.format(scale_mag) + '_SmoothDen{}'.format(alpha_den) + '_RERUN2'
    
                joint_dict={
                    'cov':cov,
                    'dwname':dwname,
                    'folder_name':folder_name,
                    'scale_mag':scale_mag,
                    'alpha_den':alpha_den
                }
    
                inversion_list.append(joint_dict)

def joint_inversion(invdict):

    os.makedirs(invdict['folder_name'],exist_ok=True)

    gmmref = utils.WeightedGaussianMixture(
        n_components=3, #number of rock units: bckgrd, PK, HK
        mesh=mesh, # inversion mesh
        actv=actv, #actv cells
        covariance_type='diag', # diagonal covariances
    )
    # required: initialization with fit
    # fake random samples, size of the mesh, number of physical properties: 2 (density and mag.susc)
    gmmref.fit(np.random.randn(nactv,2))
    # set parameters manually
    #    set phys. prop means for each unit
    gmmref.means_ = np.c_[
        [0.,0.], # BCKGRD density contrast and mag. susc
        [0.1, 0.05], # MAFIC
        [-0.2, 0.15], # SERPENTINIZED
    ].T
    # set phys. prop covariances for each unit
    gmmref.covariances_ = invdict['cov']
    # important after setting cov. manually: compute precision matrices and cholesky
    gmmref.compute_clusters_precisions()
    #set global proportions; low-impact as long as not 0 or 1 (total=1)
    gmmref.weights_ = np.r_[0.9, 0.075, 0.025]

    pickle.dump(gmmref, open(invdict['folder_name'] + os.path.sep + "GMMRF_joint.p", "wb"))
    # ## Create PGI regularization

    # In[19]:


    # Sensitivity weighting
    if invdict['dwname'] == 'SW':
        wr_grav = np.sum(simulation_grav.G**2., axis=0)**0.5 / mesh.cell_volumes 
        wr_grav = (wr_grav / np.max(wr_grav))

        wr_mag = np.sum(simulation_mag.G**2., axis=0)**0.5 / mesh.cell_volumes 
        wr_mag = (wr_mag / np.max(wr_mag))
    else:
        wr_grav = utils.depth_weighting(
            mesh, data_grav.survey.receiver_locations, 
            indActive=np.ones(mesh.n_cells, dtype=bool), 
            exponent=2
        )

        wr_mag = utils.depth_weighting(
            mesh, data_mag.survey.receiver_locations, 
            indActive=np.ones(mesh.n_cells, dtype=bool), 
            exponent=3
        )
    # create joint PGI regularization with smoothness
    reg = utils.make_PGI_regularization(
        gmmref=gmmref,
        mesh=mesh,
        wiresmap=wires,
        maplist=[idenMap, idenMap],
        mref=np.zeros_like(m0),
        indActive=actv,
        alpha_s=1.0, alpha_x=1.0, alpha_y=1.0, alpha_z=1.0,
        alpha_xx=0., alpha_yy=0., alpha_zz=0.,
        cell_weights_list=[np.asarray(wr_grav),np.asarray( wr_mag)] # weights each phys. prop. by correct sensW
    )


    # ## Directives

    # Add directives to the inversion
    # ratio to use for each phys prop. smoothness in each direction; roughly the ratio of range of each phys. prop.
    alpha0_ratio = np.r_[np.zeros(len(reg.objfcts[0].objfcts)),
                    invdict['alpha_den'] * np.ones(len(reg.objfcts[1].objfcts)),
                    1e-2 * np.ones(len(reg.objfcts[2].objfcts))]
    Alphas = directives.AlphasSmoothEstimate_ByEig(
        alpha0_ratio=alpha0_ratio,
        verbose=True
    )
    # initialize beta and beta/alpha_s schedule
    beta = directives.BetaEstimate_ByEig(beta0_ratio=1.0)
    betaIt = directives.PGI_BetaAlphaSchedule(
        verbose=True, 
        coolingFactor=2., 
        tolerance=0.2,
        progress=1.,
        betamin=1e-16,
    )
# geophy. and petro. target misfits
    targets = directives.MultiTargetMisfits(
        verbose=True,
        chiSmall=0.5,
    )
    # add learned mref in smooth once stable
    MrefInSmooth = directives.PGI_AddMrefInSmooth(
        wait_till_stable=True,
        verbose=True,
    )
    # update the parameters in smallness (L2-approx of PGI)
    update_smallness = directives.PGI_UpdateParameters(
        update_gmm = False #keep GMM model fixed
    )
    # pre-conditioner
    update_Jacobi = directives.UpdatePreconditioner()
    # iteratively balance the scaling of the data misfits
    scaling_init = directives.ScalingMultipleDataMisfits_ByEig(chi0_ratio=[1.,scale_mag])
    scale_schedule = directives.JointScalingSchedule(verbose=True)

    # Options for outputting recovered models and predicted data for each beta.
    save_iteration = directives.SaveModelEveryIteration()
    #save_iteration = directives.SaveOutputEveryIteration(save_txt=True)
    os.makedirs(invdict['folder_name'] + os.path.sep + 'EveryIterationModels', exist_ok=True)
    save_iteration.directory = invdict['folder_name'] + os.path.sep + 'EveryIterationModels' + os.path.sep
    print(invdict['folder_name'])

    save_dict = directives.SaveOutputDictEveryIteration(saveOnDisk=True)
    save_dict.directory = invdict['folder_name'] + os.path.sep + 'EveryIterationModels' + os.path.sep


    # Optimization
    # set lower and upper bounds
    lowerbound = np.r_[-1. * np.ones(actvMap.nP), 0. * np.ones(actvMap.nP)]
    upperbound = np.r_[1. * np.ones(actvMap.nP), 1. * np.ones(actvMap.nP)]
    opt = optimization.ProjectedGNCG(
        maxIter=30,
        lower=lowerbound, upper=upperbound,
        maxIterLS=20,
        maxIterCG=100, tolCG=1e-4
    )
    # create inverse problem
    invProb = inverse_problem.BaseInvProblem(dmis, reg, opt)
    inv = inversion.BaseInversion(
        invProb,
        # directives: evaluate alphas (and data misfits scales) before beta
        directiveList=[
            Alphas, 
            scaling_init, 
            beta,
            update_smallness, targets, scale_schedule,
            betaIt, MrefInSmooth, update_Jacobi,
            save_iteration, save_dict,
        ]
    )  

    pgi_model = inv.run(m0)

    quasi_geology_model = reg.objfcts[0].compute_quasi_geology_model()
    density_model = gravmap * pgi_model
    magsus_model = magmap * pgi_model

    np.save(invdict['folder_name'] + os.path.sep + 'pgi_gravity_joint', density_model)
    np.save(invdict['folder_name'] + os.path.sep + 'pgi_magnetic_joint', magsus_model)
    np.save(invdict['folder_name'] + os.path.sep + 'pgi_quasigeology_joint', quasi_geology_model)
    np.save(invdict['folder_name'] + os.path.sep + 'pgi_model', pgi_model)
    
    mdict = {
        'density_model.den':density_model,
        'magsus_model.sus':magsus_model,
        'quasi_geology.geo':quasi_geology_model
            }
    
    mesh.write_UBC('mesh.msh', models=mdict, directory=invdict['folder_name'] + os.path.sep)
    
    dpred_pgi_grav = simulation_grav.make_synthetic_data(pgi_model)
    utils.io_utils.write_grav3d_ubc(
        invdict['folder_name'] + os.path.sep + 'dpred_pgi.grav',
        dpred_pgi_grav
        )
    
    dpred_pgi_mag = simulation_mag.make_synthetic_data(pgi_model)
    utils.io_utils.write_mag3d_ubc(
        invdict['folder_name'] + os.path.sep + 'dpred_pgi.mag',
        dpred_pgi_mag
    )


for joint in inversion_list:
    print(joint['folder_name'])
    joint_inversion(joint)




