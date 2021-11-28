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

# In[15]:


actvMap = maps.IdentityMap(mesh)
actv = np.ones(mesh.nC, dtype='bool')
nactv = mesh.nC


# ## Create simulations and data misfits

# In[16]:


# Wires mapping
wires = maps.Wires(('den', actvMap.nP))
gravmap = actvMap * wires.den
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

# Initial Model
m0 = 0. * np.ones(actvMap.nP)

# # Inversion with full petrophysical information

# ## Create petrophysical GMM

# In[43]:

gmmref = utils.WeightedGaussianMixture(
    n_components=3, #number of rock units: bckgrd, PK, HK
    mesh=mesh, # inversion mesh
    actv=actv, #actv cells
    covariance_type='diag', # diagonal covariances
)
# required: initialization with fit
# fake random samples, size of the mesh, number of physical properties: 2 (density and mag.susc)
gmmref.fit(np.random.randn(nactv,1))
# set parameters manually
# set phys. prop means for each unit
gmmref.means_ = np.c_[
    [0.], # BCKGRD density contrast and mag. susc
    [0.1], # MAFIC
    [-0.2], # SERPENTINIZED
].T
# set phys. prop covariances for each unit
gmmref.covariances_ = np.array([[1e-05],
       [1e-04],
       [1e-04]])
# important after setting cov. manually: compute precision matrices and cholesky
gmmref.compute_clusters_precisions()
#set global proportions; low-impact as long as not 0 or 1 (total=1)
gmmref.weights_ = np.r_[0.9, 0.075, 0.025]

# ## Create PGI regularization

# In[19]:


wr_grav = utils.depth_weighting(
            mesh, data_grav.survey.receiver_locations, 
            indActive=np.ones(mesh.n_cells, dtype=bool), 
            exponent=2
)
print('Depth Weighting')

# create joint PGI regularization with smoothness
reg = utils.make_PGI_regularization(
    gmmref=gmmref,
    mesh=mesh,
    wiresmap=wires,
    maplist=[idenMap],
    mref=m0,
    indActive=actv,
    alpha_s=1.0, alpha_x=1.0, alpha_y=1.0, alpha_z=1.0,
    alpha_xx=0., alpha_yy=0., alpha_zz=0.,
    cell_weights_list=[np.asarray(wr_grav)] # weights each phys. prop. by correct sensW
)


# ## Directives

# In[20]:


# Add directives to the inversion
# ratio to use for each phys prop. smoothness in each direction; roughly the ratio of range of each phys. prop.
alpha0_ratio = np.r_[np.zeros(len(reg.objfcts[0].objfcts)),
                    1e-2 * np.ones(len(reg.objfcts[1].objfcts))]
Alphas = directives.AlphasSmoothEstimate_ByEig(
    alpha0_ratio=alpha0_ratio,
    verbose=True
)
# initialize beta and beta/alpha_s schedule
beta = directives.BetaEstimate_ByEig(beta0_ratio=1.)
betaIt = directives.PGI_BetaAlphaSchedule(
    verbose=True, 
    coolingFactor=2., 
    tolerance=0.2,
    progress=0.2,
)
# geophy. and petro. target misfits
targets = directives.MultiTargetMisfits(
    verbose=True,
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
scale_schedule = directives.JointScalingSchedule(verbose=True)


# ## Create inverse problem

# In[21]:


# Optimization
# set lower and upper bounds
lowerbound = -1.0
upperbound = 1.0
opt = optimization.ProjectedGNCG(
    maxIter=100,
    lower=lowerbound, upper=upperbound,
    maxIterLS=20,
    maxIterCG=100, tolCG=1e-4
)
# create inverse problem
invProb = inverse_problem.BaseInvProblem(dmis_grav, reg, opt)
inv = inversion.BaseInversion(
    invProb,
    # directives: evaluate alphas (and data misfits scales) before beta
    directiveList=[
        Alphas, 
        beta,
        update_smallness, targets,
        betaIt, MrefInSmooth, update_Jacobi
    ]
)  


# In[22]:


pgi_model = inv.run(m0)


# ## Plot the result with full petrophysical information

# In[23]:


density_model = gravmap * pgi_model
quasi_geology_model = actvMap * reg.objfcts[0].membership(reg.objfcts[0].mref)


# In[26]:


np.save('pgi_gravity-singlephysics', density_model)
np.save('pgi_quasigeology_gravity-singlephysics', quasi_geology_model)


# In[27]:


import pickle


# In[28]:


pickle.dump(gmmref, open("GMMRF-_gravity-singlephysics.p", "wb"))


# In[ ]:




