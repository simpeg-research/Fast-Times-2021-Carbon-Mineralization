# Fast-Times-2021-Carbon-Mineralization

This repository contains the scripts and notebooks used to generate the Figures in the article "Geophysical Inversions to Delineate Rocks with CO2 Sequestration Potential Through Carbon Mineralization"

https://fasttimesonline.co/geophysical-inversions-to-delineate-rocks-with-co-2-sequestration-potential-through-carbon-mineralization/

## Contents

### Scripts
- model setup 
- magnetics
   - [magnetics L2](./3_blocks_model/L2/Script/Batch_Mag_Inversion_L2.py)
   - [magnetics L01](./3_blocks_model/Lpq/Script/Batch_Mag_Inversion_LpLq_mag.py)
   - volume estimates 
- gravity 
   - [gravity L2](./3_blocks_model/L2/Script/Batch_Mag_Inversion_L2.py) (same script as the Mag L2)
   - [gravity L01](./3_blocks_model/Lpq/Script/Batch_Mag_Inversion_LpLq_grav.py)
   - volume estimates 
- one block inversions
   - [grav and mag L2](./1_block_model_4k/Scripts/Batch_Inversions_L2.py)
   - [magnetics L01](./1_block_model_4k/Scripts/Batch_Mag_Inversion_LpLq.py)
   - [gravity L01](./1_block_model_4k/Scripts/Batch_Grav_Inversion_LpLq.py)
- PGI
   - [individual magnetics](./3_blocks_model/pgi-magnetic/FINAL_DepthWeighting/CaMP_PGI_magnetic-cloudrun.py)
   - [individual gravity](./3_blocks_model/pgi-gravity/FINAL_DepthWeighting/CaMP_PGI_gravity-cloudrun.py)
   - [joint pgi](./3_blocks_model/pgi-joint/FINAL_DepthWeighting/CHOICE_GMM2_DW_ScaleMag1.0_Smoothden0.01_RERUN.py)
- cumulative volume estimates

### Notebooks for figures
- [figures for the 3 block model](./3_blocks_model/Final_Visualization/MakeFigures-3blocks.ipynb)
- volume estimates for the 3 block model
- figures for the 1 block model
- volume estimates for the 1 block model
