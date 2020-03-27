This repository contains code for the paper  'Learnable Douglas-Rachford Iteration and Its Applications in DOT Imaging'


# Dataset
The Jacobian matrix and dataset are available at 
https://www.dropbox.com/sh/0fijlgtlmlm1y45/AAD9gXtxUUlzzXjWyYUjMJbHa?dl=0



# How to use the code
## Training

python ./train_dot_recon.py --images-dataset  dosdata_sim_exp_dataset.mat   --measurements-dataset  dosdata_sim_exp_dataset.mat --Jacobian-matrix utility/J.mat  --batch-size 128  --inversion-regularize-method D --stages 6 --output-name net_configs/prediction_dot_D_stg6_.pth.tar --epochs 300

## Testing 

python ./predict_dot_recon.py   --measurements  dosdata_sim_exp_testset.mat --Jacobian-matrix utility/J.mat  --batch-size 128  --inversion-regularize-method D --stages 6 --prediction-parameter net_configs/prediction_dot_D_stg6_.pth.tar --output-prefix output/all_D_stg6_


