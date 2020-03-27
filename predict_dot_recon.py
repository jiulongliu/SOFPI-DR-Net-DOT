import sys
sys.path.append('./dos_Operators')
import argparse
import torch
import torch.utils.data


import inverse_network

import numpy as np
import scipy.io
import pydotrecon 

#parse command line input
parser = argparse.ArgumentParser(description='DOT/DOS Reconstruction given set of measurements')

##required parameters
requiredNamed = parser.add_argument_group('required named arguments')
requiredNamed.add_argument('--measurements', nargs='+', required=True, metavar=('m1', 'm2, m3...'),
                           help='List of measurements stored in .mat File names are seperated by space.')
requiredNamed.add_argument('--Jacobian-matrix', required=True, metavar=('file_name'),
                           help='Jacobian matrix input directory + name of the network parameters, in .mat format.')
requiredNamed.add_argument('--output-prefix', nargs='+', required=True, metavar=('o1', 'o2, o3...'),
                           help='List of reconstruction output prefixes for each measurement , seperated by space. Preferred to be a directory (e.g. /some_path/output_dir/)')


#parser.add_argument('--sofip-nets-version', type=str, default='ADMM', metavar='string',
#                    help='ADMM: Alternating Direction Method of Multipliers ; PD: Primal-dual. (default: ADMM)')
parser.add_argument('--inversion-regularize-method', type=str, default='D', metavar='string',
                    help='I: identity; W: gradient operator; D: CNN with learned filters. (default: D)')
parser.add_argument('--inversion-version', type=str, default='torch', metavar='string',
                    help='torch: torch ; cuda: conjugate gradient defined by cuda . (default: torch)')
parser.add_argument('--forward-operator-version', type=str, default='cuda', metavar='string',
                    help='torch: torch.fft; cuda:   fft defined by cuda. (default: torch)')
parser.add_argument('--image-size-nx', type=int, default=9, metavar='N',
                    help='image height')
parser.add_argument('--image-size-ny', type=int, default=12, metavar='N',
                    help='image width')
parser.add_argument('--image-size-nz', type=int, default=13, metavar='N',
                    help='image temporal length')
parser.add_argument('--stages', type=int, default=6, metavar='N',
                    help='number of stages of the deep network (default: 6)')
parser.add_argument('--rho-learnable', type=bool, default=True,  help='this is a True or False we want')
parser.add_argument('--features', type=int, default=4, metavar='N',
                    help='number of output features for the first layer of the deep network (default: 8)')
parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                    help='input batch size for prediction network (default: 8)')
parser.add_argument('--use-dropout', action='store_true', default=False,
                    help='Use dropout to train the network')
parser.add_argument('--n-GPU', type=int, default=1, metavar='N',
                    help='number of GPUs used for training.')
parser.add_argument('--prediction-parameter', default='../../network_configs/predict.pth.tar',
                    help="network parameters for the prediction network")
args = parser.parse_args()




def create_net(args, network_config):
    JacMat = scipy.io.loadmat(args.Jacobian_matrix)['JM']
    Maix = scipy.io.loadmat(args.Jacobian_matrix)['Maix'][:,0]
    Msix = scipy.io.loadmat(args.Jacobian_matrix)['Msix'][:,0]


    
    
    M=[]
    for i in range(0, args.stages):
        if args.forward_operator_version=='torch': 
            Mi=pydotrecon.dotoplib(JacMat = JacMat, Maix = Maix, Msix = Msix, nx=9,ny=12,nz=13,rho_upperbound=10.0-i*1.2,mu=0.0,rho=0.5,lam=0.08,N_iter=0,CG_iter=15,CG_tol=1e-8,Min_iter=10)   #   
        elif args.forward_operator_version=='cuda': 
            Mi=pydotrecon.dotoplib(JacMat = JacMat, Maix = Maix, Msix = Msix, nx=9,ny=12,nz=13,rho_upperbound=10.0-i*1.0,mu=0.0,rho=0.5,lam=0.08,N_iter=0,CG_iter=5,CG_tol=1e-8,Min_iter=10)   #  10+20*i 
        M.append(Mi)
    net_single = inverse_network.sofpi_admm_nets(1, args.features, args.use_dropout, M, args.inversion_regularize_method, args.rho_learnable, args.inversion_version, args.forward_operator_version).cuda()                    
        
    net_single.load_state_dict(network_config['state_dict'])

    if (args.n_GPU > 1) :
        device_ids=range(0, args.n_GPU)
        net = torch.nn.DataParallel(net_single, device_ids=device_ids).cuda()
    else:
        net = net_single
    
    net.train()
    return net;

def predict_recon_image(args):

    predict_network_config = torch.load(args.prediction_parameter)
    prediction_net = create_net(args, predict_network_config);

    JacMat = scipy.io.loadmat(args.Jacobian_matrix)['JM']
#    Maix = scipy.io.loadmat(args.Jacobian_matrix)['Maix'][:,0]
#    Msix = scipy.io.loadmat(args.Jacobian_matrix)['Msix'][:,0]

    
    for i in range(0, len(args.measurements)):

        print(len(args.measurements))

      
        
        Y_test = torch.from_numpy(scipy.io.loadmat(args.measurements[i])['Yst'].transpose((1,0))).view((-1,JacMat.shape[0],1))

        mvrc0_batch_np = scipy.io.loadmat(args.measurements[i])['imgtvrst'].transpose((4,3,2,1,0))

        mvrc0_part = torch.from_numpy(mvrc0_batch_np)#.view(-1,1,args.image_size_nx,args.image_size_ny).float()#.cuda()
        recon_p = np.zeros((args.image_size_nz,args.image_size_ny,args.image_size_nx, 2, Y_test.size(0),args.stages*2))

        
        torch_testset = torch.utils.data.TensorDataset(mvrc0_part,Y_test)
        
        test_loader = torch.utils.data.DataLoader(dataset=torch_testset, batch_size=args.batch_size, shuffle=False,	num_workers=4)
        
        
        for j, (mvrc0_part_batch,Y_test_batch) in enumerate(test_loader):
            
            input_batch_variable = (mvrc0_part_batch.cuda(), Y_test_batch.cuda())
            recon_result = prediction_net(input_batch_variable)#util.predict_momentum(moving_image_np, target_image_np, input_batch, batch_size, patch_size, prediction_net, predict_transform_space);

            print(len(recon_result),recon_result[2].shape)
    #        print(recon_result.shape)
            for k in range(len(recon_result)):
                recon_p[j*mvrc0_part_batch.size(0):(j+1)*mvrc0_part_batch.size(0),:,:,:,:,k] = recon_result[k][:,:,:,:,:].detach().cpu().numpy().transpose((4,3,2,1,0))#prediction_result['image_space']  .view(-1,args.image_size_nx,args.image_size_ny)      
        scipy.io.savemat(args.output_prefix[i]+'recon_res_'+args.inversion_regularize_method+'.mat', {'recon_imgs_net':recon_p,'recon_imgs_admm':mvrc0_batch_np[:,:,:,:,:].transpose((4,3,2,1,0))})




if __name__ == '__main__':
    predict_recon_image(args)
