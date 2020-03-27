import sys
sys.path.append('CT_Operators')
import argparse
import gc

import torch
import torch.utils.data

import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.autograd import Variable
import inverse_network

import numpy as np
import scipy.io
import pydotrecon 


#parse command line input
parser = argparse.ArgumentParser(description='DOT/DOS Reconstruction given set of measurements')

##required parameters
requiredNamed = parser.add_argument_group('required named arguments')
requiredNamed.add_argument('--images-dataset', nargs='+', required=True, metavar=('i1', 'i2, i3...'),
                           help='List of  images datasets stored in .pth.tar format File names are seperated by space.')
requiredNamed.add_argument('--measurements-dataset', nargs='+', required=True, metavar=('m1', 'm2, m3...'),
                           help='List of measurement datasets stored in .pth.tar File names are seperated by space.')
requiredNamed.add_argument('--output-name', required=True, metavar=('file_name'),
                           help='output directory + name of the network parameters, in .pth.tar format.')
requiredNamed.add_argument('--Jacobian-matrix', required=True, metavar=('file_name'),
                           help='Jacobian matrix input directory + name of the network parameters, in .mat format.')
##optional parameters
parser.add_argument('--sofip-nets-version', type=str, default='ADMM', metavar='string',
                    help='ADMM: Alternating Direction Method of Multipliers ; PD: Primal-dual. (default: ADMM)')
parser.add_argument('--inversion-regularize-method', type=str, default='D', metavar='string',
                    help='I: identity; W: gradient operator; D: CNN with learned filters. (default: D)')
parser.add_argument('--inversion-version', type=str, default='torch', metavar='string',
                    help='torch: torch ; cuda: conjugate gradient defined by cuda . (default: torch)')
parser.add_argument('--forward-operator-version', type=str, default='torch', metavar='string',
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
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train the network (default: 10)')
parser.add_argument('--learning-rate', type=float, default=0.0001, metavar='N',
                    help='learning rate for the adam optimization algorithm (default: 0.0001)')
parser.add_argument('--use-dropout', action='store_true', default=False,
                    help='Use dropout to train the network')
parser.add_argument('--n-GPU', type=int, default=1, metavar='N',
                    help='number of GPUs used for training.')
parser.add_argument('--continue-from-parameter', metavar=('parameter_name'),
                           help='file directory+name of the existing parameter ')
parser.add_argument('--continue-from-pretrained-parameter', metavar=('parameter_name'),
                           help='file directory+name of the pretrained parameter ')
args = parser.parse_args()
# finish command line input


# check validity of input arguments from command line

def check_args(args):
    n_img = len(args.images_dataset)
    n_mea = len(args.measurements_dataset)
    if (n_img != n_mea):
        print('The number of image datasets is not consistent with the number of measurement datasets!')
        sys.exit(1)
    if (args.n_GPU <= 0):
        print('Number of GPUs must be positive!')
        sys.exit(1)


class LossFunc(nn.Module):
    def __init__(self):
        super(LossFunc, self).__init__()
    def forward(self, rcp, mgd):  
        N = len(rcp)
        d = 1.0/2.0 * func.mse_loss(rcp[N-1], mgd, size_average=None, reduce=None, reduction='sum')
        for i in range(0, N-1): 
            d = d + 1.0/(N-i+0.0)*func.mse_loss(rcp[i], mgd, size_average=None, reduce=None, reduction='sum')
        return d





def create_net(args):    
    JacMat = scipy.io.loadmat(args.Jacobian_matrix)['JM']
    Maix = scipy.io.loadmat(args.Jacobian_matrix)['Maix'][:,0]
    Msix = scipy.io.loadmat(args.Jacobian_matrix)['Msix'][:,0]
#    mask_size = mask.shape
#    if (mask_size[0]!=args.image_size_nx):
#        print('mask width not correct')
#        sys.exit(1)
#    if (mask_size[1]!=args.image_size_ny):
#        print('mask height not correct')
#        sys.exit(1)

    
    
    M=[]
    for i in range(0, args.stages):
        if args.forward_operator_version=='torch': 
#            Mi=pydotrecon.dotoplib(JacMat = JacMat, Maix = Maix, Msix = Msix, nx=9,ny=12,nz=13,rho_upperbound=300.0,mu=0.0,rho=0.5,lam=0.08,N_iter=0,CG_iter=15,CG_tol=1e-8,Min_iter=10)   #  
#            Mi=pydotrecon.dotoplib(JacMat = JacMat, Maix = Maix, Msix = Msix, nx=9,ny=12,nz=13,rho_upperbound=2.0-i*0.20,mu=0.0,rho=0.5,lam=0.08,N_iter=0,CG_iter=15,CG_tol=1e-8,Min_iter=10)   #               
            Mi=pydotrecon.dotoplib(JacMat = JacMat, Maix = Maix, Msix = Msix, nx=9,ny=12,nz=13,rho_upperbound=10.0-i*1.2,mu=0.0,rho=0.5,lam=0.08,N_iter=0,CG_iter=15,CG_tol=1e-8,Min_iter=10)   #               

            #            Mi=pydotrecon.dotoplib(JacMat = JacMat, Maix = Maix, Msix = Msix, nx=9,ny=12,nz=13,rho_upperbound=15+i*6.0,mu=0.0,rho=0.5,lam=0.08,N_iter=0,CG_iter=15,CG_tol=1e-8,Min_iter=10)   #   
        elif args.forward_operator_version=='cuda': 
            Mi=pydotrecon.dotoplib(JacMat = JacMat, Maix = Maix, Msix = Msix, nx=9,ny=12,nz=13,rho_upperbound=300.0,mu=0.0,rho=0.5,lam=0.08,N_iter=0,CG_iter=5,CG_tol=1e-8,Min_iter=10)   #  10+20*i 
        M.append(Mi)
                
#    if args.sofip_nets_version=='ADMM':        
#        if args.inversion_regularize_method=='I':        
#            net_single = inverse_network.sofpi_admm_net_rho_I_CT(1,args.features, args.use_dropout, M).cuda()
#            print('Identity operator(Tikhonov regularization) as regularization method for inversion!')        
#        elif args.inversion_regularize_method=='W':
#            net_single = inverse_network.sofpi_admm_net_rho_W_CT(1,args.features, args.use_dropout, M).cuda() 
#            print('Predifined gradient operator applied as regularization method for inversion!')        
#        elif args.inversion_regularize_method=='D':
#            net_single = inverse_network.sofpi_admm_net_rho_D_CT(1,args.features, args.use_dropout, M).cuda() 
#            print('learnable CNN applied as regularization method for inversion!')
#        else:
#            print('Not valid regularization method for inversion!')
#            sys.exit(1)
#    elif args.sofip_nets_version=='PD':
#        if args.inversion_regularize_method=='I':        
#            net_single = inverse_network.sofpi_pd_net_I(1,args.features, args.use_dropout, M).cuda()
#            print('Identity operator(Tikhonov regularization) as regularization method for inversion!')        
        
    net_single = inverse_network.sofpi_admm_nets(1, args.features, args.use_dropout, M, args.inversion_regularize_method, args.rho_learnable, args.inversion_version, args.forward_operator_version).cuda()                    
        
        
    if (args.continue_from_parameter != None):
        print('Loading existing parameter file!')
        config = torch.load(args.continue_from_parameter)
        net_single.load_state_dict(config['state_dict'])
        
                
    if (args.continue_from_pretrained_parameter != None):
        print('Loading pretrained parameter file!')
        config = torch.load(args.continue_from_pretrained_parameter)        
        lsd = list(config['state_dict'].keys())
        for k in range(2,args.stages):
            for x in lsd:
                if x.find('.1.')!=-1:
            #        print(x,x.find('.1.'))
                    beg=x.find('.1.')
        
                    y = x[0:beg+1]+str(k)+x[beg+2:len(x)+1]
#                    print(y)
                    config['state_dict'][y]=config['state_dict'][x]        
        
        net_single.load_state_dict(config['state_dict'])

    if (args.n_GPU > 1) :
        device_ids=range(0, args.n_GPU)
        net = torch.nn.DataParallel(net_single, device_ids=device_ids).cuda()
    else:
        net = net_single
    net.train()
    return net


def train_cur_data(cur_epoch, datapart, image_file, measurement_file, output_name, net, criterion, optimizer, args):


    JacMat = scipy.io.loadmat(args.Jacobian_matrix)['JM']
    Maix = scipy.io.loadmat(args.Jacobian_matrix)['Maix'][:,0]
    Msix = scipy.io.loadmat(args.Jacobian_matrix)['Msix'][:,0]
    image_trainset = torch.from_numpy(scipy.io.loadmat(image_file)['imgsd'].transpose((4,3,2,1,0)))
    Y_trainset = torch.from_numpy(scipy.io.loadmat(measurement_file)['Ysd'].transpose((1,0))).view((-1,JacMat.shape[0],1))

    dataset_size = image_trainset.size()
#    print(dataset_size)
    mea_size = Y_trainset.size()
    if (len(dataset_size)!=5):
        print('The size of images dataset should be (nl, nx, ny, nz)')
        sys.exit(1)
    if (dataset_size[2]!=args.image_size_nx):
        print('images width not correct')
        sys.exit(1)
    if (dataset_size[3]!=args.image_size_ny):
        print('images height not correct')
        sys.exit(1)
    if (dataset_size[4]!=args.image_size_nz):
        print('images height not correct')
        sys.exit(1)        
    if (dataset_size[1]!=2):
        print('absorption coefficient and scattering coefficient are needed to be included')
        sys.exit(1)
        
    if (len(mea_size)!=3):
        print('The size of measurement dataset should be (nl, ns, 1), but got', mea_size)
        sys.exit(1)        
    if (mea_size[1]!=JacMat.shape[0]):
        print('measurement size not correct')
        sys.exit(1)


    N = dataset_size[0] / args.batch_size;    
    
#    M=pydotrecon.dotoplib(JacMat = JacMat, Maix = Maix, Msix = Msix,nx=9,ny=12,nz=13,rho_upperbound=4.0,mu=0.6,rho=0.0001,lam=0.08,N_iter=12,CG_iter=15,CG_tol=1e-8,Min_iter=10)   #   
#    mvrc0_batch_np = M.recon_admm_batch(Y_trainset.numpy(), np.zeros_like(image_trainset.numpy())) 
    mvrc0_batch_np = scipy.io.loadmat(image_file)['imgtvrsd'].transpose((4,3,2,1,0))


#    scipy.io.savemat("output/train/imtrain_mvrc0_nv_"+str(args.sinogram_size_nv)+"_ep_"+str(cur_epoch+1)+".mat", {'mvrc0_batch_np': mvrc0_batch_np,
#                                             'Y_part':Y_trainset.numpy(),
#                                             'image_trainset':image_trainset.numpy()})#enddef  
    mvrc0_part = torch.from_numpy(mvrc0_batch_np) 
#    print(image_trainset.shape,Y_trainset.shape, mvrc0_part.shape )
    torch_dataset = torch.utils.data.TensorDataset(image_trainset,Y_trainset,mvrc0_part)       
    train_loader = torch.utils.data.DataLoader(dataset=torch_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)        #
#    input_batch = torch.zeros(args.batch_size, 2, dataset_size[1], dataset_size[2])#.cuda()
    for i, (mv_batch,Y_batch,mvrc0_batch) in enumerate(train_loader):

        mvrc0_batch_variable = Variable(mvrc0_batch.view(-1,2,args.image_size_nx,args.image_size_ny,args.image_size_nz)).cuda()
        
        Y_batch_variable = Variable(Y_batch).cuda()
        mv_batch_variable = Variable(mv_batch).cuda()
        input_batch_variable = (mvrc0_batch_variable,Y_batch_variable) 
        optimizer.zero_grad()
        rcp_batch_variable = net(input_batch_variable) 
        loss = criterion(rcp_batch_variable,mv_batch_variable) 
        loss.backward()
        loss_value = loss.data.item()
        optimizer.step()
   
        print('====> Epoch: {}, data_part: {}, iter: {}/{}, loss: {:.10f}, rho1: {}, rho2: {}, rho3: {}, rho4: {}, rho5: {}, rho6: {}'.format(
            cur_epoch+1, datapart+1, i, N, loss_value/args.batch_size,net.invblks[0].M.rho, net.invblks[1].M.rho, net.invblks[2].M.rho, net.invblks[3].M.rho, net.invblks[4].M.rho, net.invblks[5].M.rho))      
        if i % 10 == 0 or i == N-1:
            if args.n_GPU > 1:
                cur_state_dict = net.module.state_dict()
            else:
                cur_state_dict = net.state_dict()            
            
            modal_name = output_name
            
            model_info = {
                'network_feature' : args.features,
                'state_dict': cur_state_dict,
                'last_epoch': cur_epoch
            }

            torch.save(model_info, modal_name)#+args.mask[-18:-4]
            # for debug

            if (cur_epoch+1)%25==0:
                torch.save(model_info, modal_name[:-8]+"_ep_"+str(cur_epoch+1)+".pth.tar")
                scipy.io.savemat(modal_name[:-8]+"_ep_"+str(cur_epoch+1)+".mat", {'rcp1': rcp_batch_variable[0].detach().view(-1,2,args.image_size_nx,args.image_size_ny,args.image_size_nz).cpu().numpy().transpose((4,3,2,1,0)),
                                                 'rcp2': rcp_batch_variable[0].detach().view(-1,2,args.image_size_nx,args.image_size_ny,args.image_size_nz).cpu().numpy().transpose((4,3,2,1,0)),                                              
                                                 'rcp3': rcp_batch_variable[0].detach().view(-1,2,args.image_size_nx,args.image_size_ny,args.image_size_nz).cpu().numpy().transpose((4,3,2,1,0)),
                                                 'rcp4':rcp_batch_variable[0].detach().view(-1,2,args.image_size_nx,args.image_size_ny,args.image_size_nz).cpu().numpy().transpose((4,3,2,1,0)),
                                                 'rcp5':rcp_batch_variable[0].detach().view(-1,2,args.image_size_nx,args.image_size_ny,args.image_size_nz).cpu().numpy().transpose((4,3,2,1,0)),
                                                 'rcp6': rcp_batch_variable[0].detach().view(-1,2,args.image_size_nx,args.image_size_ny,args.image_size_nz).cpu().numpy().transpose((4,3,2,1,0)),
#                                                     'rcp7': rcp_batch_variable[6].detach().view(-1,args.image_size_nt,args.image_size_nx,args.image_size_ny).cpu().numpy(),
#                                                     'rcp8': rcp_batch_variable[7].detach().view(-1,args.image_size_nt,args.image_size_nx,args.image_size_ny).cpu().numpy(),                                              
#                                                     'rcp9': rcp_batch_variable[8].detach().view(-1,args.image_size_nt,args.image_size_nx,args.image_size_ny).cpu().numpy(),
#                                                     'rcp10':rcp_batch_variable[9].detach().view(-1,args.image_size_nt,args.image_size_nx,args.image_size_ny).cpu().numpy(),
#                                                     'rcp11':rcp_batch_variable[10].detach().view(-1,args.image_size_nt,args.image_size_nx,args.image_size_ny).cpu().numpy(),
#                                                     'rcp12': rcp_batch_variable[11].detach().view(-1,args.image_size_nt,args.image_size_nx,args.image_size_ny).cpu().numpy(),                                                  
#                                                 'trec': rcp_batch_variable[6].detach().view(-1,args.image_size_nt,args.image_size_nx,args.image_size_ny).cpu().numpy(),                                                 
                                                 'mvrc0': mvrc0_batch_variable.detach().view(-1,2,args.image_size_nx,args.image_size_ny,args.image_size_nz).cpu().numpy().transpose((4,3,2,1,0)),
                                                 'mvgd': mv_batch_variable.detach().view(-1,2,args.image_size_nx,args.image_size_ny,args.image_size_nz).cpu().numpy().transpose((4,3,2,1,0)),
                                                 'rho1': net.invblks[0].M.rho,'rho2': net.invblks[0].M.rho,'rho3': net.invblks[0].M.rho,'rho4': net.invblks[0].M.rho,'rho5': net.invblks[0].M.rho,'rho6': net.invblks[0].M.rho})#enddef 
                
#        print('====> Epoch: {}, data_part: {}, iter: {}/{}, loss: {:.10f}, rho1: {}, rho3: {}, rho5: {}, rho6: {}'.format(
#            cur_epoch+1, datapart+1, i, N, loss_value/args.batch_size,net.invblks[0].M.rho, net.invblks[2].M.rho, net.invblks[4].M.rho, net.invblks[5].M.rho))      
#        if i % 10 == 0 or i == N-1:
#            if args.n_GPU > 1:
#                cur_state_dict = net.module.state_dict()
#            else:
#                cur_state_dict = net.state_dict()            
#            
#            modal_name = output_name
#            
#            model_info = {
#                'network_feature' : args.features,
#                'state_dict': cur_state_dict,
#                'last_epoch': cur_epoch
#            }
#
#            torch.save(model_info, modal_name)#+args.mask[-18:-4]
#            # for debug
#
#            if (cur_epoch+1)%50==0:
#                torch.save(model_info, modal_name[:-8]+"_ep_"+str(cur_epoch+1)+".pth.tar")
#                scipy.io.savemat(modal_name[:-8]+"_ep_"+str(cur_epoch+1)+".mat", {'rcp1': rcp_batch_variable[0].detach().view(-1,2,args.image_size_nx,args.image_size_ny,args.image_size_nz).cpu().numpy().transpose((4,3,2,1,0)),
#                                                 'rcp2': rcp_batch_variable[1].detach().view(-1,2,args.image_size_nx,args.image_size_ny,args.image_size_nz).cpu().numpy().transpose((4,3,2,1,0)),                                              
#                                                 'rcp3': rcp_batch_variable[2].detach().view(-1,2,args.image_size_nx,args.image_size_ny,args.image_size_nz).cpu().numpy().transpose((4,3,2,1,0)),
#                                                 'rcp4':rcp_batch_variable[3].detach().view(-1,2,args.image_size_nx,args.image_size_ny,args.image_size_nz).cpu().numpy().transpose((4,3,2,1,0)),
#                                                 'rcp5':rcp_batch_variable[4].detach().view(-1,2,args.image_size_nx,args.image_size_ny,args.image_size_nz).cpu().numpy().transpose((4,3,2,1,0)),
#                                                 'rcp6': rcp_batch_variable[5].detach().view(-1,2,args.image_size_nx,args.image_size_ny,args.image_size_nz).cpu().numpy().transpose((4,3,2,1,0)),
##                                                     'rcp7': rcp_batch_variable[6].detach().view(-1,args.image_size_nt,args.image_size_nx,args.image_size_ny).cpu().numpy(),
##                                                     'rcp8': rcp_batch_variable[7].detach().view(-1,args.image_size_nt,args.image_size_nx,args.image_size_ny).cpu().numpy(),                                              
##                                                     'rcp9': rcp_batch_variable[8].detach().view(-1,args.image_size_nt,args.image_size_nx,args.image_size_ny).cpu().numpy(),
##                                                     'rcp10':rcp_batch_variable[9].detach().view(-1,args.image_size_nt,args.image_size_nx,args.image_size_ny).cpu().numpy(),
##                                                     'rcp11':rcp_batch_variable[10].detach().view(-1,args.image_size_nt,args.image_size_nx,args.image_size_ny).cpu().numpy(),
##                                                     'rcp12': rcp_batch_variable[11].detach().view(-1,args.image_size_nt,args.image_size_nx,args.image_size_ny).cpu().numpy(),                                                  
##                                                 'trec': rcp_batch_variable[6].detach().view(-1,args.image_size_nt,args.image_size_nx,args.image_size_ny).cpu().numpy(),                                                 
#                                                 'mvrc0': mvrc0_batch_variable.detach().view(-1,2,args.image_size_nx,args.image_size_ny,args.image_size_nz).cpu().numpy().transpose((4,3,2,1,0)),
#                                                 'mvgd': mv_batch_variable.detach().view(-1,2,args.image_size_nx,args.image_size_ny,args.image_size_nz).cpu().numpy().transpose((4,3,2,1,0)),
#                                                 'rho1': net.invblks[0].M.rho,'rho2': net.invblks[1].M.rho,'rho3': net.invblks[2].M.rho,'rho4': net.invblks[3].M.rho,'rho5': net.invblks[4].M.rho,'rho6': net.invblks[5].M.rho})#enddef    



def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def train_network(args):
    net = create_net(args)
    net.train()
    criterion = LossFunc()#.cuda()
    optimizer = optim.Adam(net.parameters(), args.learning_rate)
    if (args.continue_from_parameter != None):
        print('Continue training from last train!')
        config = torch.load(args.continue_from_parameter)
        last_epoch = config['last_epoch']
    else:
        last_epoch = -1
    for cur_epoch in range(last_epoch+1, args.epochs) :

        if cur_epoch == 300:
            lr = 8e-5
            adjust_learning_rate(optimizer, lr)
        if cur_epoch == 500:
            lr = 7.5e-5
            adjust_learning_rate(optimizer, lr)            
        if cur_epoch == 700:
            lr = 3e-5
            adjust_learning_rate(optimizer, lr)            
        if cur_epoch == 900:
            lr = 1e-5         
            adjust_learning_rate(optimizer, lr)        
        for datapart in range(0, len(args.images_dataset)) :
            train_cur_data(
                cur_epoch, 
                datapart,
                args.images_dataset[datapart], 
                args.measurements_dataset[datapart], 
                args.output_name,
                net, 
                criterion, 
                optimizer,
                args
            )
            gc.collect()
            


if __name__ == '__main__':
    check_args(args)
    train_network(args)
