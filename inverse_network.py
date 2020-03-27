
import sys
sys.path.append('./dos_Operators')
import torch
import torch.nn as nn
#import torch.nn.functional as F
#import mrirecon3d
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
#in_channels = 1
#n_features = 8

import scipy.io
#import pydotrecon 
##import pyctrecon




class xnet(nn.Module):
    def __init__(self, in_channels, n_features, use_dropout):
        super(xnet, self).__init__()
        self.in_channels = in_channels
        self.conv_enc_a1 = nn.Conv3d(in_channels, n_features, 3, 1, 1, 1).cuda()
        self.conv_enc_a2 = nn.Conv3d(n_features, n_features*2, 3, 1, 1, 1).cuda() # 8 --> 16
        self.conv_enc_a3 = nn.Conv3d(n_features*2, n_features*2, 3, 1, 1, 1).cuda() # 16 --> 16        
        self.conv_enc_s1 = nn.Conv3d(in_channels, n_features, 3, 1, 1, 1).cuda()
        self.conv_enc_s2 = nn.Conv3d(n_features, n_features*2, 3, 1, 1, 1).cuda() # 8 --> 16
        self.conv_enc_s3 = nn.Conv3d(n_features*2, n_features*2, 3, 1, 1, 1).cuda() # 16 --> 16        
        self.conv_pooling_a = nn.Conv3d(n_features*2, n_features*4, 2, 2, 1, 1).cuda()  # 16 --> 32 
        self.conv_pooling_s = nn.Conv3d(n_features*2, n_features*4, 2, 2, 1, 1).cuda()
        self.conv_ma = nn.Conv3d(n_features*8, n_features*4, 3, 1, 1, 1).cuda() # 64 --> 32    
        self.conv_ms = nn.Conv3d(n_features*8, n_features*4, 3, 1, 1, 1).cuda() # 64 --> 32    
        
        self.conv_unpooling_a = nn.ConvTranspose3d(n_features*4, n_features*2, 2, 2, 1,[1,0,1]).cuda() # 32 --> 16 
        self.conv_unpooling_s = nn.ConvTranspose3d(n_features*4, n_features*2, 2, 2, 1,[1,0,1]).cuda() # 32 --> 16 
        self.conv_dec_a3 = nn.Conv3d(n_features*4, n_features*2, 3, 1, 1, 1).cuda() # 32 --> 16 
        self.conv_dec_a2 = nn.Conv3d(n_features*2, n_features*2, 3, 1, 1, 1).cuda()  
        self.conv_dec_a1 = nn.Conv3d(n_features*2, n_features*1, 3, 1, 1, 1).cuda()          
        self.conv_dec_a0 = nn.Conv3d(n_features, in_channels, 3, 1, 1, 1).cuda() 
        self.conv_dec_s3 = nn.Conv3d(n_features*4, n_features*2, 3, 1, 1, 1).cuda()
        self.conv_dec_s2 = nn.Conv3d(n_features*2, n_features*2, 3, 1, 1, 1).cuda()
        self.conv_dec_s1 = nn.Conv3d(n_features*2, n_features*1, 3, 1, 1, 1).cuda()        
        self.conv_dec_s0 = nn.Conv3d(n_features, in_channels, 3, 1, 1, 1).cuda()           
#        self.conv_inblock2 = nn.Conv3d(output_feature*2, output_feature*4, 3, 1, 1, 1).cuda()

        self.prelu_enc_a1 = nn.PReLU().cuda()
        self.prelu_enc_a2 = nn.PReLU().cuda()
        self.prelu_enc_a3 = nn.PReLU().cuda()
        self.prelu_enc_a4 = nn.PReLU().cuda()
        self.prelu_ma = nn.PReLU().cuda()
        self.prelu_enc_s1 = nn.PReLU().cuda()
        self.prelu_enc_s2 = nn.PReLU().cuda()
        self.prelu_enc_s3 = nn.PReLU().cuda()
        self.prelu_enc_s4 = nn.PReLU().cuda()
        self.prelu_ms = nn.PReLU().cuda() 
        self.prelu_dec_a1 = nn.PReLU().cuda()
        self.prelu_dec_a2 = nn.PReLU().cuda()
        self.prelu_dec_a3 = nn.PReLU().cuda()
        self.prelu_dec_a4 = nn.PReLU().cuda()        
        self.prelu_dec_s1 = nn.PReLU().cuda()
        self.prelu_dec_s2 = nn.PReLU().cuda()
        self.prelu_dec_s3 = nn.PReLU().cuda()
        self.prelu_dec_s4 = nn.PReLU().cuda()
        
        self.use_dropout = use_dropout;
        self.dropout = nn.Dropout(0.2).cuda()      
    def apply_dropout(self, input):
        if self.use_dropout:
            return self.dropout(input)
        else:
            return input;
    def forward(self, x):
#        mua_input = x[0][:,0:1,:,:,:]
#        mus_input = x[0][:,8:16,:,:,:]
#        print(x.shape, self.in_channels)
        [mua, mus] = torch.split(x, self.in_channels, 1)
#        output_a1 = self.conv_enc_a1(mua)
        output_enc_a1 = self.prelu_enc_a1(self.conv_enc_a1(mua)) 
        output_enc_a2 = self.prelu_enc_a2(self.conv_enc_a2(output_enc_a1)) 
        output_enc_a3 = self.prelu_enc_a3(self.conv_enc_a3(output_enc_a2))+output_enc_a2      
        output_enc_a4 = self.prelu_enc_a4(self.conv_pooling_a(output_enc_a3))       
#        output_s1 = self.conv_enc_s1(mus)
#        output_a1 = self.conv_enc_a1(mua)
        output_enc_s1 = self.prelu_enc_s1(self.conv_enc_s1(mus)) 
        output_enc_s2 = self.prelu_enc_s2(self.conv_enc_s2(output_enc_s1)) 
        output_enc_s3 = self.prelu_enc_s3(self.conv_enc_s3(output_enc_s2))+output_enc_s2
        output_enc_s4 = self.prelu_enc_s4(self.conv_pooling_s(output_enc_s3))      
        
        output_comb = torch.cat((output_enc_a4, output_enc_s4), 1)  
        
        output_ma = self.prelu_ma(self.conv_ma(output_comb)) 
        output_ms = self.prelu_ms(self.conv_ms(output_comb))         

        output_dec_a4u = self.prelu_dec_a4(self.conv_unpooling_a(output_ma))    
        output_dec_a4 = torch.cat((output_dec_a4u, output_enc_a3), 1)        
        output_dec_a3 = self.prelu_dec_a3(self.conv_dec_a3(output_dec_a4)) 
        output_dec_a2 = self.prelu_dec_a2(self.conv_dec_a2(output_dec_a3))+output_dec_a3
        output_dec_a1 = self.prelu_dec_a1(self.conv_dec_a1(output_dec_a2))      
        output_dec_a0 = self.conv_dec_a0(output_dec_a1)
         
        output_dec_s4u = self.prelu_dec_s4(self.conv_unpooling_s(output_ms))
        output_dec_s4 = torch.cat((output_dec_s4u, output_enc_s3), 1)
        output_dec_s3 = self.prelu_dec_s3(self.conv_dec_s3(output_dec_s4)) 
        output_dec_s2 = self.prelu_dec_s2(self.conv_dec_s2(output_dec_s3))+output_dec_s3
        output_dec_s1 = self.prelu_dec_s1(self.conv_dec_s1(output_dec_s2))
        output_dec_s0 = self.conv_dec_s0(output_dec_s1)                   
        

        
        mu = torch.cat((output_dec_a0, output_dec_s0), 1)
#        mup1 = torch.cat((mua_p1, mus_p1), 1)

        return mu








class priornet(nn.Module):
    def __init__(self, in_channels, n_features, use_dropout):
        super(priornet, self).__init__()
        self.xnet_ = xnet(in_channels, n_features, 0)
    def forward(self, x):
        m =self.xnet_(x)
        return m         
    






class inv_cg_Function(torch.autograd.Function):

    @staticmethod
    def forward(self, v,Y,rho,M):
        imshape=v.shape
#        rho = 1.0/(1.0+torch.exp(-rho))
        M.set_rho(rho.data.cpu().numpy())
        self.M = M
        self.imshape=imshape
#        print(im.size())
        v_np= v.data.cpu().numpy()#.view(-1,imshape[2],imshape[3]).numpy() #.view(-1,imshape[2],imshape[3])
        Y_np = Y.data.cpu().numpy()
        X_np=self.M.recon_cg_batch(Y_np, v_np)
        X=torch.from_numpy(X_np).cuda() #.view(-1,1,imshape[2],imshape[3])
#
        self.save_for_backward(v,Y,X)
        return X 

    @staticmethod
    def backward(self, grad_output):
        mv0,Y,X, = self.saved_tensors
#        mv= mv0.data.cpu().view(-1,self.imshape[2],self.imshape[3]).numpy()
        grad_input = grad_output.clone()
        grad_input_np = grad_input.data.cpu().numpy()*self.M.rho #view(-1,self.imshape[2],self.imshape[3]).
        grad_v_np=self.M.recon_cg_batch(np.zeros_like(Y.cpu().numpy()), grad_input_np)
        grad_v=torch.from_numpy(grad_v_np).cuda() #.view(-1,1,self.imshape[2],self.imshape[3])
        grad_v = Variable(grad_v).cuda()
#        grad_y = Variable(torch.cat((X/self.M.rho,X/self.M.rho),1)).cuda() # no need to compute gra_out2,  save computation time
#        T = (mv0 - X)*grad_input
        T = (mv0 - X)*grad_v/torch.Tensor(self.M.rho).cuda()
        grad_rho = torch.sum(T, (1, 2, 3))
#        grad_rho = 1.0/(1.0+torch.exp(-grad_rho))*(1.0-1.0/(1.0+torch.exp(-grad_rho)))
        return grad_v, None, grad_rho , None
#        return grad_out














class Ax_cuda_Function(torch.autograd.Function):

    @staticmethod
    def forward(self, X, M):
        self.save_for_backward(X)
        self.M = M
        X = X.data.cpu().numpy() #.view(-1,X.size(1),X.size(2))
        X=self.M.Ax_batch(X)
        Y=torch.from_numpy(X).cuda() #.view(-1,2,self.M.N,self.M.N)
#
        
        return Y

    @staticmethod
    def backward(self, grad_output):
        X, = self.saved_tensors
#        Y= Y.data.cpu().view(-1,2,self.M.N,self.M.N).numpy()
        grad_input = grad_output.clone()
        grad_input = grad_input.data.cpu().numpy() #.view(-1,2,self.M.N,self.M.N)
        aX=self.M.Atx_batch(grad_input)
        aX=torch.from_numpy(aX).cuda() #.view(-1,self.M.N,self.M.N) .view(-1,1,X.size(1),X.size(2))
        grad_out = Variable(aX).cuda()
#        grad_out2 = Variable(torch.cat((X/self.M.rho,X/self.M.rho),1)).cuda()
        return grad_out, None
class Ax_cuda(nn.Module):
    def __init__(self, M):
        super(Ax_cuda, self).__init__()
        self.M = M

    def forward(self, X):
        output = Ax_cuda_Function.apply(X, self.M)
        return output





class Atx_cuda_Function(torch.autograd.Function):

    @staticmethod
    def forward(self, Y, M):
        self.save_for_backward(Y)
        self.M = M  
        Y = Y.data.cpu().numpy()
        X=self.M.Atx_batch(Y)
        X=torch.from_numpy(X).cuda() #.view(-1,1,self.M.N,self.M.N) .view(X.shape[0],1,X.shape[1],X.shape[2])
        
        return X 

    @staticmethod
    def backward(self, grad_output):
        Y, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input = grad_input.data.cpu().numpy() #.view(grad_output.size(0),grad_output.size(1),grad_output.size(2))
        atX=self.M.Ax_batch(grad_input)
        atX=torch.from_numpy(atX).cuda()#.view(-1,2,self.M.N,self.M.N)
        grad_out = Variable(atX).cuda()

        return grad_out, None
    
class Atx_cuda(nn.Module):
    def __init__(self, M):
        super(Atx_cuda, self).__init__()
        self.M = M

    def forward(self, Y):
        output = Atx_cuda_Function.apply(Y, self.M)
        return output




class Ax_torch(nn.Module):
    def __init__(self, JM,nx,ny,nz):
        super(Ax_torch, self).__init__()
        self.JM_torch = JM
        self.nx=nx
        self.ny=ny    
        self.nz=nz
        self.nxyz=nx*ny*nz
    def forward(self, Xs):

        JM_torch_batch = self.JM_torch.expand(Xs.size(0), -1, -1)
        output = torch.bmm(JM_torch_batch, Xs.view(-1,2*self.nxyz,1))
        return output


class Atx_torch(nn.Module):
    def __init__(self, JM,nx,ny,nz):
        super(Atx_torch, self).__init__()
        self.JM_torch = JM
        self.nx=nx
        self.ny=ny    
        self.nz=nz
        self.nxyz=nx*ny*nz


    def forward(self, Ys):
        JM_torch_t_batch = self.JM_torch.transpose(1,0).expand(Ys.size(0), -1, -1)
        output = torch.bmm(JM_torch_t_batch, Ys).view((-1,2,self.nx,self.ny,self.nz)) #self.JacMat_torch.transpose(1,0).expand(Ys.size(0), -1, -1)
        return output








def Wxs3d_torch_batch(im):
    Dx = torch.zeros_like(im)
    Dx[:,:,:-1,:,:] = im[:,:,1:,:,:]-im[:,:,:-1,:,:]
    Dy = torch.zeros_like(im)
    Dy[:,:, :, :-1,:] = im[:,:,:,1:,:]-im[:,:,:,:-1,:]
    Dz = torch.zeros_like(im)
    Dz[:,:, :, :,:-1] = im[:,:,:,:,1:]-im[:,:,:,:,:-1]   
    res=torch.cat((Dx, Dy, Dz), 1)
    return res
    

def adjDx_torch_batch(x):
    Dtx=torch.zeros_like(x)
    Dtx[:,:,1:,:,:]=x[:,:,:-1,:,:] - x[:,:,1:,:,:]
    Dtx[:,:,0,:,:]=-x[:,:,0,:,:]
    Dtx[:,:,-1,:,:]=x[:,:,-2,:,:]
    return Dtx

def adjDy_torch_batch(x):
    Dty=torch.zeros_like(x)
    Dty[:,:,:,1:,:]=x[:,:,:,:-1,:] - x[:,:,:,1:,:]
    Dty[:,:,:,0,:]=-x[:,:,:,0,:]
    Dty[:,:,:,-1,:]=x[:,:,:,-2,:]
    return Dty

def adjDz_torch_batch(x):
    Dtz=torch.zeros_like(x)
    Dtz[:,:,:,:,1:]=x[:,:,:,:,:-1] - x[:,:,:,:,1:]
    Dtz[:,:,:,:,0]=-x[:,:,:,:,0]
    Dtz[:,:,:,:,-1]=x[:,:,:,:,-2]
    return Dtz


def Wtxs3d_torch_batch(y):
#    n=y.shape[1]
    res=adjDx_torch_batch(y[:,0:1,:,:,:]) + adjDy_torch_batch(y[:,1:2,:,:,:]) + adjDz_torch_batch(y[:,2:3,:,:,:])
    return res
def Sxs_torch_batch(Bub,lam,mu):
    res = torch.zeros_like(Bub)
    s=torch.sqrt((Bub*Bub).sum(1, keepdim=True))
    res=torch.max(s-torch.Tensor([mu/lam]).cuda(),torch.Tensor([0.0]).cuda())*Bub/(s+torch.Tensor([np.spacing(1)]).cuda()) 
    return res
   




class inv_cg_torch(nn.Module):
    def __init__(self, in_channels, n_features, M, inversion_regularize_method, rho_learnable, forward_operator_version):
        super(inv_cg_torch, self).__init__()
        self.M = M
        if forward_operator_version == 'torch':
            self.Ax = Ax_torch(M.JacMat_torch, M.nx, M.ny, M.nz)#.cuda())
            self.Atx = Atx_torch(M.JacMat_torch, M.nx, M.ny, M.nz)#.cuda()) 
        elif forward_operator_version == 'cuda':
            self.Ax = Ax_cuda(M)
            self.Atx = Atx_cuda(M)        
        self.sigmoid  = nn.Sigmoid().cuda()
        self.linear = nn.Linear(1, 1, bias=False).cuda()              
        self.rho = Variable(M.rho*torch.ones(1).cuda())
        self.CG_maxit = M.CG_maxit
        self.CG_tol = M.CG_tol
        self.rho_learnable = rho_learnable
        self.inversion_regularize_method = inversion_regularize_method
        self.in_channels = in_channels
        self.n_features = n_features
        self.DnotSharingWeight = True
        if inversion_regularize_method == 'D':  
#            self.conv_1a = nn.Conv3d(in_channels=in_channels, out_channels=n_features*8, kernel_size=5, stride=1, padding=2, dilation=1,bias=False).cuda()
#            self.conv_2a = nn.Conv3d(in_channels=n_features*8, out_channels=n_features*2, kernel_size=5, stride=1, padding=2, dilation=1,bias=False).cuda() #8->16 + n 8-->32 +
#            self.conv_1s = nn.Conv3d(in_channels=in_channels, out_channels=n_features*8, kernel_size=5, stride=1, padding=2, dilation=1,bias=False).cuda()
#            self.conv_2s = nn.Conv3d(in_channels=n_features*8, out_channels=n_features*2, kernel_size=5, stride=1, padding=2, dilation=1,bias=False).cuda() #8->16 + n 8-->32 +    
            self.conv_1a = nn.Conv3d(in_channels=in_channels, out_channels=n_features*1, kernel_size=3, stride=1, padding=1, dilation=1,bias=False).cuda()
            self.conv_2a = nn.Conv3d(in_channels=n_features*1, out_channels=n_features*2, kernel_size=3, stride=1, padding=1, dilation=1,bias=False).cuda() #8->16 + n 8-->32 +
            self.conv_1s = nn.Conv3d(in_channels=in_channels, out_channels=n_features*1, kernel_size=3, stride=1, padding=1, dilation=1,bias=False).cuda()
            self.conv_2s = nn.Conv3d(in_channels=n_features*1, out_channels=n_features*2, kernel_size=3, stride=1, padding=1, dilation=1,bias=False).cuda() #8->16 + n 8-->32 +       
            
            if self.DnotSharingWeight:
                self.conv_1ad = nn.Conv3d(in_channels=n_features, out_channels=in_channels, kernel_size=3, stride=1, padding=1, dilation=1,bias=False).cuda()
                self.conv_2ad = nn.Conv3d(in_channels=n_features*2, out_channels=n_features, kernel_size=3, stride=1, padding=1, dilation=1,bias=False).cuda() #8->16 + n 8-->32 +             
                self.conv_1sd = nn.Conv3d(in_channels=n_features, out_channels=in_channels, kernel_size=3, stride=1, padding=1, dilation=1,bias=False).cuda()
                self.conv_2sd = nn.Conv3d(in_channels=n_features*2, out_channels=n_features, kernel_size=3, stride=1, padding=1, dilation=1,bias=False).cuda() #8->16 + n 8-->32 +                   
                        
                        
#            filter3d =torch.zeros(3,3,3).cuda()
#            filter3d[1,1,1] = torch.tensor([1.0]).cuda() 
#            self.conv_1a.weight.data = filter3d.unsqueeze(0).unsqueeze(0).repeat(n_features, in_channels, 1, 1, 1)/4.0
#            self.conv_2a.weight.data = filter3d.unsqueeze(0).unsqueeze(0).repeat(2*n_features, n_features, 1, 1, 1)/torch.sqrt(torch.tensor([8.0]).cuda())#.cuda()
#            self.conv_1s.weight.data = filter3d.unsqueeze(0).unsqueeze(0).repeat(n_features, in_channels, 1, 1, 1)/4.0
#            self.conv_2s.weight.data = filter3d.unsqueeze(0).unsqueeze(0).repeat(2*n_features, n_features, 1, 1, 1)/torch.sqrt(torch.tensor([8.0]).cuda())#.cuda()           
    def forward(self, x, db, y):
        if self.rho_learnable == True:           
            rho_l = self.M.rho_upperbound*self.sigmoid(self.linear(self.rho))
        else:
            rho_l = self.rho
        self.M.set_rho(rho_l.data.cpu().numpy())
        if self.inversion_regularize_method == 'I':
            b=self.Atx(y)+rho_l*db
            r = b - self.Atx(self.Ax(x))-rho_l*x            
        elif self.inversion_regularize_method == 'W': 
            b=self.Atx(y)+rho_l*Wtxs3d_torch_batch(db)
            r = b - self.Atx(self.Ax(x))-rho_l*Wtxs3d_torch_batch(Wxs3d_torch_batch(x))              
        elif self.inversion_regularize_method == 'D':
            [db1,db2]=torch.split(db, 2*self.n_features, 1)
            if self.DnotSharingWeight:
                cdb1=self.conv_1ad(self.conv_2ad(db1))
                cdb2=self.conv_1sd(self.conv_2sd(db2))             
            else:             
                cdb1=F.conv_transpose3d(F.conv_transpose3d(db1,self.conv_2a.weight,padding=1),self.conv_1a.weight,padding=1)
                cdb2=F.conv_transpose3d(F.conv_transpose3d(db2,self.conv_2s.weight,padding=1),self.conv_1s.weight,padding=1)
            rhocdb = torch.cat((rho_l*cdb1, (rho_l*2.0)*cdb2), 1)
            b=self.Atx(y)+rhocdb 
            [xa,xs]=torch.split(x, self.in_channels, 1)
            if self.DnotSharingWeight:
                cxa=self.conv_1ad(self.conv_2ad(self.conv_2a(self.conv_1a(xa))))
                cxs=self.conv_1sd(self.conv_2sd(self.conv_2s(self.conv_1s(xs))))              
            else:                    
                cxa=F.conv_transpose3d(F.conv_transpose3d(self.conv_2a(self.conv_1a(xa)),self.conv_2a.weight,padding=1),self.conv_1a.weight,padding=1)
                cxs=F.conv_transpose3d(F.conv_transpose3d(self.conv_2s(self.conv_1s(xs)),self.conv_2s.weight,padding=1),self.conv_1s.weight,padding=1)
            rhocx = torch.cat((rho_l*cxa, (rho_l*2.0)*cxs), 1)
            r = b - self.Atx(self.Ax(x))-rhocx

        p = r
        rsold = torch.sum(r*r,(1,2,3,4))
        CG_tol=rsold.mean()*self.CG_tol
        for i in range(self.CG_maxit):
            if self.inversion_regularize_method == 'I': 
                Ap=self.Atx(self.Ax(p)) +rho_l*p             
            elif self.inversion_regularize_method == 'W':  
                Ap=self.Atx(self.Ax(p))  +rho_l*Wtxs3d_torch_batch(Wxs3d_torch_batch(p))             
            elif self.inversion_regularize_method == 'D':  
                [pa,ps]=torch.split(p, self.in_channels, 1)
                if self.DnotSharingWeight:
                    cpa=self.conv_1ad(self.conv_2ad(self.conv_2a(self.conv_1a(pa))))
                    cps=self.conv_1sd(self.conv_2sd(self.conv_2s(self.conv_1s(ps))))              
                else:                
                    cpa=F.conv_transpose3d(F.conv_transpose3d(self.conv_2a(self.conv_1a(pa)),self.conv_2a.weight,padding=1),self.conv_1a.weight,padding=1)
                    cps=F.conv_transpose3d(F.conv_transpose3d(self.conv_2s(self.conv_1s(ps)),self.conv_2s.weight,padding=1),self.conv_1s.weight,padding=1)
                rhocp = torch.cat((rho_l*cpa, (rho_l*2.0)*cps), 1)
                Ap=self.Atx(self.Ax(p))  +rhocp
            alpha=rsold/torch.sum(p*Ap,(1,2,3,4))
            x=x+alpha.view(-1,1,1,1,1)*p
            r=r-alpha.view(-1,1,1,1,1)*Ap
            rsnew=torch.sum(r*r,(1,2,3,4))
            if torch.sqrt(rsnew).mean()<CG_tol:
                break
            p=r+(rsnew/rsold).view(-1,1,1,1,1)*p
            rsold=rsnew        

        if self.inversion_regularize_method == 'D':       
            if self.DnotSharingWeight:
                return x, self.conv_1a, self.conv_2a, self.conv_1s, self.conv_2s, self.conv_1ad, self.conv_2ad, self.conv_1sd, self.conv_2sd
            else :
                return x, self.conv_1a, self.conv_2a, self.conv_1s, self.conv_2s#,cg_err,cg_n [:,:,:,0].view(x.size(0),1,x.size(1),x.size(2))
        
        else:
            return x
    
class sofpi_admm_nets(nn.Module):
    def __init__(self, in_channels, n_features, use_dropout, M, inversion_regularize_method, rho_learnable, inversion_version,forward_operator_version):
        super(sofpi_admm_nets, self).__init__()
#        self.M = M            
        self.nstages = len(M)
        self.inversion_regularize_method = inversion_regularize_method
        self.rho_learnable = rho_learnable
        self.in_channels = in_channels
        self.n_features = n_features
        self.DnotSharingWeight = True
        if self.inversion_regularize_method == 'I': 
            self.priornetblks = nn.ModuleList([priornet(in_channels, n_features, use_dropout) for i in range(0, self.nstages)]) 
        elif self.inversion_regularize_method == 'W': 
            self.priornetblks = nn.ModuleList([priornet(3*in_channels, n_features, use_dropout) for i in range(0, self.nstages)]) 
        elif self.inversion_regularize_method == 'D': 
            self.priornetblks = nn.ModuleList([priornet(2*n_features, n_features, use_dropout) for i in range(0, self.nstages)])             
        if inversion_version == 'torch':
            self.invblks = nn.ModuleList([inv_cg_torch(in_channels, n_features, M[i], inversion_regularize_method, rho_learnable, forward_operator_version) for i in range(0, self.nstages)]) 
        elif inversion_version == 'cuda':
            self.invblks = nn.ModuleList([inv_cg_torch(in_channels, n_features, M[i], inversion_regularize_method, rho_learnable) for i in range(0, self.nstages)]) 
#        self.conv_1ad = nn.Conv3d(in_channels=n_features, out_channels=in_channels, kernel_size=3, stride=1, padding=1, dilation=1,bias=False).cuda()
#        self.conv_2ad = nn.Conv3d(in_channels=n_features*2, out_channels=n_features, kernel_size=3, stride=1, padding=1, dilation=1,bias=False).cuda() #8->16 + n 8-->32 +             
#        self.conv_1sd = nn.Conv3d(in_channels=n_features, out_channels=in_channels, kernel_size=3, stride=1, padding=1, dilation=1,bias=False).cuda()
#        self.conv_2sd = nn.Conv3d(in_channels=n_features*2, out_channels=n_features, kernel_size=3, stride=1, padding=1, dilation=1,bias=False).cuda() #8->16 + n 8-->32 + 
    def forward(self, x):        
        (rc0,Y) = x
        rc0=(1.0/torch.cat((self.invblks[0].M.Maix_torch, self.invblks[0].M.Msix_torch),0)*rc0.view((-1,2*self.invblks[0].M.nxyz))).view(-1,2,self.invblks[0].M.nx,self.invblks[0].M.ny,self.invblks[0].M.nz)
        if self.inversion_regularize_method == 'I': 
            ti = torch.zeros(rc0.size(0), rc0.size(1), rc0.size(2), rc0.size(3), rc0.size(4)).cuda()        
            di = torch.zeros(rc0.size(0), rc0.size(1), rc0.size(2), rc0.size(3), rc0.size(4)).cuda()
        elif self.inversion_regularize_method == 'W': 
            ti = torch.zeros(rc0.size(0), 3*rc0.size(1), rc0.size(2), rc0.size(3), rc0.size(4)).cuda()        
            di = torch.zeros(rc0.size(0), 3*rc0.size(1), rc0.size(2), rc0.size(3), rc0.size(4)).cuda()              
        elif self.inversion_regularize_method == 'D': 
            ti = torch.zeros(rc0.size(0), 4*self.n_features, rc0.size(2), rc0.size(3), rc0.size(4)).cuda()        
            di = torch.zeros(rc0.size(0), 4*self.n_features, rc0.size(2), rc0.size(3), rc0.size(4)).cuda()
        
        ui = rc0
        u = []
        for i in range(0, self.nstages):  
            db =2*di - ti
            if self.inversion_regularize_method == 'D': 
                if self.DnotSharingWeight:
                     (ui,conv_1a,conv_2a,conv_1s,conv_2s,conv_1ad,conv_2ad,conv_1sd,conv_2sd) = self.invblks[i](ui,db,Y)
                else :
                    (ui,conv_1a,conv_2a,conv_1s,conv_2s) = self.invblks[i](ui,db,Y)
            else:
                ui=self.invblks[i](ui,db,Y)
            u.append((torch.cat((self.invblks[0].M.Maix_torch, self.invblks[0].M.Msix_torch),0)*ui.view((-1,2*self.invblks[0].M.nxyz))).view(-1,2,self.invblks[0].M.nx,self.invblks[0].M.ny,self.invblks[0].M.nz))
            if self.inversion_regularize_method == 'I': 
                ti = ti -di + ui
            elif self.inversion_regularize_method == 'W': 
                ti = ti -di +Wxs3d_torch_batch(ui)               
            elif self.inversion_regularize_method == 'D':
                [uia,uis]=torch.split(ui, self.in_channels, 1)
                cuia = conv_2a(conv_1a(uia)) 
                cuis = conv_2s(conv_1s(uis)) 
                cui = torch.cat((cuia, cuis), 1) 
                ti = ti -di +cui               
            di = self.priornetblks[i](ti)
            [dia,dis]=torch.split(di, 2*self.n_features, 1)
            if self.DnotSharingWeight:
                cdia=conv_1ad(conv_2ad(dia))
                cdis=conv_1sd(conv_2sd(dis))
            else :                
                cdia=F.conv_transpose3d(F.conv_transpose3d(dia,conv_2a.weight,padding=1),conv_1a.weight,padding=1)
                cdis=F.conv_transpose3d(F.conv_transpose3d(dis,conv_2s.weight,padding=1),conv_1s.weight,padding=1)
#            cdia = self.conv_1ad(self.conv_2ad(dia))
#            cdis = self.conv_1sd(self.conv_2sd(dis))            
            
            cdi = torch.cat((cdia, cdis), 1)            
            u.append((torch.cat((self.invblks[0].M.Maix_torch, self.invblks[0].M.Msix_torch),0)*cdi.view((-1,2*self.invblks[0].M.nxyz))).view(-1,2,self.invblks[0].M.nx,self.invblks[0].M.ny,self.invblks[0].M.nz))
            
#            d.append(di)
#            if self.inversion_regularize_method == 'I': 
#                cdi = di
#            elif self.inversion_regularize_method == 'W': 
#                cdi = Wxs3d_torch_batch(di)               
#            elif self.inversion_regularize_method == 'D': 
#                cdi = conv_2(conv_1(di))           
        
        return u











