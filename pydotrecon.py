import numpy as np
#import scipy.io as scio
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
#import copy
#__all__ = [
#    "Wxs",
#    "adjDx",
#    "adjDy",
#    "Wtxs",
#    "Sxs",
#    "fft2c",
#    "ifft2c",
#    "Amx",
#    "Atmx",
#    "conj_grad",
#    "genkspacdata",
#    "ctrecon_admm"
#]







class dotoplib():
    def __init__(self, JacMat = sio.loadmat('utility/J.mat')['JM'], Maix = sio.loadmat('utility/J.mat')['Maix'],Msix = sio.loadmat('utility/J.mat')['Msix'],nx=9,ny=12,nz=13,rho_upperbound=4.0,mu=0.6,rho=0.0001,lam=0.08,N_iter=20,CG_iter=20,CG_tol=1e-8,Min_iter=10):    
#        self.nx=np.uint32(256)
#        self.ny=nx
#        
#        self.dx=float(250.0/nx);
#        self.dy=dx;
        self.nx=np.uint32(nx)
        self.ny=np.uint32(ny)        
        self.nz=np.uint32(nz)
        self.nxyz=np.uint32(nx*ny*nz)
        self.JacMat=JacMat#sio.loadmat('utility/J.mat')['JM']
        self.JacMat_torch= torch.from_numpy(JacMat).cuda()
        self.Maix=Maix#sio.loadmat('utility/J.mat')['Maix']#[1:10,:,:]
        self.Msix=Msix#sio.loadmat('utility/J.mat')['Msix']#[1:10,:,:]
        self.Maix_torch= torch.from_numpy(Maix).cuda()
        self.Msix_torch= torch.from_numpy(Msix).cuda()
        self.mu=float(mu)
        self.rho=float(rho)
        self.rho_upperbound = float(rho_upperbound)
        if self.mu>0.000001:
            self.lam=self.mu/float(lam)
        else:
            self.lam=float(0.08)
        self.N_iter=np.uint32(N_iter)
        self.CG_maxit=np.uint32(CG_iter)
        self.CG_tol=float(CG_tol)
        self.Min_iter=np.uint32(Min_iter)


    def Ax(self,im):      
#        JM=sio.loadmat('utility/J.mat')['JM']
        Y = np.dot(self.JacMat, im.flatten())
        return Y

    def Atx(self, Y):

#        JacMat=sio.loadmat('utility/J.mat')['JM']
        aty = np.dot(self.JacMat.transpose(), Y).reshape((2,self.nx,self.ny,self.nz))
 
        return aty#.reshape(nt,self.nx,self.ny)
    def set_rho(self,rho):
        self.rho = rho
        
    def Ax_torch_batch(self,ims):
#        im = torch.torch.from_numpy(im).cuda()
        JacMat_torch_batch = self.JacMat_torch.expand(ims.size(0), -1, -1)
        output = torch.bmm(JacMat_torch_batch, ims.view(-1,2*self.nxyz,1))
        return output
    def Atx_torch_batch(self,Ys):
        JacMat_torch_t_batch = self.JacMat_torch.transpose(1,0).expand(Ys.size(0), -1, -1)
#        print(Ys.shape,JacMat_torch_t_batch.shape)
        output = torch.bmm(JacMat_torch_t_batch, Ys).view((-1,2,self.nx,self.ny,self.nz))
        return output
      
    def Ax_batch(self,ims):
        ims = torch.from_numpy(ims).cuda()#.double()
        output = self.Ax_torch_batch(ims)
        
        output = output.cpu().numpy()
        return output
    def Atx_batch(self,Ys):
        Ys = torch.from_numpy(Ys).cuda()
        output = self.Atx_torch_batch(Ys)
        output = output.cpu().numpy()
        return output
      
        
       
   


    def recon_cg_torch_batch(self, b, Xp):
#        b = torch.from_numpy(b).cuda()
#        rho = 0.4
#    def conj_grad(M,b, mu,rho,x,maxit,CG_tol0):
        x = Xp
        r = b - self.Atx_torch_batch(self.Ax_torch_batch(x))-self.rho*Xp - self.mu*Wtxs3d_torch_batch(Wxs3d_torch_batch(x))
        p = r
        rsold = torch.sum(r*r,(1,2,3,4))
        CG_tol=rsold.mean()*self.CG_tol
        for i in range(self.CG_maxit):
            Ap=self.Atx_torch_batch(self.Ax_torch_batch(p)) +self.rho*p + self.mu*Wtxs3d_torch_batch(Wxs3d_torch_batch(p))
            alpha=rsold/torch.sum(p*Ap,(1,2,3,4))
            x=x+alpha.view(-1,1,1,1,1)*p
            r=r-alpha.view(-1,1,1,1,1)*Ap
            rsnew=torch.sum(r*r,(1,2,3,4))#np.inner(r.flatten(),r.flatten().conj())
#            print(torch.sqrt(rsnew).mean())
            if torch.sqrt(rsnew).mean()<CG_tol:
                break
            p=r+(rsnew/rsold).view(-1,1,1,1,1)*p
            rsold=rsnew      

        cg_err=torch.sqrt(rsnew).mean()
        cg_n=i
        return  x, cg_err, cg_n#reshape(self.N,self.N)#+1.0j*xi#.reshape(nt,self.nv,self.nd)         


    def recon_cg(self, b, Xp):
        b = torch.from_numpy(b).cuda()
        Xp = torch.from_numpy(Xp).cuda()
        x =  self.recon_cg_torch(b, Xp)
        x = x.cpu().numpy()
 

        return  x#reshape(self.N,self.N)#+1.0j*xi#.reshape(nt,self.nv,self.nd)         


    def recon_admm_torch_batch(self, Y, XplusB):
#        assert(X.shape == XplusB.shape )#== Y.shape)

            
        Xi=(1.0/torch.cat((self.Maix_torch, self.Msix_torch),0)*XplusB.view((-1,2*self.nxyz))).view(-1,2,self.nx,self.ny,self.nz)
#        Xi=XplusB
        d_xs=torch.zeros_like(Wxs3d_torch_batch(Xi));
        v_xs=torch.zeros_like(Wxs3d_torch_batch(Xi));
#        print(d_xs.shape,XplusB.shape)
        for n_iter in range(self.N_iter):
            g=self.Atx_torch_batch(Y)+self.mu*Wtxs3d_torch_batch(d_xs+v_xs)+self.rho*XplusB
            
            
            Xi,cg_err,cg_n=self.recon_cg_torch_batch(g, Xi)

            
            if n_iter%5==0:
                print('%d -- CG: N= %d   error= %.12f' %(n_iter,cg_n,cg_err.cpu().numpy()))
            temp_xs=Wxs3d_torch_batch(Xi)-v_xs
            d_xs=Sxs_torch_batch(temp_xs,self.lam,self.mu)
#            d_xs=temp_xs        
            v_xs=d_xs-temp_xs
            
            if cg_err>100000:
                break
            

        Xi=(torch.cat((self.Maix_torch, self.Msix_torch),0)*Xi.view((-1,2*self.nxyz))).view(-1,2,self.nx,self.ny,self.nz)
        return  Xi      

    def genksdata(self, Xg, noiselevel=0.02):
        Xgn = np.zeros((Xg.shape[0], 2, self.nx, self.ny, self.nz))
        Xgn[:,:,:,:,:]=Xg+noiselevel*(np.random.standard_normal((Xg.shape[0], 2, self.nx, self.ny, self.nz)))
        
        Y = self.Ax_batch(Xgn)
        return Y
    def recon_cg_batch(self, Y, Xp):
        Y = torch.from_numpy(Y).cuda()
        Xp = torch.from_numpy(Xp).cuda()
        X = self.recon_cg_torch_batch(Y,Xp)
        X = X.cpu().numpy()        
        return X

    def recon_admm_batch(self, Y, Xp):
        Y = torch.from_numpy(Y).cuda()
        Xp = torch.from_numpy(Xp).cuda()
        X = self.recon_admm_torch_batch(Y,Xp)
        X = X.cpu().numpy()  
        return X


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












