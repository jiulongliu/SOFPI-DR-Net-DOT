function [X]=dos_recon_admm_cg_wc_tv_mams(x0,f0,para)



N=size(x0);
% X=zeros(N);
X=x0;
AmX=para.A;


% Nxy=para.nx*para.ny;
% for iz = 1:para.nz
%     idxz = (iz-1)*Nxy + (1:Nxy); %mua
%     AmX(:,idxz) = AmX(:,idxz) * para.Ma(iz);
% 
% %   idxz = Nxy*Nvz + (iz-1)*Nxy + (1:Nxy); %mus
% % 	X0(idxz) = X0(idxz) / Ms(iz);
% end




lambda=para.lambda;
mu=para.mu;
CG_tol0=para.CG_tol;
cg_iter=para.cg_iter;
N_iter=para.N_iter;
Nxyz=para.nx*para.ny*para.nz;
Ma=para.Ma;
Ms=para.Ms;
d_xs=zeros(size(cat(5,Wxs(X(1:Nxyz),para,Ma),Wxs(X(Nxyz+1:2*Nxyz),para,Ma))));
v_xs=d_xs;

Min_iter=para.Min_iter;
f=f0;
Xt=X;

n_Ind=2;ind=0;Ind=zeros(n_Ind,1);
n_Ind2=1;ind2=0;Ind2=zeros(n_Ind2,1);
d=zeros(N_iter,1);

for n_iter=1:N_iter
%     tic;
    Xi=Xt;

    g=AmX'*f+[mu(1)*Wtxs(d_xs(:,:,:,:,1)+v_xs(:,:,:,:,1),para,Ma); mu(2)*Wtxs(d_xs(:,:,:,:,2)+v_xs(:,:,:,:,2),para,Ms)];
    
    [X,cg_err,cg_n]=conj_grad(AmX, g,Xt,para);
    X=max(min(X,0.6),0.0);
    
%     disp([num2str(n_iter) ' -- CG: N=' num2str(cg_n) ' error=' num2str(cg_err)]);

    temp_xs=cat(5,Wxs(X(1:Nxyz),para,Ma),Wxs(X(Nxyz+1:2*Nxyz),para,Ms))-v_xs;

    d_xs=cat(5,Sxs(temp_xs(:,:,:,:,1),lambda(1),mu(1)),Sxs(temp_xs(:,:,:,:,2),lambda(2),mu(2)));

    v_xs=d_xs-temp_xs;

    Xt=X;

 

    d(n_iter)=sum(abs(Xi-Xt))/sum(abs(Xt));





%     toc
end





function res = Wxs(image,para,Ma)
Nvz=para.nz;
Nxy=para.nx*para.ny;
% Ma=para.Ma;
% Ms=para.Ms;
for iz = 1:Nvz
    idxz = (iz-1)*Nxy + (1:Nxy); %mua
    image(idxz) = image(idxz) / Ma(iz);

%     idxz = Nxy*Nvz + (iz-1)*Nxy + (1:Nxy); %mus
%     image(idxz) = image(idxz) / Ms(iz);
end
image=reshape(image,[para.nx,para.ny,para.nz]);

% [sx,sy] = size(image);

Dx = image([2:end,end],:,:) - image;
Dy = image(:,[2:end,end],:) - image;
Dz = image(:,:,[2:end,end]) - image;
% %res = [sum(image(:))/sqrt(sx*sy); Dx(:);  Dy(:)]; 
% imgs=reshape(image(Nxy*Nvz+1:2*Nxy*Nvz),[para.nx,para.ny,para.nz]);
% 
% % [sx,sy] = size(image);
% 
% Dxs = imgs([2:end,end],:,:) - imgs;
% Dys = imgs(:,[2:end,end],:) - imgs;
% Dzs = imgs(:,:,[2:end,end]) - imgs;

res = cat(4,Dx,Dy,Dz);


end




function res = Wtxs(y,para,Ma)
Nvz=para.nz;
Nxy=para.nx*para.ny;
% Ma=para.Ma;
% Ms=para.Ms;
% res = zeros(size(y,1),size(y,2),size(y,3));

%y1 = ones(imsize)*y(1)/sqrt(prod(imsize));
%yx = (reshape(y(2:prod(imsize)+1), imsize(1), imsize(2)));
%yy = (reshape(y(prod(imsize)+2:end), imsize(1), imsize(2)));

res =adjDx(y(:,:,:,1)) + adjDy(y(:,:,:,2))+ adjDz(y(:,:,:,3));
res=res(:);
for iz = 1:Nvz
    idxz = (iz-1)*Nxy + (1:Nxy); %mua
    res(idxz) = res(idxz) * Ma(iz);

%     idxz = Nxy*Nvz + (iz-1)*Nxy + (1:Nxy); %mus
%     res(idxz) = res(idxz) * Ms(iz);
end
end
function res = adjDz(x)
% x=reshape(x,[21,21,9]);
res = x(:,:,[1,1:end-1]) - x;
res(:,:,1) = -x(:,:,1);
res(:,:,end) = x(:,:,end-1);
end
function res = adjDy(x)
%x=reshape(x,[21,21,9]);
res = x(:,[1,1:end-1],:) - x;
res(:,1,:) = -x(:,1,:);
res(:,end,:) = x(:,end-1,:);
end
function res = adjDx(x)
%x=reshape(x,[21,21,9]);
res = x([1,1:end-1],:,:) - x;
res(1,:,:) = -x(1,:,:);
res(end,:,:) = x(end-1,:,:);


end

% 
% function d=Sxs(Bub,lambda,mu)
%     th1=mu(1)/lambda(1);
%     th2=mu(2)/lambda(2);
% %     s = sqrt(Bub(1:n).^2 + Bub(n+1:end).^2);
% %     s = sqrt(Bub(:,:,1).^2 + Bub(:,:,2).^2);
%     s1 = sqrt(Bub(:,:,:,1).*Bub(:,:,:,1) + Bub(:,:,:,2).*Bub(:,:,:,2) + Bub(:,:,:,3).*Bub(:,:,:,3))+eps;% !!!!!!!!!!!!!!!!!!!!!eps!!!!!!!!!!!!!!!
%     s2 = sqrt(Bub(:,:,:,4).*Bub(:,:,:,4) + Bub(:,:,:,5).*Bub(:,:,:,6) + Bub(:,:,:,6).*Bub(:,:,:,6))+eps;% !!!!!!!!!!!!!!!!!!!!!eps!!!!!!!!!!!!!!!
% 
%     d = cat(4,max(s1-th1,0).*Bub(:,:,:,1)./s1 , max(s1-th1,0).*Bub(:,:,:,2)./s1, max(s1-th1,0).*Bub(:,:,:,3)./s1, max(s2-th2,0).*Bub(:,:,:,4)./s2, max(s2-th2,0).*Bub(:,:,:,5)./s2, max(s2-th2,0).*Bub(:,:,:,6)./s2);
% end

function d=Sxs(Bub,lambda,mu)
    
%     s = sqrt(Bub(1:n).^2 + Bub(n+1:end).^2);
%     s = sqrt(Bub(:,:,1).^2 + Bub(:,:,2).^2);
    s = sqrt(Bub(:,:,:,1).*Bub(:,:,:,1) + Bub(:,:,:,2).*Bub(:,:,:,2) + Bub(:,:,:,3).*Bub(:,:,:,3))+eps;% !!!!!!!!!!!!!!!!!!!!!eps!!!!!!!!!!!!!!!
%     d = [max(s-mu/lambda,0).*Bub(1:n)./s ; max(s-mu/lambda,0).*Bub(n+1:end)./s ];
    d = cat(4,max(s-mu/lambda,0).*Bub(:,:,:,1)./s , max(s-mu/lambda,0).*Bub(:,:,:,2)./s, max(s-mu/lambda,0).*Bub(:,:,:,3)./s );
end

function [x,cg_err,cg_n] = conj_grad(A, b,x,para)
Nxyz=para.nx*para.ny*para.nz;    
lambda=para.lambda;
mu=para.mu;
CG_tol0=para.CG_tol;
maxit=para.cg_iter;
Ma=para.Ma;
Ms=para.Ms;

    r = b - A'*(A * x)-[mu(1)*Wtxs(Wxs(x(1:Nxyz),para,Ma),para,Ma);mu(2)*Wtxs(Wxs(x(Nxyz+1:2*Nxyz),para,Ms),para,Ms)] ;
    p = r;
    rsold = r(:)' * r(:);
    CG_tol=rsold*CG_tol0;
    for i = 1:maxit
%         Ap = A * p;

        Ap=A'*(A * p)+[mu(1)*Wtxs(Wxs(p(1:Nxyz),para,Ma),para,Ma);mu(2)*Wtxs(Wxs(p(Nxyz+1:2*Nxyz),para,Ms),para,Ms)];
        alpha = rsold / (p(:)' * Ap(:));
        x = x + alpha * p;
        r = r - alpha * Ap;
        rsnew = r(:)' * r(:);
        if sqrt(rsnew) < CG_tol
              break;
        end
        p = r + (rsnew / rsold) * p;
        rsold = rsnew;
%         sum(abs(x(:)))
    end
    cg_err=sqrt(rsnew);
    cg_n=i;
end
    
    
end