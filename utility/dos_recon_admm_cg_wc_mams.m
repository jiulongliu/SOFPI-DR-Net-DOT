function [X]=dos_recon_admm_cg_wc_mams(x0,f0,para)



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
N_iter=para.N_iter;
cg_iter=para.cg_iter;
d_xs=zeros(size(Wxs(X)));
v_xs=d_xs;
CG_tol0=para.CG_tol;
Min_iter=para.Min_iter;
f=f0;
Xt=X;

n_Ind=2;ind=0;Ind=zeros(n_Ind,1);
n_Ind2=1;ind2=0;Ind2=zeros(n_Ind2,1);
d=zeros(N_iter,1);

for n_iter=1:N_iter
%     tic;
    Xi=Xt;

    g=AmX'*f+mu*Wtxs(d_xs+v_xs);
    
    [X,cg_err,cg_n]=conj_grad(AmX, g, mu,Xt,cg_iter,CG_tol0);
    X=max(min(X,0.5),0.0);
    
    disp([num2str(n_iter) ' -- CG: N=' num2str(cg_n) ' error=' num2str(cg_err)]);
%     X=abs(X);
    temp_xs=Wxs(X)-v_xs;

    d_xs=Sxs(temp_xs,lambda,mu);

    v_xs=d_xs-temp_xs;

    Xt=X;

 
%     if (cg_err>5)
%         break;
%     end
    d(n_iter)=sum(abs(Xi-Xt))/sum(abs(Xt));
%     if(n_iter>Min_iter&&(d(n_iter)>d(n_iter-1)))
%         ind=ind+1;Ind(ind)=n_iter;
%         disp(['Ind ' num2str(ind) '=' num2str(n_iter)]);
%     end
%     if(n_iter>Min_iter&&(d(n_iter)>d(n_iter-1)&&d(n_iter-1)>d(n_iter-2)))
%         ind2=ind2+1;Ind2(ind2)=n_iter;
%         disp(['Ind2 ' num2str(ind2) '=' num2str(n_iter)]);
%     end

%     if(n_iter>Min_iter&&(d(n_iter)<ip.threshold||ind>=n_Ind||ind2>=n_Ind2))
% %         save(['D:\documents\works\results_i_' num2str(i) '_j_' num2str(j) '_loop_' num2str(n_iter) '_final'],'X');
%         break
%     end


%     toc
end





function res = Wxs(image)
image=reshape(image,[13,12,9*2]);

% [sx,sy] = size(image);

Dx = image([2:end,end],:,:) - image;
Dy = image(:,[2:end,end],:) - image;
Dz = image(:,:,[2:end,end]) - image;
%res = [sum(image(:))/sqrt(sx*sy); Dx(:);  Dy(:)]; 
res = cat(4,Dx,Dy,Dz);
end




function res = Wtxs(y)

% res = zeros(size(y,1),size(y,2),size(y,3));

%y1 = ones(imsize)*y(1)/sqrt(prod(imsize));
%yx = (reshape(y(2:prod(imsize)+1), imsize(1), imsize(2)));
%yy = (reshape(y(prod(imsize)+2:end), imsize(1), imsize(2)));

res = adjDx(y(:,:,:,1)) + adjDy(y(:,:,:,2))+ adjDz(y(:,:,:,3));
res=res(:);
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


function d=Sxs(Bub,lambda,mu)
    
%     s = sqrt(Bub(1:n).^2 + Bub(n+1:end).^2);
%     s = sqrt(Bub(:,:,1).^2 + Bub(:,:,2).^2);
    s = sqrt(Bub(:,:,:,1).*Bub(:,:,:,1) + Bub(:,:,:,2).*Bub(:,:,:,2) + Bub(:,:,:,3).*Bub(:,:,:,3))+eps;% !!!!!!!!!!!!!!!!!!!!!eps!!!!!!!!!!!!!!!
%     d = [max(s-mu/lambda,0).*Bub(1:n)./s ; max(s-mu/lambda,0).*Bub(n+1:end)./s ];
    d = cat(4,max(s-mu/lambda,0).*Bub(:,:,:,1)./s , max(s-mu/lambda,0).*Bub(:,:,:,2)./s, max(s-mu/lambda,0).*Bub(:,:,:,3)./s );
end


function [x,cg_err,cg_n] = conj_grad(A, b, mu,x,maxit,CG_tol0)

    r = b - A'*(A * x)-mu*Wtxs(Wxs(x));
    p = r;
    rsold = r(:)' * r(:);
    CG_tol=rsold*CG_tol0;
    for i = 1:maxit
%         Ap = A * p;

        Ap=A'*(A * p)+mu*Wtxs(Wxs(p));
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