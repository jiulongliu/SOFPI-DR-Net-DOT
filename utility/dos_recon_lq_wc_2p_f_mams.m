function [X]=dos_recon_lq_wc_2p_f_mams(f0,para)

AmX=para.A;
Nxy=para.nx*para.ny;
Nvz=para.nz;
for iz = 1:para.nz
    idxz = (iz-1)*Nxy + (1:Nxy); %mua
    AmX(:,idxz) = AmX(:,idxz) * para.Ma(iz);

  idxz = Nxy*Nvz + (iz-1)*Nxy + (1:Nxy); %mus
    AmX(:,idxz) = AmX(:,idxz) * para.Ms(iz);
end

i=6;
j=5;
k=para.k;

[sx,sy]=size(AmX);
X=zeros(sy,1);
si=(k-1)*para.nx*para.ny+(j-1)*para.nx+i;

sj=(k-1)*para.nx*para.ny+(j-1)*para.nx+i+1;

sis=(k-1)*para.nx*para.ny+(j-1)*para.nx+i+Nxy*Nvz;

sjs=(k-1)*para.nx*para.ny+(j-1)*para.nx+i+1+Nxy*Nvz;
k
si
sj




a=AmX(:,[si,sj,sis,sjs]);
c=a'*f0./diag(a'*a);
X([si,sj,sis,sjs])=c;
% size(AmX)
% size(X)
disp(['G D error:' num2str(norm(AmX*X(:)-f0,2))]);




