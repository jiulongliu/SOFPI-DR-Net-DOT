close all
clear all
addpath('utility/');



n_tissue = 1.350; %water at 850nm.
C0 = 0.6; % solution concentration in percent
mua0 = 0.04; % per 1%
mus0 = 10; % per 1%
mua = mua0; % Background absorption coefficient in cm-1; background absorption is mainly from water.
musr = mus0 * C0;
D0 = 1/3/(mua+musr); % Background diffusion coefficient in cm-1

pl = (-0.010:0.005:0.030).'; %Laplace parameter; divide by light velocity in cm/s.
p00 = pl*30/n_tissue;
Np = length(p00);

L0 = [17.4, 14.7, 6.3]; %thickness in cm.

r0 = 13;
c0 = 13;
DelRow = 1; % delete 1st row.

% 1.3. geometric param of source & detector
Sxyd = [c0, r0-DelRow, 0.5]; % [x,y,dx&dy], number of source locations and scan step during the 2D scanning process
Sxy0 = [5.8, 2.1] + [0, DelRow*Sxyd(3)]; % starting point
Sx = Sxy0(1) + (0:1:(Sxyd(1)-1))*Sxyd(3);
Sy = Sxy0(2) + (0:1:(Sxyd(2)-1))*Sxyd(3);

Nsx = length(Sx);
Nsy = length(Sy);
Nsxy = Nsx*Nsy;

d_shift = [0,0; 0,2; -2,1].'; % relative detector locations; currently three detector channels
% d_shift = [1,-1; 1,1; -1,0].';
if max(abs(d_shift(:))) > 1
    StartPoint = 'CH1'; % or Ch2 or Ch3
else
    StartPoint = 'Central'; 
end

Nd = size(d_shift, 2); %number of detectors.

% 1.4. geometric param of image reconstruction
dv = [1, 1, 1]*0.5; % [dx,dy,dz], voxel size in cm
dNN = 0;
V00 = Sxy0 - [dv(1), dv(2)]*dNN; % shift starting point to expand reconstruction area
VI = Sxyd + [dNN*2, dNN*2, 0]; % expand reconstruction area by padding dNN
Vx = V00(1) + (0:1:(VI(1)-1))*dv(1);%+ dv(1)/2
Vy = V00(2) + (0:1:(VI(2)-1))*dv(2);% + dv(2)/2
Vz1 = 1.0; Vz2 = 5.0;
Vz = Vz1:dv(3):Vz2;

Nvx = length(Vx);
Nvy = length(Vy);
Nvz = length(Vz);
Nv = Nvx*Nvy*Nvz;
Nxy = Nvx*Nvy;

%% 4. Reconstruct image by using CS-L1
load('J.mat');
% JM = cat(2,JaM,JsM);
JM=JM;%JaM;
% JM = JaM;
% load('data_exp/Ydata_n_Phantom5mm_13.mat');
% idx0 = (log(sum(I1c,2)./sum(I0c,2)) > 0);
% idx1 = repmat(idx0, Np, 1);
% 
% Y0 = log(LapI1./LapI0);
% 
% % Y0(idx1) = [];
% % JM(idx1,:) = []; 
% 
% % JM = JM(468*5+1:end, :);
% % Y0 = Y0(468*5+1:end);
% 
% Y1 = (JM.')*Y0;
% MM = (JM.')*JM;
% Delta = 0.01*max(max(MM));
% 
% % X0 = (MM + Delta*diag(ones(1,Nv)))\Y1;
% 
% % X0 = (MM + Delta*diag(ones(1,Nv*2)))^(-1)*Y1;
% X0 = (MM + Delta*diag(ones(1,Nv*2)))^(-1)*Y1;
% X00=X0;
% % X0=[X0;X0];
% 
% for iz = 1:Nvz
% 	idxz = (iz-1)*Nxy + (1:Nxy); %mua
% 	X0(idxz) = X0(idxz) / Ma(iz);
%     
%     idxz = Nxy*Nvz + (iz-1)*Nxy + (1:Nxy); %mus
% 	X0(idxz) = X0(idxz) / Ms(iz);
% end
% 
% 
% Iat = X0(1:Nxy*Nvz);
% I3d = reshape(Iat, [Nvx,Nvy,Nvz]);
% figure; imagesc(Vx,Vy,squeeze(sum(I3d,3)).');
% colorbar; colormap jet;
% xlabel('x (cm)'); ylabel('y (cm)'); title('mua');
% figure; imagesc(Vx,Vz,squeeze(sum(I3d,2)).');
% colorbar; colormap jet;
% xlabel('x (cm)'); ylabel('z (cm)'); title('mua');
% figure; imagesc(Vy,Vz,squeeze(sum(I3d,1)).');
% colorbar; colormap jet;
% xlabel('y (cm)'); ylabel('z (cm'); title('mua');
% % figure; imshow(reshape(Iat, [Nvx,Nvy*Nvz]),[]);
% % figure; imshow(reshape(X00, [Nvx,Nvy*Nvz]),[]);
% I2 = squeeze(I3d(:,:,5)).';
% [rr,cc] = size(I2);
% I2(I2<0) = 0;
% I2 = padarray(I2,[rr,cc],0,'both');
% fI2 = fftshift(fft2(ifftshift(I2)));
% fI3 = padarray(fI2,2*size(fI2),0,'both') * (2*2+1)^2;
% I3 = real(fftshift(ifft2(ifftshift(fI3))));
% I3(I3<0) = 0;
% I3 = I3(rr*5+1:rr*10, cc*5+1:cc*10);
% figure; imagesc(Vx,Vy,I3);
% colormap jet; xlabel('x (cm)'); ylabel('y (cm)');
% 
% I2 = squeeze(I3d(:,4,:)).';
% [rr,cc] = size(I2);
% I2(I2<0) = 0;
% I2 = padarray(I2,[rr,cc],0,'both');
% fI2 = fftshift(fft2(ifftshift(I2)));
% fI3 = padarray(fI2,2*size(fI2),0,'both') * (2*2+1)^2;
% I3 = real(fftshift(ifft2(ifftshift(fI3))));
% I3(I3<0) = 0;
% I3 = I3(rr*5+1:rr*10, cc*5+1:cc*10);
% figure; imagesc(Vx,Vz,I3);
% colormap jet; xlabel('x (cm)'); ylabel('z (cm)');
% 
% I2 = squeeze(I3d(7,:,:)).';
% [rr,cc] = size(I2);
% I2(I2<0) = 0;
% I2 = padarray(I2,[rr,cc],0,'both');
% fI2 = fftshift(fft2(ifftshift(I2)));
% fI3 = padarray(fI2,2*size(fI2),0,'both') * (2*2+1)^2;
% I3 = real(fftshift(ifft2(ifftshift(fI3))));
% I3(I3<0) = 0;
% I3 = I3(rr*5+1:rr*10, cc*5+1:cc*10);
% figure; imagesc(Vz,Vy,I3.');
% colormap jet; xlabel('z (cm)'); ylabel('y (cm');




%%%%%L1 recon

mu =[0.8 1.0]; 	% Weight for TV penalty
lambda=[mu/0.08 mu/(0.10)];

% Number of iterations
N_iter = 20;	
cg_iter=100;
CG_tol=1e-8;
Min_iter=5;

nx=13;
ny=12;
nz=9;




param.nx=nx;

param.ny=ny;

param.nz=nz;
% initialize Parameters for reconstruction
% param = init;
param.A = JM;
% param.A=eye(3969);
% param.TV = TVOP;
param.mu =mu;     % TV penalty 
param.lambda =lambda; 
param.N_iter = N_iter;
param.cg_iter = cg_iter;
param.CG_tol=CG_tol;
param.Min_iter=Min_iter;
param.Ma=Ma;
param.Ms=Ms;
noiselev=0.05;
x0 = zeros(nx*ny*nz,1);
errs=[];
load('data_sim/Ydata_sim_all.mat');%,'Ys','imgds'

[~,lds]=size(Ys);
% imgs=zeros([nx ny nz 2 lds]);
% imgrs=zeros([nx ny nz 2 lds]);
imgtvrs=zeros([nx ny nz lds]);
% imgsd=zeros([lds 2 nx ny nz],'single');
imgtvrsd=zeros([  nx ny nz 2 lds],'single');
Ysd=Ys;%zeros([lds 4212],'single');
imgsd=imgds;

i=0;
for j=1:lds%lds-35


        y=Ysd(:,j);

        tic
%         param.A = JaM;
        imgtvr = dos_recon_admm_cg_wc_tv_mams(zeros(13*12*9*2,1),y,param);
        disp(['TV recon error:' num2str(norm(param.A*imgtvr(:)-y,2))]);
        for iz = 1:Nvz
            idxz = (iz-1)*Nxy + (1:Nxy); %mua
            imgtvr(idxz) = imgtvr(idxz) / Ma(iz);

            idxz = Nxy*Nvz + (iz-1)*Nxy + (1:Nxy); %mus
        	imgtvr(idxz) = imgtvr(idxz) / Ms(iz);
        end
        
        disp('tv cg computation time');
        toc;
        
        
        imgtvr=reshape(imgtvr,[nx,ny,nz,2]);
        imgtvrs(:,:,:,j)=imgtvr(:,:,:,1);
%         imgtvrsd(j,1,:,:,:)=single(imgtvr(:,:,:,1));
%         imgtvrsd(j,2,:,:,:)=single(imgtvr(:,:,:,2));
        imgtvrsd(:,:,:,:,j)=single(imgtvr);
        
        
        

%         imgtvr=imgtvr(:,:,:,1);
%         imgtvrn=[imgtvr(:,:,1)  imgtvr(:,:,2) imgtvr(:,:,3)  imgtvr(:,:,4)  imgtvr(:,:,5) imgtvr(:,:,6)  imgtvr(:,:,7) imgtvr(:,:,8)  imgtvr(:,:,9)];
%         imwrite([imgtvrn],['data_sim/imgr_ms_' num2str(i) '.png']);%img;
%         imgtvr=reshape(imgtvr,nx,ny*nz);
%         img=reshape(img(1+nx*ny*nz:2*nx*ny*nz),nx,ny*nz);
%         imshow([img imgtvr],[0 1]);





%         figure;
    if mod(j,1)==10
            imgr=reshape(imgtvr(:,:,:,1),nx,ny,nz);
            imgd=reshape(imgsd(j,:,:,:,1),nx,ny,nz);
            lt=0;
            ht=0.4;
            figure;
            subplot(2,9,1)
            imshow(imgr(:,:,1),[lt ht]);
            subplot(2,9,2)
            imshow(imgr(:,:,2),[lt ht]);
            subplot(2,9,3)
            imshow(imgr(:,:,3),[lt ht]);
            subplot(2,9,4)
            imshow(imgr(:,:,4),[lt ht]);
            subplot(2,9,5)
            imshow(imgr(:,:,5),[lt ht]);
            subplot(2,9,6)
            imshow(imgr(:,:,6),[lt ht]);
            subplot(2,9,7)
            imshow(imgr(:,:,7),[lt ht]);
            subplot(2,9,8)
            imshow(imgr(:,:,8),[lt ht]);
            subplot(2,9,9)
            imshow(imgr(:,:,9),[lt ht]);

            subplot(2,9,10)
            imshow(imgd(:,:,1),[lt ht]);
            subplot(2,9,11)
            imshow(imgd(:,:,2),[lt ht]);
            subplot(2,9,12)
            imshow(imgd(:,:,3),[lt ht]);
            subplot(2,9,13)
            imshow(imgd(:,:,4),[lt ht]);
            subplot(2,9,14)
            imshow(imgd(:,:,5),[lt ht]);
            subplot(2,9,15)
            imshow(imgd(:,:,6),[lt ht]);
            subplot(2,9,16)
            imshow(imgd(:,:,7),[lt ht]);
            subplot(2,9,17)
            imshow(imgd(:,:,8),[lt ht]);
            subplot(2,9,18)
            imshow(imgd(:,:,9),[lt ht]);        



            set(gcf,'color','white','paperpositionmode','auto', 'position',[400 300 600 400]);
            saveas(gcf,['data_sim/imgrs_mams_plt_'  num2str(j)],'png');
%             save(['data_sim/imgrs_mams_plt_' num2str(5*(2*li-1)) 'mm_' num2str(ti) '.mat'] ,'imgr','imgd')
            close all
    end
    
    
    
    

end
% errs
% save('dosdata_g.mat','imgs','imgrs','imgtvrs')
% save('dosdata_g_rs_wc_2p.mat','imgs','imgtvrs')
% imgst=single(imgst);
% imgtvrst=single(imgtvrst);
testsub=lds-35:lds;
imgst=imgsd(:,:,:,:,testsub);
imgsd(:,:,:,:,testsub)=[];
imgtvrst=imgtvrsd(:,:,:,:,testsub);
imgtvrsd(:,:,:,:,testsub)=[];
Yst=Ysd(:,testsub);
Ysd(:,testsub)=[];
save('data_sim/dosdata_g_rs_wc_sim_mams_2_dataset.mat','imgsd','imgtvrsd','Ysd');
save('data_sim/dosdata_g_rs_wc_sim_mams_2_testset.mat','imgst','imgtvrst','Yst');
% save('errs.mat','errs')

figure;imshow([reshape(imgsd(:,:,:,1,1),nx,ny*nz);reshape(imgtvrsd(:,:,:,1,1),nx,ny*nz)],[0 0.4])
X0=imgtvrs(:,:,:,1);
% for iz = 1:Nvz
% 	idxz = (iz-1)*Nxy + (1:Nxy); %mua
% 	X0(idxz) = X0(idxz) / Ma(iz);
%     
% %     idxz = Nxy*Nvz + (iz-1)*Nxy + (1:Nxy); %mus
% % 	X0(idxz) = X0(idxz) / Ms(iz);
% end


Iat = X0(1:Nxy*Nvz);
I3d = reshape(Iat, [Nvx,Nvy,Nvz]);
figure; imagesc(Vx,Vy,squeeze(sum(I3d,3)).');
colorbar; colormap jet;
xlabel('x (cm)'); ylabel('y (cm)'); title('mua');
figure; imagesc(Vx,Vz,squeeze(sum(I3d,2)).');
colorbar; colormap jet;
xlabel('x (cm)'); ylabel('z (cm)'); title('mua');
figure; imagesc(Vy,Vz,squeeze(sum(I3d,1)).');
colorbar; colormap jet;
xlabel('y (cm)'); ylabel('z (cm'); title('mua');
% figure; imshow(reshape(Iat, [Nvx,Nvy*Nvz]),[]);

% I2 = squeeze(I3d(:,:,5)).';
% [rr,cc] = size(I2);
% I2(I2<0) = 0;
% I2 = padarray(I2,[rr,cc],0,'both');
% fI2 = fftshift(fft2(ifftshift(I2)));
% fI3 = padarray(fI2,2*size(fI2),0,'both') * (2*2+1)^2;
% I3 = real(fftshift(ifft2(ifftshift(fI3))));
% I3(I3<0) = 0;
% I3 = I3(rr*5+1:rr*10, cc*5+1:cc*10);
% figure; imagesc(Vx,Vy,I3);
% colormap jet; xlabel('x (cm)'); ylabel('y (cm)');
% 
% I2 = squeeze(I3d(:,4,:)).';
% [rr,cc] = size(I2);
% I2(I2<0) = 0;
% I2 = padarray(I2,[rr,cc],0,'both');
% fI2 = fftshift(fft2(ifftshift(I2)));
% fI3 = padarray(fI2,2*size(fI2),0,'both') * (2*2+1)^2;
% I3 = real(fftshift(ifft2(ifftshift(fI3))));
% I3(I3<0) = 0;
% I3 = I3(rr*5+1:rr*10, cc*5+1:cc*10);
% figure; imagesc(Vx,Vz,I3);
% colormap jet; xlabel('x (cm)'); ylabel('z (cm)');
% 
% I2 = squeeze(I3d(7,:,:)).';
% [rr,cc] = size(I2);
% I2(I2<0) = 0;
% I2 = padarray(I2,[rr,cc],0,'both');
% fI2 = fftshift(fft2(ifftshift(I2)));
% fI3 = padarray(fI2,2*size(fI2),0,'both') * (2*2+1)^2;
% I3 = real(fftshift(ifft2(ifftshift(fI3))));
% I3(I3<0) = 0;
% I3 = I3(rr*5+1:rr*10, cc*5+1:cc*10);
% figure; imagesc(Vz,Vy,I3.');
% colormap jet; xlabel('z (cm)'); ylabel('y (cm');



% imshow([reshape(data_test(:,:,:,1),nx,ny*nz);reshape(datar_test(:,:,:,1),nx,ny*nz);reshape(recon_imgs(:,:,:,1),nx,ny*nz);reshape(output_imgs(:,:,:,1),nx,ny*nz)],[0 1])


imgst2=squeeze(imgst(:,:,:,1,3));
imgtv2=squeeze(imgtvrst(:,:,:,1,3));


% fig = figure;
% nImages=9;
% for idx = 1:nImages
%  
% %     imshow(imgst2(:,:,idx));
% %     imshow(imgtv2(:,:,idx),[0 0.08]);  colormap(mycmap); %colorbar;
%     imagesc(imgtv2(:,:,idx));  colormap(mycmap); %colorbar;
%     drawnow
%     frame = getframe(fig);
%     im{idx} = frame2im(frame);
% end



 

%  imwrite(im{1},'imgtv2.tif')
% for i=2:9
% imwrite(im{i},'imgtv2.tif','WriteMode','append')
% end
%     
 
%   figure;imagesc(imgst2(:,:,3));colormap jet; colorbar;
%  
%  imagesc(imgtv2(:,:,1));  colorbar;colormap(mycmap);
% figure;imshow(im{2},[])