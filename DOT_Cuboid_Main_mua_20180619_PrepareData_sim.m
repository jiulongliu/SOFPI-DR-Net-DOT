close all
clear all

addpath('utility/');
load('J.mat');
AmX=JM;
Nxy=13*12;
Nvz=9;
for iz = 1:Nvz
    idxz = (iz-1)*Nxy + (1:Nxy); %mua
    AmX(:,idxz) = AmX(:,idxz) * Ma(iz);

  idxz = Nxy*Nvz + (iz-1)*Nxy + (1:Nxy); %mus
    AmX(:,idxz) = AmX(:,idxz) * Ms(iz);
end
JM=AmX;

MainFolder = 'data_sim/';
ns=13*12*9*3+12*12*9*3+13*11*9*3+12*11*9*3+4*9;
imgds=zeros(13,12,9,2,ns,'single');
Ys=zeros(4212,ns,'single');
sigma=0.01;
i=0;
for j=1:9
    for ix=1:13
        for iy=1:12
            for ne=1:3
                i=i+1;
                imgds(ix,iy,j,1,i)=0.4+0.2*randn(1);
                imgds(ix,iy,j,2,i)=0.2+0.1*randn(1);
                Ys(:,i)=JM*([reshape(imgds(:,:,:,1,i),13*12*9,1)+sigma*randn(13*12*9,1);reshape(imgds(:,:,:,2,i),13*12*9,1)+sigma*0.5*randn(13*12*9,1)]);   
                if ix<13
                    i=i+1;
                    imgds(ix,iy,j,1,i)=0.4+0.2*randn(1);
                    imgds(ix+1,iy,j,1,i)=0.4+0.2*randn(1);
                    imgds(ix,iy,j,2,i)=0.2+0.1*randn(1);
                    imgds(ix+1,iy,j,2,i)=0.2+0.1*randn(1);                    
                    Ys(:,i)=JM*([reshape(imgds(:,:,:,1,i),13*12*9,1)+sigma*randn(13*12*9,1);reshape(imgds(:,:,:,2,i),13*12*9,1)+sigma*0.5*randn(13*12*9,1)]); 
                end
                 if iy<12
                    i=i+1;
                    imgds(ix,iy,j,1,i)=0.4+0.2*randn(1);
                    imgds(ix,iy+1,j,1,i)=0.4+0.2*randn(1);
                    imgds(ix,iy,j,2,i)=0.2+0.1*randn(1);
                    imgds(ix,iy+1,j,2,i)=0.2+0.2*randn(1);                    
                    Ys(:,i)=JM*([reshape(imgds(:,:,:,1,i),13*12*9,1)+sigma*randn(13*12*9,1);reshape(imgds(:,:,:,2,i),13*12*9,1)+sigma*0.5*randn(13*12*9,1)]); 
                end           
                 if ix<13&&iy<12
                    i=i+1;
                    imgds(ix,iy,j,1,i)=0.4+0.2*randn(1);
                    imgds(ix+1,iy,j,1,i)=0.4+0.2*randn(1);
                    imgds(ix,iy+1,j,1,i)=0.4+0.2*randn(1);
                    imgds(ix+1,iy+1,j,1,i)=0.4+0.2*randn(1);
                    imgds(ix,iy,j,2,i)=0.2+0.1*randn(1);
                    imgds(ix+1,iy,j,2,i)=0.2+0.1*randn(1);
                    imgds(ix,iy+1,j,2,i)=0.2+0.1*randn(1);
                    imgds(ix+1,iy+1,j,2,i)=0.2+0.1*randn(1);                    
                    Ys(:,i)=JM*([reshape(imgds(:,:,:,1,i),13*12*9,1)+sigma*randn(13*12*9,1);reshape(imgds(:,:,:,2,i),13*12*9,1)+sigma*0.5*randn(13*12*9,1)]);  
                end              
            


            end
        end
    end
    
      

end
for j=1:9
    
        ix=7;
        iy=5;
        i=i+1;
        imgds(ix,iy,j,1,i)=0.4+0.2*randn(1);
        imgds(ix,iy,j,2,i)=0.2+0.1*randn(1);
        Ys(:,i)=JM*([reshape(imgds(:,:,:,1,i),13*12*9,1)+sigma*randn(13*12*9,1);reshape(imgds(:,:,:,2,i),13*12*9,1)+sigma*0.5*randn(13*12*9,1)]); 
        if ix<13
            i=i+1;
            imgds(ix,iy,j,1,i)=0.4+0.2*randn(1);
            imgds(ix+1,iy,j,1,i)=0.4+0.2*randn(1);
            imgds(ix,iy,j,2,i)=0.2+0.1*randn(1);
            imgds(ix+1,iy,j,2,i)=0.2+0.1*randn(1);                    
            Ys(:,i)=JM*([reshape(imgds(:,:,:,1,i),13*12*9,1)+sigma*randn(13*12*9,1);reshape(imgds(:,:,:,2,i),13*12*9,1)+sigma*0.5*randn(13*12*9,1)]); 
        end
         if iy<12
            i=i+1;
            imgds(ix,iy,j,1,i)=0.4+0.2*randn(1);
            imgds(ix,iy+1,j,1,i)=0.4+0.2*randn(1);
            imgds(ix,iy,j,2,i)=0.2+0.1*randn(1);
            imgds(ix,iy+1,j,2,i)=0.2+0.2*randn(1);                    
            Ys(:,i)=JM*([reshape(imgds(:,:,:,1,i),13*12*9,1)+sigma*randn(13*12*9,1);reshape(imgds(:,:,:,2,i),13*12*9,1)+sigma*0.5*randn(13*12*9,1)]); 
        end           
         if ix<13&&iy<12
            i=i+1;
            imgds(ix,iy,j,1,i)=0.4+0.2*randn(1);
            imgds(ix+1,iy,j,1,i)=0.4+0.2*randn(1);
            imgds(ix,iy+1,j,1,i)=0.4+0.2*randn(1);
            imgds(ix+1,iy+1,j,1,i)=0.4+0.2*randn(1);
            imgds(ix,iy,j,2,i)=0.2+0.1*randn(1);
            imgds(ix+1,iy,j,2,i)=0.2+0.1*randn(1);
            imgds(ix,iy+1,j,2,i)=0.2+0.1*randn(1);
            imgds(ix+1,iy+1,j,2,i)=0.2+0.1*randn(1);                    
            Ys(:,i)=JM*([reshape(imgds(:,:,:,1,i),13*12*9,1)+sigma*randn(13*12*9,1);reshape(imgds(:,:,:,2,i),13*12*9,1)+sigma*0.5*randn(13*12*9,1)]);         
        end                  

    
end

save('data_sim/Ydata_sim_all.mat','Ys','imgds');



