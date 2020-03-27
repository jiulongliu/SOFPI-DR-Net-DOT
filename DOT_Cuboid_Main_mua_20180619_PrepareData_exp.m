close all
clear all
addpath('utility/');

%% 1. Set parameters
% 1.1. physical
%wave = 850;
% n_tissue = 1.325; %water at 850nm.
n_tissue = 1.350; %water at 850nm.
C0 = 0.6; % solution concentration in percent
mua0 = 0.04; % per 1%
mus0 = 10; % per 1%
mua = mua0; % Background absorption coefficient in cm-1; background absorption is mainly from water.
musr = mus0 * C0;
D0 = 1/3/(mua+musr); % Background diffusion coefficient in cm-1

pl = (-0.010:0.005:0.030).'; %Laplace parameter; divide by light velocity in cm/s.


%% 3. Import data
MainFolder = 'data_exp/';
DataFolder = [MainFolder, '20180619/'];

DelRow = 1; % delete 1st row.

SubFolder = 'NoPhantom/';
iData = 102:111;
for i = 1:length(iData)
    DataFile = ['d',int2str(iData(i)),'.txt'];
    Input = load([DataFolder, SubFolder, DataFile]);
    I0(i,:,:) = Input;
end
I0(:, 4:4:end, :) = [];
%I0(I0<0) = 0;close all

I0m = squeeze(mean(I0,1));
% figure; imagesc(I0m); colorbar;
% figure; plot(I0m.')

I0ch1 = I0m(1:3:end, :);
I0ch2 = I0m(2:3:end, :);
I0ch3 = I0m(3:3:end, :);
% figure; plot(I0ch1.');
% figure; plot(I0ch2.');
% figure; plot(I0ch3.');

r0 = 13;
c0 = 13;

I0ch1p = reshape(sum(I0ch1,2),[c0,r0]);
I0ch2p = reshape(sum(I0ch2,2),[c0,r0]);
I0ch3p = reshape(sum(I0ch3,2),[c0,r0]);
% figure; imagesc(I0ch1p.'); colorbar; colormap jet; title('ch1');
% figure; imagesc(I0ch2p.'); colorbar; colormap jet; title('ch2');
% figure; imagesc(I0ch3p.'); colorbar; colormap jet; title('ch3');

idxS = 7; idxE = 30;

I0c = I0m(3*c0*DelRow+1:end, idxS:idxE);
% I0c = I0m(:, idxS:idxE);
for j=1:5
    if j==1
        SubFolder = 'Phantom5mm/';
        iD=[112:116];
        iDataS = ListB(iD);%{iD(1),iD(2),iD(3),iD(4),iD(1:2),iD(2:3),iD(3:4),iD([1 3]),iD([2 4]),iD([3 4]),iD([2 4]),iD([1 2]),iD(2:4),iD([1 3 4]),iD([1 3 5]),iD([2 4 1]),iD([3 4 2]),iD([4 2 3]),iD,iD([1 4]),iD([1 3 4]),iD(5),iD(4:5)};
    elseif j==2
        SubFolder = 'Phantom15mm/';
        iD=[117:124];
        iDataS = ListB(iD);%{iD(1),iD(2),iD(3),iD(4),iD(5),iD(6),iD(7),iD(8),iD(1:2),iD(2:3),iD(3:4),iD(4:5),iD(5:6),iD(6:7),iD(7:8),iD(1:3),iD(2:4),iD(3:5),iD(4:6),iD(5:7),iD(6:8),iD([1 3]),iD([2 4]),iD([1 3]),iD([2 4]),iD([3 5]),iD([4 6]),iD([5 7]),iD([6 8]),iD([1 3 5]),iD([2 4 6]),iD([3 5 7]),iD([4 6 8]),iD([5 7 8]),iD([5 6 8]),iD,iD([2 5]),iD([3 6]),iD([1 5 6]),iD([2 5 6])};  
    elseif j==3
        SubFolder = 'Phantom25mm/';

        iD=[125:129];
        iDataS = ListB(iD);%{iD(1),iD(2),iD(3),iD(4),iD(5),iD(1:2),iD(2:3),iD(3:4),iD(4:5),iD([1 3]),iD([2 4]),iD([3 5]),iD([2 5]),iD([1 5]),iD(2:4),iD(3:5),iD([1 3 5]),iD([2 4 1]),iD([3 5 2]),iD([4 2 3]),iD,iD([1 4]),iD([1 3 4])}
     elseif j==4
        SubFolder = 'Phantom35mm/';

        iD=[130:134];
        iDataS = ListB(iD);       
     elseif j==5
        SubFolder = 'Phantom45mm/';

        iD=[135:139];
        iDataS = ListB(iD);       
    end

    % SubFolder = 'Phantom35mm\';
    % iData = 130:134;

    % SubFolder = 'Phantom45mm\';
    % iData = 135:139;
    
    for ii=1:length(iDataS)
        I1=[];
        iData=iDataS{ii}
        for i = 1:length(iData)
            DataFile = ['d',int2str(iData(i)),'.txt'];
            Input = load([DataFolder, SubFolder, DataFile]);
            I1(i,:,:) = Input;
        end
        I1(:, 4:4:end, :) = [];
        %I1(I1<0) = 0;
        I1m = squeeze(mean(I1,1));
        % figure; imagesc(I1m); colorbar;
        % figure; plot(I1m.')

        I1c = I1m(3*c0*DelRow+1:end, idxS:idxE);
        % I1c = I1m(:, idxS:idxE);
        % figure; plot(I1c.');

        % 3.1. Laplace Transform
        p00 = pl*30/n_tissue;
        Np = length(p00);
        N = idxE-idxS+1;
        dt = 0.4;
        t0 = (0:N-1)*dt;
        Ns = size(I0c,1);
        LapI0 = zeros(Ns*Np, 1);
        LapI1 = zeros(Ns*Np, 1);
        for i1 = 1:Np
            p0 = p00(i1);
            for i2 = 1:size(I0c,1)
                i = (i1-1)*Ns + i2;
                LapI0(i) = sum(I0c(i2,:).*exp(-p0*t0))*dt;
                LapI1(i) = sum(I1c(i2,:).*exp(-p0*t0))*dt;
            end
        end
        if 0
            figure; plot(t0,I1c.');
            xlabel('Time (ns)'); ylabel('Intensity (a.u.)'); title('TPSF');
            axis([0 8 0 0.16]);
        end

        % figure; plot(LapI1);
        % idx = find(pl==0);
        % I3d = reshape(LapI1((idx-1)*Ns+(1:3:Ns)), [c0,r0-DelRow]);
        % figure; imagesc(Sx,Sy,squeeze(I3d).'/max(max(I3d))); colorbar; colormap jet;
        % xlabel('x (cm)'); ylabel('y (cm)'); title('channel 1');
        % I3d = reshape(LapI1((idx-1)*Ns+(2:3:Ns)), [c0,r0-DelRow]);
        % figure; imagesc(Sx,Sy,squeeze(I3d).'/max(max(I3d))); colorbar; colormap jet;
        % xlabel('x (cm)'); ylabel('y (cm)'); title('channel 2');
        % I3d = reshape(LapI1((idx-1)*Ns+(3:3:Ns)), [c0,r0-DelRow]);
        % figure; imagesc(Sx,Sy,squeeze(I3d).'/max(max(I3d))); colorbar; colormap jet;
        % xlabel('x (cm)'); ylabel('y (cm)'); title('channel 3');

        % LapI0(LapI0<0) = 0;
        % LapI1(LapI1<0) = 0;


        %%
        % 1.2. geometric param of cuboid
        L0 = [17.4, 14.7, 6.3]; %thickness in cm.

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



        %% 3.1. Laplace Transform
        p00 = pl*30/n_tissue;
        Np = length(p00);
        N = idxE-idxS+1;
        dt = 0.4;
        t0 = (0:N-1)*dt;
        Ns = size(I0c,1);
        LapI0 = zeros(Ns*Np, 1);
        LapI1 = zeros(Ns*Np, 1);
        for i1 = 1:Np
            p0 = p00(i1);
            for i2 = 1:size(I0c,1)
                i = (i1-1)*Ns + i2;
                LapI0(i) = sum(I0c(i2,:).*exp(-p0*t0))*dt;
                LapI1(i) = sum(I1c(i2,:).*exp(-p0*t0))*dt;
            end
        end

        % figure; plot(LapI1);
        idx = find(pl==0);
        I3d = reshape(LapI1((idx-1)*Ns+(1:3:Ns)), [Nvx,Nvy]);
%         figure; imagesc(Vx,Vy,squeeze(I3d).'/max(max(I3d))); colorbar; colormap jet;
%         xlabel('x (cm)'); ylabel('y (cm)'); title('channel 1');
%         I3d = reshape(LapI1((idx-1)*Ns+(2:3:Ns)), [Nvx,Nvy]);
%         figure; imagesc(Vx,Vy,squeeze(I3d).'/max(max(I3d))); colorbar; colormap jet;
%         xlabel('x (cm)'); ylabel('y (cm)'); title('channel 2');
%         I3d = reshape(LapI1((idx-1)*Ns+(3:3:Ns)), [Nvx,Nvy]);
%         figure; imagesc(Vx,Vy,squeeze(I3d).'/max(max(I3d))); colorbar; colormap jet;
%         xlabel('x (cm)'); ylabel('y (cm)'); title('channel 3');


        idx = find(pl==0);
        I3d = reshape(LapI1((idx-1)*Ns+(1:3:Ns)), [Nvx,Nvy]);
        Cmin = 0.13; Cmax = 0.43;
%         figure; imagesc(Vx,Vy,squeeze(I3d).'/max(I3d(:))); colorbar; colormap jet; %caxis([Cmin, Cmax]);
%         xlabel('x (cm)'); ylabel('y (cm)'); title('channel 1');
%         I3d = reshape(LapI1((idx-1)*Ns+(2:3:Ns)), [Nvx,Nvy]);
%         figure; imagesc(Vx,Vy,squeeze(I3d).'/max(I3d(:))); colorbar; colormap jet; %caxis([Cmin, Cmax]);
%         xlabel('x (cm)'); ylabel('y (cm)'); title('channel 2');
%         I3d = reshape(LapI1((idx-1)*Ns+(3:3:Ns)), [Nvx,Nvy]);
%         figure; imagesc(Vx,Vy,squeeze(I3d).'/max(I3d(:))); colorbar; colormap jet; %caxis([Cmin, Cmax]);
%         xlabel('x (cm)'); ylabel('y (cm)'); title('channel 3');


        % pl = (-0.010:0.005:0.010).';
        % p00 = pl*30/n_tissue;
        % Np = length(p00);
        % for i1 = 1:Np
        %     p0 = p00(i1);
        %     figure; plot(t0,exp(-p0*t0));
        % end

        LapI0(LapI0<0) = 0;
        LapI1(LapI1<0) = 0;

        Y0 = log(LapI1./LapI0);
        save(['data_exp/Ydata_2_' SubFolder(1:end-1) '_' int2str(ii) '.mat'],'Y0','I1c','I0c')
    end
end



% LapI0ch1 = LapI0(3:3:end);
% figure; imagesc(reshape(LapI0ch1,[12,8])); colorbar;
% LapI1ch1 = LapI1(3:3:end);
% figure; imagesc(reshape(LapI1ch1,[12,8])); colorbar;
% 
% figure; imagesc(reshape(LapI1ch1./LapI0ch1,[12,8])); colorbar;
% 
% temp = LapI1ch1./LapI0ch1;
% temp(temp>1) = 0;
% figure; imagesc(reshape(temp,[12,8])); colorbar;

% for i1 = 1:Np
%     p0 = p00(i1);
%     figure; plot(exp(-p0*t0));
% end




