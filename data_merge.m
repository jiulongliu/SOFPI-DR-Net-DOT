if 0    
    clear all
    load('data_sim/dosdata_g_rs_wc_sim_mams_2_dataset.mat')
    imgsd_sim = imgsd;
    imgtvrsd_sim = imgtvrsd;
    Ysd_sim = Ysd;
    load('data_exp/dosdata_g_rs_wc_2p_mams_2_dataset.mat')
    imgsd_exp = imgsd;
    imgtvrsd_exp = imgtvrsd;
    Ysd_exp = Ysd;
    imgsd = cat(5,imgsd_sim, imgsd_exp);
    imgtvrsd = cat(5,imgtvrsd_sim, imgtvrsd_exp);
    Ysd = cat(5,Ysd_sim, Ysd_exp);
    save('dosdata_all_mams_2_dataset.mat','imgsd','imgtvrsd','Ysd');

    clear all
    load('data_sim/dosdata_g_rs_wc_sim_mams_2_testset.mat')
    imgst_sim = imgst;
    imgtvrst_sim = imgtvrst;
    Yst_sim = Yst;
    load('data_exp/dosdata_g_rs_wc_2p_mams_2_testset.mat')
    imgst_exp = imgst;
    imgtvrst_exp = imgtvrst;
    Yst_exp = Yst;
    imgst = cat(5,imgst_sim, imgst_exp);
    imgtvrst = cat(5,imgtvrst_sim, imgtvrst_exp);
    Yst = cat(5,Yst_sim, Yst_exp);

    save('dosdata_all_mams_2_testset.mat','imgst','imgtvrst','Yst');
end
if 0

clear all
    load('data_sim/dosdata_g_rs_wc_sim_mams_2_dataset.mat')
    imgsd_sim = imgsd;
    imgtvrsd_sim = imgtvrsd;
    Ysd_sim = Ysd;
    load('data_exp/dosdata_g_rs_wc_2p_mams_2_dataset.mat')
    testid=(31+255+1):(31+255+31);
    imgst_exp = imgsd(:,:,:,:,testid);
    imgsd(:,:,:,:,testid)=[];
    imgtvrst_exp = imgtvrsd(:,:,:,:,testid);
    imgtvrsd(:,:,:,:,testid)=[];
    Yst_exp = Ysd(:,testid);
    Ysd(:,testid)=[];
    
    imgsd_exp = imgsd;
    imgtvrsd_exp = imgtvrsd;
    Ysd_exp = Ysd;
    imgsd_sim = cat(5,imgsd_sim, imgsd_exp);
    imgtvrsd_sim = cat(5,imgtvrsd_sim, imgtvrsd_exp);
    Ysd_sim = cat(2,Ysd_sim, Ysd_exp);


%     clear all
    load('data_sim/dosdata_g_rs_wc_sim_mams_2_testset.mat')
    imgst_sim = imgst;
    imgtvrst_sim = imgtvrst;
    Yst_sim = Yst;
    load('data_exp/dosdata_g_rs_wc_2p_mams_2_testset.mat')
    imgsd_exp = imgst;
    imgtvrsd_exp = imgtvrst;
    Ysd_exp = Yst;
    
    imgsd = cat(5,imgsd_sim, imgsd_exp);
    imgtvrsd = cat(5,imgtvrsd_sim, imgtvrsd_exp);
    Ysd = cat(2,Ysd_sim, Ysd_exp);
    
    imgst = cat(5,imgst_sim, imgst_exp);
    imgtvrst = cat(5,imgtvrst_sim, imgtvrst_exp);
    Yst = cat(2,Yst_sim, Yst_exp);
    
    save('dosdata_all_mams_2_dataset_mid.mat','imgsd','imgtvrsd','Ysd');
    save('dosdata_all_mams_2_testset_mid.mat','imgst','imgtvrst','Yst');
end

if 1

clear all
    load('data_sim/dosdata_g_rs_wc_sim_mams_2_dataset.mat')
    imgsd_sim = imgsd;
    imgtvrsd_sim = imgtvrsd;
    Ysd_sim = Ysd;
    load('data_exp/dosdata_g_rs_wc_2p_mams_2_dataset.mat')
    testid=(31+255+1):(31+255+31);
    C0=0.6;
    imgst_exp = imgsd(:,:,:,:,testid)*C0;
    imgsd(:,:,:,:,testid)=[];
    imgtvrst_exp = imgtvrsd(:,:,:,:,testid)*C0;
    imgtvrsd(:,:,:,:,testid)=[];
    Yst_exp = Ysd(:,testid)*C0;
    Ysd(:,testid)=[];
    
    imgsd_exp = imgsd*C0;
    imgtvrsd_exp = imgtvrsd*C0;
    Ysd_exp = Ysd*C0;
    imgsd_sim = cat(5,imgsd_sim, imgsd_exp);
    imgtvrsd_sim = cat(5,imgtvrsd_sim, imgtvrsd_exp);
    Ysd_sim = cat(2,Ysd_sim, Ysd_exp);


%     clear all
    load('data_sim/dosdata_g_rs_wc_sim_mams_2_testset.mat')
    imgst_sim = imgst;
    imgtvrst_sim = imgtvrst;
    Yst_sim = Yst;
    load('data_exp/dosdata_g_rs_wc_2p_mams_2_testset.mat')
    imgsd_exp = imgst*0.6;
    imgtvrsd_exp = imgtvrst*0.6;
    Ysd_exp = Yst*0.6;
    
    imgsd = cat(5,imgsd_sim, imgsd_exp);
    imgtvrsd = cat(5,imgtvrsd_sim, imgtvrsd_exp);
    Ysd = cat(2,Ysd_sim, Ysd_exp);
    
    imgst = cat(5,imgst_sim, imgst_exp);
    imgtvrst = cat(5,imgtvrst_sim, imgtvrst_exp);
    Yst = cat(2,Yst_sim, Yst_exp);
    
    save('dosdata_all_mams_2_dataset_mid_rc.mat','imgsd','imgtvrsd','Ysd');
    save('dosdata_all_mams_2_testset_mid_rc.mat','imgst','imgtvrst','Yst');
end