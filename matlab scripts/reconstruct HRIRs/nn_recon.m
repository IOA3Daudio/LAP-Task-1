function [sofa_out] = nn_recon(sofa_path,hrir_pred,pos_126)
sofa_in = SOFAload(sofa_path);
pos = sofa_in.SourcePosition;
ori_hrirs = sofa_in.Data.IR;
ori_fs = sofa_in.Data.SamplingRate;

pos(:,1) = mod(pos(:,1) + 180,360) - 180;

svm_ind = [];
for j = 1:126
    temp_dis = sum(abs(pos(:,1:2) - pos_126(j,:)),2);
    [mindis,minind] = min(temp_dis);
    if mindis > 1
        disp(pos_126(j,:));
    end
    svm_ind = [svm_ind;minind];
end

HRIR_L = size(ori_hrirs,3);

recon_hrirs = ori_hrirs;

for i = 1:length(svm_ind)
    for j = 1:2
        temp_hrir_recon = squeeze(hrir_pred(i,j,:));
        
        temp_ori_hrir = squeeze(ori_hrirs(svm_ind(i),j,:));
        
        recon_svm_hrirs(i,j,:) = temp_hrir_recon;
    end
    % actual ITD estimation and reconstruction
    temp_ori_itd = itdestimator(ori_hrirs(svm_ind(i),:,:), 'MaxIACCe', 'fs',ori_fs);
    temp_ori_itd = round(temp_ori_itd*ori_fs);
    
    recon_svm_hrirs(i,:,:) = modify_itd(squeeze(recon_svm_hrirs(i,:,:)),temp_ori_itd,ori_fs);
end

recon_hrirs(svm_ind,:,:) = recon_svm_hrirs;

front_pos_dis = sum(abs(pos(:,1:2)),2);
[~,front_pos_ind] = min(front_pos_dis);
recon_hrirs(front_pos_ind,:,:) = recon_hrirs(front_pos_ind,:,:)*max(abs(ori_hrirs(front_pos_ind,:,:)),[],'all')...
    /max(abs(recon_hrirs(front_pos_ind,:,:)),[],'all');

sofa_out = sofa_in;

sofa_out.Data.IR = recon_hrirs;

end


