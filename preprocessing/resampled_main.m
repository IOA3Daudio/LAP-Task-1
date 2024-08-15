function [re_hrir_all,pos,re_sofa,hrtf_all,onset_all] = resampled_main(filename)
TARGET_fs = 48000;
HRIR_L = 512;

datatemp = SOFAload(filename);
datatemp = extract_only_farthest_distance(datatemp);

pos = datatemp.SourcePosition;
ori_hrirs = datatemp.Data.IR;
ori_fs = datatemp.Data.SamplingRate;

pos_num = length(pos);

re_hrir_all = zeros(pos_num,2,HRIR_L);
onset_all = zeros(pos_num,2);

for i = 1:pos_num
    for subj_ear = 1:2
        temp_hrir = squeeze(ori_hrirs(i,subj_ear,:));
        temp_hrir = resample(temp_hrir,TARGET_fs,ori_fs);
        if length(temp_hrir) < HRIR_L
            temp_hrir = [temp_hrir;zeros(HRIR_L-length(temp_hrir),1)];
        end
        temp_hrir = temp_hrir(1:HRIR_L);
        
        [peak, idx] = max(abs(temp_hrir));
        onset = find(abs(temp_hrir(1:idx))> peak/10);
        onset = onset(1);
        
        onset_all(i,subj_ear) = onset;
        
        temp_hrir(1:onset) = 0;
        temp_hrir(end-100:end) = 0;
        
        re_hrir_all(i,subj_ear,:) = temp_hrir;
    end
end


% front_pos_dis = sum(abs(pos(:,1:2)),2);
% [~,front_pos_ind] = min(front_pos_dis);
% front_hrirs(front_pos_ind,:,:) = re_hrir_all(front_pos_ind,:,:);
max_v = max(abs(re_hrir_all),[],'all');

for i = 1:pos_num
    for subj_ear = 1:2
        temp_hrir = squeeze(re_hrir_all(i,subj_ear,:));
        temp_hrir = temp_hrir./max_v;
        
        temp_hrtf = hrtf_processing(temp_hrir);
        
        re_hrir_all(i,subj_ear,:) = temp_hrir;
        
        hrtf_all(i,subj_ear,:) = temp_hrtf;
    end
end


re_sofa = datatemp;
re_sofa.Data.SamplingRate = TARGET_fs;
re_sofa.Data.IR = re_hrir_all;
re_sofa.API.N = HRIR_L;

end


function sofa_output = extract_only_farthest_distance(sofa_input)
    sofa_output = sofa_input;

    % test presence multiple distances
    if length(unique(round(sofa_output.SourcePosition(:, 3), 2))) > 1
        % select maximum distance
        far_dist = max(sofa_output.SourcePosition(:, 3));
    
        % remove HRIRs by updating sofa object
        idx_remove = sofa_output.SourcePosition(:, 3) ~= far_dist;
        sofa_output.SourcePosition(idx_remove, :) = [];
        sofa_output.Data.IR(idx_remove, :, :) = []; 
        sofa_output.API.M = size(sofa_output.SourcePosition, 1);
    end
end

