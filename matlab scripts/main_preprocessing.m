clc
clear all
close all
% first addpath for sofatools

dataset_path = 'D:/matlab_code/240406_LAPchallenge_task1/ori_sofa/'; % sofa files
allDirs = dir(fullfile(dataset_path));

hrtfs_all = [];
hrirs_all = [];
labels_all = [];
onsets_all = [];

pos_126 = load('pos_126.mat'); % index for target 126 positions
pos_126 = pos_126.pos;

for folder_id = 1:8
    temp_folder = allDirs(folder_id+2,1);
    temp_folder_name = temp_folder.name;
    
    fileExt = '*.sofa';
    folder_path = [dataset_path,temp_folder_name,'/'];
    allFiles = dir(fullfile(folder_path,fileExt));
    numofsofa = size(allFiles,1);
    
    hrtfs_temp_dataset = [];
    hrirs_temp_dataset = [];
    onsets_temp_dataset = [];
    
    for i = 1:numofsofa
        sofa_name = allFiles(i,1).name;
        disp(sofa_name);
        sofa_path = strcat(folder_path,sofa_name);
        
        % resample the original HRIRs to harmonized sampling rate
        [hrirs,pos,re_sofa,hrtfs,onsets] = resampled_main(sofa_path);
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
        
        SOFAsave(['./resampled_sofa/',temp_folder_name,'/',sofa_name],re_sofa);
        
        hrtfs_temp_dataset = [hrtfs_temp_dataset;hrtfs(svm_ind,:,:)];
        hrirs_temp_dataset = [hrirs_temp_dataset;hrirs(svm_ind,:,:)];
        onsets_temp_dataset = [onsets_temp_dataset;onsets(svm_ind,:)];
    end
    
    hrtfs_all = [hrtfs_all;hrtfs_temp_dataset];
    hrirs_all = [hrirs_all;hrirs_temp_dataset];
    labels_all = [labels_all;ones(size(hrtfs_temp_dataset,1),1)*folder_id];
    onsets_all = [onsets_all;onsets_temp_dataset];
    
end


save('train_data.mat','hrtfs_all','hrirs_all','labels_all','onsets_all','-v6');

