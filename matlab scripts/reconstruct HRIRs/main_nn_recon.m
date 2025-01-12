clc
clear all
close all
% first sofatools

dataset_path = '../resampled_sofa/';
allDirs = dir(fullfile(dataset_path));

pos_126 = load('pos_126.mat');
pos_126 = pos_126.pos;

load('hrir_pred.mat');

load('../train_data.mat');

for folder_id = 1:8
    temp_folder = allDirs(folder_id+2,1);
    temp_folder_name = temp_folder.name;
    
    fileExt = '*.sofa';
    folder_path = [dataset_path,temp_folder_name,'/'];
    allFiles = dir(fullfile(folder_path,fileExt));
    numofsofa = size(allFiles,1);
    
    hrir_pred_temp = hrir_pred(labels_all == folder_id,:,:);
    
    for i = 1:numofsofa
        sofa_name = allFiles(i,1).name;
        disp(sofa_name);
        sofa_path = strcat(folder_path,sofa_name);
        
        hrir_pred_temp_temp = hrir_pred_temp((i-1)*126+1:i*126,:,:);

        re_sofa = nn_recon(sofa_path,hrir_pred_temp_temp,pos_126);
        
        
        SOFAsave(['../harmonised_sofa/',temp_folder_name,'/',sofa_name],re_sofa);
    end
    
end


