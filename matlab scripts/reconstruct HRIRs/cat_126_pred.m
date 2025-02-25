clc
clear all
close all

hrir_pred = zeros(126*80,2,256);

for i = 1:126
    filename = ['./hrir_pred_ojsp/hrir_pred_',num2str(i-1),'.mat'];
    t_hrir_pred = load(filename);
    t_hrir_pred = t_hrir_pred.hrir_pred;
    
    hrir_pred(i:126:end,:,:) = t_hrir_pred;
    
end

save('hrir_pred.mat','hrir_pred','-v6');


