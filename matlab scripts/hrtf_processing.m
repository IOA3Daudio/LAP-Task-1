function [hrtf] = hrtf_processing(hrir)

temp_hrtf = fft(hrir,length(hrir));
temp_phase = angle(temp_hrtf);
temp_amp = abs(temp_hrtf);
hrtf = temp_amp(1:length(hrir)/2+1);


end