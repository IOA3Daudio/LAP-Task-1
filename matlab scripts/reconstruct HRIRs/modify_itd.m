function [hrirs] = modify_itd(hrirs,itd,fs)
%hrirs:2x256

itd_now = itdestimator(reshape(hrirs,1,2,length(hrirs)),'fs',fs,'MaxIACCe');
itd_now = round(itd_now*fs);

if itd_now > itd
    bias = itd_now - itd;
    hrirs(2,:) = [zeros(1,bias),hrirs(2,1:end-bias)];
elseif itd_now < itd
    bias = itd - itd_now;
    hrirs(1,:) = [zeros(1,bias),hrirs(1,1:end-bias)];
end

itd_after = itdestimator(reshape(hrirs,1,2,length(hrirs)),'fs',fs,'MaxIACCe');
itd_after = round(itd_after*fs);

end

