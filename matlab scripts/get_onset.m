function onset_sample = get_onset(HRIR)
onset_sample = AKonsetDetect(HRIR, 10, -20, 'rel', [3000 44100]);
onset_sample = floor(onset_sample);
end