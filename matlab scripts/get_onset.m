function onset_sample = get_onset(HRIR)
% Onsets obtained here are only used to guide the model training. They are not the harmonized ITDs and not used to reconstruct the final binaural HRIRs.
onset_sample = AKonsetDetect(HRIR, 10, -20, 'rel', [3000 44100]);
onset_sample = floor(onset_sample);
end
