# Rerun of the MotionSense swap experiment (activity target, subject nuisance) with
# --disc_heldout now wired in, so Table 6 (tab:swap) can report the discriminator's
# held-out balanced accuracy alongside the held-out probe, the same fix applied to the
# central HHAR condition (Section 4.1a). Fresh output dir to force a full rerun rather
# than resuming the original checkpoint, which has no disc_heldout fields.
Set-Location $PSScriptRoot
"[{0:HH:mm:ss}] start subject_nuisance_discheldout" -f (Get-Date) | Out-File subject_nuisance_discheldout_progress.txt -Append -Encoding utf8
python subject_nuisance_experiment.py --lambda_sweep 0 0.02 0.05 0.1 0.5 1 2 --seeds 0 1 2 --out_dir ./gait_experiment_output_v3_subject_nuisance_discheldout *> subject_nuisance_discheldout.log
"[{0:HH:mm:ss}] SUBJECT NUISANCE DISCHELDOUT DONE" -f (Get-Date) | Out-File subject_nuisance_discheldout_progress.txt -Append -Encoding utf8
