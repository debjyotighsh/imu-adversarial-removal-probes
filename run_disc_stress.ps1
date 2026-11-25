# Disambiguates the "5x faster, 4x wider, or both" robustness check in Section 4.4
# (sec:dynamics): tests the discriminator at settings BEYOND the paper's standard
# disc_lr_mult=5, disc_hidden=128 -- specifically 25x the backbone learning rate
# (5x the standard discriminator rate) and/or 512 hidden units (4x the standard width).
Set-Location $PSScriptRoot
$hhar = "gait_experiment_output_v2\windows_hz50_win2.56_ov0.5_bike-stairsdown-stairsup-walk_ex-s3mini_2.npz"
$common = @('--npz',$hhar,'--held_out_device','s3mini','--held_out_activity','walk',
            '--lambda_sweep','1','--seeds','0','1','2','--epochs','25','--pooling','stats',
            '--mode','adversarial','--test_cell_probe','--probe_fit_cap','10000','--experiment','sweep')

"[{0:HH:mm:ss}] start disc_stress" -f (Get-Date) | Out-File disc_stress_progress.txt -Append -Encoding utf8
python gait_dual_adversarial_v2.py @common --disc_lr_mult 25 --disc_hidden 128 --out_dir ./gait_experiment_output_v3_stress_faster *> disc_stress_faster.log
"[{0:HH:mm:ss}] faster done" -f (Get-Date) | Out-File disc_stress_progress.txt -Append -Encoding utf8
python gait_dual_adversarial_v2.py @common --disc_lr_mult 5 --disc_hidden 512 --out_dir ./gait_experiment_output_v3_stress_wider *> disc_stress_wider.log
"[{0:HH:mm:ss}] wider done" -f (Get-Date) | Out-File disc_stress_progress.txt -Append -Encoding utf8
python gait_dual_adversarial_v2.py @common --disc_lr_mult 25 --disc_hidden 512 --out_dir ./gait_experiment_output_v3_stress_both *> disc_stress_both.log
"[{0:HH:mm:ss}] DISC STRESS DONE" -f (Get-Date) | Out-File disc_stress_progress.txt -Append -Encoding utf8
