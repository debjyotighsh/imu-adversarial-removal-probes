# Held-out discriminator evaluation for HHAR's central condition (the paper's flagship
# result), requested by review: the discriminator is normally scored on TRAINING accuracy
# while the probe is scored held-out, an apples-to-oranges comparison. This reruns the
# exact same HHAR sweep (same npz, same held-out device/activity, same dlm5, same 8-point
# lambda grid, same 5 seeds as the merged central table) with --disc_heldout added, which
# additionally scores the trained discriminator HEAD ITSELF on the same held-out cells the
# probe uses. Separate output dir; does not touch any existing checkpoint.
Set-Location $PSScriptRoot
$out = './gait_experiment_output_v3_disc_heldout'
$hhar = "gait_experiment_output_v2\windows_hz50_win2.56_ov0.5_bike-stairsdown-stairsup-walk_ex-s3mini_2.npz"
$c = @('--held_out_device','s3mini','--held_out_activity','walk','--disc_lr_mult','5',
       '--lambda_sweep','0','0.02','0.05','0.1','0.15','0.25','0.5','1',
       '--seeds','0','1','2','3','4','--epochs','25','--pooling','stats','--mode','adversarial',
       '--nonlinear_probe','--knn_probe','--test_cell_probe','--disc_heldout',
       '--probe_fit_cap','10000','--out_dir',$out,'--experiment','sweep','--npz',$hhar)

"[{0:HH:mm:ss}] start hhar_disc_heldout" -f (Get-Date) | Out-File disc_heldout_progress.txt -Append -Encoding utf8
python gait_dual_adversarial_v2.py @c *> disc_heldout.log
"[{0:HH:mm:ss}] DISC HELDOUT DONE" -f (Get-Date) | Out-File disc_heldout_progress.txt -Append -Encoding utf8
