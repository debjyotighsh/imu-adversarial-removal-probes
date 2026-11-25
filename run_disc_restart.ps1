# Anti-collapse fix check: rerun the central HHAR condition (Table tab:r1's grid) with
# the discriminator head periodically reinitialised (--disc_restart_every 5, i.e. at
# epochs 5, 10, 15, 20 of the 25-epoch run), a standard fix for a discriminator stuck in
# a degenerate constant-prediction local optimum. Tests whether collapse and probe-level
# recoverability persist under it, matched to the same protocol as Table tab:r1.
Set-Location $PSScriptRoot
$out = './gait_experiment_output_v3_disc_restart'
$hhar = "gait_experiment_output_v2\windows_hz50_win2.56_ov0.5_bike-stairsdown-stairsup-walk_ex-s3mini_2.npz"
$c = @('--npz',$hhar,'--held_out_device','s3mini','--held_out_activity','walk','--disc_lr_mult','5',
       '--lambda_sweep','0','0.02','0.05','0.1','0.15','0.25','0.5','1',
       '--seeds','0','1','2','--epochs','25','--disc_restart_every','5','--pooling','stats',
       '--mode','adversarial','--nonlinear_probe','--knn_probe','--test_cell_probe','--disc_heldout',
       '--probe_fit_cap','10000','--out_dir',$out,'--experiment','sweep')

"[{0:HH:mm:ss}] start hhar_disc_restart" -f (Get-Date) | Out-File disc_restart_progress.txt -Append -Encoding utf8
python gait_dual_adversarial_v2.py @c *> disc_restart.log
"[{0:HH:mm:ss}] DISC RESTART DONE" -f (Get-Date) | Out-File disc_restart_progress.txt -Append -Encoding utf8
