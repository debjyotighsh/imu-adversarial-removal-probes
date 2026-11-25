# One frozen configuration, every condition rerun end to end.
#   - identical training config, three seeds, three probe families
#   - probes on the training pool AND scored on the held-out evaluation cells (--test_cell_probe)
#   - every probe family fitted on at most 10,000 windows, at every training-set size
#   - the same lambda grid for RealWorld at both training-set sizes
#   - discriminator confusion counts stored for every run
# Output: ./gait_experiment_output_v3_frozen  (logs: frozen_*.log)
Set-Location $PSScriptRoot
$out = './gait_experiment_output_v3_frozen'
$c = @('--seeds','0','1','2','--epochs','25','--pooling','stats','--mode','adversarial',
       '--nonlinear_probe','--knn_probe','--test_cell_probe','--probe_fit_cap','10000',
       '--out_dir',$out,'--experiment','sweep')
$hhar = "gait_experiment_output_v2\windows_hz50_win2.56_ov0.5_bike-stairsdown-stairsup-walk_ex-s3mini_2.npz"
$rw   = "har_cache\realworld_hz50_w2.56_ov0.5_gap1_min4_acts-jog-stairsdown-stairsup-walk.npz"
$rwl  = @('0','0.05','0.1','0.15','0.2','0.3','1','2')

function Step($name, $argsList) {
    $t0 = Get-Date
    "[{0:HH:mm:ss}] start {1}" -f $t0, $name | Out-File frozen_progress.txt -Append -Encoding utf8
    & python gait_dual_adversarial_v2.py @argsList *> "frozen_$name.log"
    "[{0:HH:mm:ss}] done  {1}  ({2:N0} min)" -f (Get-Date), $name, ((Get-Date) - $t0).TotalMinutes | Out-File frozen_progress.txt -Append -Encoding utf8
}

Step 'hhar_dlm5' (@('--npz',$hhar,'--held_out_device','s3mini','--held_out_activity','walk','--disc_lr_mult','5',
                    '--lambda_sweep','0','0.02','0.05','0.1','0.15','0.25','0.5','1') + $c)
Step 'realworld_cap' (@('--npz',$rw,'--held_out_device','waist','--held_out_activity','walk','--disc_lr_mult','5',
                        '--max_train_windows','15315','--lambda_sweep') + $rwl + $c)
Step 'mobiact_B' (@('--npz',"har_cache\mobiact_hz50_w2.56_ov0.5_gap1_min4_acts-stairsdown-stairsup-walk.npz",
                    '--held_out_activity','walk','--disc_lr_mult','5',
                    '--lambda_sweep','0','0.05','0.1','0.25','0.5','1','2') + $c)
Step 'mobiact_A' (@('--npz',"har_cache\mobiact_hz50_w2.56_ov0.5_gap1_min4_acts-jog-stand-walk.npz",
                    '--held_out_activity','walk','--disc_lr_mult','5','--max_train_windows','3579',
                    '--lambda_sweep','0','0.05','0.1','0.25','0.5','1','2') + $c)
Step 'motionsense_amb' (@('--npz',"har_cache\motionsense_hz50_w2.56_ov0.5_gap1_min4_acts-jog-stairsdown-stairsup-walk.npz",
                          '--held_out_activity','walk','--disc_lr_mult','5',
                          '--lambda_sweep','0','0.05','0.1','0.15','0.2','0.3','0.5','1') + $c)
Step 'motionsense_all' (@('--npz',"har_cache\motionsense_hz50_w2.56_ov0.5_gap1_min4_acts-all.npz",
                          '--held_out_activity','walk','--disc_lr_mult','5','--max_train_windows','5234',
                          '--lambda_sweep','0','0.05','0.1','0.15','0.2','0.3','0.5','1') + $c)
# the standard discriminator learning rate (multiplier 1), central HHAR condition
Step 'hhar_dlm1' (@('--npz',$hhar,'--held_out_device','s3mini','--held_out_activity','walk','--disc_lr_mult','1',
                    '--lambda_sweep','0','0.02','0.05','0.1','0.15','0.25','0.5','1') + $c)
Step 'realworld_full' (@('--npz',$rw,'--held_out_device','waist','--held_out_activity','walk','--disc_lr_mult','5',
                         '--lambda_sweep') + $rwl + $c)
"[{0:HH:mm:ss}] ALL DONE" -f (Get-Date) | Out-File frozen_progress.txt -Append -Encoding utf8
