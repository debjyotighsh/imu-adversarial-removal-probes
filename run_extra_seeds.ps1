# Two additional seeds (3, 4) for the highest-priority conditions, to bring n from 3 to 5:
#   1. HHAR central collapse experiment (disc_lr_mult=5, the paper's headline result)
#   2. RealWorld full pool (covers both the nonlinear-probe table and the lambda sweep,
#      since both are read from the same run)
# Same flags as run_frozen.ps1, new seeds, separate output dir so the existing seeds 0-2
# checkpoints are untouched; merge_extra_seeds.py combines the two afterward.
Set-Location $PSScriptRoot
$out = './gait_experiment_output_v3_frozen_extra'
$c = @('--seeds','3','4','--epochs','25','--pooling','stats','--mode','adversarial',
       '--nonlinear_probe','--knn_probe','--test_cell_probe','--probe_fit_cap','10000',
       '--out_dir',$out,'--experiment','sweep')
$hhar = "gait_experiment_output_v2\windows_hz50_win2.56_ov0.5_bike-stairsdown-stairsup-walk_ex-s3mini_2.npz"
$rw   = "har_cache\realworld_hz50_w2.56_ov0.5_gap1_min4_acts-jog-stairsdown-stairsup-walk.npz"
$rwl  = @('0','0.05','0.1','0.15','0.2','0.3','1','2')

function Step($name, $argsList) {
    $t0 = Get-Date
    "[{0:HH:mm:ss}] start {1}" -f $t0, $name | Out-File extra_seeds_progress.txt -Append -Encoding utf8
    & python gait_dual_adversarial_v2.py @argsList *> "extra_$name.log"
    "[{0:HH:mm:ss}] done  {1}  ({2:N0} min)" -f (Get-Date), $name, ((Get-Date) - $t0).TotalMinutes | Out-File extra_seeds_progress.txt -Append -Encoding utf8
}

Step 'hhar_dlm5' (@('--npz',$hhar,'--held_out_device','s3mini','--held_out_activity','walk','--disc_lr_mult','5',
                    '--lambda_sweep','0','0.02','0.05','0.1','0.15','0.25','0.5','1') + $c)
Step 'realworld_full' (@('--npz',$rw,'--held_out_device','waist','--held_out_activity','walk','--disc_lr_mult','5',
                         '--lambda_sweep') + $rwl + $c)
"[{0:HH:mm:ss}] EXTRA SEEDS DONE" -f (Get-Date) | Out-File extra_seeds_progress.txt -Append -Encoding utf8
