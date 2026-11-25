# Fill-in for run_selection_degrade.ps1: that script's skip test matched on the level tag only, so the HHAR and MotionSense parts,
# running together, skipped levels the other part had already written (tags are shared, e.g. 'noise2'). This version tests for
# the dataset as well. Missing after the first pass: HHAR noise4, mix0.3, mix0.5 and MotionSense noise2.
param([string]$Part = 'H')
Set-Location $PSScriptRoot
$out = './gait_experiment_output_v3_select_degrade'
$c = @('--seeds','0','1','2','--disc_lr_mult','5','--epochs','25','--pooling','stats','--mode','adversarial',
       '--nonlinear_probe','--knn_probe','--test_cell_probe','--split_probe','--probe_fit_cap','10000',
       '--out_dir',$out,'--experiment','sweep')
$hh = "gait_experiment_output_v2\windows_hz50_win2.56_ov0.5_bike-stairsdown-stairsup-walk_ex-s3mini_2.npz"
$ms = "har_cache\motionsense_hz50_w2.56_ov0.5_gap1_min4_acts-jog-stairsdown-stairsup-walk.npz"
function Go($ds, $tag, $argsList) {
    $name = "${ds}_$tag"
    "[{0:HH:mm:ss}] start {1}" -f (Get-Date), $name | Out-File seldeg_progress.txt -Append -Encoding utf8
    & python gait_dual_adversarial_v2.py @argsList *> "seldeg_fill_$name.log"
    "[{0:HH:mm:ss}] done  {1}" -f (Get-Date), $name | Out-File seldeg_progress.txt -Append -Encoding utf8
}
if ($Part -eq 'H') {
    foreach ($d in @(@('noise',4),@('mix',0.3),@('mix',0.5))) {
        Go 'hhar' "$($d[0])$($d[1])" (@('--npz',$hh,'--held_out_device','s3mini','--held_out_activity','walk','--lambda_sweep','0','0.05','0.1','0.25','1','--degrade',$d[0],'--degrade_level',$d[1]) + $c)
    }
} else {
    Go 'ms' 'noise2' (@('--npz',$ms,'--held_out_activity','walk','--lambda_sweep','0','0.05','0.1','0.15','0.2','0.3','0.5','1','--degrade','noise','--degrade_level',2) + $c)
}
"[{0:HH:mm:ss}] FILL $Part DONE" -f (Get-Date) | Out-File seldeg_progress.txt -Append -Encoding utf8
