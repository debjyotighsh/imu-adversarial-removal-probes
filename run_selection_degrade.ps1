# Controlled-degradation runs (paper Section "A Controlled Manipulation of Separability") repeated with --split_probe, so the
# reduction can be re-estimated with lambda chosen on one half of the evaluation cell and read on the other (see
# run_selection.ps1 / selection_report.py). Same configuration as run_frozen_degrade.ps1, three seeds.
# Run in two parts so they share the GPU:  .\run_selection_degrade.ps1 -Part H   (HHAR)   and   -Part M   (MotionSense)
param([string]$Part = 'H')
Set-Location $PSScriptRoot
$out = './gait_experiment_output_v3_select_degrade'
$c = @('--seeds','0','1','2','--disc_lr_mult','5','--epochs','25','--pooling','stats','--mode','adversarial',
       '--nonlinear_probe','--knn_probe','--test_cell_probe','--split_probe','--probe_fit_cap','10000',
       '--out_dir',$out,'--experiment','sweep')
$hh = "gait_experiment_output_v2\windows_hz50_win2.56_ov0.5_bike-stairsdown-stairsup-walk_ex-s3mini_2.npz"
$ms = "har_cache\motionsense_hz50_w2.56_ov0.5_gap1_min4_acts-jog-stairsdown-stairsup-walk.npz"
function Go($tag, $argsList) {
    if (Get-ChildItem "$out\*_deg$tag.pkl" -ErrorAction SilentlyContinue) { return }
    $t0 = Get-Date
    "[{0:HH:mm:ss}] start {1}" -f $t0, $tag | Out-File seldeg_progress.txt -Append -Encoding utf8
    & python gait_dual_adversarial_v2.py @argsList *> "seldeg_$tag.log"
    "[{0:HH:mm:ss}] done  {1}  ({2:N0} min)" -f (Get-Date), $tag, ((Get-Date) - $t0).TotalMinutes | Out-File seldeg_progress.txt -Append -Encoding utf8
}
if ($Part -eq 'H') {
    foreach ($d in @(@('noise',2),@('noise',4),@('noise',6),@('noise',8),@('mix',0.3),@('mix',0.4),@('mix',0.5))) {
        Go "$($d[0])$($d[1])" (@('--npz',$hh,'--held_out_device','s3mini','--held_out_activity','walk','--lambda_sweep','0','0.05','0.1','0.25','1','--degrade',$d[0],'--degrade_level',$d[1]) + $c)
    }
} else {
    foreach ($d in @(@('noise',1),@('noise',2),@('noise',4),@('mix',0.3),@('mix',0.5))) {
        Go "$($d[0])$($d[1])" (@('--npz',$ms,'--held_out_activity','walk','--lambda_sweep','0','0.05','0.1','0.15','0.2','0.3','0.5','1','--degrade',$d[0],'--degrade_level',$d[1]) + $c)
    }
}
"[{0:HH:mm:ss}] PART $Part DONE" -f (Get-Date) | Out-File seldeg_progress.txt -Append -Encoding utf8
