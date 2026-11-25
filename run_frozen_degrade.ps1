# Controlled-degradation runs under the same frozen configuration; starts after run_frozen.ps1 finishes.
Set-Location $PSScriptRoot
while (-not ((Test-Path frozen_progress.txt) -and (Select-String -Path frozen_progress.txt -Pattern 'ALL DONE' -Quiet))) { Start-Sleep -Seconds 60 }
$out = './gait_experiment_output_v3_frozen'
$c = @('--seeds','0','1','2','--disc_lr_mult','5','--epochs','25','--pooling','stats','--mode','adversarial',
       '--nonlinear_probe','--knn_probe','--test_cell_probe','--probe_fit_cap','10000',
       '--out_dir',$out,'--experiment','sweep')
$hh = "gait_experiment_output_v2\windows_hz50_win2.56_ov0.5_bike-stairsdown-stairsup-walk_ex-s3mini_2.npz"
$ms = "har_cache\motionsense_hz50_w2.56_ov0.5_gap1_min4_acts-jog-stairsdown-stairsup-walk.npz"
foreach ($d in @(@('noise',2),@('noise',4),@('noise',6),@('noise',8),@('mix',0.3),@('mix',0.4),@('mix',0.5))) {
    "[{0:HH:mm:ss}] start degrade hhar {1}{2}" -f (Get-Date), $d[0], $d[1] | Out-File frozen_progress.txt -Append -Encoding utf8
    & python gait_dual_adversarial_v2.py --npz $hh --held_out_device s3mini --held_out_activity walk --lambda_sweep 0 0.05 0.1 0.25 1 --degrade $d[0] --degrade_level $d[1] @c *> "frozen_degrade_hhar_$($d[0])$($d[1]).log"
}
foreach ($d in @(@('noise',1),@('noise',2),@('noise',4),@('mix',0.3),@('mix',0.5))) {
    "[{0:HH:mm:ss}] start degrade ms {1}{2}" -f (Get-Date), $d[0], $d[1] | Out-File frozen_progress.txt -Append -Encoding utf8
    & python gait_dual_adversarial_v2.py --npz $ms --held_out_activity walk --lambda_sweep 0 0.05 0.1 0.15 0.2 0.3 0.5 1 --degrade $d[0] --degrade_level $d[1] @c *> "frozen_degrade_ms_$($d[0])$($d[1]).log"
}
"[{0:HH:mm:ss}] DEGRADE DONE" -f (Get-Date) | Out-File frozen_progress.txt -Append -Encoding utf8
