# Recoverability profile: for each removed factor, the discriminator head, the three probe families, a probe grid (capacity, fit-set size,
# attacker epochs, attacker restarts), the unseen-device cell, a segment-bootstrap 95% interval for G, and the chance/majority baselines.
# --probe_grid now also scores the device / position factor and stores the segment bootstrap (gait_dual_adversarial_v2.py).
# Three seeds, 25 epochs, held-out discriminator. Run in three parts so they share the GPU:
#   .\run_profile.ps1 -Part A (HHAR)   -Part B (MobiAct B, MotionSense ambulatory)   -Part C (RealWorld, 15.3k training windows)
# profile_report.py reads the results.
param([string]$Part = 'A')
Set-Location $PSScriptRoot
$hhar = "gait_experiment_output_v2\windows_hz50_win2.56_ov0.5_bike-stairsdown-stairsup-walk_ex-s3mini_2.npz"
$rw   = "har_cache\realworld_hz50_w2.56_ov0.5_gap1_min4_acts-jog-stairsdown-stairsup-walk.npz"
$base = @('--seeds','0','1','2','--epochs','25','--pooling','stats','--mode','adversarial','--nonlinear_probe','--knn_probe','--test_cell_probe',
          '--disc_heldout','--probe_grid','--probe_fit_cap','10000','--experiment','sweep','--disc_lr_mult','5')
function Cell($name, $dataArgs, $lams) {
    if (Get-ChildItem "gait_experiment_output_v3_profile_$name\*.pkl" -ErrorAction SilentlyContinue) { return }
    $t0 = Get-Date
    "[{0:HH:mm:ss}] start {1}" -f $t0, $name | Out-File profile_progress.txt -Append -Encoding utf8
    & python gait_dual_adversarial_v2.py @dataArgs @base '--lambda_sweep' @lams '--out_dir' "./gait_experiment_output_v3_profile_$name" *> "profile_$name.log"
    "[{0:HH:mm:ss}] done  {1}  ({2:N0} min)" -f (Get-Date), $name, ((Get-Date) - $t0).TotalMinutes | Out-File profile_progress.txt -Append -Encoding utf8
}
if ($Part -eq 'A') {
    Cell 'hhar' @('--npz',$hhar,'--held_out_device','s3mini','--held_out_activity','walk') @('0','1')
} elseif ($Part -eq 'B') {
    Cell 'mobiact_B' @('--npz',"har_cache\mobiact_hz50_w2.56_ov0.5_gap1_min4_acts-stairsdown-stairsup-walk.npz",'--held_out_activity','walk') @('0','1')
    Cell 'motionsense_amb' @('--npz',"har_cache\motionsense_hz50_w2.56_ov0.5_gap1_min4_acts-jog-stairsdown-stairsup-walk.npz",'--held_out_activity','walk') @('0','0.15','1')
} else {
    Cell 'realworld_cap' @('--npz',$rw,'--held_out_device','waist','--held_out_activity','walk','--max_train_windows','15315') @('0','0.15','1')
}
"[{0:HH:mm:ss}] PROFILE $Part DONE" -f (Get-Date) | Out-File profile_progress.txt -Append -Encoding utf8
