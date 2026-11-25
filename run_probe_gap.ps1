# Probe-generalisation-gap study: how the gap between the training-time discriminator and the post-hoc attacker changes with
#   probe capacity and probe fit-set size      (E1: --probe_grid on the central HHAR sweep and MobiAct B)
#   encoder training-set size                  (E2: --max_train_windows 1000 / 3000 / full)
#   optimisation budget                        (E3: epochs 5, 10, 25 (= E1), 50, 100)
#   evaluation split                           (E4: read from the --disc_heldout cells of E1; no extra run)
# All runs: three seeds, held-out discriminator (balanced accuracy), linear/MLP/k-NN probes on the seen/seen and unseen cells,
# --probe_grid for the capacity / fit-size grid. probe_gap_report.py reads the results.
# Run in three parts so they share the GPU:  .\run_probe_gap.ps1 -Part A  (E1)   -Part B  (E2 and short E3)   -Part C  (long E3)
param([string]$Part = 'A')
Set-Location $PSScriptRoot
$hhar = "gait_experiment_output_v2\windows_hz50_win2.56_ov0.5_bike-stairsdown-stairsup-walk_ex-s3mini_2.npz"
$mob  = "har_cache\mobiact_hz50_w2.56_ov0.5_gap1_min4_acts-stairsdown-stairsup-walk.npz"
$hh   = @('--npz',$hhar,'--held_out_device','s3mini','--held_out_activity','walk')
$mb   = @('--npz',$mob,'--held_out_activity','walk')
$base = @('--seeds','0','1','2','--pooling','stats','--mode','adversarial','--nonlinear_probe','--knn_probe','--test_cell_probe',
          '--disc_heldout','--probe_grid','--probe_fit_cap','10000','--experiment','sweep','--disc_lr_mult','5')
function Cell($name, $dataArgs, $lams, $epochs, $extra) {
    if (Get-ChildItem "gait_experiment_output_v3_gap_$name\*.pkl" -ErrorAction SilentlyContinue) { return }
    $t0 = Get-Date
    "[{0:HH:mm:ss}] start {1}" -f $t0, $name | Out-File gap_progress.txt -Append -Encoding utf8
    & python gait_dual_adversarial_v2.py @dataArgs @base '--epochs' "$epochs" '--lambda_sweep' @lams @extra '--out_dir' "./gait_experiment_output_v3_gap_$name" *> "gap_$name.log"
    "[{0:HH:mm:ss}] done  {1}  ({2:N0} min)" -f (Get-Date), $name, ((Get-Date) - $t0).TotalMinutes | Out-File gap_progress.txt -Append -Encoding utf8
}
if ($Part -eq 'A') {
    Cell 'hhar_ep25'   $hh @('0','0.1','0.25','1') 25 @()
    Cell 'mobiact_ep25' $mb @('0','0.5','1') 25 @()
} elseif ($Part -eq 'B') {
    Cell 'hhar_n1000' $hh @('0','1') 25 @('--max_train_windows','1000')
    Cell 'hhar_n3000' $hh @('0','1') 25 @('--max_train_windows','3000')
    Cell 'hhar_ep5'   $hh @('0','1') 5 @()
    Cell 'hhar_ep10'  $hh @('0','1') 10 @()
} else {
    Cell 'hhar_ep50'  $hh @('0','1') 50 @()
    Cell 'hhar_ep100' $hh @('0','1') 100 @()
}
"[{0:HH:mm:ss}] GAP $Part DONE" -f (Get-Date) | Out-File gap_progress.txt -Append -Encoding utf8
