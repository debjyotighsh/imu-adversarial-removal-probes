# Validation of the protocol on conditions not used to develop it. The decision rule (protocol_rule.py) was frozen first
# (protocol_frozen.txt). Three new conditions, new seeds (10, 11, 12), same flags as run_selection.ps1 plus --disc_heldout:
#   N1  HHAR with all six activities (sit and stand added; walk held out)             a training-pool composition not used before
#   N2  HHAR ambulatory with BIKE held out instead of walk                             a different held-out evaluation cell
#   N3  HHAR ambulatory (walk held out), RMSprop with a constant learning rate         an independently configured training setup
# Run in three parts:  .\run_protocol_validation.ps1 -Part A | B | C
param([string]$Part = 'A')
Set-Location $PSScriptRoot
$all  = "gait_experiment_output_v2\windows_hz50_win2.56_ov0.5_all_ex-s3mini_2.npz"
$amb  = "gait_experiment_output_v2\windows_hz50_win2.56_ov0.5_bike-stairsdown-stairsup-walk_ex-s3mini_2.npz"
$lams = @('0','0.05','0.1','0.25','0.5','1')
$base = @('--seeds','10','11','12','--epochs','25','--pooling','stats','--mode','adversarial','--nonlinear_probe','--knn_probe','--test_cell_probe',
          '--disc_heldout','--split_probe','--probe_fit_cap','10000','--experiment','sweep','--disc_lr_mult','5','--lambda_sweep') + $lams
function Cell($name, $extra) {
    if (Get-ChildItem "gait_experiment_output_v3_validate_$name\*.pkl" -ErrorAction SilentlyContinue) { return }
    $t0 = Get-Date
    "[{0:HH:mm:ss}] start {1}" -f $t0, $name | Out-File validate_progress.txt -Append -Encoding utf8
    & python gait_dual_adversarial_v2.py @extra @base '--out_dir' "./gait_experiment_output_v3_validate_$name" *> "validate_$name.log"
    "[{0:HH:mm:ss}] done  {1}  ({2:N0} min)" -f (Get-Date), $name, ((Get-Date) - $t0).TotalMinutes | Out-File validate_progress.txt -Append -Encoding utf8
}
if ($Part -eq 'A') { Cell 'hhar_all'  @('--npz',$all,'--held_out_device','s3mini','--held_out_activity','walk') }
elseif ($Part -eq 'B') { Cell 'hhar_bike' @('--npz',$amb,'--held_out_device','s3mini','--held_out_activity','bike') }
else { Cell 'hhar_rmsprop' @('--npz',$amb,'--held_out_device','s3mini','--held_out_activity','walk','--optimizer','rmsprop','--constant_lr') }
"[{0:HH:mm:ss}] VALIDATE $Part DONE" -f (Get-Date) | Out-File validate_progress.txt -Append -Encoding utf8
