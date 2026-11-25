# Selection-bias check (review: lambda was chosen as the minimum probe accuracy on the same held-out cell that was then
# reported). Reruns every condition behind the paper's probe-accuracy-reduction numbers with --split_probe, which scores the
# same fitted probes separately on two disjoint halves (by segment, stratified) of the seen/seen evaluation cell:
# half A chooses lambda, half B reports. selection_report.py then computes, per seed, the reduction at the lambda chosen on
# A and scored on B, next to the naive minimum on B.
# Same flags as run_frozen.ps1 / run_extra_seeds.ps1 (five seeds for HHAR central and RealWorld full, three elsewhere).
# Run in two parts so they share the GPU:   .\run_selection.ps1 -Part A   (RealWorld full pool, longest)
#                                           .\run_selection.ps1 -Part B   (everything else)
param([string]$Part = 'B')
Set-Location $PSScriptRoot
$hhar = "gait_experiment_output_v2\windows_hz50_win2.56_ov0.5_bike-stairsdown-stairsup-walk_ex-s3mini_2.npz"
$rw   = "har_cache\realworld_hz50_w2.56_ov0.5_gap1_min4_acts-jog-stairsdown-stairsup-walk.npz"
$rwl  = @('0','0.05','0.1','0.15','0.2','0.3','1','2')
$base = @('--epochs','25','--pooling','stats','--mode','adversarial','--nonlinear_probe','--knn_probe','--test_cell_probe',
          '--split_probe','--probe_fit_cap','10000','--experiment','sweep','--disc_lr_mult','5')
$s3 = @('--seeds','0','1','2'); $s5 = @('--seeds','0','1','2','3','4')

function Step($name, $argsList) {
    if (Get-ChildItem "gait_experiment_output_v3_select_$name\*.pkl" -ErrorAction SilentlyContinue) {
        "[{0:HH:mm:ss}] skip  {1}" -f (Get-Date), $name | Out-File select_progress.txt -Append -Encoding utf8; return
    }
    $t0 = Get-Date
    "[{0:HH:mm:ss}] start {1}" -f $t0, $name | Out-File select_progress.txt -Append -Encoding utf8
    & python gait_dual_adversarial_v2.py @argsList '--out_dir' "./gait_experiment_output_v3_select_$name" *> "select_$name.log"
    "[{0:HH:mm:ss}] done  {1}  ({2:N0} min)" -f (Get-Date), $name, ((Get-Date) - $t0).TotalMinutes | Out-File select_progress.txt -Append -Encoding utf8
}

if ($Part -eq 'A') {
    Step 'realworld_full' (@('--npz',$rw,'--held_out_device','waist','--held_out_activity','walk','--lambda_sweep') + $rwl + $s5 + $base)
} else {
    Step 'hhar'            (@('--npz',$hhar,'--held_out_device','s3mini','--held_out_activity','walk','--lambda_sweep','0','0.02','0.05','0.1','0.15','0.25','0.5','1') + $s5 + $base)
    Step 'mobiact_B'       (@('--npz',"har_cache\mobiact_hz50_w2.56_ov0.5_gap1_min4_acts-stairsdown-stairsup-walk.npz",'--held_out_activity','walk','--lambda_sweep','0','0.05','0.1','0.25','0.5','1','2') + $s3 + $base)
    Step 'mobiact_A'       (@('--npz',"har_cache\mobiact_hz50_w2.56_ov0.5_gap1_min4_acts-jog-stand-walk.npz",'--held_out_activity','walk','--max_train_windows','3579','--lambda_sweep','0','0.05','0.1','0.25','0.5','1','2') + $s3 + $base)
    Step 'motionsense_amb' (@('--npz',"har_cache\motionsense_hz50_w2.56_ov0.5_gap1_min4_acts-jog-stairsdown-stairsup-walk.npz",'--held_out_activity','walk','--lambda_sweep','0','0.05','0.1','0.15','0.2','0.3','0.5','1') + $s3 + $base)
    Step 'motionsense_all' (@('--npz',"har_cache\motionsense_hz50_w2.56_ov0.5_gap1_min4_acts-all.npz",'--held_out_activity','walk','--max_train_windows','5234','--lambda_sweep','0','0.05','0.1','0.15','0.2','0.3','0.5','1') + $s3 + $base)
    Step 'realworld_cap'   (@('--npz',$rw,'--held_out_device','waist','--held_out_activity','walk','--max_train_windows','15315','--lambda_sweep') + $rwl + $s3 + $base)
}
"[{0:HH:mm:ss}] PART $Part DONE" -f (Get-Date) | Out-File select_progress.txt -Append -Encoding utf8
