# Held-out SUBJECT evaluation (reviewer request: findings on a cell of people the encoder and probes never saw).
# For each condition, three different random sets of subjects are held out entirely (--hold_subjects_n / --hold_subjects_seed 1..3),
# one encoder seed per split, so the three runs give three subject splits. Held-out subjects are removed from encoder training, the probe
# fit pool and all other cells; cell 'unseen_subject' (their windows on seen devices and activities) is scored for the discriminator heads and
# the probes, with the segment bootstrap of --probe_grid. Same flags as run_profile.ps1 otherwise (25 epochs, held-out discriminator).
# Run in three parts:  .\run_subject_holdout.ps1 -Part A (HHAR)   -Part B (MobiAct B, MotionSense)   -Part C (RealWorld, 15.3k windows)
# subject_holdout_report.py reads the results.
param([string]$Part = 'A')
Set-Location $PSScriptRoot
$hhar = "gait_experiment_output_v2\windows_hz50_win2.56_ov0.5_bike-stairsdown-stairsup-walk_ex-s3mini_2.npz"
$rw   = "har_cache\realworld_hz50_w2.56_ov0.5_gap1_min4_acts-jog-stairsdown-stairsup-walk.npz"
$base = @('--seeds','0','--epochs','25','--pooling','stats','--mode','adversarial','--nonlinear_probe','--knn_probe','--test_cell_probe',
          '--disc_heldout','--probe_grid','--probe_fit_cap','10000','--experiment','sweep','--disc_lr_mult','5')
function Cell($name, $dataArgs, $nsub, $lams) {
    foreach ($sp in 1,2,3) {
        $out = "gait_experiment_output_v3_subj_${name}_s$sp"
        if (Get-ChildItem "$out\*.pkl" -ErrorAction SilentlyContinue) { continue }
        $t0 = Get-Date
        "[{0:HH:mm:ss}] start {1} split {2}" -f $t0, $name, $sp | Out-File subject_progress.txt -Append -Encoding utf8
        & python gait_dual_adversarial_v2.py @dataArgs @base '--hold_subjects_n' "$nsub" '--hold_subjects_seed' "$sp" '--lambda_sweep' @lams '--out_dir' "./$out" *> "subj_${name}_s$sp.log"
        "[{0:HH:mm:ss}] done  {1} split {2}  ({3:N0} min)" -f (Get-Date), $name, $sp, ((Get-Date) - $t0).TotalMinutes | Out-File subject_progress.txt -Append -Encoding utf8
    }
}
if ($Part -eq 'A') {
    Cell 'hhar' @('--npz',$hhar,'--held_out_device','s3mini','--held_out_activity','walk') 3 @('0','1')
} elseif ($Part -eq 'B') {
    Cell 'mobiact_B' @('--npz',"har_cache\mobiact_hz50_w2.56_ov0.5_gap1_min4_acts-stairsdown-stairsup-walk.npz",'--held_out_activity','walk') 12 @('0','1')
    Cell 'motionsense_amb' @('--npz',"har_cache\motionsense_hz50_w2.56_ov0.5_gap1_min4_acts-jog-stairsdown-stairsup-walk.npz",'--held_out_activity','walk') 5 @('0','0.15','1')
} else {
    Cell 'realworld_cap' @('--npz',$rw,'--held_out_device','waist','--held_out_activity','walk','--max_train_windows','15315') 3 @('0','0.15','1')
}
"[{0:HH:mm:ss}] SUBJECT $Part DONE" -f (Get-Date) | Out-File subject_progress.txt -Append -Encoding utf8
