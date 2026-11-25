# Intervention test of the moving-target account (Section "Training Dynamics" says drift causes collapse is
# not shown). On the central HHAR condition at lambda = 0.1 and 1 (three seeds), slow the encoder
# (--backbone_lr_mult) and/or give the discriminator more updates per encoder step (--disc_inner_steps), then
# compare the discriminator's held-out balanced accuracy with the post-hoc probes. The matched baseline is the
# existing run_disc_heldout.ps1 output (same code path, seeds 0-2 of its five).
Set-Location $PSScriptRoot
$hhar = "gait_experiment_output_v2\windows_hz50_win2.56_ov0.5_bike-stairsdown-stairsup-walk_ex-s3mini_2.npz"
$common = @('--npz',$hhar,'--held_out_device','s3mini','--held_out_activity','walk','--disc_lr_mult','5',
            '--lambda_sweep','0.1','1','--seeds','0','1','2','--epochs','25','--pooling','stats','--mode','adversarial',
            '--nonlinear_probe','--knn_probe','--test_cell_probe','--disc_heldout','--probe_fit_cap','10000',
            '--experiment','sweep')

function Step($name, $extra) {
    if (Get-ChildItem "gait_experiment_output_v3_intervene_$name\*.pkl" -ErrorAction SilentlyContinue) {
        "[{0:HH:mm:ss}] skip  {1} (results already present)" -f (Get-Date), $name | Out-File intervene_progress.txt -Append -Encoding utf8
        return
    }
    $t0 = Get-Date
    "[{0:HH:mm:ss}] start {1}" -f $t0, $name | Out-File intervene_progress.txt -Append -Encoding utf8
    & python gait_dual_adversarial_v2.py @common @extra '--out_dir' "./gait_experiment_output_v3_intervene_$name" *> "intervene_$name.log"
    "[{0:HH:mm:ss}] done  {1}  ({2:N0} min)" -f (Get-Date), $name, ((Get-Date) - $t0).TotalMinutes | Out-File intervene_progress.txt -Append -Encoding utf8
}

Step 'k5'         @('--disc_inner_steps','5')
Step 'bb0p1'      @('--backbone_lr_mult','0.1')
Step 'bb0p03'     @('--backbone_lr_mult','0.03')
Step 'bb0p1_k5'   @('--backbone_lr_mult','0.1','--disc_inner_steps','5')
"[{0:HH:mm:ss}] INTERVENE DONE" -f (Get-Date) | Out-File intervene_progress.txt -Append -Encoding utf8
