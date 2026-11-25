# Factorial test of the moving-target account (review: slowing the encoder also changes the optimisation trajectory and weakens the
# adversarial push per update, so it cannot isolate representation drift). Central HHAR condition at lambda = 1, three seeds, 25 epochs,
# same budget (epochs, batch size, encoder steps) in every cell:
#   encoder learning-rate multiplier  {1, 0.1}
# x discriminator updates per encoder update K  {1, 5}   (extra steps on the detached embedding)
# x discriminator restart every 5 epochs  {no, yes}
# plus two cells that raise the adversarial weight to lambda = 10 with the slow encoder (restores the push relative to the identity
# loss while the encoder stays slow). --track_dynamics records the realised representation drift in every cell.
# Run in two parts:  .\run_factorial.ps1 -Part A   and   .\run_factorial.ps1 -Part B
param([string]$Part = 'A')
Set-Location $PSScriptRoot
$hhar = "gait_experiment_output_v2\windows_hz50_win2.56_ov0.5_bike-stairsdown-stairsup-walk_ex-s3mini_2.npz"
$common = @('--npz',$hhar,'--held_out_device','s3mini','--held_out_activity','walk','--disc_lr_mult','5',
            '--seeds','0','1','2','--epochs','25','--pooling','stats','--mode','adversarial','--nonlinear_probe','--knn_probe',
            '--test_cell_probe','--disc_heldout','--probe_fit_cap','10000','--track_dynamics','--experiment','sweep')
function Cell($name, $lam, $lr, $k, $r) {
    if (Get-ChildItem "gait_experiment_output_v3_factorial_$name\*.pkl" -ErrorAction SilentlyContinue) { return }
    $t0 = Get-Date
    "[{0:HH:mm:ss}] start {1}" -f $t0, $name | Out-File factorial_progress.txt -Append -Encoding utf8
    $extra = @('--lambda_sweep',"$lam",'--backbone_lr_mult',"$lr",'--disc_inner_steps',"$k")
    if ($r -gt 0) { $extra += @('--disc_restart_every',"$r") }
    & python gait_dual_adversarial_v2.py @common @extra '--out_dir' "./gait_experiment_output_v3_factorial_$name" *> "factorial_$name.log"
    "[{0:HH:mm:ss}] done  {1}  ({2:N0} min)" -f (Get-Date), $name, ((Get-Date) - $t0).TotalMinutes | Out-File factorial_progress.txt -Append -Encoding utf8
}
if ($Part -eq 'A') {
    Cell 'lr1_k1_r0'   1 1   1 0
    Cell 'lr0.1_k1_r0' 1 0.1 1 0
    Cell 'lr1_k5_r0'   1 1   5 0
    Cell 'lr0.1_k5_r0' 1 0.1 5 0
    Cell 'lr0.1_k1_r0_lam10' 10 0.1 1 0
} else {
    Cell 'lr1_k1_r5'   1 1   1 5
    Cell 'lr0.1_k1_r5' 1 0.1 1 5
    Cell 'lr1_k5_r5'   1 1   5 5
    Cell 'lr0.1_k5_r5' 1 0.1 5 5
    Cell 'lr0.1_k5_r0_lam10' 10 0.1 5 0
}
"[{0:HH:mm:ss}] FACTORIAL $Part DONE" -f (Get-Date) | Out-File factorial_progress.txt -Append -Encoding utf8
