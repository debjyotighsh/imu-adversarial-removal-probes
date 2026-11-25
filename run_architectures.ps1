# Architecture-generality check (review: one CNN encoder family). Reruns the central HHAR sweep and MobiAct condition B with two
# further encoders, a small Transformer and a bidirectional LSTM (gait_dual_adversarial_v2.py --encoder transformer|lstm), under the
# same protocol as run_selection.ps1 (three seeds, held-out discriminator, linear/MLP/k-NN probes, split-half scoring). The CNN
# reference runs are run_selection.ps1's 'hhar' and 'mobiact_B' (first three seeds).
# Run in two parts:  .\run_architectures.ps1 -Part T   and   .\run_architectures.ps1 -Part L
param([string]$Part = 'T')
Set-Location $PSScriptRoot
$enc = if ($Part -eq 'T') { 'transformer' } else { 'lstm' }
$hhar = "gait_experiment_output_v2\windows_hz50_win2.56_ov0.5_bike-stairsdown-stairsup-walk_ex-s3mini_2.npz"
$mb   = "har_cache\mobiact_hz50_w2.56_ov0.5_gap1_min4_acts-stairsdown-stairsup-walk.npz"
$base = @('--seeds','0','1','2','--epochs','25','--pooling','stats','--mode','adversarial','--nonlinear_probe','--knn_probe',
          '--test_cell_probe','--disc_heldout','--split_probe','--probe_fit_cap','10000','--experiment','sweep','--disc_lr_mult','5',
          '--encoder',$enc)
function Go($ds, $argsList) {
    $name = "${enc}_$ds"
    if (Get-ChildItem "gait_experiment_output_v3_arch_$name\*.pkl" -ErrorAction SilentlyContinue) { return }
    "[{0:HH:mm:ss}] start {1}" -f (Get-Date), $name | Out-File arch_progress.txt -Append -Encoding utf8
    & python gait_dual_adversarial_v2.py @argsList '--out_dir' "./gait_experiment_output_v3_arch_$name" *> "arch_$name.log"
    "[{0:HH:mm:ss}] done  {1}" -f (Get-Date), $name | Out-File arch_progress.txt -Append -Encoding utf8
}
Go 'mobiact_B' (@('--npz',$mb,'--held_out_activity','walk','--lambda_sweep','0','0.05','0.1','0.25','0.5','1','2') + $base)
Go 'hhar'      (@('--npz',$hhar,'--held_out_device','s3mini','--held_out_activity','walk','--lambda_sweep','0','0.02','0.05','0.1','0.15','0.25','0.5','1') + $base)
"[{0:HH:mm:ss}] ARCH $Part DONE" -f (Get-Date) | Out-File arch_progress.txt -Append -Encoding utf8
