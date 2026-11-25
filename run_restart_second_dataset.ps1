# Replication of the discriminator-restart check (Section "Preventing Collapse") on two further datasets:
# RealWorld (position nuisance, training pool capped at the HHAR-matched size 15,315) and MobiAct B
# (activity only). For each dataset a matched no-restart baseline and a restart run (--disc_restart_every 5)
# are fitted with --disc_heldout, so both are scored with the same held-out balanced metric.
# Same flags as run_frozen.ps1 otherwise; three seeds.
Set-Location $PSScriptRoot
$common = @('--seeds','0','1','2','--epochs','25','--pooling','stats','--mode','adversarial',
            '--nonlinear_probe','--knn_probe','--test_cell_probe','--disc_heldout','--probe_fit_cap','10000',
            '--experiment','sweep','--disc_lr_mult','5')
$rw = "har_cache\realworld_hz50_w2.56_ov0.5_gap1_min4_acts-jog-stairsdown-stairsup-walk.npz"
$mb = "har_cache\mobiact_hz50_w2.56_ov0.5_gap1_min4_acts-stairsdown-stairsup-walk.npz"

function Step($name, $argsList) {
    $t0 = Get-Date
    "[{0:HH:mm:ss}] start {1}" -f $t0, $name | Out-File restart2_progress.txt -Append -Encoding utf8
    & python gait_dual_adversarial_v2.py @argsList *> "restart2_$name.log"
    "[{0:HH:mm:ss}] done  {1}  ({2:N0} min)" -f (Get-Date), $name, ((Get-Date) - $t0).TotalMinutes | Out-File restart2_progress.txt -Append -Encoding utf8
}

# MobiAct B (fast) first
$mbl = @('--lambda_sweep','0','0.1','0.25','0.5','1','2')
Step 'mobiact_B_base'    (@('--npz',$mb,'--held_out_activity','walk','--out_dir','./gait_experiment_output_v3_restart2_mobiact_base') + $mbl + $common)
Step 'mobiact_B_restart' (@('--npz',$mb,'--held_out_activity','walk','--disc_restart_every','5','--out_dir','./gait_experiment_output_v3_restart2_mobiact_restart') + $mbl + $common)

# RealWorld, capped pool
$rwl = @('--lambda_sweep','0','0.1','0.3','1','2')
Step 'realworld_base'    (@('--npz',$rw,'--held_out_device','waist','--held_out_activity','walk','--max_train_windows','15315','--out_dir','./gait_experiment_output_v3_restart2_realworld_base') + $rwl + $common)
Step 'realworld_restart' (@('--npz',$rw,'--held_out_device','waist','--held_out_activity','walk','--max_train_windows','15315','--disc_restart_every','5','--out_dir','./gait_experiment_output_v3_restart2_realworld_restart') + $rwl + $common)
"[{0:HH:mm:ss}] RESTART2 DONE" -f (Get-Date) | Out-File restart2_progress.txt -Append -Encoding utf8
