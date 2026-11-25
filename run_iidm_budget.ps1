# Follow-up to run_iidm_native.ps1, answering two objections:
#  (a) the faithful IIDM configuration may be limited by its training budget (25 epochs ~ 3,000 steps on HHAR vs
#      20,000-30,000 iterations in the original; 100 epochs raised HHAR device removal from 5.7% to 13.4%)
#      -> faithful IIDM for 200 epochs (~24,000 steps) on HHAR (device and activity) and MotionSense subject.
#  (b) Table 11 compared best cases over different lambda grids -> gradient reversal at the IIDM grid points it
#      lacked (0.001 and 0.01 on HHAR and MotionSense, 0.01 on RealWorld), same seeds and protocol.
# Run in two halves so they can share the GPU:   .\run_iidm_budget.ps1 -Part A   and   .\run_iidm_budget.ps1 -Part B
param([string]$Part = 'A')
Set-Location $PSScriptRoot
$faithful = @('--adv_method','iidm','--iidm_constant_lambda','--iidm_gp','10','--iidm_init_centroids',
              '--optimizer','rmsprop','--constant_lr','--disc_lr_mult','1')
$hhar = "gait_experiment_output_v2\windows_hz50_win2.56_ov0.5_bike-stairsdown-stairsup-walk_ex-s3mini_2.npz"
$rw   = "har_cache\realworld_hz50_w2.56_ov0.5_gap1_min4_acts-jog-stairsdown-stairsup-walk.npz"
$sweep = @('--seeds','0','1','2','--pooling','stats','--mode','adversarial','--nonlinear_probe','--knn_probe',
           '--test_cell_probe','--disc_heldout','--probe_fit_cap','10000','--experiment','sweep')

function Run($name, $script, $argsList) {
    if (Get-ChildItem "gait_experiment_output_v3_budget_$name\*.pkl" -ErrorAction SilentlyContinue) {
        "[{0:HH:mm:ss}] skip  {1}" -f (Get-Date), $name | Out-File budget_progress.txt -Append -Encoding utf8
        return
    }
    $t0 = Get-Date
    "[{0:HH:mm:ss}] start {1}" -f $t0, $name | Out-File budget_progress.txt -Append -Encoding utf8
    & python $script @argsList '--out_dir' "./gait_experiment_output_v3_budget_$name" *> "budget_$name.log"
    "[{0:HH:mm:ss}] done  {1}  ({2:N0} min)" -f (Get-Date), $name, ((Get-Date) - $t0).TotalMinutes | Out-File budget_progress.txt -Append -Encoding utf8
}

if ($Part -eq 'A') {
    # gradient reversal at the small lambdas (paper's GRL setting: Adam, cosine, annealed lambda, dlm 5)
    Run 'grl_hhar'       'gait_dual_adversarial_v2.py' (@('--npz',$hhar,'--held_out_device','s3mini','--held_out_activity','walk','--disc_lr_mult','5','--epochs','25','--lambda_sweep','0.001','0.01') + $sweep)
    Run 'grl_ms'         'subject_nuisance_experiment.py' @('--epochs','25','--seeds','0','1','2','--lambda_sweep','0.001','0.01')
    Run 'grl_rw'         'gait_dual_adversarial_v2.py' (@('--npz',$rw,'--held_out_device','waist','--held_out_activity','walk','--max_train_windows','15315','--disc_lr_mult','5','--epochs','25','--lambda_sweep','0.01') + $sweep)
    Run 'hhar_faithful200' 'gait_dual_adversarial_v2.py' (@('--npz',$hhar,'--held_out_device','s3mini','--held_out_activity','walk','--epochs','200','--lambda_sweep','1') + $faithful + $sweep)
} else {
    Run 'ms_faithful200' 'subject_nuisance_experiment.py' (@('--epochs','200','--seeds','0','1','2','--lambda_sweep','0.1','1') + $faithful)
}
"[{0:HH:mm:ss}] PART $Part DONE" -f (Get-Date) | Out-File budget_progress.txt -Append -Encoding utf8
