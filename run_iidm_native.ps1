# Follow-up to run_iidm.ps1, answering three objections to the first IIDM baseline:
#  (1) it was run only on HHAR activity, where prior decodability is ~0.99 and removal is near zero for every
#      method, so it could not succeed anywhere -> also run on IIDM's native use case, user anonymisation
#      (MotionSense with subject identity as the nuisance, Section "Swapping the Roles"), and on RealWorld
#      (activity p0 ~ 0.80, position, with room to remove);
#  (2) it dropped three details of the original -> a "faithful" configuration restores RMSprop (lr 1e-3, constant),
#      an R1 gradient penalty (weight 10) on the discriminator, centroids initialised from the whole training set,
#      constant lambda, one learning rate for all parts;
#  (3) it was short -> a 100-epoch HHAR run (the original used 20k-30k iterations; 100 epochs here is ~11.8k).
# Three seeds throughout; discriminator scored held-out; probes held-out (linear / MLP / k-NN).
Set-Location $PSScriptRoot
$faithful = @('--adv_method','iidm','--iidm_constant_lambda','--iidm_gp','10','--iidm_init_centroids',
              '--optimizer','rmsprop','--constant_lr','--disc_lr_mult','1')
$simplified = @('--adv_method','iidm','--disc_lr_mult','5')   # Adam, cosine, annealed lambda, uniform centroids, no GP
$hhar = "gait_experiment_output_v2\windows_hz50_win2.56_ov0.5_bike-stairsdown-stairsup-walk_ex-s3mini_2.npz"
$rw   = "har_cache\realworld_hz50_w2.56_ov0.5_gap1_min4_acts-jog-stairsdown-stairsup-walk.npz"
$sweep = @('--seeds','0','1','2','--pooling','stats','--mode','adversarial','--nonlinear_probe','--knn_probe',
           '--test_cell_probe','--disc_heldout','--probe_fit_cap','10000','--experiment','sweep')

function Run($name, $script, $argsList) {
    if (Get-ChildItem "gait_experiment_output_v3_iidmn_$name\*.pkl" -ErrorAction SilentlyContinue) {
        "[{0:HH:mm:ss}] skip  {1}" -f (Get-Date), $name | Out-File iidmn_progress.txt -Append -Encoding utf8
        return
    }
    $t0 = Get-Date
    "[{0:HH:mm:ss}] start {1}" -f $t0, $name | Out-File iidmn_progress.txt -Append -Encoding utf8
    & python $script @argsList '--out_dir' "./gait_experiment_output_v3_iidmn_$name" *> "iidmn_$name.log"
    "[{0:HH:mm:ss}] done  {1}  ({2:N0} min)" -f (Get-Date), $name, ((Get-Date) - $t0).TotalMinutes | Out-File iidmn_progress.txt -Append -Encoding utf8
}

$l4 = @('--lambda_sweep','0.001','0.01','0.1','1')
Run 'ms_simplified' 'subject_nuisance_experiment.py' (@('--epochs','25','--seeds','0','1','2') + $l4 + $simplified)
Run 'ms_faithful'   'subject_nuisance_experiment.py' (@('--epochs','25','--seeds','0','1','2') + $l4 + $faithful)
Run 'hhar_faithful' 'gait_dual_adversarial_v2.py' (@('--npz',$hhar,'--held_out_device','s3mini','--held_out_activity','walk','--epochs','25') + $l4 + $faithful + $sweep)
Run 'rw_faithful'   'gait_dual_adversarial_v2.py' (@('--npz',$rw,'--held_out_device','waist','--held_out_activity','walk','--max_train_windows','15315','--epochs','25','--lambda_sweep','0.01','0.1','1','2') + $faithful + $sweep)
Run 'hhar_long'     'gait_dual_adversarial_v2.py' (@('--npz',$hhar,'--held_out_device','s3mini','--held_out_activity','walk','--epochs','100','--lambda_sweep','0.1','1') + $faithful + $sweep)
"[{0:HH:mm:ss}] IIDMN DONE" -f (Get-Date) | Out-File iidmn_progress.txt -Append -Encoding utf8
