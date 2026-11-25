# Training-time baseline requested by review: Iwasawa, Akuzawa & Matsuo (IJCAI 2020), "invariance induction by
# discriminator matching" (IIDM), run through this paper's protocol on the central HHAR condition (three seeds,
# held-out discriminator scoring, linear / MLP / k-NN probes on held-out segments). Their lambda grid is
# {0.001, 0.01, 0.1, 1}. Two variants: lambda annealed by the DANN schedule used elsewhere in this paper, and
# lambda held constant as in the original. Not reproduced: their gradient penalty and RMSprop (Adam is used here).
Set-Location $PSScriptRoot
$hhar = "gait_experiment_output_v2\windows_hz50_win2.56_ov0.5_bike-stairsdown-stairsup-walk_ex-s3mini_2.npz"
$common = @('--npz',$hhar,'--held_out_device','s3mini','--held_out_activity','walk','--disc_lr_mult','5',
            '--lambda_sweep','0.001','0.01','0.1','1','--seeds','0','1','2','--epochs','25','--pooling','stats',
            '--mode','adversarial','--nonlinear_probe','--knn_probe','--test_cell_probe','--disc_heldout',
            '--probe_fit_cap','10000','--experiment','sweep','--adv_method','iidm')

function Step($name, $extra) {
    if (Get-ChildItem "gait_experiment_output_v3_iidm_$name\*.pkl" -ErrorAction SilentlyContinue) {
        "[{0:HH:mm:ss}] skip  {1} (results already present)" -f (Get-Date), $name | Out-File iidm_progress.txt -Append -Encoding utf8
        return
    }
    $t0 = Get-Date
    "[{0:HH:mm:ss}] start {1}" -f $t0, $name | Out-File iidm_progress.txt -Append -Encoding utf8
    & python gait_dual_adversarial_v2.py @common @extra '--out_dir' "./gait_experiment_output_v3_iidm_$name" *> "iidm_$name.log"
    "[{0:HH:mm:ss}] done  {1}  ({2:N0} min)" -f (Get-Date), $name, ((Get-Date) - $t0).TotalMinutes | Out-File iidm_progress.txt -Append -Encoding utf8
}

Step 'annealed' @()
Step 'constant' @('--iidm_constant_lambda')
"[{0:HH:mm:ss}] IIDM DONE" -f (Get-Date) | Out-File iidm_progress.txt -Append -Encoding utf8
