# RECONSTRUCTED, NOT THE ORIGINAL COMMANDS. The training-dynamics run behind the dynamics figure and the
# TCN robustness check were launched interactively before this release was assembled, and their exact
# command lines were not saved. The settings below follow the paper's description (Sections "Training
# Dynamics" and "Scope of the Measurement"): central HHAR condition, three seeds, standard flags.
# Re-run them before relying on a bit-for-bit match, and compare against the numbers in the paper.
Set-Location $PSScriptRoot
$hhar = "gait_experiment_output_v2\windows_hz50_win2.56_ov0.5_bike-stairsdown-stairsup-walk_ex-s3mini_2.npz"
$common = @('--npz',$hhar,'--held_out_device','s3mini','--held_out_activity','walk','--disc_lr_mult','5',
            '--seeds','0','1','2','--epochs','25','--pooling','stats','--mode','adversarial',
            '--nonlinear_probe','--knn_probe','--test_cell_probe','--probe_fit_cap','10000','--experiment','sweep')

# dynamics: lambda in {0, 0.1, 1}; plot with  python plot_dynamics.py --pkl "<dir>\*_dynamics.pkl"
python gait_dual_adversarial_v2.py @common --lambda_sweep 0 0.1 1 --track_dynamics --out_dir ./gait_experiment_output_v2_dynamics

# TCN replication: same eight-point lambda grid as the CNN sweep
python gait_dual_adversarial_v2.py @common --encoder tcn --lambda_sweep 0 0.02 0.05 0.1 0.15 0.25 0.5 1 --out_dir ./gait_experiment_output_v2_stats
