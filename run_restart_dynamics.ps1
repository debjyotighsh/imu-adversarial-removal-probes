# Instrumented rerun of the restart condition at lambda=1 (three seeds): per-epoch discriminator
# accuracy and recovery curves on the frozen final encoder. Feeds Figure "restart" via
#   python plot_restart_dynamics.py --pkl "gait_experiment_output_v3_restart_dynamics\*_dynamics.pkl"
Set-Location $PSScriptRoot
$hhar = "gait_experiment_output_v2\windows_hz50_win2.56_ov0.5_bike-stairsdown-stairsup-walk_ex-s3mini_2.npz"
python gait_dual_adversarial_v2.py --npz $hhar --held_out_device s3mini --held_out_activity walk --disc_lr_mult 5 `
  --lambda_sweep 1 --seeds 0 1 2 --epochs 25 --disc_restart_every 5 --track_dynamics --pooling stats --mode adversarial `
  --test_cell_probe --disc_heldout --probe_fit_cap 10000 --out_dir ./gait_experiment_output_v3_restart_dynamics --experiment sweep
