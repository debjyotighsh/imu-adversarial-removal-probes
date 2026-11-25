Set-Location $PSScriptRoot
"[{0:HH:mm:ss}] start subject_nuisance_motionsense" -f (Get-Date) | Out-File subject_nuisance_progress.txt -Append -Encoding utf8
python subject_nuisance_experiment.py --lambda_sweep 0 0.02 0.05 0.1 0.5 1 2 --seeds 0 1 2 *> subject_nuisance.log
"[{0:HH:mm:ss}] SUBJECT NUISANCE DONE" -f (Get-Date) | Out-File subject_nuisance_progress.txt -Append -Encoding utf8
