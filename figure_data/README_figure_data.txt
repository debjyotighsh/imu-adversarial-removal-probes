Source data for the figures in the manuscript
=============================================
"A Probe-Based Analysis of Adversarial Feature Removal in Inertial Sensing for Subject Identification and Activity Recognition"

One CSV file per data figure. Figure 1 of the manuscript (overview diagram) has no data.

  fig1_divergence.csv   Figure 2: discriminator vs post-hoc probe against lambda, HHAR
  fig_probe_gap.csv     Figure 3: the probe-discriminator gap G along four axes
  fig5_dynamics.csv     Figure 4: training dynamics, six panels (HHAR, per epoch)
  fig7_restart.csv      Figure 5: restart condition, per-epoch discriminator and recovery
  fig3_datasize.csv     Figure 6: activity-probe accuracy vs lambda at two training-set sizes (RealWorld)
  fig4_lambda_band.csv  Figure 7: probe vs discriminator across lambda, four datasets

(The file names are the internal figure names; the figure numbers are those of the manuscript.)

Columns
  panel     panel label as in the figure
  series    plotted curve
  x         x-axis value (lambda, epoch, extra epochs, probe setting, number of windows, ...)
  n_seeds   number of independent seeds
  mean      mean over seeds (the plotted marker or line)
  sd        sample standard deviation over seeds (n-1; the plotted error bar or band)
  seed_1..  the individual seed values (the thin markers or lines in the figures)

Accuracy values are fractions in [0, 1]. The gap G is the difference of two balanced accuracies. Seeds are the only
independent replicates; the figures show no confidence intervals from windows.
The files are produced by export_figure_data.py in the code release, through the same loaders as the figure scripts.

