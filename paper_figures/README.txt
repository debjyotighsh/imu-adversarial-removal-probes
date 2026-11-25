The figures as printed in the paper (PDF, all text in Times New Roman).

  overview.pdf          Figure 1   overview diagram (TikZ source: overview/, compile with XeLaTeX:
                                   cd overview && xelatex overview_times_standalone.tex)
  fig1_divergence.pdf   Figure 2   discriminator vs post-hoc probe over lambda, HHAR
  fig_probe_gap.pdf     Figure 3   the probe-discriminator gap G along four axes
  fig5_dynamics.pdf     Figure 4   training dynamics, six panels
  fig7_restart.pdf      Figure 5   restart condition: discriminator and recovery
  fig3_datasize.pdf     Figure 6   activity-probe accuracy vs lambda at two training-set sizes (RealWorld)
  fig4_lambda_band.pdf  Figure 7   probe vs discriminator across lambda, four datasets

Regenerate Figures 2-7 from the stored results:  python make_times_figures.py --out paper_figures_regenerated
(see the top-level README). The values behind every plotted point are in ../figure_data/.
