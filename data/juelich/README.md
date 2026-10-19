# Jülich crowd experiment samples (real data)

Source: demo data shipped with PedPy (https://github.com/PedestrianDynamics/PedPy, notebooks/demo-data),
the analysis library of Forschungszentrum Jülich. The trajectories come from Jülich pedestrian experiments
(Pedestrian Dynamics Data Archive). Check the archive's licence/citation before publishing.

Format (tab-separated, 25 fps): id  frame  x[m]  y[m]  z[m]   (one row per person per frame)

- 040_c_56_h-.txt          bottleneck experiment, 75 people through a narrow bottleneck (see experimental_setup.png
                           and the overhead photo 040_c_56_h-.png). Measured: all 75 through in 64.5 s, ~1.15 persons/s.
- traj_UNI_CORR_500_01.txt one-way corridor experiment (low density)
- uo-070-180-180.txt, uo-145-180-180.txt   one-way flow experiments with different widths

Used for: Improving ML step 1-2 (calibrate door flow and speed-density) and step 5 (real-data test).

Added in the Improving-ML session (same source): uo-050/060/100-180-180, uo-180-180-070/095/120/180 (corridor 1.8 m,
exit narrowed to 0.70-1.80 m), traj_UNI_CORR_500_03/_08. Units: uo-* in cm at 16 fps; UNI_CORR in m at 25 fps.
Experiments: Zhang et al. (2011) J. Stat. Mech. P06004 (corridors); 040_c_56_h- = Adrian, Seyfried & Sieben (2020)
J. R. Soc. Interface 17:20190871 (Wuppertal, 0.5 m entrance, 5.6 m corridor, low motivation). PedPy: MIT licence.

Added in the Improving-ML session (same source): uo-050/060/100-180-180, uo-180-180-070/095/120/180 (corridor 1.8 m,
exit narrowed to 0.70-1.80 m), traj_UNI_CORR_500_03/_08. Units: uo-* in cm at 16 fps; UNI_CORR in m at 25 fps.
Experiments: Zhang et al. (2011) J. Stat. Mech. P06004 (corridors); 040_c_56_h- = Adrian, Seyfried & Sieben (2020)
J. R. Soc. Interface 17:20190871 (Wuppertal, 0.5 m entrance, 5.6 m corridor, low motivation). PedPy: MIT licence.
