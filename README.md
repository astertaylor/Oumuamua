# Oumuamua
---
A set of files and codes for Taylor et al. (2023), which perform the analysis and produce the results used in the paper. 

---

## Analytics
A folder containing files which are utilized in the analytics.
>**ana_num_res.csv**: A file containing the numerically-computed tidal torques at different values, for use in validation of the tidal torques.\
>\
>**Comparative Analytics.ipynb**: Jupyter Notebook which compares the tidal and outgassing torques and saves the figure.\
>\
>**Jet Rotation Comparison.ipynb**: Jupyter Notebook which computes the non-normal outgassing torque, compares it to the tidal torque, and saves the figure. \
>\
>**Tidal Analytics Validation.ipynb**:  Jupyter Notebook which validates the use of the 'dumbbell' model, reads in the numerically-computed tidal torques and compares it to the analytically-evaluated model values. Creates and saves a figure.\
>\
>**Jet Torque Analytics.nb**: Mathematica file which demonstrates the outgassing jet torque formulae.\
>\
>**Rotated Jet Torque Analytics.nb**:  Mathematica file which demonstrates the non-normal outgassing jet torque formulae.\
>\
>**Tidal Torque Analytics.nb**: Mathematica file which demonstrates the 'dumbbell' tidal torque formulae.
>\
>**Tidal and Jet Torque Analytics Comparisons.nb**:  Mathematica file which compares the outgassing and the tidal torque magnitudes.\

## Figures and Paper
A folder containing all figures used in the paper.
>**arbitrary_axis_lightcurve.pdf**: Figure with the optimized light curve for the fixed-position, arbitrary-axis model, plotted with photometric data.\
>\
>**arbitrary_axis_residual.pdf**: Figure with the residual of the optimized fixed-position, arbitrary-axis light curve versus the photometric data.\
>\
>**axis_classes.pdf**: Figure demonstrating the unique classes of axis arrangement.\
>\
>**dumbbellmodel.pdf**: Figure demonstrating the arrangement and variables of the 'dumbbell' model.\
>\
>**evolving_axis_lightcurve.pdf**: Figure with the optimized light curve for the evolving-position, arbitrary-axis model, plotted with photometric data.\
>\
>**evolving_axis_residual.pdf**: Figure with the residual of the optimized evolving-position, arbitrary-axis light curve versus the photometric data.\
>\
>**fixed_axis_lightcurve.pdf**: Figure with the optimized light curve for the fixed-position, fixed-axis model, plotted with photometric data.\
>\
>**fixed_axis_residual.pdf**: Figure with the residual of the optimized fixed-position, fixed-axis light curve versus the photometric data.\
>\
>**lomb_scargle_power.pdf**: Figure with the Lomb-Scargle periodogram for the October and November 2017 photometric data.\
>\
>**ml_comp.pdf**: Figure comparing the Muinonen-Lumme evolving-position, arbitrary-axis model to the fixed-position, arbitrary-axis model for different parameters.\
>\
>**num_lightcurve_comp.pdf**: Figure comparing the fixed-position, arbitrary-axis (numerical) model to the fixed-position, fixed-axis model for different numerical parameters.\
>\
>**optimal_axis_aspect_ratio.pdf**: Figure showing the evolution in the aspect ratios for simulations utilizing the optimized rotation axis.\
>\
>**optimal_axis_heatmap.pdf**: Figure with a heatmap showing the convergence and change in the moment of inertia for simulations of viscosity and initial primary sizes, with optimally chosen rotation axis.\
>\
>**optimal_axis_lightcurve_sims.pdf**: Figure showing simulated light curves at 5 points in the trajectory, incorporating changes in period and aspect ratios. Utilizes the simulations with optimally chosen rotation axis. \
>\
>**optimal_axis_periods.pdf**: Figure showing the evolution in the periods for simulations utilizing the optimized rotation axis.\
>\
>**optimal_sims_lightcurve_comp.pdf**: Figure showing simulated light curves over the photometric data, exclusively incorporating changes in aspect ratio. Utilizes the simulations with optimally chosen rotation axis.\
>\
>**outgassing_diagram.pdf**: Figure showing the non-normal outgassing forces.\
>\
>**period_comp_hist.pdf**: Figure with a histogram showing the optimized fits for the period, in October and November. A) is the periods, B) is the statistical significance of the difference between each pair.\
>\
>**phased_lightcurve.pdf**: Figure showing the light curve datas phased against the October and November best-fit periods, demonstrating the change in period.\
>\
>**principal_axis_aspect_ratio.pdf**: Figure showing the evolution in the aspect ratios for simulations utilizing the classes with principal rotation axis.\
>\
>**principal_axis_heatmap.pdf**: Figure with a heatmap showing the convergence and change in the moment of inertia for simulations of viscosity and initial primary sizes, with the principal rotation axis.\
>\
>**principal_axis_lightcurve_sims.pdf**: Figure showing simulated light curves at 5 points in the trajectory, incorporating changes in period and aspect ratios. Utilizes the simulations with classes of principal rotation axis. \
>\
>**principal_axis_periods.pdf**: Figure showing the evolution in the periods for simulations utilizing the classes of principal rotation axis.\
>\
>**rot_torque_ratio.pdf**: Figure comparing the tidal torque and the rotated outgassing torque.\
>\
>**SAMUSfig.pdf**: Figure with a flowchart showing the general structure of SAMUS.\
>\
>**sim_chi2_heatmap.pdf**: Figure with a heatmap of the Χ<sup>2</sup> values for the optimal-axis simulated lightcurves when compared to the photometric data. \
>\
>**simflowchart.pdf**: Figure with a flowchart of the simulation sets run in Taylor et al.\
>\
>**torque_ratio.pdf**: Figure showing the ratio between the outgassing and tidal torques.\
>\
>**validation_chi2.pdf**: Figure showing the Χ<sup>2</sup> values between the best-fit curve and the synthetic data it is generated from. Used to validate the optimization method for finding the rotation axis.\
>\
>**validation_lightcurve.pdf**: Figure showing a randomized light curve, the synthetic data from that light curve, and the best-fit curve to that data.\
>\
>**validation_ratio.pdf**: Figure showing the ratio between the numerically computed tidal torque and the analytically computed tidal torque from the 'dumbbell' model, demonstrating their equivalence.
>
>### Figure PPTs
> Folder containing all of the PowerPoint files used to create figures.
>>**Class Figure.pptx**: PPT which is used to create the figure showing the classes of unique axis arrangement.\
>>\
>>**Dumbbell Figure.pptx**: PPT which is used to create the figure showing the structure of the 'dumbbell' model.\
>>\
>>**Rotating Figure.pptx**: PPT which is used to create the figure showing the non-normal outgassing torque.\
>>\
>>**SAMUS Figure.pptx**: PPT with a flowchart showing the structure of SAMUS.\
>>\
>>**Sim Flow Chart.pptx**: PPT with a flowchart showing the use of the SAMUS model in the simulations used in Taylor et al. 
>>

## Lightcurves
A folder containing files which are used to compute and model the light curves.

## Simulations
A folder containing files which are used in the numerical simulations of `Oumaumua.

---
Aster Taylor\
astertaylor@uchicago.edu | aster.taylor8587@gmail.com \
University of Chicago, Department of Astrophysics
