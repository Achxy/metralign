# Evidence map and verified references

This document records the literature basis for the physical nuisance mechanisms and estimators used in the submission. It is deliberately a claim map, not a claim that the renderer is a calibrated SEM simulator.

## Scope of the claims

- The cited literature establishes that the named mechanism, measurement effect, or estimator is real and relevant. It does **not** validate the numerical distributions or amplitudes selected for this challenge.
- Every configured augmentation range (roughness amplitude and correlation length, width/radius/placement spread, blur width, dose, Gaussian-noise level, gain/offset, drift, row jitter, scale, rotation, anisotropy, and interpolation choice) is an **engineering stress-test range** unless a separate calibration source is explicitly attached to that value. The submission does not claim those ranges are typical of a particular fab, tool, detector, process node, dose, dwell time, or magnification.
- The image renderer is phenomenological. In particular, a gradient-derived bright edge is a compact surrogate for secondary-electron topographic edge contrast; Gaussian blur is a compact surrogate for a finite probe/transfer function; and simple random variables stand in for more complicated detector and scan chains.
- Physical layout variation is persistent in shared specimen/world coordinates; acquisition nuisances are sampled per image. The papers below support that conceptual separation but do not prescribe the implementation's random seeds or probability laws.
- Some detector and scan references are from STEM/TEM rather than top-down SEM. Those papers are cited only for generic scanned-electron or electron-detector effects, and are labeled as such; SEM-specific evidence is included wherever available.

## Augmentation evidence matrix

This table maps every enabled augmentation family to code and primary-source evidence. `U[a,b]` denotes a uniform draw on the closed engineering interval shown. The difficulty multiplier `f` is 0.60 (`easy`), 1.00 (`medium`), or 1.45 (`hard`). Values below describe the generator exactly; they are challenge stress-test settings, not ranges inferred from the cited papers.

| Mechanism | Implementation/module | Parameter rationale and implemented range | Primary-source evidence |
|---|---|---|---|
| DRAM contact pitch, size, placement, and edge variation | `src/drift_sense/architectures.py::render_dram_geometry` | Shared world-coordinate variation makes both captures observe the same specimen. Default global pitch spread is ±1% (geometry OOD: ±3.5%); nominal radius spread is ±2.5% (OOD: ±10%); per-contact radius is ±7.5%, center displacement is bounded by ±0.46/±0.40 world units in x/y, and radial edge modulation has amplitude 0.10 world unit. These are stress ranges. | [Vijaya-Kumar et al. 2011](https://doi.org/10.1016/j.mee.2011.02.003)<br>[Kim et al. 2014](https://doi.org/10.1117/12.2048282)<br>[Severi et al. 2021](https://doi.org/10.1117/12.2585308) |
| Fin/gate pitch, width, center, and line-edge/line-width variation | `src/drift_sense/architectures.py::render_finfet_geometry` | Default pitch spread is ±1% (geometry OOD: ±4%); nominal width spread is ±2.5% (OOD: ±10%). Persistent line and segment center terms have amplitudes 0.18/0.20 world unit for fins and 0.18/0.22 for gates; width terms are 4.5% + 5.5% for fins and 4% + 5% for gates. These are stress ranges, not a fabrication covariance model. | [Constantoudis et al. 2004](https://doi.org/10.1116/1.1776561)<br>[Xiong and Bokor 2003](https://doi.org/10.1109/TED.2003.818594)<br>[Patel et al. 2009](https://doi.org/10.1109/TED.2009.2032605) |
| Secondary-electron edge response | `src/drift_sense/distortions.py::secondary_electron_edge_response`; alternate morphology in `src/drift_sense/sem_render_alt.py` | Gradient-edge strength is `U[0.12,0.28]`; the alternate path uses 0.55 times that strength on a morphological range response. This is a phenomenological contrast stressor, not Monte Carlo electron transport. | [Li et al. 2013](https://doi.org/10.1002/sca.21042)<br>[Zou et al. 2018](https://doi.org/10.1016/j.measurement.2018.02.069) |
| Finite probe / effective PSF blur | `src/drift_sense/distortions.py::apply_acquisition`; Fourier-domain path in `src/drift_sense/sem_render_alt.py` | Search sigma is `U[0.55,1.0]` sensor pixel; reference sigma is 0.68 times that interval. Gaussian blur is a bounded effective transfer-function stressor, not a stated beam diameter. | [Yano and Nomura 1993](https://doi.org/10.1002/sca.4950150103)<br>[Zotta et al. 2018](https://doi.org/10.1017/S1431927618012412) |
| Poisson/statistical electron counts | `src/drift_sense/distortions.py::apply_acquisition`; reordered in `src/drift_sense/sem_render_alt.py` | Effective peak count is `U[0.8,1.25] × B/n`, with `B=230` for reference and `B=120` for search. `n=f`, multiplied by 1.8 in `high_noise` and 1.25 in `cross_generator`. It is an effective count scale, not calibrated dose. | [Sim et al. 2004](https://doi.org/10.1002/sca.4950260106)<br>[Timischl et al. 2012](https://doi.org/10.1002/sca.20282) |
| Additive Gaussian/electronic noise | `src/drift_sense/distortions.py::apply_acquisition`; reordered before counting in `src/drift_sense/sem_render_alt.py` | Search sigma is `U[0.010,0.025] × n`; reference sigma is 0.68 times that interval, with `n` defined above. The alternate ordering intentionally changes the capture model. Values are stress settings. | [Timischl et al. 2012](https://doi.org/10.1002/sca.20282)<br>[Zietlow and Lindner 2025](https://doi.org/10.1038/s41598-025-85982-4)<br>[Du 2015](https://doi.org/10.1016/j.ultramic.2014.11.012) |
| Global gain and offset | `src/drift_sense/distortions.py::apply_acquisition`; `src/drift_sense/sem_render_alt.py` | Gain is `U[0.88,1.12]`; offset is `U[-0.035,0.035]`. These affine ranges test intensity invariance and do not model detector saturation or a calibrated operating point. | [Everhart and Thornley 1960](https://doi.org/10.1088/0950-7671/37/7/307)<br>[LeBeau and Stemmer 2008](https://doi.org/10.1016/j.ultramic.2008.07.001) |
| Slowly varying intensity field | `src/drift_sense/distortions.py::apply_acquisition`; alternate bilinear field in `src/drift_sense/sem_render_alt.py` | Coefficient is `U[-0.055,0.055]`; the primary path applies a linear x/y field and the alternate adds an x·y term. The bound is a smooth-background stress range, not a calibrated charging or collection-efficiency distribution. | [Watkins et al. 2023](https://doi.org/10.3389/fnins.2023.1281098)<br>[Findlay and LeBeau 2013](https://doi.org/10.1016/j.ultramic.2012.09.001) |
| Scan-line shift / jitter | `src/drift_sense/distortions.py::scan_line_shifts` and `apply_scan_line_shift` | Prefilter draw sigma is `U[0.12,0.34] × j` pixel and Gaussian correlation is `U[3,10]` rows. `j=f`, multiplied by 3 in `scan_distortion` and 1.35 in `cross_generator`; realized RMS is recorded per image. This is a correlated row-shift stress model. | [Maraghechi et al. 2019](https://doi.org/10.1007/s11340-018-00469-w)<br>[Jones and Nellist 2013](https://doi.org/10.1017/S1431927613001402) |
| Scale, rotation, anisotropy, and smooth geometric drift | `src/drift_sense/dataset.py::_geometry_params`; `src/drift_sense/geometry.py::CaptureGeometry` | Per capture: scale is ±0.008`f` around its nominal value (transform OOD: ±0.025`f`); rotation is ±0.35`f`° (transform OOD: ±1.15`f`°); anisotropy is ±0.0015`f` (transform OOD: ±0.006`f`). Linear drift is ±0.45`f` pixel-equivalent (scan-distortion suite: ±1.8`f`), with quadratic coefficient bounded at 0.4 times that limit. These are transform stress ranges. | [Ghosh 1975](https://doi.org/10.1016/0031-8663(75)90008-3)<br>[Marschallinger and Topa 1997](https://doi.org/10.1002/sca.4950190105)<br>[Cizmar et al. 2011](https://doi.org/10.1017/S1431927610094250) |
| Independent reference/search acquisition | `src/drift_sense/dataset.py::_acquisition_params` and `generate_pair` | Reference and search draw independent acquisition seeds; the reference uses a 0.68 noise/blur factor while search uses 1.0. This implements the challenge requirement that search is normally noisier; it is not a physical assertion about every inspection workflow. | [Sim et al. 2004](https://doi.org/10.1002/sca.4950260106)<br>[Timischl et al. 2012](https://doi.org/10.1002/sca.20282)<br>[Jin and Li 2015](https://doi.org/10.1111/jmi.12293) |
| Supersampling, anti-aliasing, and cross-renderer sampling | `src/drift_sense/sem_render.py`; `src/drift_sense/sem_render_alt.py`; routing in `src/drift_sense/dataset.py` | Supersampling factor is an integer ≥1 (default 2). With supersampling enabled, the primary reference/search use different area and Lanczos paths; the alternate pair uses Kaiser- and Hann-windowed polyphase paths. The choices reduce shared interpolation fingerprints; they are not unique physically correct SEM kernels. | [Thévenaz et al. 2000](https://doi.org/10.1109/42.875199)<br>[Seidner 2005](https://doi.org/10.1109/TIP.2005.854493) |

The table's ranges come from the checked-in implementation, not from the linked publications. The detailed claim map below states the narrower evidentiary role of each source.

## 1. Persistent line-edge and line-width variation

**Literature-backed mechanism.** Lithographic line edges have stochastic spatial structure, and line-width variation depends on the two edge processes rather than on one independent pixel-noise field. This supports drawing a specimen geometry once and retaining it across views.

**Implementation abstraction.** Correlated edge displacement and shared line-width/center variation are compact geometric models, not a process-specific resist/etch model. Their spectra, amplitudes, and correlation lengths are challenge stress tests.

1. V. Constantoudis, G. P. Patsis, L. H. A. Leunissen, and E. Gogolides, “Line edge roughness and critical dimension variation: Fractal characterization and comparison using model functions,” *Journal of Vacuum Science & Technology B*, 22(4), 2004. [https://doi.org/10.1116/1.1776561](https://doi.org/10.1116/1.1776561)
2. K. Patel, T.-J. King Liu, and C. J. Spanos, “Gate Line Edge Roughness Model for Estimation of FinFET Performance Variability,” *IEEE Transactions on Electron Devices*, 56(12), 3055–3063, 2009. [https://doi.org/10.1109/TED.2009.2032605](https://doi.org/10.1109/TED.2009.2032605)

## 2. Persistent contact-hole size, edge, and placement variation

**Literature-backed mechanism.** Contact-hole populations exhibit contact-edge roughness, critical-dimension variation, and local placement error. Local placement error is a shift of a printed contact relative to its intended position, distinct from diameter variation.

**Implementation abstraction.** Per-contact center, radius, and rough-boundary perturbations capture these distinct modes. Their distributions and magnitudes are challenge stress tests, not reported process statistics.

1. M. K. Vijaya-Kumar, V. Constantoudis, and E. Gogolides, “Contact Edge Roughness: Characterization and modeling,” *Microelectronic Engineering*, 88(8), 2492–2495, 2011. [https://doi.org/10.1016/j.mee.2011.02.003](https://doi.org/10.1016/j.mee.2011.02.003)
2. S. M. Kim, S. Koo, J.-T. Park, C.-M. Lim, M. Kim, C.-N. Ahn, A. Fumar-Pici, and A. C. Chen, “EUV stochastic noise analysis and LCDU mitigation by etching on dense contact-hole array patterns,” *Proceedings of SPIE*, 9048, 90480A, 2014. The paper separately analyzes local CD uniformity and local placement error. [https://doi.org/10.1117/12.2048282](https://doi.org/10.1117/12.2048282)
3. J. Severi, C. A. Mack, G. F. Lorusso, and D. De Simone, “Measuring and Analyzing Contact Hole Variations in EUV Lithography,” *Proceedings of SPIE*, 11609, 1160913, 2021. [https://doi.org/10.1117/12.2585308](https://doi.org/10.1117/12.2585308)

## 3. Persistent fin and gate width/center variation

**Literature-backed mechanism.** FinFET performance is sensitive to fin/body thickness, gate length, line-edge roughness, and lateral gate-edge mismatch. This supports persistent fin/gate geometric variation in a specimen model.

**Implementation abstraction.** Independent compact perturbations of fin and gate centers and widths are not claimed to reproduce a particular fabrication covariance matrix. Their ranges are challenge stress tests.

1. S. Xiong and J. Bokor, “Sensitivity of Double-Gate and FinFET Devices to Process Variations,” *IEEE Transactions on Electron Devices*, 50(11), 2255–2261, 2003. [https://doi.org/10.1109/TED.2003.818594](https://doi.org/10.1109/TED.2003.818594)
2. K. Patel, T.-J. King Liu, and C. J. Spanos, “Gate Line Edge Roughness Model for Estimation of FinFET Performance Variability,” *IEEE Transactions on Electron Devices*, 56(12), 3055–3063, 2009. [https://doi.org/10.1109/TED.2009.2032605](https://doi.org/10.1109/TED.2009.2032605)

## 4. Secondary-electron edge contrast

**Literature-backed mechanism.** Secondary-electron intensity depends on topography and produces characteristic edge-sensitive line profiles in CD-SEM images. Monte Carlo image-formation work shows that these profiles depend on line geometry, corner rounding, sidewall angle, material, probe, and detection conditions.

**Implementation abstraction.** Adding a scaled image-gradient magnitude is only a fast qualitative edge-response surrogate. It is not a quantitative Monte Carlo treatment of electron transport or detector geometry; its strength is a challenge stress-test parameter.

1. Y. G. Li, P. Zhang, and Z. J. Ding, “Monte Carlo simulation of CD-SEM images for linewidth and critical dimension metrology,” *Scanning*, 35(2), 127–139, 2013. [https://doi.org/10.1002/sca.21042](https://doi.org/10.1002/sca.21042)
2. Y. B. Zou, M. S. S. Khan, H. M. Li, Y. G. Li, W. Li, S. T. Gao, L. S. Liu, and Z. J. Ding, “Use of model-based library in critical dimension measurement by CD-SEM,” *Measurement*, 123, 150–162, 2018. [https://doi.org/10.1016/j.measurement.2018.02.069](https://doi.org/10.1016/j.measurement.2018.02.069)
3. H. Seiler, “Secondary electron emission in the scanning electron microscope,” *Journal of Applied Physics*, 54(11), R1–R18, 1983. This is a foundational survey of emission, detection, contrast, and resolution rather than an original augmentation model. [https://doi.org/10.1063/1.332840](https://doi.org/10.1063/1.332840)

## 5. Finite probe and point-spread blur

**Literature-backed mechanism.** A finite electron beam and instrument response blur SEM images; SEM point-spread functions can be estimated and used for deconvolution or image-quality analysis.

**Implementation abstraction.** An isotropic Gaussian convolution is a deliberately simple effective PSF. Its sigma is not a measured beam diameter or a calibrated modulation-transfer function and is varied only for challenge stress testing.

1. F. Yano and S. Nomura, “Deconvolution of scanning electron microscopy images,” *Scanning*, 15(1), 19–24, 1993. The paper explicitly treats blur caused by finite beam size. [https://doi.org/10.1002/sca.4950150103](https://doi.org/10.1002/sca.4950150103)
2. M. D. Zotta, M. C. Nevins, R. K. Hailstone, and E. Lifshin, “The Determination and Application of the Point Spread Function in the Scanning Electron Microscope,” *Microscopy and Microanalysis*, 24(4), 396–405, 2018. [https://doi.org/10.1017/S1431927618012412](https://doi.org/10.1017/S1431927618012412)

## 6. Poisson/statistical electron noise

**Literature-backed mechanism.** Random incident-electron counts, secondary-electron generation, and cascaded detection produce signal-dependent statistical noise. A Poisson count draw is therefore a useful first-order low-dose/counting abstraction.

**Implementation abstraction.** The configured “dose” is an effective count scale. It is not calibrated to beam current, dwell time, yield, detector efficiency, or a named SEM operating point; its range is a challenge stress test.

1. K. S. Sim, J. T. L. Thong, and J. C. H. Phang, “Effect of shot noise and secondary emission noise in scanning electron microscope images,” *Scanning*, 26(1), 36–40, 2004. [https://doi.org/10.1002/sca.4950260106](https://doi.org/10.1002/sca.4950260106)
2. F. Timischl, M. Date, and S. Nemoto, “A statistical model of signal–noise in scanning electron microscopy,” *Scanning*, 34(3), 137–144, 2012. The model follows noise through five conversion stages of an Everhart–Thornley SEM detector. [https://doi.org/10.1002/sca.20282](https://doi.org/10.1002/sca.20282)

## 7. Additive Gaussian/electronic noise

**Literature-backed mechanism.** Detector and signal-conversion chains add non-counting noise; additive Gaussian noise is a common effective approximation after aggregation or at sufficiently high counts. It must not be confused with the exact low-count Poisson law.

**Implementation abstraction.** Independent zero-mean Gaussian pixels are a coarse nuisance model. They do not reproduce all correlations, readout structure, scintillator statistics, saturation, or nonlinear response. The standard-deviation range is a challenge stress test.

1. F. Timischl, M. Date, and S. Nemoto, “A statistical model of signal–noise in scanning electron microscopy,” *Scanning*, 34(3), 137–144, 2012. [https://doi.org/10.1002/sca.20282](https://doi.org/10.1002/sca.20282)
2. C. Zietlow and J. K. N. Lindner, “An applied noise model for scintillation-based CCD detectors in transmission electron microscopy,” *Scientific Reports*, 15, 3815, 2025. This TEM-detector paper distinguishes quantized-beam statistics from detector contributions and discusses gain nonlinearity; it is used only for generic detector evidence. [https://doi.org/10.1038/s41598-025-85982-4](https://doi.org/10.1038/s41598-025-85982-4)
3. H. Du, “A nonlinear filtering algorithm for denoising HR(S)TEM micrographs,” *Ultramicroscopy*, 151, 62–67, 2015. This electron-microscopy paper explicitly discusses additive Poisson or Gaussian approximations. [https://doi.org/10.1016/j.ultramic.2014.11.012](https://doi.org/10.1016/j.ultramic.2014.11.012)

## 8. Gain and offset variation

**Literature-backed mechanism.** Electron detectors contain amplification stages and background/offset terms; quantitative work calibrates linear response, incident-beam normalization, background, and saturation.

**Implementation abstraction.** A global affine intensity transform is a first-order exposure/electronics nuisance. It omits spatial nonuniformity and nonlinear saturation. Its gain and offset ranges are challenge stress tests.

1. T. E. Everhart and R. F. M. Thornley, “Wide-band detector for micro-microampere low-energy electron currents,” *Journal of Scientific Instruments*, 37(7), 246–248, 1960. This is the original scintillator–photomultiplier detector paper underlying a common SEM secondary-electron detector. [https://doi.org/10.1088/0950-7671/37/7/307](https://doi.org/10.1088/0950-7671/37/7/307)
2. J. M. LeBeau and S. Stemmer, “Experimental quantification of annular dark-field images in scanning transmission electron microscopy,” *Ultramicroscopy*, 108(12), 1653–1658, 2008. This STEM study measures detector linearity, background, saturation, and normalized response; it is used only for generic gain/offset evidence. [https://doi.org/10.1016/j.ultramic.2008.07.001](https://doi.org/10.1016/j.ultramic.2008.07.001)

## 9. Slowly varying intensity field

**Literature-backed mechanism.** Electron-microscopy mosaics can contain within-image brightness gradients and between-image offsets, and detector response can be spatially nonuniform.

**Implementation abstraction.** A multiplicative linear ramp is a low-dimensional smooth-field surrogate. It does not identify whether a particular change came from illumination, collection efficiency, charging, scintillator response, or electronics, and its amplitude is a challenge stress test.

1. P. V. Watkins, E. Jelli, and K. L. Briggman, “msemalign: a pipeline for serial section multibeam scanning electron microscopy volume alignment,” *Frontiers in Neuroscience*, 17, 1281098, 2023. The paper explicitly addresses intra-tile brightness gradients and inter-tile offsets in multibeam SEM data. [https://doi.org/10.3389/fnins.2023.1281098](https://doi.org/10.3389/fnins.2023.1281098)
2. S. D. Findlay and J. M. LeBeau, “Detector non-uniformity in scanning transmission electron microscopy,” *Ultramicroscopy*, 124, 52–60, 2013. This STEM paper provides direct evidence for spatially nonuniform detector response and normalization; it is used only for the generic smooth-field nuisance. [https://doi.org/10.1016/j.ultramic.2012.09.001](https://doi.org/10.1016/j.ultramic.2012.09.001)

## 10. Slow geometric drift

**Literature-backed mechanism.** Relative motion during raster acquisition creates smooth, spatially varying distortion rather than only a rigid between-frame translation. SEM drift correction is therefore an established metrology problem.

**Implementation abstraction.** Linear or quadratic displacement versus scan coordinate is a compact drift trajectory, not a measured thermal or charging time series. Coefficients are challenge stress tests.

1. P. Cizmar, A. E. Vladár, and M. T. Postek, “Real-Time Scanning Charged-Particle Microscope Image Composition with Correction of Drift,” *Microscopy and Microanalysis*, 17(2), 302–308, 2011. [https://doi.org/10.1017/S1431927610094250](https://doi.org/10.1017/S1431927610094250)
2. P. Jin and X. Li, “Correction of image drift and distortion in a scanning electron microscopy,” *Journal of Microscopy*, 260(3), 268–280, 2015. [https://doi.org/10.1111/jmi.12293](https://doi.org/10.1111/jmi.12293)

## 11. Scan-line jitter and line-shift distortion

**Literature-backed mechanism.** Raster acquisition can contain line-dependent position errors (“scan line shifts” or scan noise) that differ from smooth drift and global lens distortion.

**Implementation abstraction.** A correlated per-row horizontal displacement is a parsimonious scan-line model. It is not a full beam-positioning, vibration, mains-frequency, charging, or flyback model; its RMS and correlation are challenge stress tests.

1. S. Maraghechi, J. P. M. Hoefnagels, and R. H. J. Peerlings, “Correction of Scanning Electron Microscope Imaging Artifacts in a Novel Digital Image Correlation Framework,” *Experimental Mechanics*, 59(4), 489–516, 2019. The paper separately models SEM line shifts, static spatial distortion, and drift distortion. [https://doi.org/10.1007/s11340-018-00469-w](https://doi.org/10.1007/s11340-018-00469-w)
2. L. Jones and P. D. Nellist, “Identifying and correcting scan noise and drift in the scanning transmission electron microscope,” *Microscopy and Microanalysis*, 19(4), 1050–1060, 2013. This STEM paper supports the generic line-scan noise mechanism. [https://doi.org/10.1017/S1431927613001402](https://doi.org/10.1017/S1431927613001402)

## 12. Scale, rotation, and other geometric mismatch

**Literature-backed mechanism.** SEM calibration work explicitly treats magnification/scale, rotation, affine deformation, and systematic geometric distortion. Relative scale and rotation must therefore be allowed when matching separately acquired views.

**Implementation abstraction.** Independent global scale, rotation, and optional anisotropy are stress-test nuisances, not calibrated error bars for a specific microscope or stage.

1. S. K. Ghosh, “Photogrammetric calibration of a scanning electron microscope,” *Photogrammetria*, 31(3), 91–114, 1975. The calibration solves for magnification, rotation, and systematic distortion parameters. [https://doi.org/10.1016/0031-8663(75)90008-3](https://doi.org/10.1016/0031-8663(75)90008-3)
2. R. Marschallinger and D. Topa, “Assessment and correction of geometric distortions in low-magnification scanning electron microscopy images,” *Scanning*, 19(1), 36–41, 1997. [https://doi.org/10.1002/sca.4950190105](https://doi.org/10.1002/sca.4950190105)
3. B. S. Reddy and B. N. Chatterji, “An FFT-based technique for translation, rotation, and scale-invariant image registration,” *IEEE Transactions on Image Processing*, 5(8), 1266–1271, 1996. This is algorithmic support for estimating these transformations, not evidence for a particular SEM error magnitude. [https://doi.org/10.1109/83.506761](https://doi.org/10.1109/83.506761)

## 13. Resampling and anti-aliasing

**Literature-backed mechanism.** Geometric transformation requires interpolation, and changing sampling density without appropriate filtering creates aliasing and interpolation artifacts.

**Implementation abstraction.** Supersampled area integration and Lanczos-like resizing are engineering choices to reduce pixel-grid artifacts. The references support filtered resampling in general; they do not establish that the chosen kernel or supersampling factor is uniquely correct for SEM.

1. P. Thévenaz, T. Blu, and M. Unser, “Interpolation Revisited,” *IEEE Transactions on Medical Imaging*, 19(7), 739–758, 2000. [https://doi.org/10.1109/42.875199](https://doi.org/10.1109/42.875199)
2. D. Seidner, “Polyphase Antialiasing in Resampling of Images,” *IEEE Transactions on Image Processing*, 14(11), 1876–1889, 2005. [https://doi.org/10.1109/TIP.2005.854493](https://doi.org/10.1109/TIP.2005.854493)

## 14. Fourier/reciprocal-lattice analysis

**Literature-backed estimator.** Periodic specimen structure produces reciprocal-lattice peaks in the Fourier domain. Peak geometry can be used to estimate lattice bases and spatial variations in periodic structure.

**Implementation abstraction.** Windowing, robust normalization, peak thresholds, symmetry checks, and notch widths are implementation choices tuned and validated by synthetic tests. The papers support the estimator class, not those thresholds.

1. X. Zeng, B. Gipson, Z. Y. Zheng, L. Renault, and H. Stahlberg, “Automatic lattice determination for two-dimensional crystal images,” *Journal of Structural Biology*, 160(3), 353–361, 2007. [https://doi.org/10.1016/j.jsb.2007.08.008](https://doi.org/10.1016/j.jsb.2007.08.008)
2. M. J. Hytch, “Analysis of Variations in Structure from High Resolution Electron Microscope Images by Combining Real Space and Fourier Space Information,” *Microscopy Microanalysis Microstructures*, 8(1), 41–57, 1997. [https://doi.org/10.1051/mmm:1997105](https://doi.org/10.1051/mmm:1997105)

## 15. Phase correlation and subpixel translation

**Literature-backed estimator.** Phase correlation supplies translation estimates in the Fourier domain, and established extensions recover subpixel shifts efficiently.

**Implementation abstraction.** Peak fitting, local upsampling, masks, confidence rules, and fallback logic are engineering choices. Their numerical thresholds are challenge settings rather than universal statistical guarantees.

1. H. Foroosh, J. B. Zerubia, and M. Berthod, “Extension of phase correlation to subpixel registration,” *IEEE Transactions on Image Processing*, 11(3), 188–200, 2002. [https://doi.org/10.1109/83.988953](https://doi.org/10.1109/83.988953)
2. M. Guizar-Sicairos, S. T. Thurman, and J. R. Fienup, “Efficient subpixel image registration algorithms,” *Optics Letters*, 33(2), 156–158, 2008. [https://doi.org/10.1364/OL.33.000156](https://doi.org/10.1364/OL.33.000156)

## 16. Repeated-pattern ambiguity

**Literature-backed limitation.** Repetitive structures create multiple locally plausible correspondences. A strong correlation peak can therefore represent the wrong lattice-equivalent displacement unless nonperiodic evidence or geometric consistency resolves the ambiguity.

**Implementation consequence.** Confidence must reflect competing peaks and lattice-equivalent alternatives; ambiguity thresholds and any nonperiodic weighting are challenge-specific design choices.

1. B. Fan, F. Wu, and Z. Hu, “Towards reliable matching of images containing repetitive patterns,” *Pattern Recognition Letters*, 32(14), 1851–1859, 2011. [https://doi.org/10.1016/j.patrec.2011.07.029](https://doi.org/10.1016/j.patrec.2011.07.029)
2. B. Barrois, M. Konrad, C. Wöhler, and H.-M. Groß, “Resolving stereo matching errors due to repetitive structures using model information,” *Pattern Recognition Letters*, 31(12), 1683–1692, 2010. [https://doi.org/10.1016/j.patrec.2010.05.020](https://doi.org/10.1016/j.patrec.2010.05.020)

## Verification note

Bibliographic metadata above was cross-checked against publisher records and/or authoritative indexes (including PubMed, NIST, institutional publication repositories, and IEEE/Optica/SPIE records). DOI links intentionally use the canonical `https://doi.org/…` form. On 2026-08-16, all 34 unique DOI URLs in this file returned an HTTP 302 response from the DOI resolver, confirming that the identifiers resolve; the redirect is expected because DOI targets are publisher landing pages.
