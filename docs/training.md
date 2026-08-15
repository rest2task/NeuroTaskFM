# Training

## Cluster topology

The reference configuration uses 32 NVIDIA B200 GPUs across four eight-GPU NVL8 domains.

- expert parallel size: 8 within each NVLink/NVSwitch domain;
- data parallel size: 4 across nodes;
- FSDP sharding across corresponding expert ranks;
- BF16 parameters and activations;
- delayed FP8 linear execution after BF16 stabilization;
- activation checkpointing at every backbone block;
- hierarchical participant-aware sampling.

Training examples are sampled as

\[
\text{dataset}\rightarrow\text{participant/family}\rightarrow\text{visit}\rightarrow\text{run}\rightarrow\text{window}.
\]

Frame-uniform sampling is not used because it would over-weight long scans and short-TR datasets.

## Stage 1: raw-space teacher

Configuration: `configs/training/pretrain_teacher.yaml`

NeuroTaskFM-T is trained for 1.2 million optimizer updates. The configured global batch is 256 observations. Objectives include:

- masked multimodal latent prediction;
- future latent prediction at several temporal offsets;
- T1–fMRI representation alignment;
- same-participant alignment across rest, tasks, and repeated visits;
- signed task-effect and individualized residual prediction;
- behavior, quality, dynamic-state, clinical, and longitudinal outcomes when available;
- sparse-expert load balancing and router stabilization.

The context curriculum gradually increases DICOM tokens, slices, representative fMRI frames, spatial channels, and temporal context.

## Stage 2: teacher probing and NeuroCompiler search

Configuration: `configs/training/compiler_search.yaml`

The selected teacher checkpoint is frozen. Probe and compiler-validation cohorts remain distinct. The optional Gemini planner proposes bounded operator graphs; continuous parameters are optimized numerically. Candidate promotion uses deterministic thresholds for:

- spatial and temporal representation fidelity;
- downstream output consistency;
- alignment and motion accuracy;
- runtime and peak memory;
- failure rate.

The final graph and teacher-derived HDF5 artifact are frozen before compiled-space training.

## Stage 3: compiled-space model

Configuration: `configs/training/train_compiled.yaml`

NeuroTaskFM-C is trained for 800,000 optimizer updates. Each sample can contain DICOM tokens, image slices, compiled spatial and temporal channels, optional clinical values, and raw teacher inputs used only for frozen-teacher distillation.

The objective combines:

- real task, behavior, quality, and longitudinal targets;
- hidden-state alignment with the teacher;
- teacher output distillation;
- future latent prediction;
- incomplete-modality consistency;
- calibration and sparse-router regularization.

Real labels remain the primary anchor so the clinical model can correct teacher errors.

## Stage 4: NeuroSignature training

Configuration: `configs/training/train_signature.yaml`

The 35B shared backbone is frozen while the signature updater and selected output modules are trained on multi-task and longitudinal episodes.

A training episode:

1. selects one participant;
2. withholds a target task or later visit;
3. assimilates a subset of available earlier or same-visit observations;
4. predicts the withheld target;
5. repeats with different subsets and observation orders.

Future visits are never used to construct an earlier signature.

## Stage 5: clinical post-training

Configurations:

- `configs/training/clinical_ad.yaml`
- `configs/training/clinical_pd.yaml`

Clinical post-training updates disease adapters, selected upper blocks, the signature updater, concept heads, clinical heads, and trajectory heads. Normative replay is included in 35% of observations to limit catastrophic forgetting.

Recommended fixed roles are:

- ADNI for AD adaptation and longitudinal discovery;
- OASIS-3/OASIS-4 for external AD transportability;
- PPMI for PD adaptation and longitudinal discovery;
- an independent PD imaging cohort for external replication.

Dataset roles must be fixed before analysis.

## Optimization

The reference configurations use:

- AdamW with fused CUDA kernels;
- gradient clipping at 1.0;
- cosine learning-rate decay;
- RMSNorm and SwiGLU;
- grouped-query attention and diagonal state-space blocks;
- sparse top-2 expert routing with a shared expert;
- dataset-temperature sampling with exponent 0.5;
- modality, task-query, T1, DICOM, slice, temporal, and spatial masking;
- shortened-run simulation and motion corruption;
- partial-observation and held-out-visit episodes.

## Optional heterogeneous MR inputs

Direct MR resources are opt-in and therefore do not alter existing runs, parameter counts, or checkpoint loading. The `resources` block in the teacher and compiled-space configurations is set to `enabled: false`. Enabling it adds three small front ends before the shared observation resampler:

1. a typed 2D encoder for independent MR screenshots, exported views, and slice collections;
2. a typed, position-aware 2D encoder for cine/video frames;
3. a typed, position-aware 3D encoder for structural contrasts, quantitative maps, and sampled 4D acquisitions.

Each encoder normalizes all sources into a fixed token budget, so batches may mix pack-only, NIfTI-only, image-only, video-only, and multimodal observations. Padding masks prevent a missing source in one sample from becoming evidence for that sample.

The primary `t1_nifti` and `fmri_nifti` fields accept both `.nii` and `.nii.gz`. `primary_nifti_destination` controls their route:

- `typed` (recommended) sends them through `mr_volumes`; this works for both teacher and compiled model variants;
- `native` sends them through the teacher's original T1/fMRI volume encoders;
- `both` exposes both routes for a prespecified comparison and intentionally duplicates the evidence.

A typical opt-in block is:

```yaml
resources:
  enabled: true
  include_primary_nifti: true
  primary_nifti_destination: typed
  allow_missing_pack: false
  image_size: [224, 224]
  volume_shape: [96, 112, 96]
  max_images: 48
  max_video_frames: 32
  max_volumes: 8
  max_fmri_volumes: 8
  frame_stride: 1
  image_patch_size: 16
  tokens_per_image: 4
  tokens_per_video_frame: 4
  volume_patch_size: [8, 8, 8]
  tokens_per_volume: 64
  eligible_resources: [dicom, slices, compiled_spatial, compiled_temporal, mr_images, mr_video, mr_volumes]
  sample_combinations: true
  combinations:
    all_available: {weight: 0.30, include: ["*"]}
    structural_multicontrast: {weight: 0.10, include: [slices, mr_images, mr_volumes], modalities: [t1, t2, flair, dwi, swi]}
    functional_sequences: {weight: 0.10, include: [slices, mr_video, mr_volumes], modalities: [bold, fmri, epi, asl, dsc]}
```

`tokens_per_image` and `tokens_per_video_frame` must be perfect squares; `tokens_per_volume` must be a perfect cube. Adaptive spatial pooling covers the complete image or volume instead of retaining only the first patches.

### Resource-combination training

When `sample_combinations` is on, one weighted, usable resource subset is selected separately for every sample and separately for each consistency view. A preset is considered usable when at least one listed source exists for that sample; absent sources are ignored. `"*"` means every currently available eligible source. An optional `modalities` list filters typed slices/images/videos/volumes by MR contrast while leaving format-independent compiler or acquisition tokens intact. The implementation always restores a genuinely available eligible observation if subsequent dropout masks everything.

The released opt-in presets exercise these observation regimes:

| Regime | Included evidence |
|---|---|
| all available | every eligible source on the row |
| generic MR | still images + cine/video + typed volumes |
| single-family | images only, video only, or typed volumes only |
| 2D + 3D | images or video paired with volumes |
| compiler only | compiled spatial + temporal tokens |
| compiler + pixels | compiler tokens paired with images, video, or volumes |
| acquisition + pixels | DICOM acquisition tokens plus direct pixel resources |
| native raw | original T1 + sampled fMRI teacher volumes |
| structural multicontrast | available T1/T2/FLAIR/DWI/SWI/QSM/magnitude/angiographic/parametric pixels |
| functional sequences | available BOLD/fMRI/EPI/perfusion/ASL/DSC/spectroscopy pixels |
| T1 + BOLD | paired structural and functional pixel routes when both exist, or the available member under missingness |

The relative `weight` values in the YAML are sampling weights, not loss coefficients. `eligible_resources` must match the model variant: the compiled model excludes the original `t1_volume` and `fmri_volume` branches but can ingest the same NIfTI files through `mr_volumes`.

For stage 3 distillation, enable compatible resource settings in the stage-1 teacher run and its checkpoint; otherwise leave the feature off. Signature and clinical stages also default to off and should only enable it when their incoming checkpoint contains the corresponding resource encoders. If physics is enabled at the same time, every training batch must still supply compiled temporal tokens and the required parcel BOLD/anatomy fields; direct images or videos do not manufacture those observations.

The full source/shape contract and manifest examples are in [data.md](data.md#direct-mr-resource-routes).

## Optional physics-informed objective

The physics objective is a soft prior over an auxiliary parcel-level trajectory. It is disabled in every released training configuration, so existing runs and checkpoints retain exactly the previous objective. When enabled, a small physics head decodes a neural state (x), a vasoactive signal (s), normalized blood inflow (f), normalized blood volume (v), and normalized deoxyhemoglobin (q) for each atlas parcel and time point.

The useful mental model is a chain:

\[
\text{latent temporal evidence}
\longrightarrow \text{neural activity on the anatomy graph}
\longrightarrow \text{blood flow and oxygenation}
\longrightarrow \text{observed parcel BOLD}.
\]

The full auxiliary objective is

\[
\mathcal L_{\mathrm{physics}}=
0.15\,\mathcal L_{\mathrm{dyn}}+
0.08\,\mathcal L_{\mathrm{BOLD}}+
0.05\,\mathcal L_{\mathrm{stab}}+
0.04\,\mathcal L_{\mathrm{equiv}}.
\]

These four coefficients are locked in `src/neurotaskfm_cpp/src/losses.cpp`. A physics ablation disables a component without redistributing or retuning its weight.

### Neural dynamics and anatomy

For normalized symmetric parcel adjacency \(\widetilde A\), the neural vector field is

\[
\dot x_t=-x_t/\tau_n+g\widetilde A\tanh(x_t)+u_t.
\]

The optional \(u_t\) is an externally supplied neural drive, such as an event-derived input; it is zero when absent. The dynamics loss compares this vector field with the finite-difference derivative \((x_{t+1}-x_t)/\Delta t\) using a masked smooth-L1 residual. The anatomy prior is therefore part of the dynamics operator, not a separately tunable fifth loss. Setting `terms.anatomy: false` removes the graph-coupling term while retaining local relaxation and any supplied drive.

The compiler constructs a non-negative, undirected parcel graph from face contacts in the configured atlas. The loss symmetrically degree-normalizes it and removes self-edges. A study-specific structural-connectivity graph can replace the compiler graph at `/physics/anatomy_graph`, provided it uses the same parcel order.

### Hemodynamics and BOLD observation

The differentiable Balloon--Windkessel residual uses

\[
\begin{aligned}
\dot s &= x-\kappa s-\gamma(f-1), & \dot f &= s,\\
\tau_0\dot v &= f-v^{1/\alpha}, &
\tau_0\dot q &= \frac{f}{E_0}\left[1-(1-E_0)^{1/f}\right]-qv^{1/\alpha-1}.
\end{aligned}
\]

The corresponding observation operator is

\[
\widehat y_{\mathrm{BOLD}}=V_0\left[k_1(1-q)+k_2(1-q/v)+k_3(1-v)\right],
\]

with \(k_1=7E_0\), \(k_2=2\), and \(k_3=2E_0-0.2\). `physics_bold` combines the four hemodynamic equation residuals and the masked discrepancy between this observation and parcel BOLD. By default both predicted and observed series are time-demeaned. `bold_scale` is the explicit conversion from the dimensionless observation operator to the units stored in a study pack; use `100.0` for percent-signal-change targets.

The head maps unconstrained logits through sigmoid or tanh functions, so optimization cannot leave the following physiological domain:

| Quantity | Range |
|---|---:|
| Neural relaxation \(\tau_n\) | 0.05–2.00 s |
| Anatomy coupling \(g\) | 0.00–1.50 s\(^{-1}\) |
| Signal decay \(\kappa\) | 0.40–1.00 s\(^{-1}\) |
| Flow feedback \(\gamma\) | 0.10–0.50 s\(^{-2}\) |
| Transit time \(\tau_0\) | 0.50–3.00 s |
| Grubb exponent \(\alpha\) | 0.20–0.50 |
| Resting extraction \(E_0\) | 0.20–0.60 |
| Resting blood volume \(V_0\) | 0.01–0.08 |
| Normalized \(f,v,q\) | 0.20–3.00 |

The initialization is the physiological resting fixed point: \(x=s=0\) and \(f=v=q=1\).

### Stability and time-shift equivariance

For the symmetrically normalized anatomy operator, the spectral radius is at most one. The sufficient local contraction condition \(g < 1/\tau_n\) motivates `physics_stability`; the loss softly penalizes violations with a small configured margin.

For time-shift equivariance, the temporal encoder processes a cropped copy beginning `equivariance_shift_steps` frames later, with positions reset to zero. `physics_equivariance` aligns the overlapping neural and hemodynamic states with the original output shifted by the same number of frames. This tests time-translation consistency without a second backbone pass.

All residuals respect the temporal and optional parcel masks. Differential residuals require two adjacent valid frames, and the TR is read per sample from feature-pack metadata (or from `physics_dt` when explicitly supplied). Physics arithmetic is evaluated in FP32 even when the surrounding model uses BF16 or FP8.

### Enabling and ablating

Feature-pack version 3 supplies `/compiled/parcel_series` as the parcel BOLD observation and `/physics/anatomy_graph` as atlas adjacency. Optional overrides are `/physics/bold`, `/physics/neural_drive`, `/physics/mask`, and `/physics/dt_seconds`. Packs created by an older compiler must be regenerated or augmented with a matching anatomy graph before the full objective can be enabled.

The reference configuration is deliberately off:

```yaml
loss:
  physics:
    enabled: false                 # change only for a prespecified physics run
    regions: 512                   # must match parcel BOLD and anatomy graph
    lambda_dyn: 0.15               # locked
    lambda_bold: 0.08              # locked
    lambda_stab: 0.05              # locked
    lambda_equiv: 0.04             # locked
    equivariance_shift_steps: 1
    stability_margin: 0.02
    bold_scale: 1.0
    demean_bold: true
    terms:
      dynamics: true
      anatomy: true
      hemodynamics: true
      stability: true
      equivariance: true
```

For ablations, switch exactly one entry under `terms` to `false`. `enabled: false` is the no-physics condition. Enabling anatomy or hemodynamics without the required graph or BOLD observation fails with a clear error instead of silently dropping the term.

## Launching

```bash
sbatch workflows/slurm/train_teacher.sbatch
sbatch workflows/slurm/train_compiled.sbatch
sbatch workflows/slurm/train_signature.sbatch
sbatch workflows/slurm/train_clinical_ad.sbatch
sbatch workflows/slurm/train_clinical_pd.sbatch
```

The shared launcher is `workflows/slurm/launch_training.sh`.

## Planning estimates

| Stage | Configured duration per run |
|---|---:|
| NeuroTaskFM-T pretraining | 55–78 days |
| NeuroCompiler probing and search | 16–26 days |
| NeuroTaskFM-C pretraining | 38–55 days |
| NeuroSignature training | 18–28 days |
| AD post-training | 12–20 days |
| PD post-training | 12–20 days |

With the configured final replicate counts, the sequential programme is approximately 469–709 cluster-days. Scaling experiments, failed runs, architecture ablations, compiler comparisons, external validation, and statistical analysis can extend the full research programme to approximately 18–30 months.

These are capacity-planning estimates, not measured performance claims. Record actual optimizer steps, scan windows, GPU-hours, throughput, interruptions, and final successful runs.

## Model selection

Freeze the compiler, model configuration, final seeds, calibration procedure, subgroup analyses, and output semantics before locked evaluation. Select checkpoints on development-validation and external-validation endpoints, not on the test set.
