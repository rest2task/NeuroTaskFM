# Data and preprocessing

## Input contract

One JSONL manifest row represents one imaging observation: a resting-state run, task run, or visit-specific acquisition. Multiple rows may share a participant and visit.

The formal schema is `contracts/manifest.schema.json`; an example is provided in `examples/manifest.jsonl`.

Core fields include:

- dataset, site, participant, family, visit, and run identifiers;
- task and directed contrast;
- optional T1 and fMRI NIfTI paths, including uncompressed `.nii` and compressed `.nii.gz`;
- optional de-identified T1 and fMRI DICOM directories;
- atlas path, TR, acquisition metadata, and split;
- target files and an optional output feature-pack path;
- zero or more typed `mr_resources` for images, videos, volumes, DICOM pixels, arrays, or k-space.

A row must provide at least one of `output_pack`, `t1_nifti`, `fmri_nifti`, or `mr_resources`. The standard feature pack remains the source of compiler features and attached supervised targets. With `resources.enabled: true`, direct resources can supplement a pack or form a resource-only self-supervised observation.

## Direct MR resource routes

The native training loader consumes NIfTI volumes, OpenCV-decoded still images,
ordered image directories, and cine/video. Quantitative arrays, DICOM, and
k-space are converted before training so expensive reconstruction never runs in
data-loader workers.

| Manifest `kind` | Accepted sources | Model interpretation |
|---|---|---|
| `image` | OpenCV-readable PNG, JPEG, TIFF, or BMP | one normalized 2D MR view |
| `image_series` | directory of OpenCV-readable images | ordered sampled frames |
| `video` | OpenCV/FFmpeg-readable cine or video | ordered sampled frames |
| `volume` | `.nii` or `.nii.gz` | one native 3D acquisition |
| `volume_series` | 4D `.nii` or `.nii.gz` | sampled ordered 3D frames |
| `dicom` / `dicom_series` | de-identified DICOM directory | convert with the bundled C++ dcm2niix executable |
| `array` | numeric HDF5 dataset | convert with `hdf5-to-nifti` |
| `kspace` | complex HDF5 dataset | centered CUDA inverse FFT, root-sum-of-squares coil combination, then NIfTI |

Uncompressed and compressed NIfTI are read through the native NIfTI C library.

Example resource declarations:

```json
{
  "t1_nifti": "anat/sub-01_T1w.nii",
  "fmri_nifti": "func/sub-01_task-rest_bold.nii",
  "mr_resources": [
    {"path": "anat/sub-01_FLAIR.nii.gz", "kind": "volume", "modality": "flair"},
    {"path": "perf/sub-01_asl.nii", "kind": "volume_series", "modality": "asl", "frame_axis": 3},
    {"path": "raw/sub-01_kspace.h5", "kind": "kspace", "modality": "magnitude", "dataset": "/kspace", "coil_axis": 0, "reconstruction_dims": 2, "layout": "image"}
  ]
}
```

Paths may be absolute or relative to the manifest. `dataset` selects an HDF5
array. Complex arrays may use an HDF5 compound with `real`/`imaginary` (or
`r`/`i`) members, a trailing dimension of length two, or separate datasets
selected with `--real-dataset` and `--imag-dataset`. Spatial axes must be the
trailing two or three axes; optional frame and coil axes precede them.

The loader robustly clips and normalizes frames and NIfTI volumes, resizes them
to the native model grids, samples long series deterministically, and pads
batches with explicit masks. Lossless quantitative inputs are preferred.

`examples/multiresource_manifest.jsonl` shows mixed and resource-only rows. `examples/mr_resource_index.json` can be supplied to `ntfm-tool build-manifest --resource-index`; accepted lookup keys are sample ID, `subject/visit/task`, `subject/visit`, or subject, in that priority order.

Training resolves relative resources against the manifest directory. Validation
and k-space conversion accept `--resource-root` for an external raw-data tree;
the converted manifest records model-ready outputs explicitly.

## General processing algorithm

### 1. Secure ingest and de-identification

- De-identify DICOM before it reaches the training repository.
- Preserve only acquisition fields required for geometry, protocol control, and quality analysis.
- Keep protected identifiers outside model inputs and manifests.
- Confirm that each source agreement permits the intended compute environment and derived-data handling.

### 2. Source conversion

- For the compiler path, convert T1 and fMRI pixel data to NIfTI using a validated site workflow.
- Direct training may additionally retain lossless rendered slices and cine/video exports in `mr_resources`; DICOM, HDF5 arrays, and k-space are converted first.
- Retain de-identified DICOM directories for the compiler acquisition-token branch; a direct DICOM resource supplies pixels, not the compiler's whitelisted metadata token stream.
- Record converter version, phase-encoding direction, TR, field strength, scanner, and site.

NeuroTaskFM does not attempt to decode every vendor-private compressed DICOM representation. Unsupported or encrypted vendor raw formats should be converted through a validated site pipeline before they are declared as a direct resource.

### Native conversion commands

```bash
# Maintained C++ DICOM converter, built into the B200 image.
build/bin/ntfm-tool dicom-to-nifti \
  --input /data/dicom/series --output-dir /data/nifti

# Canonical numeric array interchange.
build/bin/ntfm-tool nifti-to-hdf5 \
  --input /data/anat.nii.gz --output /data/anat.h5 --dataset /volume
build/bin/ntfm-tool hdf5-to-nifti \
  --input /data/anat.h5 --dataset /volume --output /data/anat.nii.gz

# B200 CUDA/cuFFT Cartesian reconstruction. Input defaults to /kspace with
# [frame,coil,(depth),height,width] logical order.
build/bin/ntfm-tool kspace-reconstruct \
  --input /data/raw.h5 --dataset /kspace --fft-dims 3 \
  --frame-axis 0 --coil-axis 1 --output /data/recon.nii.gz

# Convert every k-space resource once and rewrite the manifest to NIfTI.
build/bin/ntfm-tool convert-kspace-manifest \
  --manifest /data/manifests/raw.jsonl --output-root /data/reconstructed \
  --output-manifest /data/manifests/reconstructed.jsonl

build/bin/ntfm-tool normalize-volume \
  --input /data/recon.nii.gz --output /data/recon_norm.nii.gz --mode robust
build/bin/ntfm-tool inspect-data --input /data/recon.nii.gz
build/bin/ntfm-tool inspect-data \
  --input /data/raw.h5 --dataset /kspace --complex
```

The bundled k-space path is Cartesian FFT reconstruction, not a replacement for
GRAPPA, SENSE, compressed sensing, trajectory correction, density compensation,
or vendor-specific calibration. Convert vendor raw data to a validated HDF5/MRD
array and use a protocol-qualified reconstruction when those operations are
required.

### 3. Corpus inventory

Report separately:

- unique participants and families;
- visits and sessions;
- resting-state and task runs;
- tasks and directed contrasts;
- T1 examinations;
- BOLD volumes and scan hours;
- sites, scanners, field strengths, TRs, and spatial resolutions.

BOLD volumes are nested observations, not independent participants. The intended foundation corpus contains more than 35 million three-dimensional BOLD volumes, corresponding to over 12,000 hours of fMRI, together with T1-weighted examinations. The final inventory must be generated from frozen manifests.

### 4. Leakage-safe partitioning

Partition before temporal-window sampling or feature-pack generation.

1. Group by family identifier where available, otherwise participant identifier.
2. Assign complete groups to training, development-validation, external validation, or locked test partitions.
3. Keep all modalities, tasks, visits, derivatives, and repeated runs for one participant in the same partition.
4. Reserve complete sites or datasets for transportability evaluation.
5. Fix ADNI, OASIS, PPMI, and independent-cohort roles before clinical adaptation.
6. Freeze manifests and store checksums.

`ntfm-tool validate-manifest` checks split isolation, missing files, corpus counts, and optional HDF5 packs.

### 5. Reference targets

Targets are constructed outside NeuroCompiler and attached to feature packs:

- signed task-effect maps from a prespecified GLM or FIR workflow;
- task-state labels derived from event timing with an HRF-tolerant target construction;
- in-scanner and out-of-scanner behavioral outcomes;
- deterministic imaging-quality measurements and adjudicated labels;
- clinical and future outcomes;
- prespecified mechanism concepts.

Event timing is used only to create supervision and reference labels. It is withheld during event-log-free evaluation.

### 6. NeuroCompiler feature packs

For each observation, NeuroCompiler:

1. loads T1, fMRI, atlas, and optional DICOM inputs;
2. robustly normalizes intensities;
3. constructs a low-motion EPI reference;
4. estimates residual EPI–T1 alignment and per-frame motion;
5. composes transforms and resamples the fMRI sequence once;
6. performs GPU nuisance projection;
7. computes parcel time series;
8. derives an undirected face-adjacency graph from the atlas labels;
9. applies the frozen spatial and temporal compiler basis;
10. extracts tri-planar T1 and EPI slices;
11. creates bounded DICOM token streams;
12. writes quality values, provenance, and representations to HDF5.

The output pack is the standard interface between the native compiler and the neural training/inference stack. Direct MR resources can be fused with it when enabled. Version 3 stores nuisance-residual parcel BOLD at `/compiled/parcel_series` and the matching atlas graph at `/physics/anatomy_graph`. The native LibTorch dataset loads these larger arrays only when the optional physics loss is enabled.

### 7. Teacher-only raw channels

`ntfm-tool pack-raw-inputs` adds a fixed-grid T1 volume and representative fMRI frames for NeuroTaskFM-T. Resizing runs on CUDA. These channels are disabled for the compiled-space clinical model.

### 8. Task-template residuals

`ntfm-tool build-map-residuals` builds training-fold task templates and participant-specific residual targets. Templates must be derived only from training participants for each task and contrast.

## Typical preparation commands

```bash
build/bin/ntfm-tool build-manifest \
  --root /data/bids --dataset study-a --atlas /data/atlas.nii.gz \
  --compiled-root /data/compiled --resource-index /data/mr-resources.json \
  --output /data/manifests/foundation_train.jsonl
build/bin/ntfm-tool validate-manifest --manifest /data/manifests/foundation_train.jsonl
build/bin/ntfm-tool prepare-packs --manifest /data/manifests/foundation_train.jsonl --gpus 0,1,2,3,4,5,6,7
build/bin/ntfm-tool pack-raw-inputs --manifest /data/manifests/foundation_train.jsonl
build/bin/ntfm-tool attach-targets --manifest /data/manifests/foundation_train.jsonl --index /data/targets/index.json
build/bin/ntfm-tool build-map-residuals --train-manifest /data/manifests/foundation_train.jsonl --artifact /data/targets/task-templates.h5 --overwrite
```

Omit `--compiled-root` to create direct-NIfTI/resource-backed rows without an expected pack. For a manifest that names packs which are not present for every row, explicitly set `resources.allow_missing_pack: true`; missing packs otherwise fail fast.

## Data governance

This repository does not redistribute source imaging, protected health information, or controlled clinical records. Users are responsible for obtaining data from the original custodians and complying with consent, de-identification, data-use, export, cloud/API, and publication conditions.
