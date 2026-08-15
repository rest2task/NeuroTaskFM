# Deployment and API

## Hardware contract

NeuroTaskFM deployment requires two NVIDIA B200 GPUs in one NVLink domain. The process is CUDA-only and rejects unsupported devices. The model remains resident between requests in service mode.

Recommended host resources:

- local NVMe input staging;
- sufficient host memory for DICOM/NIfTI parsing and HDF5 queues;
- protected storage mounts;
- TLS termination, authentication, audit logging, and network isolation outside the supplied service.

## Build

The B200 container uses NVIDIA's CUDA Deep Learning image and the official
CUDA LibTorch C++ distribution. The shipped stage contains only C++/CUDA and
Go executables, LibTorch shared objects, and the CUDA deep-learning runtime.

```bash
docker build -f deploy/Dockerfile.b200 -t neurotaskfm:0.1.0 .
```

or on a configured DGX host:

```bash
workflows/build.sh
```

Outputs:

```text
build/bin/neurocompile
build/bin/dcm2niix
build/bin/ntfm-tool
bin/ntfm
bin/neurotaskd
bin/neurocompiler-agent
```

## Compile endpoint

Request schema: `contracts/compile_request.schema.json`

```bash
bin/ntfm compile --request examples/compile_request.json
```

The request specifies T1, fMRI, atlas, optional de-identified DICOM directories, task metadata, TR, compiler config, and output HDF5 path.

## Inference

Request schema: `contracts/inference_request.schema.json`

```bash
bin/ntfm infer --request examples/inference_request.json
```

or directly:

```bash
mpirun -np 2 build/bin/ntfm-infer \
  --config configs/deployment/product_b200x2.yaml \
  --request examples/inference_request.json
```

An inference request can contain one or more compiled observations, an existing NeuroSignature, and a target task/contrast query.

Typical outputs are:

```text
task_effect_mean.nii.gz
task_effect_std.nii.gz
task_effect_template.nii.gz
task_effect_individual_residual.nii.gz
neurosignature.pt
result.json
```

`result.json` contains source observations, query, named behavior and clinical distributions, quality measurements, OOD score, calibration state, and report/partial/abstain status.

## Personalization

Request schema: `contracts/personalization_request.schema.json`

```bash
bin/ntfm personalize --request examples/personalization_request.json
```

Shared model parameters remain frozen. Only the participant signature is optimized. Held-out tasks and future outcomes used for evaluation must not be included.

## Service

```bash
NTFM_API_TOKEN='replace-with-a-secret' bin/neurotaskd --listen :8080
```

Endpoints:

- `POST /v1/compile`
- `POST /v1/infer`
- `POST /v1/personalize`
- `GET /v1/jobs/{id}`
- `GET /healthz`

When `NTFM_API_TOKEN` is set, all endpoints except `/healthz` require a bearer token.

The supplied Go server is a GPU-job orchestrator. It is not a complete multi-tenant clinical security system.

## Runtime profile

`configs/deployment/product_b200x2.yaml` defines a 170-second warm-path budget. A final deployment benchmark should report:

- scan length, TR, and image dimensions;
- compiler, model, and output-writing time separately;
- median, interquartile range, and 95th percentile;
- cold-start and warm-start performance;
- peak GPU memory;
- failure, fallback-to-abstention, and unsuccessful-job rates.

The repository does not encode a fabricated runtime result.

## Calibration and abstention

`artifacts/calibration-v1.json` is a bootstrap record. Fit the final calibration on a development-validation cohort using `ntfm-tool fit-calibration`, inspect subgroup coverage, freeze the file and checksum, and do not refit after test-set access.

The deployment output distinguishes:

- reportable estimates;
- partially constrained domains;
- out-of-distribution observations;
- abstention.

## Model weights

Training checkpoints are not included. The expected distributed checkpoint topology is one local shard per rank during 32-GPU training and two expert partitions for deployment. `ntfm-tool export-expert-checkpoint` and `ntfm-tool repartition-checkpoint` convert between these layouts.

Public weight release should include the exact model config, compiler artifact, calibration file, checkpoint checksums, intended use, validated acquisition range, and limitations. Controlled imaging data must not be redistributed with the repository.

## Agent boundary

The Gemini client is used only for offline bounded compiler planning. Frozen NeuroCompiler execution and clinical inference do not require an external language-model endpoint. Candidate graphs are selected by numerical metrics rather than model prose.
