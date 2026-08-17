# NeuroTaskFM

NeuroTaskFM is a CUDA-native research system for learning relationships among
brain structure, resting-state activity, task function, behavior, and
longitudinal clinical state.

> Research use only. This project is not a medical device and must not be used
> for autonomous diagnosis or treatment decisions.

## Main components

- **NeuroTaskFM-T**: large multimodal teacher model.
- **NeuroCompiler**: deterministic C++/CUDA MRI preprocessing and feature-pack compiler.
- **NeuroTaskFM-C**: compiled-space model designed for two-B200 inference.
- **NeuroSignature**: participant-specific state updated across observations and visits.
- **Go platform**: command, API, job, and bounded compiler-agent services.

The model and GPU tools use C++20, LibTorch, CUDA, cuFFT, cuDNN, NCCL, HDF5,
NIfTI, DCMTK, and OpenCV. Platform services use Go. No Python runtime is
required by the project.

## Hardware

NeuroTaskFM is CUDA-only and targets NVIDIA B200:

- training: 32 B200 GPUs across four NVL8 nodes;
- inference and personalization: two B200 GPUs in one NVLink domain.

There is no CPU or alternate-GPU compute path.

## Build

The recommended build is the DGX B200 container:

```bash
docker build -f deploy/Dockerfile.b200 -t neurotaskfm:0.1.0 .
```

On a configured DGX host with LibTorch under `/opt/libtorch`:

```bash
make build
```

Important binaries:

```text
build/bin/neurocompile
build/bin/ntfm-train
build/bin/ntfm-infer
build/bin/ntfm-personalize
build/bin/ntfm-tool
build/bin/neurotask-web
bin/ntfm
bin/neurotaskd
```

## Prepare data

Each observation is one JSONL manifest row. Keep every task, visit, and
derivative from the same participant or family in one split.

```bash
build/bin/ntfm-tool validate-manifest \
  --manifest /data/manifests/train.jsonl

build/bin/ntfm-tool prepare-packs \
  --manifest /data/manifests/train.jsonl \
  --gpus 0,1,2,3,4,5,6,7
```

The primary numerical inputs are NIfTI volumes and compiled HDF5 feature packs.
Native tools also support DICOM conversion, image/video decoding, HDF5
conversion, volume normalization, and Cartesian CUDA k-space reconstruction:

```bash
build/bin/ntfm-tool --help
```

K-space reconstruction is limited to dense Cartesian 2D/3D inverse FFT with
root-sum-of-squares coil combination. Use a protocol-qualified reconstruction
pipeline for non-Cartesian, parallel-imaging, or compressed-sensing data.

## Compile one observation

```bash
build/bin/neurocompile \
  --t1 /data/sub-0001/t1w.nii.gz \
  --fmri /data/sub-0001/rest_bold.nii.gz \
  --atlas /data/atlas.nii.gz \
  --config configs/compiler/neurocompiler.yaml \
  --subject-key sub-0001 --task rest --tr 0.8 \
  --out /data/compiled/sub-0001_rest.h5
```

## Train

```bash
sbatch workflows/slurm/train_teacher.sbatch
sbatch workflows/slurm/train_compiled.sbatch
sbatch workflows/slurm/train_signature.sbatch
sbatch workflows/slurm/train_clinical_ad.sbatch
sbatch workflows/slurm/train_clinical_pd.sbatch
```

## Infer on two B200 GPUs

```bash
mpirun -np 2 build/bin/ntfm-infer \
  --config configs/deployment/product_b200x2.yaml \
  --request examples/inference_request.json
```

## Go command and service

```bash
bin/ntfm compile --request examples/compile_request.json
bin/ntfm infer --request examples/inference_request.json
NTFM_API_TOKEN='replace-with-a-secret' bin/neurotaskd --listen :8080
```

## Repository guide

```text
src/neurotaskfm_cpp/  LibTorch model, training, inference, and data tools
src/neurocompiler/    C++/CUDA MRI compiler
src/platform/         Go CLI and services
configs/              model, training, deployment, and cluster settings
contracts/            JSON schemas
workflows/            local and Slurm launchers
docs/                 detailed design and operating documentation
```

Read [data preparation](docs/data.md), [training](docs/training.md),
[deployment](docs/deployment.md), and [architecture](docs/architecture.md) for
details.

## Model Release

🔗 **Download the model here:** [markjiang1/NeuroTaskFM](https://huggingface.co/markjiang1/NeuroTaskFM)

⚠️ **DISCLAIMER: This model is for research purposes only. It is strictly NOT intended for clinical, diagnostic, or therapeutic use.**