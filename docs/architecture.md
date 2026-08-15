# Architecture

## Design principle

NeuroTaskFM separates expensive representation discovery from deterministic deployment. A large raw-space teacher learns anatomical–functional correspondence, acquisition invariances, temporal bases, cross-task relationships, and task-conditioned outputs. Stable low-level computations are then compiled into a fixed GPU graph. A second model is retrained in that compiled representation and conditioned on an individualized longitudinal signature.

The system has three principal components:

1. **NeuroTaskFM-T**, the raw-space teacher;
2. **NeuroCompiler**, the deterministic model-aligned image compiler;
3. **NeuroTaskFM-C + NeuroSignature**, the shared clinical model and patient-specific digital-twin state.

## NeuroTaskFM-T

The teacher configuration defines 73.22B total parameters and approximately 15.46B active parameters per observation. It accepts:

- de-identified DICOM token streams;
- tri-planar T1 and EPI slices;
- representative three-dimensional T1 and fMRI volumes;
- optional typed still images, ordered cine/video frames, and mixed-contrast 3D/4D MR resources;
- optional compiled spatial and temporal channels;
- task and directed-contrast queries;
- behavioral, biomarker, and clinical values;
- an existing NeuroSignature.

The backbone contains 48 multimodal blocks. Twelve blocks use sparse top-2 mixture-of-experts feed-forward layers; the remaining blocks use dense SwiGLU layers. Grouped-query attention is interleaved with diagonal state-space blocks for long temporal context. A Perceiver-style resampler bounds the core sequence length when DICOM tokens, slices, volumes, and long functional sequences are combined.

The optional direct-resource front ends are deliberately shallow. A 2D patch encoder handles independent images, a second position-aware 2D encoder handles cine/video, and a position-aware 3D encoder handles mixed contrasts and sampled time volumes. Contrast embeddings preserve source meaning, explicit masks preserve missingness, and balanced sampling prevents one long run from displacing every other declared contrast. All three feed the same observation resampler as the established inputs and are absent when `resources.enabled` is false.

## NeuroCompiler

The teacher is frozen and interrogated using several complementary probes:

- cross-modal attention correspondences;
- perturbation sensitivity;
- gradient sensitivity;
- layer-wise representational similarity;
- spatial and temporal equivariance;
- motion and acquisition sensitivity;
- downstream output consistency.

Attention is evidence, not the sole basis of compilation.

A bounded planning agent can propose candidate operator graphs from an audited vocabulary. It cannot introduce arbitrary executable source code. Candidate graphs are compiled and ranked using representation fidelity, downstream consistency, geometric alignment, temporal stability, failure rate, and runtime.

The native graph supports:

1. robust intensity normalization;
2. low-motion EPI reference construction;
3. T1/EPI edge extraction and residual rigid alignment;
4. per-frame motion estimation;
5. single composed resampling;
6. GPU nuisance projection;
7. parcel and network projection;
8. fixed teacher-derived temporal projection;
9. tri-planar slice extraction;
10. de-identified DICOM tokenization;
11. quality measurement and provenance recording.

Its output is a **model-aligned neuroimaging representation**. It is not a general-purpose substitute for all conventional neuroimaging workflows and is not interpreted directly as a biological activation map.

## NeuroTaskFM-C

The compiled-space model configuration defines 35.40B total parameters and approximately 9.45B active parameters. It contains 40 multimodal blocks and ten sparse top-2 mixture-of-experts blocks. It accepts:

- compiled spatial and temporal channels;
- DICOM tokens;
- image slices;
- optional direct images, cine/video, and typed volumes when trained with matching resource encoders;
- task queries;
- biomarkers and clinical measurements;
- the participant's NeuroSignature.

The model is retrained in four phases:

1. representation alignment with the frozen teacher;
2. self-supervised compiled-space pretraining;
3. output distillation from the teacher;
4. supervised consolidation against real task maps, behavior, event-derived states, quality labels, and longitudinal outcomes.

Real targets remain the primary scientific anchor.

## Functional cognitive digital twin

The digital twin is not a separate per-person network. It is the pair

\[
\mathcal T_{i,v}=(F_\theta,S_{i,v}),
\]

where `F_theta` is the shared NeuroTaskFM-C model and `S_i,v` is the participant's NeuroSignature after all accepted observations through visit `v`.

The signature contains dedicated token groups for:

- stable traits;
- current functional state;
- observed task evidence;
- longitudinal change;
- observation provenance;
- uncertainty.

Each observation updates the signature through quality-gated cross-attention. Multiple tasks from the same visit can be assimilated without concatenating every raw sequence. Visit order remains explicit for longitudinal modeling.

Optional participant refinement freezes all shared weights and optimizes only the signature. Evaluation targets and future visits must remain withheld.

## Output heads

### Task-effect decoder

The signed task-effect estimate is decomposed as

\[
\hat\beta_{i,c}=\hat\beta^{\mathrm{template}}_c+\Delta\hat\beta_{i,c}.
\]

The first term is task-conditioned; the second is participant-conditioned. The decoder also predicts voxel-wise uncertainty. This decomposition supports direct testing against a training-fold task template and prevents evaluation from being reduced to group-map copying.

### Dynamic task-state head

For a task run that was actually acquired, temporal tokens produce task-state probabilities without requiring event timing at inference. Event files may be used to create training labels and evaluation references. The system does not claim an observed temporal sequence for a task that was never performed.

### Optional physics head

Physics-enabled training adds an auxiliary parcel decoder for neural, flow, volume, and deoxyhemoglobin trajectories plus bounded subject-conditioned physiological parameters. It is used only when an observed temporal run is present and does not fabricate dynamics for signature-only task queries. The head and its loss are disabled by default and can be omitted at deployment; the task, behavior, and clinical output semantics are unchanged. The equations, bounds, and pack contract are specified in `docs/training.md`.

### Distribution heads

Behavior, image quality, task fidelity, engagement, clinical state, and longitudinal trajectory are represented by separate probabilistic heads. The implementation does not equate low ability, low engagement, and poor image quality.

## Clinical adaptation

AD and PD post-training uses low-rank adapters, selected upper-layer updates, normative replay, and mechanism-oriented concept heads. Clinical adaptation estimates disease-associated deviations from the normative relationship among anatomy, intrinsic function, task states, and behavior.

The intended formulation is

\[
\hat y_{clinical}=\hat y_{norm}+\Delta_{shared}+\Delta_{disease}.
\]

Mechanism-oriented outputs are prespecified quantitative concepts rather than post hoc attention labels. Longitudinal associations and mediation analyses are treated as mechanism-consistent evidence, not proof of causality.
