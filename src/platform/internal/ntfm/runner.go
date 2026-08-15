package ntfm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

type Runner struct {
	RepoRoot       string
	MPIRun         string
	InferBin       string
	PersonalizeBin string
	TrainBin       string
	Compiler       string
}

func NewRunner(root string) *Runner {
	if root == "" {
		root, _ = os.Getwd()
	}
	bin := filepath.Join(root, "build", "bin")
	return &Runner{RepoRoot: root, MPIRun: valueOr("NTFM_MPIRUN", "mpirun"), InferBin: valueOr("NTFM_INFER", filepath.Join(bin, "ntfm-infer")), PersonalizeBin: valueOr("NTFM_PERSONALIZE", filepath.Join(bin, "ntfm-personalize")), TrainBin: valueOr("NTFM_TRAIN", filepath.Join(bin, "ntfm-train")), Compiler: valueOr("NTFM_COMPILER", filepath.Join(bin, "neurocompile"))}
}

func valueOr(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

func (r *Runner) environment() []string {
	env := append([]string{}, os.Environ()...)
	env = append(env, "CUDA_DEVICE_ORDER=PCI_BUS_ID", "NCCL_ASYNC_ERROR_HANDLING=1")
	return env
}

func (r *Runner) run(ctx context.Context, args []string) (string, error) {
	if len(args) == 0 {
		return "", fmt.Errorf("empty command")
	}
	cmd := exec.CommandContext(ctx, args[0], args[1:]...)
	cmd.Dir = r.RepoRoot
	cmd.Env = r.environment()
	var output bytes.Buffer
	cmd.Stdout = &output
	cmd.Stderr = &output
	err := cmd.Run()
	text := output.String()
	if err != nil {
		return text, fmt.Errorf("%s: %w\n%s", strings.Join(args, " "), err, text)
	}
	return text, nil
}

func (r *Runner) Compile(ctx context.Context, req CompileRequest) (string, []string, error) {
	args := []string{r.Compiler, "--t1", req.T1, "--fmri", req.FMRI, "--atlas", req.Atlas, "--config", req.Config, "--out", req.Output, "--subject-key", req.SubjectKey}
	if req.T1DICOM != "" {
		args = append(args, "--t1-dicom", req.T1DICOM)
	}
	if req.FMRIDICOM != "" {
		args = append(args, "--fmri-dicom", req.FMRIDICOM)
	}
	if req.Task != "" {
		args = append(args, "--task", req.Task)
	}
	if req.TRSeconds > 0 {
		args = append(args, "--tr", strconv.FormatFloat(req.TRSeconds, 'f', -1, 64))
	}
	out, err := r.run(ctx, args)
	return out, args, err
}

func (r *Runner) Infer(ctx context.Context, req InferRequest) (string, []string, error) {
	if err := os.MkdirAll(req.OutputDir, 0o750); err != nil {
		return "", nil, err
	}
	requestPath := filepath.Join(req.OutputDir, "request.json")
	raw, _ := json.MarshalIndent(req, "", "  ")
	if err := os.WriteFile(requestPath, raw, 0o640); err != nil {
		return "", nil, err
	}
	args := []string{r.MPIRun, "-np", "2", r.InferBin, "--config", req.Config, "--request", requestPath, "--out", req.OutputDir}
	out, err := r.run(ctx, args)
	return out, args, err
}

func (r *Runner) Personalize(ctx context.Context, req PersonalizeRequest) (string, []string, error) {
	args := []string{r.MPIRun, "-np", "2", r.PersonalizeBin, "--config", req.Config, "--output", req.Output}
	for _, pack := range req.Packs {
		args = append(args, "--pack", pack)
	}
	for _, query := range req.Queries {
		args = append(args, "--query", query)
	}
	if req.Steps > 0 {
		args = append(args, "--steps", strconv.Itoa(req.Steps))
	}
	if req.LearningRate > 0 {
		args = append(args, "--learning-rate", strconv.FormatFloat(req.LearningRate, 'g', -1, 64))
	}
	out, err := r.run(ctx, args)
	return out, args, err
}

func (r *Runner) Train(ctx context.Context, config string, nodes int) (string, []string, error) {
	if nodes != 1 {
		return "", nil, fmt.Errorf("multi-node jobs are launched through Slurm scripts")
	}
	args := []string{r.MPIRun, "-np", "8", r.TrainBin, "--config", config}
	out, err := r.run(ctx, args)
	return out, args, err
}

func DefaultTimeout(kind string) time.Duration {
	switch kind {
	case "compile", "infer", "personalize":
		return 30 * time.Minute
	default:
		return 48 * time.Hour
	}
}
