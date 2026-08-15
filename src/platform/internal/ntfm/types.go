package ntfm

import "time"

type CompileRequest struct {
	RequestID  string  `json:"request_id"`
	T1         string  `json:"t1"`
	FMRI       string  `json:"fmri"`
	Atlas      string  `json:"atlas"`
	T1DICOM    string  `json:"t1_dicom,omitempty"`
	FMRIDICOM  string  `json:"fmri_dicom,omitempty"`
	SubjectKey string  `json:"subject_key"`
	Task       string  `json:"task,omitempty"`
	TRSeconds  float64 `json:"tr_seconds,omitempty"`
	Config     string  `json:"config"`
	Output     string  `json:"output"`
}

type Observation struct {
	Pack     string `json:"pack"`
	Task     string `json:"task,omitempty"`
	Contrast string `json:"contrast,omitempty"`
	Visit    string `json:"visit,omitempty"`
}

type InferRequest struct {
	RequestID    string         `json:"request_id"`
	Pack         string         `json:"pack,omitempty"`
	Observations []Observation  `json:"observations,omitempty"`
	Signature    string         `json:"signature,omitempty"`
	Query        map[string]any `json:"query"`
	OutputDir    string         `json:"output_dir"`
	Config       string         `json:"config"`
}

type PersonalizeRequest struct {
	RequestID    string   `json:"request_id"`
	Packs        []string `json:"packs"`
	Queries      []string `json:"queries"`
	Output       string   `json:"output"`
	Config       string   `json:"config"`
	Steps        int      `json:"steps,omitempty"`
	LearningRate float64  `json:"learning_rate,omitempty"`
}

type Job struct {
	ID        string            `json:"id"`
	Kind      string            `json:"kind"`
	Status    string            `json:"status"`
	Command   []string          `json:"command,omitempty"`
	Output    string            `json:"output,omitempty"`
	Error     string            `json:"error,omitempty"`
	Metadata  map[string]string `json:"metadata,omitempty"`
	CreatedAt time.Time         `json:"created_at"`
	StartedAt *time.Time        `json:"started_at,omitempty"`
	EndedAt   *time.Time        `json:"ended_at,omitempty"`
}

type CompilerCandidate struct {
	CandidateID string         `json:"candidate_id"`
	Rationale   string         `json:"rationale"`
	Graph       map[string]any `json:"graph"`
	Expected    map[string]any `json:"expected_tradeoffs,omitempty"`
}
