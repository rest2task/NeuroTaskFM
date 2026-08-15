package ntfm

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"sync"
	"time"
)

type JobStore struct {
	mu   sync.RWMutex
	jobs map[string]*Job
}

func NewJobStore() *JobStore { return &JobStore{jobs: map[string]*Job{}} }

func jobID() string { var data [8]byte; _, _ = rand.Read(data[:]); return hex.EncodeToString(data[:]) }

func (s *JobStore) New(kind string) *Job {
	job := &Job{ID: jobID(), Kind: kind, Status: "queued", CreatedAt: time.Now().UTC(), Metadata: map[string]string{}}
	s.mu.Lock()
	s.jobs[job.ID] = job
	s.mu.Unlock()
	return job
}

func (s *JobStore) Get(id string) (*Job, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	job, ok := s.jobs[id]
	if !ok {
		return nil, false
	}
	copy := *job
	return &copy, true
}

func (s *JobStore) update(id string, fn func(*Job)) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if job := s.jobs[id]; job != nil {
		fn(job)
	}
}

func (s *JobStore) Run(parent context.Context, job *Job, timeout time.Duration, work func(context.Context) (string, []string, error)) {
	go func() {
		started := time.Now().UTC()
		s.update(job.ID, func(j *Job) { j.Status = "running"; j.StartedAt = &started })
		ctx, cancel := context.WithTimeout(parent, timeout)
		defer cancel()
		output, command, err := work(ctx)
		ended := time.Now().UTC()
		s.update(job.ID, func(j *Job) {
			j.Command, j.Output, j.EndedAt = command, output, &ended
			if err != nil {
				j.Status, j.Error = "failed", err.Error()
			} else {
				j.Status = "complete"
			}
		})
	}()
}
