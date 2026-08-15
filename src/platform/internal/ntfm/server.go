package ntfm

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"time"
)

type Server struct {
	Runner  *Runner
	Jobs    *JobStore
	MaxBody int64
}

func NewServer(runner *Runner) *Server {
	return &Server{Runner: runner, Jobs: NewJobStore(), MaxBody: 4 << 20}
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func decodeJSON(w http.ResponseWriter, r *http.Request, dst any, max int64) bool {
	r.Body = http.MaxBytesReader(w, r.Body, max)
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(dst); err != nil {
		writeJSON(w, 400, map[string]string{"error": err.Error()})
		return false
	}
	return true
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, 200, map[string]any{"status": "ok", "cuda_only": true})
	})
	mux.HandleFunc("POST /v1/compile", s.compile)
	mux.HandleFunc("POST /v1/infer", s.infer)
	mux.HandleFunc("POST /v1/personalize", s.personalize)
	mux.HandleFunc("GET /v1/jobs/", s.job)
	handler := requestLog(mux)
	if token := strings.TrimSpace(os.Getenv("NTFM_API_TOKEN")); token != "" {
		handler = bearerAuth(handler, token)
	}
	return handler
}

func (s *Server) compile(w http.ResponseWriter, r *http.Request) {
	var req CompileRequest
	if !decodeJSON(w, r, &req, s.MaxBody) {
		return
	}
	job := s.Jobs.New("compile")
	s.Jobs.Run(context.Background(), job, DefaultTimeout("compile"), func(ctx context.Context) (string, []string, error) { return s.Runner.Compile(ctx, req) })
	writeJSON(w, 202, job)
}

func (s *Server) infer(w http.ResponseWriter, r *http.Request) {
	var req InferRequest
	if !decodeJSON(w, r, &req, s.MaxBody) {
		return
	}
	job := s.Jobs.New("infer")
	s.Jobs.Run(context.Background(), job, DefaultTimeout("infer"), func(ctx context.Context) (string, []string, error) { return s.Runner.Infer(ctx, req) })
	writeJSON(w, 202, job)
}

func (s *Server) personalize(w http.ResponseWriter, r *http.Request) {
	var req PersonalizeRequest
	if !decodeJSON(w, r, &req, s.MaxBody) {
		return
	}
	job := s.Jobs.New("personalize")
	s.Jobs.Run(context.Background(), job, DefaultTimeout("personalize"), func(ctx context.Context) (string, []string, error) { return s.Runner.Personalize(ctx, req) })
	writeJSON(w, 202, job)
}

func (s *Server) job(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, "/v1/jobs/")
	if id == "" {
		writeJSON(w, 400, map[string]string{"error": "missing job id"})
		return
	}
	job, ok := s.Jobs.Get(id)
	if !ok {
		writeJSON(w, 404, map[string]string{"error": "job not found"})
		return
	}
	writeJSON(w, 200, job)
}

func bearerAuth(next http.Handler, token string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/healthz" {
			next.ServeHTTP(w, r)
			return
		}
		if r.Header.Get("Authorization") != "Bearer "+token {
			writeJSON(w, http.StatusUnauthorized, map[string]string{"error": "unauthorized"})
			return
		}
		next.ServeHTTP(w, r)
	})
}

func requestLog(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		started := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(started).Round(time.Millisecond))
	})
}

func Listen(ctx context.Context, address string, server *Server) error {
	httpServer := &http.Server{Addr: address, Handler: server.Handler(), ReadHeaderTimeout: 10 * time.Second, IdleTimeout: 60 * time.Second}
	go func() {
		<-ctx.Done()
		shutdown, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		_ = httpServer.Shutdown(shutdown)
	}()
	log.Printf("neurotaskd listening on %s", address)
	if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		return fmt.Errorf("serve: %w", err)
	}
	return nil
}
