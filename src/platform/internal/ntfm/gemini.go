package ntfm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

type GeminiPlanner struct {
	Endpoint string
	APIKey   string
	Model    string
	Client   *http.Client
}

func NewGeminiPlanner(endpoint, model string) (*GeminiPlanner, error) {
	key := os.Getenv("GEMINI_API_KEY")
	if key == "" {
		return nil, fmt.Errorf("GEMINI_API_KEY is required")
	}
	return &GeminiPlanner{Endpoint: endpoint, APIKey: key, Model: model, Client: &http.Client{Timeout: 10 * time.Minute}}, nil
}

func (g *GeminiPlanner) Propose(ctx context.Context, prompt string, diagnostics map[string]any) (CompilerCandidate, error) {
	payload := map[string]any{
		"contents":         []any{map[string]any{"role": "user", "parts": []any{map[string]any{"text": prompt}, map[string]any{"text": mustJSON(diagnostics)}}}},
		"generationConfig": map[string]any{"temperature": 0.2, "maxOutputTokens": 32768, "responseMimeType": "application/json"},
	}
	raw, _ := json.Marshal(payload)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, g.Endpoint, bytes.NewReader(raw))
	if err != nil {
		return CompilerCandidate{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-goog-api-key", g.APIKey)
	response, err := g.Client.Do(req)
	if err != nil {
		return CompilerCandidate{}, err
	}
	defer response.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(response.Body, 16<<20))
	if response.StatusCode/100 != 2 {
		return CompilerCandidate{}, fmt.Errorf("gemini status %d: %s", response.StatusCode, body)
	}
	var envelope struct {
		Candidates []struct {
			Content struct {
				Parts []struct {
					Text string `json:"text"`
				} `json:"parts"`
			} `json:"content"`
		} `json:"candidates"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil {
		return CompilerCandidate{}, err
	}
	if len(envelope.Candidates) == 0 || len(envelope.Candidates[0].Content.Parts) == 0 {
		return CompilerCandidate{}, fmt.Errorf("empty Gemini response")
	}
	var candidate CompilerCandidate
	if err := json.Unmarshal([]byte(envelope.Candidates[0].Content.Parts[0].Text), &candidate); err != nil {
		return CompilerCandidate{}, fmt.Errorf("invalid candidate JSON: %w", err)
	}
	if candidate.CandidateID == "" || len(candidate.Graph) == 0 {
		return CompilerCandidate{}, fmt.Errorf("candidate is missing id or graph")
	}
	return candidate, nil
}

func mustJSON(value any) string { raw, _ := json.Marshal(value); return string(raw) }
