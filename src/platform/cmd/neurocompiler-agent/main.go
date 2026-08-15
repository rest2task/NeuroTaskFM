package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/ymjiang/neurotaskfm/src/platform/internal/ntfm"
)

type agentConfig struct {
	Model    string `json:"model"`
	Endpoint string `json:"endpoint"`
	Prompt   string `json:"prompt"`
}

func main() {
	configPath := flag.String("config", "configs/agent/gemini.json", "agent config")
	diagnosticsPath := flag.String("diagnostics", "", "probe diagnostics JSON")
	output := flag.String("output", "candidate.json", "candidate output")
	flag.Parse()
	var config agentConfig
	read(*configPath, &config)
	diagnostics := map[string]any{}
	read(*diagnosticsPath, &diagnostics)
	planner, err := ntfm.NewGeminiPlanner(config.Endpoint, config.Model)
	if err != nil {
		fail(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
	defer cancel()
	candidate, err := planner.Propose(ctx, config.Prompt, diagnostics)
	if err != nil {
		fail(err)
	}
	raw, _ := json.MarshalIndent(candidate, "", "  ")
	if err := os.WriteFile(*output, raw, 0o640); err != nil {
		fail(err)
	}
	fmt.Println(*output)
}

func read(path string, dst any) {
	raw, err := os.ReadFile(path)
	if err != nil {
		fail(err)
	}
	if err := json.Unmarshal(raw, dst); err != nil {
		fail(err)
	}
}
func fail(err error) { fmt.Fprintln(os.Stderr, "neurocompiler-agent:", err); os.Exit(1) }
