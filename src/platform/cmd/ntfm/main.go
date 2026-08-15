package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/ymjiang/neurotaskfm/src/platform/internal/ntfm"
)

func readJSON(path string, dst any) {
	raw, err := os.ReadFile(path)
	if err != nil {
		fatal(err)
	}
	if err := json.Unmarshal(raw, dst); err != nil {
		fatal(err)
	}
}
func fatal(err error) { fmt.Fprintln(os.Stderr, "ntfm:", err); os.Exit(1) }

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: ntfm <compile|infer|personalize|train|serve> ...")
		os.Exit(2)
	}
	runner := ntfm.NewRunner("")
	switch os.Args[1] {
	case "compile":
		fs := flag.NewFlagSet("compile", flag.ExitOnError)
		request := fs.String("request", "", "compile request JSON")
		_ = fs.Parse(os.Args[2:])
		var req ntfm.CompileRequest
		readJSON(*request, &req)
		out, _, err := runner.Compile(context.Background(), req)
		if err != nil {
			fatal(err)
		}
		fmt.Print(out)
	case "infer":
		fs := flag.NewFlagSet("infer", flag.ExitOnError)
		request := fs.String("request", "", "inference request JSON")
		_ = fs.Parse(os.Args[2:])
		var req ntfm.InferRequest
		readJSON(*request, &req)
		out, _, err := runner.Infer(context.Background(), req)
		if err != nil {
			fatal(err)
		}
		fmt.Print(out)
	case "personalize":
		fs := flag.NewFlagSet("personalize", flag.ExitOnError)
		request := fs.String("request", "", "personalization request JSON")
		_ = fs.Parse(os.Args[2:])
		var req ntfm.PersonalizeRequest
		readJSON(*request, &req)
		out, _, err := runner.Personalize(context.Background(), req)
		if err != nil {
			fatal(err)
		}
		fmt.Print(out)
	case "train":
		fs := flag.NewFlagSet("train", flag.ExitOnError)
		config := fs.String("config", "", "training config")
		_ = fs.Parse(os.Args[2:])
		out, _, err := runner.Train(context.Background(), *config, 1)
		if err != nil {
			fatal(err)
		}
		fmt.Print(out)
	case "serve":
		runServer(runner, os.Args[2:])
	default:
		fatal(fmt.Errorf("unknown command %q", os.Args[1]))
	}
}

func runServer(runner *ntfm.Runner, args []string) {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	address := fs.String("listen", ":8080", "listen address")
	_ = fs.Parse(args)
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := ntfm.Listen(ctx, *address, ntfm.NewServer(runner)); err != nil {
		fatal(err)
	}
}
