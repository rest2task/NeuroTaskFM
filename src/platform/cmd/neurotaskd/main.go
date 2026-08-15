package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/ymjiang/neurotaskfm/src/platform/internal/ntfm"
)

func main() {
	listen := flag.String("listen", ":8080", "listen address")
	root := flag.String("repo", "", "repository root")
	flag.Parse()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := ntfm.Listen(ctx, *listen, ntfm.NewServer(ntfm.NewRunner(*root))); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
