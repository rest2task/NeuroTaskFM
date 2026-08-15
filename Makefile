SHELL := /bin/bash
BUILD ?= build
TORCH_PREFIX ?= /opt/libtorch
.PHONY: build native platform check image clean

build: native platform

native:
	cmake -S . -B $(BUILD) -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=100a -DCMAKE_PREFIX_PATH=$(TORCH_PREFIX)
	cmake --build $(BUILD) -j $${NPROC:-32}

platform:
	mkdir -p bin
	go build -trimpath -ldflags "-s -w" -o bin/ntfm ./src/platform/cmd/ntfm
	go build -trimpath -ldflags "-s -w" -o bin/neurotaskd ./src/platform/cmd/neurotaskd
	go build -trimpath -ldflags "-s -w" -o bin/neurocompiler-agent ./src/platform/cmd/neurocompiler-agent

check:
	$(BUILD)/bin/ntfm-tool check-repo --root .
	gofmt -w src/platform
	go build ./src/platform/...

image:
	docker build -f deploy/Dockerfile.b200 -t neurotaskfm:0.1.0 .

clean:
	rm -rf $(BUILD) bin dist
