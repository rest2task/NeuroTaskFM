# NeuroTaskFM web interface

The browser interface is served by the native C++ `neurotask-web` binary.
Vite is used only to build the static frontend.

## Build

From the repository root:

```bash
make build
cd web_app
npm ci
npm run build
```

## Run

Return to the repository root, then run:

```bash
build/bin/neurotask-web \
  --root web_app \
  --address 0.0.0.0 \
  --port 8000
```

Open `http://127.0.0.1:8000`.

The convenience launcher starts the same service in GNU Screen:

```bash
web_app/run.sh
screen -r neurotaskfm
screen -S neurotaskfm -X quit
```

Set `NEUROTASKFM_PORT` or `NEUROTASKFM_SCREEN_NAME` to override the defaults.

The native server currently provides static assets, health information, and
basic functional-contrast data. Advanced analysis and external model endpoints
are not implemented by this server.
