## Systemd Autostart Setup for Orchestrator on Edge

This document describes how to configure the Jetson (Ubuntu with systemd) to automatically start the orchestrator stack on boot using `run_all.sh` and a `systemd` service.

### Prerequisites

- Jetson board running Ubuntu with systemd (e.g., Jetson Orin AGX).
- Project checked out at:
  - `/home/acevn/Desktop/project/orchestrator-on-edge`
- Docker and `docker compose` already installed and working.
- `tmux` installed.

### Script Entry Point

The main entry point used by systemd is:

- `run_all.sh`

This script:

- Creates a tmux session (default name: `orch_edge`).
- Starts:
  - A `llama-server` container via `jetson-containers`.
  - The main Docker stack via `docker compose`.
  - Bluetooth setup via `setup_blue.sh`.
- Keeps running while the tmux session exists so that systemd does not kill the tmux processes.

### Service File

Service file path on the system:

- `/etc/systemd/system/orchestrator-on-edge.service`

Recommended contents:

```ini
[Unit]
Description=Orchestrator-on-edge (run_all.sh)
Wants=network-online.target docker.service bluetooth.service
After=network-online.target docker.service bluetooth.service

[Service]
Type=simple
User=acevn
WorkingDirectory=/home/acevn/Desktop/project/orchestrator-on-edge
EnvironmentFile=-/home/acevn/Desktop/project/orchestrator-on-edge/.env
Environment=XDG_RUNTIME_DIR=/run/user/1000
Environment=DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
ExecStart=/home/acevn/Desktop/project/orchestrator-on-edge/run_all.sh
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Initial Setup

From a terminal:

```bash
sudo cp /home/acevn/Desktop/project/orchestrator-on-edge/orchestrator-on-edge.service /etc/systemd/system/orchestrator-on-edge.service
sudo systemctl daemon-reload
sudo systemctl enable orchestrator-on-edge.service
sudo systemctl start orchestrator-on-edge.service
```

### Verifying the Service

Check service status:

```bash
systemctl status orchestrator-on-edge.service
```

Attach to the tmux session created by `run_all.sh`:

```bash
tmux attach -t "orch_edge"
```

If the session does not exist, check logs:

```bash
journalctl -u orchestrator-on-edge.service -e
```

### Stopping and Disabling Autostart

To stop the service and prevent it from starting automatically on boot:

```bash
sudo systemctl stop orchestrator-on-edge.service
sudo systemctl disable orchestrator-on-edge.service
```

To completely remove the service file:

```bash
sudo rm /etc/systemd/system/orchestrator-on-edge.service
sudo systemctl daemon-reload
sudo systemctl reset-failed
```
