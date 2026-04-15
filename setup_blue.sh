#!/bin/bash

set -u

ENV_FILE=".env"

_read_env_var() {
  local env_file="$1"
  local key="$2"
  [[ -f "$env_file" ]] || return 1
  local line
  line="$(
    sed -nE "s/^[[:space:]]*${key}[[:space:]]*=[[:space:]]*(.*)[[:space:]]*$/\\1/p" "$env_file" | tail -n 1
  )"
  [[ -n "${line:-}" ]] || return 1
  line="${line%\"}"
  line="${line#\"}"
  line="${line%\'}"
  line="${line#\'}"
  printf '%s' "$line"
}

if [[ -f "$ENV_FILE" ]]; then
  SPEAKER_MAC_ADDRESS="${SPEAKER_MAC_ADDRESS:-$(_read_env_var "$ENV_FILE" SPEAKER_MAC_ADDRESS || true)}"
  SUDO_PWD="${SUDO_PWD:-$(_read_env_var "$ENV_FILE" SUDO_PWD || true)}"
fi

: "${SPEAKER_MAC_ADDRESS:?SPEAKER_MAC_ADDRESS must be set (check your .env file)}"

### ---- CONFIG ----
MAC_UNDERSCORE=${SPEAKER_MAC_ADDRESS//:/_}
CARD="bluez_card.${MAC_UNDERSCORE}"
SOURCE="bluez_source.${MAC_UNDERSCORE}.handsfree_head_unit"
SINK="bluez_sink.${MAC_UNDERSCORE}.handsfree_head_unit"
A2DP_MON="bluez_sink.${MAC_UNDERSCORE}.a2dp_sink.monitor"
HFP_MON="bluez_sink.${MAC_UNDERSCORE}.handsfree_head_unit.monitor"
### -----------------
 
echo "[INFO] Checking Bluetooth block state..."
if rfkill list bluetooth | grep -q "Soft blocked: yes"; then
  echo "[INFO] Bluetooth is blocked. Unblocking..."
  echo "$SUDO_PWD" | sudo -S rfkill unblock bluetooth
  sleep 2
fi
 
echo "[INFO] Restarting PulseAudio and Bluetooth..."
pulseaudio -k || true
systemctl --user restart pulseaudio
echo "$SUDO_PWD" | sudo -S systemctl restart bluetooth
sleep 5

echo "[INFO] Connecting to Bluetooth device $SPEAKER_MAC_ADDRESS..."

bluetoothctl << EOF
power on
agent on
default-agent
trust $SPEAKER_MAC_ADDRESS
pair $SPEAKER_MAC_ADDRESS
connect $SPEAKER_MAC_ADDRESS
EOF

sleep 5

echo "[INFO] Please PRESS the Bluetooth button on the speaker"
echo "[INFO] Waiting for Bluetooth audio device..."
 
### ---- MAIN LOOP ----
while true; do
  SOURCES="$(pactl list sources short)"
 
  # Case 1: Only A2DP exists → force HFP
  if echo "$SOURCES" | grep -q "$A2DP_MON" && \
     ! echo "$SOURCES" | grep -q "$HFP_MON"; then
    echo "[INFO] A2DP detected only → forcing handsfree profile"
    pactl set-card-profile "$CARD" handsfree_head_unit || true
    sleep 3
  fi
 
  # Case 2: HFP monitor exists → configure mic + sink
  if echo "$SOURCES" | grep -q "$HFP_MON"; then
    echo "[INFO] Handsfree profile detected"
 
    pactl set-default-source "$SOURCE" || true
    pactl set-default-sink "$SINK" || true

    pactl set-source-volume "$SOURCE" 100% || true
    pactl set-sink-volume "$SINK" 100% || true
 
    # Final verification
    if pactl list sources short | grep -q "$SOURCE"; then
      parec -d "$SOURCE" > /dev/null 2>&1 || true
      echo "[DONE] Device connected and configured"
      echo "Output: A2DP | Input: Headset microphone"
      pactl list sources short
      break
    fi
  fi
 
  sleep 4
done

