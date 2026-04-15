import os
import subprocess
import numpy as np
import sounddevice as sd


def find_device_id(name="default"):
    devices = sd.query_devices()
    for d in devices:
        if name in d["name"]:
            return d["index"]
    return None


def get_valid_input_device_id(preferred_name=None):
    """
    Return (device_id, device_info) for an input with max_input_channels >= 1.
    Prefer device name containing preferred_name; else try pulse, default, alsa; else first valid.
    """
    devices = sd.query_devices()
    if preferred_name:
        key = preferred_name.lower()
        for index, dev in enumerate(devices):
            if key in dev.get("name", "").lower() and dev.get("max_input_channels", 0) >= 1:
                return index, dev
    for name in ("pulse", "default", "alsa"):
        dev_id = find_device_id(name)
        if dev_id is not None:
            dev = sd.query_devices(dev_id)
            if dev.get("max_input_channels", 0) >= 1:
                return dev_id, dev
    for index, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) >= 1:
            return index, dev
    return None, None


def get_pulse_default_source():
    """Query Pulse server (host) for Default Source name; confirms which device we record from."""
    try:
        out = subprocess.run(
            ["pactl", "info"],
            capture_output=True,
            text=True,
            timeout=2,
            env={**os.environ},
        )
        if out.returncode != 0:
            return None
        for line in out.stdout.splitlines():
            if line.strip().startswith("Default Source:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


class CircularBuffer:
    def __init__(self, maxsize):
        self.buffer = np.zeros(maxsize, dtype=np.int16)
        self.maxsize = maxsize
        self.size = 0
        self.head = 0
        self.tail = 0

    def put(self, data):
        data_len = len(data)
        if data_len >= self.maxsize:
            self.buffer = data[-self.maxsize:]
            self.head = 0
            self.tail = 0
            self.size = self.maxsize
        else:
            space_right = self.maxsize - self.tail
            if data_len <= space_right:
                self.buffer[self.tail : self.tail + data_len] = data
                self.tail = (self.tail + data_len) % self.maxsize
            else:
                self.buffer[self.tail :] = data[:space_right]
                self.buffer[: data_len - space_right] = data[space_right:]
                self.tail = data_len - space_right
            self.size = min(self.size + data_len, self.maxsize)
            if self.size == self.maxsize:
                self.head = self.tail

    def get(self, size):
        if size > self.size:
            return None
        if size <= self.maxsize - self.head:
            data = self.buffer[self.head : self.head + size].copy()
            self.head = (self.head + size) % self.maxsize
        else:
            data = np.concatenate(
                (
                    self.buffer[self.head :],
                    self.buffer[: size - (self.maxsize - self.head)],
                )
            )
            self.head = size - (self.maxsize - self.head)
        self.size -= size
        return data
