"""Read-only Linux discovery. Paths are injectable for hardware-free tests."""

from __future__ import annotations

import platform
import re
from pathlib import Path


def text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def cpu_name(proc: Path = Path("/proc")) -> str:
    try:
        for line in text_file(proc / "cpuinfo").splitlines():
            key, _, value = line.partition(":")
            if key.strip() in {"model name", "Hardware"} and value.strip():
                return value.strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown"


def pci_name(vendor: str, device: str) -> str | None:
    for path in (Path("/usr/share/misc/pci.ids"), Path("/usr/share/hwdata/pci.ids")):
        try:
            active = False
            for line in text_file(path).splitlines():
                if re.match(r"^[0-9a-f]{4}  ", line):
                    active = line[:4] == vendor.removeprefix("0x")
                elif active and line.startswith("\t" + device.removeprefix("0x") + "  "):
                    return line.split("  ", 1)[1].strip()
        except OSError:
            continue
    return None


def discover_gpus(sys: Path = Path("/sys")) -> list[dict[str, str]]:
    devices: list[dict[str, str]] = []
    seen: set[Path] = set()
    for card in sorted((sys / "class/drm").glob("card*")):
        if not re.fullmatch(r"card\d+", card.name):
            continue
        root = (card / "device").resolve()
        if root in seen:
            continue
        try:
            vendor, device = text_file(root / "vendor"), text_file(root / "device")
        except OSError:
            continue
        seen.add(root)
        name = None
        for filename in ("product_name", "product"):
            try:
                name = text_file(root / filename)
                if name:
                    break
            except OSError:
                pass
        name = name or pci_name(vendor, device)
        devices.append(
            {
                "device": root.name,
                "name": name or f"GPU {vendor}:{device}",
                "vendor_id": vendor,
                "device_id": device,
                "driver": (root / "driver").resolve().name,
                "path": str(root),
            }
        )
    return devices
