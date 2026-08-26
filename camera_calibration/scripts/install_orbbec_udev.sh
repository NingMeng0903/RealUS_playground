#!/usr/bin/env bash
# Grant the current user libusb access to Orbbec (VID 2bc5), including Gemini
# 0614/0511. Requires sudo once; then unplug/replug the camera (or reboot).
set -euo pipefail

RULES_SRC=""
for c in \
  /media/camp/EXT_DRIVE/envs/camera_calib/lib/python3.10/site-packages/pyorbbecsdk/shared/99-obsensor-libusb.rules \
  "$(python - <<'PY' 2>/dev/null
import pyorbbecsdk, pathlib
print(pathlib.Path(pyorbbecsdk.__file__).parent / "shared" / "99-obsensor-libusb.rules")
PY
)"
do
  if [[ -n "${c}" && -f "${c}" ]]; then
    RULES_SRC="${c}"
    break
  fi
done

if [[ -z "${RULES_SRC}" ]]; then
  echo "99-obsensor-libusb.rules not found. In camera_calib: pip install --no-deps pyorbbecsdk2" >&2
  exit 1
fi

echo "Installing ${RULES_SRC} → /etc/udev/rules.d/99-obsensor-libusb.rules"
sudo cp "${RULES_SRC}" /etc/udev/rules.d/99-obsensor-libusb.rules
sudo udevadm control --reload
sudo udevadm trigger

if ! id -nG "${USER}" | grep -qw video; then
  echo "Adding ${USER} to group video (needed for /dev/video* after re-login)"
  sudo usermod -aG video "${USER}"
fi

echo
echo "Done. Unplug and replug the Orbbec (or reboot), then Open again in Stage 3/4."
echo "If you were just added to group video, log out and back in."
