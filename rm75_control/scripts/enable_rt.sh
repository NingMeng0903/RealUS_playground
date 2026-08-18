#!/usr/bin/env bash
# Best-effort realtime setup for the 200 Hz arm thread.
# Needs sudo.  After this, log out/in so ulimit -r picks up the new rtprio.
set -euo pipefail

USER_NAME="${SUDO_USER:-${USER:-$(id -un)}}"
LIMITS_FILE=/etc/security/limits.d/99-rm75-rt.conf
UDEV_FILE=/etc/udev/rules.d/99-cpu-dma-latency.rules

if [[ "$(id -u)" -ne 0 ]]; then
  echo "enable_rt.sh: re-running under sudo" >&2
  exec sudo -- "$0" "$@"
fi

cat >"$LIMITS_FILE" <<EOF
# rm75_control: SCHED_FIFO for the 200 Hz QPIK thread
${USER_NAME}    hard    rtprio      95
${USER_NAME}    soft    rtprio      95
${USER_NAME}    hard    memlock     unlimited
${USER_NAME}    soft    memlock     unlimited
EOF
echo "wrote $LIMITS_FILE (rtprio 95, memlock unlimited for ${USER_NAME})"

if [[ -e /dev/cpu_dma_latency ]]; then
  chmod a+rw /dev/cpu_dma_latency || true
fi
cat >"$UDEV_FILE" <<'EOF'
KERNEL=="cpu_dma_latency", MODE="0666"
EOF
udevadm control --reload-rules >/dev/null 2>&1 || true
echo "wrote $UDEV_FILE (cpu_dma_latency world-writable)"

if [[ -d /sys/devices/system/cpu ]]; then
  for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    [[ -w "$gov" ]] || continue
    echo performance >"$gov" || true
  done
  echo "set cpufreq governors to performance"
fi

echo
echo "Now: log out/in, then check:"
echo "  ulimit -r          # expect 95"
echo "  cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
echo "  ls -l /dev/cpu_dma_latency"
echo "Optional: isolate a core (isolcpus=N nohz_full=N) and set timing.control_cpu: N"
