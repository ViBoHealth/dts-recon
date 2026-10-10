#!/usr/bin/env bash
# M6 end-to-end on mock data, as a user would run it (docs/USAGE.md). Writes under data/m6/ (gitignored);
# timings and peak memory (/usr/bin/time -l, macOS) to reports/m6/e2e_run.log. Existing scans are reused.
set -euo pipefail
cd "$(dirname "$0")/.."
D=data/m6
mkdir -p "$D" reports/m6
run() {
  echo "\$ $*"
  /usr/bin/time -l "$@" 2> >(grep -vE '^ *[0-9]+ +(average|involuntary|voluntary|page|block|messages|signals|instructions|cycles|peak|swaps)|Warning: for' >&2)
}
{
[ -f "$D/cal_scan/scan.yaml" ] || run .venv/bin/dts mock-scan "$D/cal_scan" --phantom calibration
run .venv/bin/dts calibrate "$D/cal_scan" -o "$D/stage_geometry.yaml"
[ -f "$D/dog_scan/scan.yaml" ] || run .venv/bin/dts mock-scan "$D/dog_scan" --phantom dental
run .venv/bin/dts info "$D/dog_scan"
for m in os-sart asd-pocs dts-fbp; do
  run .venv/bin/dts recon "$D/dog_scan" -o "$D/recon_calibrated" --method "$m" --calibration "$D/stage_geometry.yaml"
done
run .venv/bin/dts recon "$D/dog_scan" -o "$D/recon_encoders" --method os-sart --dicom
} 2>&1 | tee reports/m6/e2e_run.log
