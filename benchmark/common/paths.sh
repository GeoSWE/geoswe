# Shell counterpart to common/paths.py -- source this from any run script.
#
#   source "$(dirname "${BASH_SOURCE[0]}")/../../common/paths.sh"
#
# Exports GEOSWE_BENCH_ROOT, GEOSWE_DATA_ROOT, GEOSWE_OUT_ROOT, honouring any
# value already set in the environment. See common/paths.py for what each root
# is for and how large the data ones get.

# Resolve benchmark/ from this file's location unless the caller set it.
if [ -z "${GEOSWE_BENCH_ROOT:-}" ]; then
  _geoswe_paths_self="${BASH_SOURCE[0]:-$0}"
  GEOSWE_BENCH_ROOT="$(cd "$(dirname "$_geoswe_paths_self")/.." && pwd)"
fi
: "${GEOSWE_DATA_ROOT:=$GEOSWE_BENCH_ROOT/data}"
: "${GEOSWE_OUT_ROOT:=$GEOSWE_DATA_ROOT/out}"

export GEOSWE_BENCH_ROOT GEOSWE_DATA_ROOT GEOSWE_OUT_ROOT

# GEOSWE_CASE_DIR: the case folder of the script that sourced this file.
# Run scripts live at <case>/run/x.sh, so it is the caller's parent directory.
if [ -z "${GEOSWE_CASE_DIR:-}" ] && [ -n "${0:-}" ] && [ "$0" != "bash" ] && [ "$0" != "-bash" ]; then
  GEOSWE_CASE_DIR="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)" || GEOSWE_CASE_DIR=""
fi
export GEOSWE_CASE_DIR



# The case scripts import the GeoSWE library as an installed package
# (`pip install geoswe`). If you are running against a source checkout instead,
# point PYTHONPATH at its src/ before launching:
#   export PYTHONPATH=/path/to/GeoSWE/src:$PYTHONPATH

# Make common/paths.py importable from any script without packaging it.
export PYTHONPATH="$GEOSWE_BENCH_ROOT/common${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$GEOSWE_DATA_ROOT" "$GEOSWE_OUT_ROOT"

geoswe_paths_describe() {
  echo "GEOSWE_BENCH_ROOT=$GEOSWE_BENCH_ROOT"
  echo "GEOSWE_DATA_ROOT =$GEOSWE_DATA_ROOT"
  echo "GEOSWE_OUT_ROOT  =$GEOSWE_OUT_ROOT"
}
