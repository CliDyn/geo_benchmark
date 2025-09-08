#!/bin/bash
#SBATCH --job-name=ollama_%a
#SBATCH --array=1-3
##SBATCH --job-name=ollama        # will be overridden by sbatch --job-name
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL
#SBATCH --account=ba1254
#SBATCH --output=slurm-%j.out
##SBATCH --exclusive

#set -euo pipefail

# --- positional argument ---
#N="${1:?Usage: $0 N}"
N="${SLURM_ARRAY_TASK_ID}"

# resource limits (adjust as needed)
ulimit -s unlimited
# ulimit -c 0

# env
source .venv/bin/activate

# --- choose a free TCP port on this node ---
PORT=$(python - <<'PY'
import socket
s = socket.socket()
s.bind(('', 0))           # OS picks a free port
print(s.getsockname()[1])
s.close()
PY
)

export OLLAMA_HOST="127.0.0.1:${PORT}"
echo "OLLAMA_HOST: $OLLAMA_HOST"

# start ollama in the background and make sure we clean it up on exit
ollama serve & 
#SERVER_PID=$!
#trap 'kill ${SERVER_PID} 2>/dev/null || true' EXIT

# wait for server to be ready (better than a fixed sleep)
for i in {1..30}; do
  if curl -s http://${OLLAMA_HOST}/api/tags >/dev/null; then
    break
  fi
  sleep 2
done

# (optional) quick warm-up call
curl -s http://${OLLAMA_HOST}/api/generate -d '{ "model": "gpt-oss:20b", "prompt": "Hi"}' #>/dev/null || true

date
python climate_llm_benchmark.py "$N"
RC=$?
date

exit "$RC"