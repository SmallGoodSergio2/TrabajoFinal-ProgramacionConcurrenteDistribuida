#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

NET=cc65-ml
NODES=${NODES:-5}
DATA=${DATA:-/data/DenunciasPoliciales_1000000.csv}
NAMES=(mongo redis api $(seq -f "nodo-%g" 1 "$NODES"))

if [[ "${1:-}" == "down" ]]; then
  podman rm -f --ignore "${NAMES[@]}" 2>/dev/null || true
  podman network rm -f "$NET" 2>/dev/null || true
  exit 0
fi

podman build -t localhost/cc65-api  --build-arg CMD=api  -f Containerfile .
podman build -t localhost/cc65-nodo --build-arg CMD=nodo -f Containerfile .

podman network exists "$NET" || podman network create "$NET"
podman rm -f --ignore "${NAMES[@]}" 2>/dev/null || true

VOL="$PWD/Dataset:/data:ro,Z"

podman run -d --name mongo --network "$NET" -p 27017:27017 \
  docker.io/library/mongo:7

podman run -d --name redis --network "$NET" -p 6379:6379 \
  docker.io/library/redis:7-alpine

sleep 3

podman run -d --name api --network "$NET" -p 8080:8080 -v "$VOL" \
  localhost/cc65-api \
  -http=:8080 -tcp=:9000 -data="$DATA" \
  -mongo=mongodb://mongo:27017 -redis=redis:6379 \
  -nodes="$NODES" -trees=30 -auto-train

for i in $(seq 1 "$NODES"); do
  podman run -d --name "nodo-$i" --hostname "nodo-$i" --network "$NET" \
    -e NODE_ID="nodo-$i" --restart=on-failure -v "$VOL" \
    localhost/cc65-nodo -coord=api:9000 -workers=2
done

echo
echo "Swagger:  http://localhost:8080/swagger/"
echo "Health:   curl http://localhost:8080/health"
echo "Logs:     podman logs -f api"
echo "Down:     ./scripts/run_local.sh down"
