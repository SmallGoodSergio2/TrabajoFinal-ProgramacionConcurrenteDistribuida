
set -euo pipefail
cd "$(dirname "$0")/.."

NET=cc65-ml
NODES=${NODES:-5}
DATA=${DATA:-/data/DenunciasPoliciales_1000000.csv}
NAMES=(coordinador $(seq -f "nodo-%g" 1 "$NODES"))

if [[ "${1:-}" == "down" ]]; then
  podman rm -f --ignore "${NAMES[@]}"
  podman network rm -f "$NET" || true
  exit 0
fi

podman build -t localhost/cc65-coordinador --build-arg CMD=coordinador -f Containerfile .
podman build -t localhost/cc65-nodo        --build-arg CMD=nodo        -f Containerfile .

podman network exists "$NET" || podman network create "$NET"
podman rm -f --ignore "${NAMES[@]}"

# ":Z" reetiqueta el volumen para SELinux (Fedora/RHEL); en otros sistemas es inofensivo.
VOL="$PWD/Dataset:/data:ro,Z"

# No se publica ningún puerto: el 9000 (TCP) es solo interno de la red $NET.
podman run -d --name coordinador --network "$NET" -v "$VOL" \
  localhost/cc65-coordinador \
  -listen=:9000 -nodes="$NODES" -data="$DATA" -trees=30 -hold

for i in $(seq 1 "$NODES"); do
  podman run -d --name "nodo-$i" --hostname "nodo-$i" --network "$NET" \
    -e NODE_ID="nodo-$i" --restart=on-failure -v "$VOL" \
    localhost/cc65-nodo -coord=coordinador:9000 -workers=2
done

echo
echo "Listo. Para ver el entrenamiento y la latencia:  podman logs -f coordinador"
echo "Probar tolerancia a fallos:                       podman stop nodo-3"
echo "Apagar todo:                                      ./scripts/run_podman.sh down"
