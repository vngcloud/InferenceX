#!/bin/bash
# run.sbatch -- does Mooncake work over RDMA on this node? Starts a mooncake_master and one store service on the
# ConnectX-6 (rocep41s0f0), then probe.py writes and reads 4 GiB through it with zero-copy batch put/get and checks
# every byte. Runs each protocol in PROTOS (rdma, then tcp for comparison) with its own master and store. No GPU,
# no engine: a few minutes.
#
# Setup as ai-system/distributed-kvcache-bench/experiments/l3-what-matters-h200/mooncake.sbatch (job 1777, Mooncake
# RDMA through the CX-6): only the CX-6, never the E810s (irdma: <= 1 GiB per registration, needs a shim); the
# store service owns the memory and the client registers no segment; ulimit -l unlimited; the enroot hooks expose
# the RDMA devices (MELLANOX_VISIBLE_DEVICES / IRDMA_VISIBLE_DEVICES).
#
# Copy this directory to /shared and submit from it (the node reads probe.py from $SLURM_SUBMIT_DIR):
#   cp -r utils/serving/mooncake-rdma-test /shared/$USER/ && cd /shared/$USER/mooncake-rdma-test
#   sbatch run.sbatch                     # rdma then tcp on hgx-h200-01
#   PROTOS=rdma SEGMENT=64gb sbatch run.sbatch
#SBATCH -J mc-rdma-test
#SBATCH -p hardware-hcm
#SBATCH -w hgx-h200-01
#SBATCH -c 16
#SBATCH --mem=200G
#SBATCH -t 00:30:00
#SBATCH -o mc-rdma-test-%j.out
set -euo pipefail
ulimit -l unlimited   # slurm copies the login node's ~1 GB soft limit; any larger registration fails with ENOMEM

PROTOS=${PROTOS:-rdma tcp}
DEV=${DEV:-rocep41s0f0}             # the ConnectX-6 Dx (mlx5); rocep41s0f1 is bond0 on the public address
SEGMENT=${SEGMENT:-32gb}            # the store's segment ("gb" = GiB, as Mooncake reads it)
OBJECTS=${OBJECTS:-64}; OBJECT_MB=${OBJECT_MB:-64}; ROUNDS=${ROUNDS:-3}
IMAGE=${IMAGE:-/mnt/containers/lmsysorg_sglang_v0.5.21.sqsh}
HERE=$SLURM_SUBMIT_DIR
OUT=$HERE/out-$SLURM_JOB_ID; mkdir -p $OUT

NETDEV=$(ls /sys/class/infiniband/$DEV/device/net | head -1)
ME=$(ip -4 -o addr show dev "$NETDEV" | awk '{ split($4, a, "/"); print a[1]; exit }')
echo "host=$(hostname) kernel=$(uname -r) memlock=$(ulimit -l) dev=$DEV netdev=$NETDEV ip=$ME"
echo "driver=$(basename "$(readlink /sys/class/infiniband/$DEV/device/driver)") fw=$(cat /sys/class/infiniband/$DEV/fw_ver)"
echo "port state: $(cat /sys/class/infiniband/$DEV/ports/1/state) rate: $(cat /sys/class/infiniband/$DEV/ports/1/rate)"
[ -f $IMAGE ] || { echo "$IMAGE missing on $(hostname)"; exit 1; }

CONTAINER=mc-rdma-test-$(basename $IMAGE .sqsh)
MOUNTS=$HERE:$HERE,/dev/shm
export MELLANOX_VISIBLE_DEVICES=all IRDMA_VISIBLE_DEVICES=all
srun --overlap -N1 -n1 --container-image=$IMAGE --container-name=$CONTAINER --container-mounts=$MOUNTS \
  --container-remap-root --container-writable --no-container-mount-home true
IN="srun --overlap -N1 -n1 -c $SLURM_CPUS_ON_NODE --container-name=$CONTAINER --container-mounts=$MOUNTS
  --container-remap-root --container-writable --no-container-mount-home"
$IN bash -c 'echo "in container: memlock=$(ulimit -l)"; ls /sys/class/infiniband | tr "\n" " "; echo;
  python3 -c "import importlib.metadata as m; print([d.metadata[\"Name\"] + \"==\" + d.version for d in m.distributions() if \"mooncake\" in d.metadata[\"Name\"].lower()])"'

stop() { kill $(jobs -p) 2>/dev/null || true; wait 2>/dev/null || true; }
trap stop EXIT
rc=0
for PROTO in $PROTOS; do
  echo "================ $PROTO"
  export MOONCAKE_PROTOCOL=$PROTO MOONCAKE_DEVICE=$DEV MC_ENABLE_DEST_DEVICE_AFFINITY=1 \
    MOONCAKE_TE_META_DATA_SERVER=http://$ME:8080/metadata MOONCAKE_MASTER=$ME:50051 \
    MOONCAKE_LOCAL_HOSTNAME=$ME MC_TCP_BIND_ADDRESS=$ME
  $IN mooncake_master --enable_http_metadata_server=true --http_metadata_server_port=8080 --client_ttl=300 \
    > $OUT/master-$PROTO.log 2>&1 &
  sleep 5
  $IN env MOONCAKE_GLOBAL_SEGMENT_SIZE=$SEGMENT MOONCAKE_LOCAL_BUFFER_SIZE=0 \
    python3 -m mooncake.mooncake_store_service --port=8081 --max-wait-time 600 > $OUT/store-$PROTO.log 2>&1 &
  for t in $(seq 1 300); do
    grep -q "Mooncake Store Service is running" $OUT/store-$PROTO.log 2>/dev/null && break
    [ $t -lt 300 ] || { echo "store did not come up in 300 s"; tail -n 30 $OUT/store-$PROTO.log; exit 1; }
    sleep 1
  done
  echo "store up ($SEGMENT on $DEV, $ME)"
  $IN python3 $HERE/probe.py --proto $PROTO --device $DEV --host $ME --master $ME:50051 \
    --metadata http://$ME:8080/metadata --objects $OBJECTS --object-mb $OBJECT_MB --rounds $ROUNDS \
    2>&1 | tee $OUT/probe-$PROTO.log || rc=1
  echo "-- transport / errors in the store and probe logs:"
  grep -hE "Using (RDMA|TCP) transport|RDMA device:|Failed|ERROR|error" $OUT/store-$PROTO.log $OUT/probe-$PROTO.log \
    | sed -E 's/^[IWE][0-9]+ [0-9:.]+ +[0-9]+ //' | sort | uniq -c | head -20 || true
  stop
  sleep 3
done
echo "================ summary"
grep -h "^RESULT\|^PROBE" $OUT/probe-*.log || true
exit $rc
