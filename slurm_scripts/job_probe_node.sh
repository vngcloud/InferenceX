#!/bin/bash
#SBATCH --job-name=probe-node
#SBATCH --partition=hardware-hcm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=hgx-h200-02
#SBATCH --time=00:10:00
#SBATCH --output=/shared/phucnlt2/job-logs/probe-node_%j.out
#SBATCH --error=/shared/phucnlt2/job-logs/probe-node_%j.err
hostname; df -h /tmp | tail -1; ls /tmp/phucnlt2 2>&1 | head; ls -d /mnt/hf_hub_cache/models--RedHat* /mnt/hf_hub_cache/models--*gemma-4-31B* 2>&1 | head
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -2
ls /shared/phucnlt2/venv-py310/bin/python* 2>&1 | head -3; ls /shared/phucnlt2/*.txt /shared/phucnlt2/*freeze* 2>/dev/null | head
which uv python3.10 python3 2>&1; python3 --version
