#!/usr/bin/env bash
# Production manual deployment script for HTR-ViTRNN Stack

set -euo pipefail

cd "$(dirname "$0")/.."

echo "Preparing production stack deployment..."

# 1. Setup local environment if missing
if [[ ! -f .env ]]; then
  echo "Copying .env.example to .env..."
  cp .env.example .env
  echo "IMPORTANT: Please modify .env with secure production values!"
fi

# 2. Make scripts executable
chmod +x scripts/*.sh

# 3. Pull latest docker images or build
echo "Building/building-cache Docker images..."
docker compose -f docker-compose.yml -f docker-compose.prod.yml build

# 4. Stop and recreate running container stack
echo "Deploying container services..."
docker compose -f docker-compose.yml -f docker-compose.prod.yml down --remove-orphans || true
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

echo "Services started successfully!"
docker compose ps
