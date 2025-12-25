#!/bin/bash
# =============================================================================
# Deployment Script
# =============================================================================
# Deploy the Adaptive LoRA system to various environments
# Usage: ./deploy.sh [environment] [options]
# =============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-}"
IMAGE_NAME="adaptive-lora"
VERSION="${VERSION:-latest}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Parse arguments
ENVIRONMENT="${1:-development}"
shift || true

# Validate environment
case "$ENVIRONMENT" in
    development|staging|production)
        log_info "Deploying to: $ENVIRONMENT"
        ;;
    *)
        log_error "Unknown environment: $ENVIRONMENT. Use: development, staging, or production"
        ;;
esac

# Load environment-specific config
CONFIG_FILE="$PROJECT_ROOT/configs/serving/${ENVIRONMENT}.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    log_error "Configuration file not found: $CONFIG_FILE"
fi

log_info "Using configuration: $CONFIG_FILE"

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    
    docker build \
        -t "$IMAGE_NAME:$VERSION" \
        -f "$PROJECT_ROOT/Dockerfile" \
        --build-arg ENVIRONMENT="$ENVIRONMENT" \
        "$PROJECT_ROOT"
    
    log_info "Image built: $IMAGE_NAME:$VERSION"
}

# Push to registry
push_image() {
    if [ -z "$DOCKER_REGISTRY" ]; then
        log_warn "DOCKER_REGISTRY not set, skipping push"
        return
    fi
    
    log_info "Pushing image to registry..."
    
    FULL_TAG="$DOCKER_REGISTRY/$IMAGE_NAME:$VERSION"
    docker tag "$IMAGE_NAME:$VERSION" "$FULL_TAG"
    docker push "$FULL_TAG"
    
    log_info "Image pushed: $FULL_TAG"
}

# Deploy locally
deploy_local() {
    log_info "Deploying locally with Docker Compose..."
    
    cd "$PROJECT_ROOT"
    
    # Use environment-specific compose file if exists
    COMPOSE_FILE="docker-compose.yml"
    if [ -f "docker-compose.$ENVIRONMENT.yml" ]; then
        COMPOSE_FILE="docker-compose.$ENVIRONMENT.yml"
    fi
    
    docker-compose -f "$COMPOSE_FILE" up -d
    
    log_info "Local deployment complete"
    log_info "API available at: http://localhost:8000"
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    KUBE_DIR="$PROJECT_ROOT/kubernetes"
    
    if [ ! -d "$KUBE_DIR" ]; then
        log_warn "Kubernetes manifests not found at: $KUBE_DIR"
        return
    fi
    
    # Apply namespace
    kubectl apply -f "$KUBE_DIR/namespace.yaml" || true
    
    # Apply config maps
    kubectl apply -f "$KUBE_DIR/configmap-$ENVIRONMENT.yaml" || true
    
    # Apply secrets (should already exist)
    # kubectl apply -f "$KUBE_DIR/secrets.yaml"
    
    # Apply deployments
    kubectl apply -f "$KUBE_DIR/deployment.yaml"
    kubectl apply -f "$KUBE_DIR/service.yaml"
    kubectl apply -f "$KUBE_DIR/ingress.yaml"
    
    # Wait for rollout
    kubectl rollout status deployment/adaptive-lora -n adaptive-lora
    
    log_info "Kubernetes deployment complete"
}

# Health check
health_check() {
    log_info "Running health check..."
    
    local URL="http://localhost:8000/health"
    local MAX_RETRIES=30
    local RETRY_INTERVAL=2
    
    for i in $(seq 1 $MAX_RETRIES); do
        if curl -s "$URL" > /dev/null 2>&1; then
            log_info "Health check passed!"
            return 0
        fi
        
        log_info "Waiting for service... ($i/$MAX_RETRIES)"
        sleep $RETRY_INTERVAL
    done
    
    log_error "Health check failed after $MAX_RETRIES attempts"
}

# Rollback
rollback() {
    log_warn "Rolling back deployment..."
    
    if command -v kubectl &> /dev/null; then
        kubectl rollout undo deployment/adaptive-lora -n adaptive-lora
    else
        cd "$PROJECT_ROOT"
        docker-compose down
        docker-compose up -d --no-build
    fi
    
    log_info "Rollback complete"
}

# Main deployment flow
main() {
    log_info "Starting deployment..."
    
    # Build
    build_image
    
    # Push (if registry configured)
    push_image
    
    # Deploy based on environment
    case "$ENVIRONMENT" in
        development)
            deploy_local
            ;;
        staging|production)
            if command -v kubectl &> /dev/null; then
                deploy_kubernetes
            else
                deploy_local
            fi
            ;;
    esac
    
    # Health check
    health_check
    
    log_info "Deployment complete!"
}

# Handle arguments
case "${1:-}" in
    --rollback)
        rollback
        ;;
    --build-only)
        build_image
        ;;
    --health-check)
        health_check
        ;;
    *)
        main
        ;;
esac
