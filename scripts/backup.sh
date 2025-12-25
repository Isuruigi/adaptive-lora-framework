#!/bin/bash
# =============================================================================
# Backup Script
# =============================================================================
# Backup adapters, checkpoints, configurations, and data
# Usage: ./backup.sh [backup_type] [options]
# =============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Backup settings
BACKUP_DIR="${BACKUP_DIR:-$PROJECT_ROOT/backups}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RETENTION_DAYS=${RETENTION_DAYS:-30}

# S3 settings (optional)
S3_BUCKET="${S3_BUCKET:-}"
S3_PREFIX="${S3_PREFIX:-adaptive-lora/backups}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Backup adapters
backup_adapters() {
    log_info "Backing up adapters..."
    
    local ADAPTERS_DIR="$PROJECT_ROOT/models/adapters"
    local BACKUP_FILE="$BACKUP_DIR/adapters_$TIMESTAMP.tar.gz"
    
    if [ -d "$ADAPTERS_DIR" ]; then
        tar -czf "$BACKUP_FILE" -C "$(dirname "$ADAPTERS_DIR")" "$(basename "$ADAPTERS_DIR")"
        log_info "Adapters backup: $BACKUP_FILE ($(du -h "$BACKUP_FILE" | cut -f1))"
    else
        log_warn "Adapters directory not found: $ADAPTERS_DIR"
    fi
}

# Backup checkpoints
backup_checkpoints() {
    log_info "Backing up checkpoints..."
    
    local CHECKPOINTS_DIR="$PROJECT_ROOT/checkpoints"
    local BACKUP_FILE="$BACKUP_DIR/checkpoints_$TIMESTAMP.tar.gz"
    
    if [ -d "$CHECKPOINTS_DIR" ]; then
        tar -czf "$BACKUP_FILE" -C "$(dirname "$CHECKPOINTS_DIR")" "$(basename "$CHECKPOINTS_DIR")"
        log_info "Checkpoints backup: $BACKUP_FILE ($(du -h "$BACKUP_FILE" | cut -f1))"
    else
        log_warn "Checkpoints directory not found: $CHECKPOINTS_DIR"
    fi
}

# Backup configuration
backup_configs() {
    log_info "Backing up configurations..."
    
    local CONFIGS_DIR="$PROJECT_ROOT/configs"
    local BACKUP_FILE="$BACKUP_DIR/configs_$TIMESTAMP.tar.gz"
    
    if [ -d "$CONFIGS_DIR" ]; then
        tar -czf "$BACKUP_FILE" -C "$(dirname "$CONFIGS_DIR")" "$(basename "$CONFIGS_DIR")"
        log_info "Configs backup: $BACKUP_FILE ($(du -h "$BACKUP_FILE" | cut -f1))"
    else
        log_warn "Configs directory not found: $CONFIGS_DIR"
    fi
}

# Backup data
backup_data() {
    log_info "Backing up data..."
    
    local DATA_DIR="$PROJECT_ROOT/data"
    local BACKUP_FILE="$BACKUP_DIR/data_$TIMESTAMP.tar.gz"
    
    if [ -d "$DATA_DIR" ]; then
        # Exclude large raw files, only backup processed data
        tar -czf "$BACKUP_FILE" \
            --exclude="$DATA_DIR/raw/*" \
            -C "$(dirname "$DATA_DIR")" "$(basename "$DATA_DIR")"
        log_info "Data backup: $BACKUP_FILE ($(du -h "$BACKUP_FILE" | cut -f1))"
    else
        log_warn "Data directory not found: $DATA_DIR"
    fi
}

# Backup router
backup_router() {
    log_info "Backing up router..."
    
    local ROUTER_DIR="$PROJECT_ROOT/outputs/router"
    local BACKUP_FILE="$BACKUP_DIR/router_$TIMESTAMP.tar.gz"
    
    if [ -d "$ROUTER_DIR" ]; then
        tar -czf "$BACKUP_FILE" -C "$(dirname "$ROUTER_DIR")" "$(basename "$ROUTER_DIR")"
        log_info "Router backup: $BACKUP_FILE ($(du -h "$BACKUP_FILE" | cut -f1))"
    else
        log_warn "Router directory not found: $ROUTER_DIR"
    fi
}

# Upload to S3
upload_to_s3() {
    if [ -z "$S3_BUCKET" ]; then
        log_info "S3_BUCKET not set, skipping S3 upload"
        return
    fi
    
    log_info "Uploading to S3..."
    
    for file in "$BACKUP_DIR"/*_$TIMESTAMP.tar.gz; do
        if [ -f "$file" ]; then
            aws s3 cp "$file" "s3://$S3_BUCKET/$S3_PREFIX/$(basename "$file")"
            log_info "Uploaded: $(basename "$file")"
        fi
    done
}

# Cleanup old backups
cleanup_old_backups() {
    log_info "Cleaning up backups older than $RETENTION_DAYS days..."
    
    find "$BACKUP_DIR" -name "*.tar.gz" -mtime +$RETENTION_DAYS -delete
    
    log_info "Cleanup complete"
}

# Full backup
full_backup() {
    log_info "Starting full backup..."
    
    backup_adapters
    backup_checkpoints
    backup_configs
    backup_data
    backup_router
    
    upload_to_s3
    cleanup_old_backups
    
    log_info "Full backup complete!"
    log_info "Backup location: $BACKUP_DIR"
    
    # List backups
    log_info "Backup files:"
    ls -lh "$BACKUP_DIR"/*_$TIMESTAMP.tar.gz 2>/dev/null || true
}

# Restore from backup
restore() {
    local BACKUP_FILE="$1"
    
    if [ -z "$BACKUP_FILE" ]; then
        log_error "Usage: ./backup.sh --restore <backup_file>"
    fi
    
    if [ ! -f "$BACKUP_FILE" ]; then
        log_error "Backup file not found: $BACKUP_FILE"
    fi
    
    log_warn "This will overwrite existing files. Continue? (y/N)"
    read -r confirm
    
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        log_info "Restore cancelled"
        exit 0
    fi
    
    log_info "Restoring from: $BACKUP_FILE"
    
    tar -xzf "$BACKUP_FILE" -C "$PROJECT_ROOT"
    
    log_info "Restore complete"
}

# List backups
list_backups() {
    log_info "Available backups:"
    
    if [ -d "$BACKUP_DIR" ]; then
        ls -lhtr "$BACKUP_DIR"/*.tar.gz 2>/dev/null || log_warn "No backups found"
    else
        log_warn "Backup directory not found: $BACKUP_DIR"
    fi
}

# Main
case "${1:-full}" in
    adapters)
        backup_adapters
        ;;
    checkpoints)
        backup_checkpoints
        ;;
    configs)
        backup_configs
        ;;
    data)
        backup_data
        ;;
    router)
        backup_router
        ;;
    full)
        full_backup
        ;;
    --restore)
        restore "$2"
        ;;
    --list)
        list_backups
        ;;
    --cleanup)
        cleanup_old_backups
        ;;
    *)
        echo "Usage: $0 [adapters|checkpoints|configs|data|router|full|--restore|--list|--cleanup]"
        exit 1
        ;;
esac
