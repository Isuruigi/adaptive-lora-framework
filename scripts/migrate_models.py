#!/usr/bin/env python3
"""
Model Migration Script

Migrate models and adapters between versions, formats, or storage backends:
- Convert between PEFT versions
- Migrate to/from cloud storage
- Update adapter configurations
- Merge adapters
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List

import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelMigrator:
    """Migrate models and adapters between versions and formats."""
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
    
    def migrate_adapter(
        self,
        source_path: Path,
        target_path: Path,
        target_format: str = 'peft'
    ) -> bool:
        """Migrate a single adapter."""
        logger.info(f"Migrating adapter: {source_path} -> {target_path}")
        
        if self.dry_run:
            logger.info("[DRY RUN] Would migrate adapter")
            return True
        
        # Create target directory
        target_path.mkdir(parents=True, exist_ok=True)
        
        # Copy adapter files
        adapter_files = [
            'adapter_config.json',
            'adapter_model.bin',
            'adapter_model.safetensors',
        ]
        
        for filename in adapter_files:
            src = source_path / filename
            if src.exists():
                dst = target_path / filename
                shutil.copy2(src, dst)
                logger.info(f"  Copied: {filename}")
        
        # Update configuration if needed
        config_path = target_path / 'adapter_config.json'
        if config_path.exists():
            self._update_adapter_config(config_path, target_format)
        
        return True
    
    def _update_adapter_config(
        self,
        config_path: Path,
        target_format: str
    ) -> None:
        """Update adapter configuration for compatibility."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Update PEFT version compatibility
        if target_format == 'peft':
            # Ensure required fields exist
            if 'peft_type' not in config:
                config['peft_type'] = 'LORA'
            
            if 'base_model_name_or_path' not in config:
                config['base_model_name_or_path'] = ''
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"  Updated config: {config_path}")
    
    def migrate_to_s3(
        self,
        local_path: Path,
        s3_uri: str
    ) -> bool:
        """Migrate adapter to S3."""
        logger.info(f"Migrating to S3: {local_path} -> {s3_uri}")
        
        if self.dry_run:
            logger.info("[DRY RUN] Would upload to S3")
            return True
        
        try:
            import boto3
            
            # Parse S3 URI
            # s3://bucket/prefix/
            parts = s3_uri.replace('s3://', '').split('/', 1)
            bucket = parts[0]
            prefix = parts[1] if len(parts) > 1 else ''
            
            s3 = boto3.client('s3')
            
            for file_path in local_path.rglob('*'):
                if file_path.is_file():
                    key = f"{prefix}/{file_path.relative_to(local_path)}"
                    s3.upload_file(str(file_path), bucket, key)
                    logger.info(f"  Uploaded: {file_path.name}")
            
            return True
            
        except ImportError:
            logger.error("boto3 not installed. Run: pip install boto3")
            return False
    
    def migrate_from_s3(
        self,
        s3_uri: str,
        local_path: Path
    ) -> bool:
        """Migrate adapter from S3."""
        logger.info(f"Migrating from S3: {s3_uri} -> {local_path}")
        
        if self.dry_run:
            logger.info("[DRY RUN] Would download from S3")
            return True
        
        try:
            import boto3
            
            parts = s3_uri.replace('s3://', '').split('/', 1)
            bucket = parts[0]
            prefix = parts[1] if len(parts) > 1 else ''
            
            s3 = boto3.client('s3')
            
            local_path.mkdir(parents=True, exist_ok=True)
            
            # List and download objects
            paginator = s3.get_paginator('list_objects_v2')
            
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    relative_key = key[len(prefix):].lstrip('/')
                    local_file = local_path / relative_key
                    
                    local_file.parent.mkdir(parents=True, exist_ok=True)
                    s3.download_file(bucket, key, str(local_file))
                    logger.info(f"  Downloaded: {relative_key}")
            
            return True
            
        except ImportError:
            logger.error("boto3 not installed. Run: pip install boto3")
            return False
    
    def merge_adapters(
        self,
        adapter_paths: List[Path],
        output_path: Path,
        weights: Optional[List[float]] = None
    ) -> bool:
        """Merge multiple adapters into one."""
        logger.info(f"Merging {len(adapter_paths)} adapters -> {output_path}")
        
        if self.dry_run:
            logger.info("[DRY RUN] Would merge adapters")
            return True
        
        try:
            import torch
            
            if weights is None:
                weights = [1.0 / len(adapter_paths)] * len(adapter_paths)
            
            # Load all adapter states
            states = []
            for path in adapter_paths:
                state_path = path / 'adapter_model.bin'
                if not state_path.exists():
                    state_path = path / 'adapter_model.safetensors'
                
                if state_path.exists():
                    state = torch.load(state_path, map_location='cpu')
                    states.append(state)
                    logger.info(f"  Loaded: {path.name}")
            
            if not states:
                logger.error("No adapter states found")
                return False
            
            # Merge states
            merged_state = {}
            for key in states[0].keys():
                merged_state[key] = sum(
                    w * s[key] for w, s in zip(weights, states)
                )
            
            # Save merged adapter
            output_path.mkdir(parents=True, exist_ok=True)
            torch.save(merged_state, output_path / 'adapter_model.bin')
            
            # Copy config from first adapter
            config_src = adapter_paths[0] / 'adapter_config.json'
            if config_src.exists():
                shutil.copy2(config_src, output_path / 'adapter_config.json')
            
            logger.info(f"Merged adapter saved: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Merge failed: {e}")
            return False
    
    def verify_adapter(self, adapter_path: Path) -> Dict[str, Any]:
        """Verify adapter integrity and compatibility."""
        logger.info(f"Verifying adapter: {adapter_path}")
        
        result = {
            'path': str(adapter_path),
            'valid': True,
            'issues': [],
            'info': {}
        }
        
        # Check required files
        required_files = ['adapter_config.json']
        model_files = ['adapter_model.bin', 'adapter_model.safetensors']
        
        for filename in required_files:
            if not (adapter_path / filename).exists():
                result['valid'] = False
                result['issues'].append(f"Missing: {filename}")
        
        has_model = any((adapter_path / f).exists() for f in model_files)
        if not has_model:
            result['valid'] = False
            result['issues'].append("Missing model file")
        
        # Load and check config
        config_path = adapter_path / 'adapter_config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            result['info']['peft_type'] = config.get('peft_type', 'unknown')
            result['info']['r'] = config.get('r', 'unknown')
            result['info']['lora_alpha'] = config.get('lora_alpha', 'unknown')
            result['info']['target_modules'] = config.get('target_modules', [])
        
        # Check model file size
        for model_file in model_files:
            model_path = adapter_path / model_file
            if model_path.exists():
                size_mb = model_path.stat().st_size / (1024 * 1024)
                result['info']['model_size_mb'] = round(size_mb, 2)
                break
        
        if result['valid']:
            logger.info("  Adapter is valid")
        else:
            logger.warn(f"  Issues found: {result['issues']}")
        
        return result


def main():
    parser = argparse.ArgumentParser(description='Model Migration Tool')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Migrate command
    migrate_parser = subparsers.add_parser('migrate', help='Migrate adapter')
    migrate_parser.add_argument('source', type=str, help='Source path')
    migrate_parser.add_argument('target', type=str, help='Target path')
    migrate_parser.add_argument('--format', type=str, default='peft', help='Target format')
    
    # S3 upload command
    upload_parser = subparsers.add_parser('upload', help='Upload to S3')
    upload_parser.add_argument('local_path', type=str, help='Local path')
    upload_parser.add_argument('s3_uri', type=str, help='S3 URI')
    
    # S3 download command
    download_parser = subparsers.add_parser('download', help='Download from S3')
    download_parser.add_argument('s3_uri', type=str, help='S3 URI')
    download_parser.add_argument('local_path', type=str, help='Local path')
    
    # Merge command
    merge_parser = subparsers.add_parser('merge', help='Merge adapters')
    merge_parser.add_argument('--adapters', type=str, nargs='+', required=True, help='Adapter paths')
    merge_parser.add_argument('--output', type=str, required=True, help='Output path')
    merge_parser.add_argument('--weights', type=float, nargs='+', help='Merge weights')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify adapter')
    verify_parser.add_argument('adapter_path', type=str, help='Adapter path')
    
    # Global options
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode')
    
    args = parser.parse_args()
    
    migrator = ModelMigrator(dry_run=args.dry_run)
    
    if args.command == 'migrate':
        migrator.migrate_adapter(
            Path(args.source),
            Path(args.target),
            args.format
        )
    elif args.command == 'upload':
        migrator.migrate_to_s3(Path(args.local_path), args.s3_uri)
    elif args.command == 'download':
        migrator.migrate_from_s3(args.s3_uri, Path(args.local_path))
    elif args.command == 'merge':
        migrator.merge_adapters(
            [Path(p) for p in args.adapters],
            Path(args.output),
            args.weights
        )
    elif args.command == 'verify':
        result = migrator.verify_adapter(Path(args.adapter_path))
        print(json.dumps(result, indent=2))
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
