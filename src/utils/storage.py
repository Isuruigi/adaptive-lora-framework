"""
Storage utilities for model and data management.

Features:
- Local and cloud storage (S3, GCS)
- Checkpoint management
- Model versioning
- Async upload/download
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Union

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class StorageMetadata:
    """Metadata for stored objects."""
    
    name: str
    size_bytes: int
    created_at: datetime
    modified_at: datetime
    checksum: Optional[str] = None
    content_type: Optional[str] = None
    custom_metadata: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "checksum": self.checksum,
            "content_type": self.content_type,
            "custom_metadata": self.custom_metadata,
        }


class StorageBackend(ABC):
    """Abstract storage backend."""
    
    @abstractmethod
    async def upload(
        self,
        source: Union[str, Path, BinaryIO],
        destination: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> StorageMetadata:
        """Upload file to storage."""
        pass
    
    @abstractmethod
    async def download(
        self,
        source: str,
        destination: Union[str, Path]
    ) -> Path:
        """Download file from storage."""
        pass
    
    @abstractmethod
    async def delete(self, path: str) -> bool:
        """Delete file from storage."""
        pass
    
    @abstractmethod
    async def list(
        self,
        prefix: str = "",
        max_results: int = 1000
    ) -> List[StorageMetadata]:
        """List files in storage."""
        pass
    
    @abstractmethod
    async def exists(self, path: str) -> bool:
        """Check if file exists."""
        pass


class LocalStorage(StorageBackend):
    """Local filesystem storage."""
    
    def __init__(self, base_path: Union[str, Path]):
        """Initialize local storage.
        
        Args:
            base_path: Base directory for storage.
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    async def upload(
        self,
        source: Union[str, Path, BinaryIO],
        destination: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> StorageMetadata:
        """Copy file to storage location."""
        dest_path = self.base_path / destination
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(source, (str, Path)):
            shutil.copy2(source, dest_path)
        else:
            # File-like object
            with open(dest_path, 'wb') as f:
                shutil.copyfileobj(source, f)
                
        # Save metadata
        if metadata:
            meta_path = dest_path.with_suffix(dest_path.suffix + '.meta.json')
            with open(meta_path, 'w') as f:
                json.dump(metadata, f)
                
        stat = dest_path.stat()
        return StorageMetadata(
            name=destination,
            size_bytes=stat.st_size,
            created_at=datetime.fromtimestamp(stat.st_ctime),
            modified_at=datetime.fromtimestamp(stat.st_mtime),
            checksum=self._compute_checksum(dest_path),
            custom_metadata=metadata or {}
        )
        
    async def download(
        self,
        source: str,
        destination: Union[str, Path]
    ) -> Path:
        """Copy file from storage."""
        source_path = self.base_path / source
        dest_path = Path(destination)
        
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest_path)
        
        return dest_path
        
    async def delete(self, path: str) -> bool:
        """Delete file from storage."""
        file_path = self.base_path / path
        
        try:
            if file_path.is_file():
                file_path.unlink()
                
                # Also delete metadata file if exists
                meta_path = file_path.with_suffix(file_path.suffix + '.meta.json')
                if meta_path.exists():
                    meta_path.unlink()
                    
                return True
            elif file_path.is_dir():
                shutil.rmtree(file_path)
                return True
        except Exception as e:
            logger.error(f"Failed to delete {path}: {e}")
            
        return False
        
    async def list(
        self,
        prefix: str = "",
        max_results: int = 1000
    ) -> List[StorageMetadata]:
        """List files in directory."""
        search_path = self.base_path / prefix
        results = []
        
        if not search_path.exists():
            return results
            
        for path in search_path.rglob("*"):
            if path.is_file() and not path.name.endswith('.meta.json'):
                if len(results) >= max_results:
                    break
                    
                stat = path.stat()
                rel_path = path.relative_to(self.base_path)
                
                results.append(StorageMetadata(
                    name=str(rel_path),
                    size_bytes=stat.st_size,
                    created_at=datetime.fromtimestamp(stat.st_ctime),
                    modified_at=datetime.fromtimestamp(stat.st_mtime)
                ))
                
        return results
        
    async def exists(self, path: str) -> bool:
        """Check if file exists."""
        return (self.base_path / path).exists()
        
    def _compute_checksum(self, path: Path) -> str:
        """Compute MD5 checksum."""
        hash_md5 = hashlib.md5()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()


class S3Storage(StorageBackend):
    """AWS S3 storage."""
    
    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        region: Optional[str] = None
    ):
        """Initialize S3 storage.
        
        Args:
            bucket: S3 bucket name.
            prefix: Key prefix.
            region: AWS region.
        """
        self.bucket = bucket
        self.prefix = prefix
        self.region = region
        self._client = None
        
    async def _get_client(self):
        """Get S3 client."""
        if self._client is None:
            try:
                import aioboto3
                session = aioboto3.Session()
                self._client = await session.client('s3', region_name=self.region).__aenter__()
            except ImportError:
                logger.error("aioboto3 not installed")
                raise RuntimeError("S3 storage requires aioboto3")
        return self._client
        
    async def upload(
        self,
        source: Union[str, Path, BinaryIO],
        destination: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> StorageMetadata:
        """Upload to S3."""
        client = await self._get_client()
        key = f"{self.prefix}/{destination}".lstrip('/')
        
        if isinstance(source, (str, Path)):
            with open(source, 'rb') as f:
                await client.upload_fileobj(
                    f, self.bucket, key,
                    ExtraArgs={'Metadata': metadata or {}}
                )
            size = Path(source).stat().st_size
        else:
            await client.upload_fileobj(
                source, self.bucket, key,
                ExtraArgs={'Metadata': metadata or {}}
            )
            size = source.seek(0, 2)
            source.seek(0)
            
        return StorageMetadata(
            name=destination,
            size_bytes=size,
            created_at=datetime.utcnow(),
            modified_at=datetime.utcnow(),
            custom_metadata=metadata or {}
        )
        
    async def download(
        self,
        source: str,
        destination: Union[str, Path]
    ) -> Path:
        """Download from S3."""
        client = await self._get_client()
        key = f"{self.prefix}/{source}".lstrip('/')
        dest_path = Path(destination)
        
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(dest_path, 'wb') as f:
            await client.download_fileobj(self.bucket, key, f)
            
        return dest_path
        
    async def delete(self, path: str) -> bool:
        """Delete from S3."""
        try:
            client = await self._get_client()
            key = f"{self.prefix}/{path}".lstrip('/')
            await client.delete_object(Bucket=self.bucket, Key=key)
            return True
        except Exception as e:
            logger.error(f"Failed to delete from S3: {e}")
            return False
            
    async def list(
        self,
        prefix: str = "",
        max_results: int = 1000
    ) -> List[StorageMetadata]:
        """List S3 objects."""
        client = await self._get_client()
        full_prefix = f"{self.prefix}/{prefix}".lstrip('/')
        
        results = []
        paginator = client.get_paginator('list_objects_v2')
        
        async for page in paginator.paginate(
            Bucket=self.bucket,
            Prefix=full_prefix,
            MaxKeys=max_results
        ):
            for obj in page.get('Contents', []):
                results.append(StorageMetadata(
                    name=obj['Key'].replace(self.prefix + '/', '', 1),
                    size_bytes=obj['Size'],
                    created_at=obj['LastModified'],
                    modified_at=obj['LastModified']
                ))
                
                if len(results) >= max_results:
                    return results
                    
        return results
        
    async def exists(self, path: str) -> bool:
        """Check if S3 object exists."""
        try:
            client = await self._get_client()
            key = f"{self.prefix}/{path}".lstrip('/')
            await client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False


class CheckpointManager:
    """Manage model checkpoints."""
    
    def __init__(
        self,
        storage: StorageBackend,
        max_checkpoints: int = 5,
        checkpoint_prefix: str = "checkpoints"
    ):
        """Initialize checkpoint manager.
        
        Args:
            storage: Storage backend.
            max_checkpoints: Maximum checkpoints to keep.
            checkpoint_prefix: Storage prefix for checkpoints.
        """
        self.storage = storage
        self.max_checkpoints = max_checkpoints
        self.checkpoint_prefix = checkpoint_prefix
        
    async def save_checkpoint(
        self,
        model_state: Dict[str, Any],
        optimizer_state: Optional[Dict[str, Any]] = None,
        epoch: int = 0,
        step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        name: Optional[str] = None
    ) -> str:
        """Save model checkpoint.
        
        Args:
            model_state: Model state dict.
            optimizer_state: Optimizer state dict.
            epoch: Current epoch.
            step: Current step.
            metrics: Metrics dict.
            name: Optional checkpoint name.
            
        Returns:
            Checkpoint path.
        """
        try:
            import torch
            
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = name or f"checkpoint_epoch{epoch}_step{step}_{timestamp}.pt"
            checkpoint_path = f"{self.checkpoint_prefix}/{checkpoint_name}"
            
            checkpoint = {
                "model_state_dict": model_state,
                "optimizer_state_dict": optimizer_state,
                "epoch": epoch,
                "step": step,
                "metrics": metrics or {},
                "timestamp": timestamp,
            }
            
            # Save to temporary file first
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
                torch.save(checkpoint, f.name)
                temp_path = f.name
                
            # Upload to storage
            await self.storage.upload(
                temp_path,
                checkpoint_path,
                metadata={
                    "epoch": str(epoch),
                    "step": str(step),
                    "timestamp": timestamp,
                }
            )
            
            # Cleanup temp file
            os.unlink(temp_path)
            
            # Cleanup old checkpoints
            await self._cleanup_old_checkpoints()
            
            logger.info(f"Saved checkpoint: {checkpoint_path}")
            return checkpoint_path
            
        except ImportError:
            logger.error("PyTorch required for checkpoint saving")
            raise
            
    async def load_checkpoint(
        self,
        path: str,
        map_location: Optional[str] = None
    ) -> Dict[str, Any]:
        """Load model checkpoint.
        
        Args:
            path: Checkpoint path.
            map_location: Device to map tensors to.
            
        Returns:
            Checkpoint dict.
        """
        try:
            import torch
            import tempfile
            
            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
                local_path = await self.storage.download(path, f.name)
                
            checkpoint = torch.load(local_path, map_location=map_location)
            os.unlink(local_path)
            
            logger.info(f"Loaded checkpoint: {path}")
            return checkpoint
            
        except ImportError:
            logger.error("PyTorch required for checkpoint loading")
            raise
            
    async def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint."""
        checkpoints = await self.storage.list(self.checkpoint_prefix)
        
        if not checkpoints:
            return None
            
        # Sort by modified time, most recent first
        checkpoints.sort(key=lambda x: x.modified_at, reverse=True)
        
        return checkpoints[0].name
        
    async def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints beyond max limit."""
        checkpoints = await self.storage.list(self.checkpoint_prefix)
        
        if len(checkpoints) <= self.max_checkpoints:
            return
            
        # Sort by modified time, oldest first
        checkpoints.sort(key=lambda x: x.modified_at)
        
        to_delete = len(checkpoints) - self.max_checkpoints
        for ckpt in checkpoints[:to_delete]:
            await self.storage.delete(ckpt.name)
            logger.debug(f"Deleted old checkpoint: {ckpt.name}")


class ModelVersioner:
    """Track model versions."""
    
    def __init__(
        self,
        storage: StorageBackend,
        models_prefix: str = "models"
    ):
        """Initialize versioner.
        
        Args:
            storage: Storage backend.
            models_prefix: Storage prefix for models.
        """
        self.storage = storage
        self.models_prefix = models_prefix
        
    async def register_model(
        self,
        model_path: Union[str, Path],
        model_name: str,
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Register model version.
        
        Args:
            model_path: Local path to model.
            model_name: Model name.
            version: Version string (auto-generated if None).
            metadata: Model metadata.
            
        Returns:
            Version string.
        """
        version = version or datetime.utcnow().strftime("v%Y%m%d%H%M%S")
        storage_path = f"{self.models_prefix}/{model_name}/{version}"
        
        # Upload model files
        model_path = Path(model_path)
        
        if model_path.is_file():
            await self.storage.upload(model_path, f"{storage_path}/model.pt")
        else:
            # Upload directory contents
            for file_path in model_path.rglob("*"):
                if file_path.is_file():
                    rel_path = file_path.relative_to(model_path)
                    await self.storage.upload(file_path, f"{storage_path}/{rel_path}")
                    
        # Save metadata
        full_metadata = {
            "model_name": model_name,
            "version": version,
            "registered_at": datetime.utcnow().isoformat(),
            **(metadata or {})
        }
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(full_metadata, f)
            temp_path = f.name
            
        await self.storage.upload(temp_path, f"{storage_path}/metadata.json")
        os.unlink(temp_path)
        
        logger.info(f"Registered model {model_name} version {version}")
        return version
        
    async def get_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """Get all versions of a model."""
        prefix = f"{self.models_prefix}/{model_name}"
        items = await self.storage.list(prefix)
        
        versions = {}
        for item in items:
            parts = item.name.split('/')
            if len(parts) >= 2:
                version = parts[1]
                if version not in versions:
                    versions[version] = {"version": version, "modified_at": item.modified_at}
                    
        return list(versions.values())
        
    async def get_latest_version(self, model_name: str) -> Optional[str]:
        """Get latest version of model."""
        versions = await self.get_model_versions(model_name)
        
        if not versions:
            return None
            
        versions.sort(key=lambda x: x["version"], reverse=True)
        return versions[0]["version"]
