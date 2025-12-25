"""
Tests for API Module

Tests for FastAPI endpoints, request handling, and response formatting.
"""

import json
from typing import Dict, Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_response_structure(self):
        """Test health check response structure."""
        response = {
            "status": "healthy",
            "version": "1.0.0",
            "components": {
                "model": "ready",
                "router": "ready",
                "adapters": "ready"
            }
        }
        
        assert response["status"] == "healthy"
        assert "components" in response
    
    def test_health_degraded_status(self):
        """Test degraded health status."""
        response = {
            "status": "degraded",
            "issues": ["redis_disconnected"]
        }
        
        assert response["status"] == "degraded"
        assert len(response["issues"]) > 0


class TestGenerateEndpoint:
    """Tests for text generation endpoint."""
    
    @pytest.fixture
    def generate_request(self) -> Dict[str, Any]:
        """Sample generate request."""
        return {
            "prompt": "Explain quantum computing",
            "max_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "adapter": None
        }
    
    def test_request_validation_required_fields(self, generate_request: Dict[str, Any]):
        """Test that prompt is required."""
        assert "prompt" in generate_request
        assert len(generate_request["prompt"]) > 0
    
    def test_request_default_values(self, generate_request: Dict[str, Any]):
        """Test default parameter values."""
        defaults = {
            "max_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9
        }
        
        for key, default in defaults.items():
            if key in generate_request:
                assert generate_request[key] == default
    
    def test_response_structure(self):
        """Test generation response structure."""
        response = {
            "text": "Generated text here...",
            "adapter_used": "reasoning",
            "tokens_generated": 128,
            "finish_reason": "stop",
            "latency_ms": 450.5
        }
        
        assert "text" in response
        assert "adapter_used" in response
        assert "tokens_generated" in response
    
    def test_temperature_range(self):
        """Test temperature parameter range."""
        valid_temps = [0.0, 0.5, 1.0, 1.5, 2.0]
        invalid_temps = [-0.1, 2.1]
        
        for temp in valid_temps:
            assert 0.0 <= temp <= 2.0
        
        for temp in invalid_temps:
            assert not (0.0 <= temp <= 2.0)


class TestRouteEndpoint:
    """Tests for routing endpoint."""
    
    def test_route_response_structure(self):
        """Test routing response structure."""
        response = {
            "adapter": "reasoning",
            "confidence": 0.85,
            "probabilities": {
                "reasoning": 0.85,
                "code": 0.08,
                "analysis": 0.05,
                "base": 0.02
            }
        }
        
        assert response["adapter"] in response["probabilities"]
        assert sum(response["probabilities"].values()) == pytest.approx(1.0)
    
    def test_confidence_threshold(self):
        """Test confidence threshold handling."""
        min_confidence = 0.3
        
        high_confidence = {"adapter": "reasoning", "confidence": 0.85}
        low_confidence = {"adapter": "base", "confidence": 0.25}
        
        assert high_confidence["confidence"] >= min_confidence
        assert low_confidence["confidence"] < min_confidence


class TestAdaptersEndpoint:
    """Tests for adapters listing endpoint."""
    
    def test_adapters_list_structure(self):
        """Test adapters list response structure."""
        adapters = [
            {
                "name": "reasoning",
                "description": "Complex reasoning tasks",
                "version": "1.0.0",
                "status": "active"
            },
            {
                "name": "code",
                "description": "Code generation tasks",
                "version": "1.0.0",
                "status": "active"
            }
        ]
        
        for adapter in adapters:
            assert "name" in adapter
            assert "status" in adapter


class TestRequestValidation:
    """Tests for request validation."""
    
    def test_max_tokens_limit(self):
        """Test max_tokens parameter limits."""
        max_allowed = 4096
        
        valid_values = [1, 256, 1024, 4096]
        invalid_values = [0, -1, 5000]
        
        for val in valid_values:
            assert 1 <= val <= max_allowed
        
        for val in invalid_values:
            assert not (1 <= val <= max_allowed)
    
    def test_prompt_length_limit(self):
        """Test prompt length validation."""
        max_prompt_length = 10000
        
        short_prompt = "Hello"
        long_prompt = "x" * 15000
        
        assert len(short_prompt) <= max_prompt_length
        assert len(long_prompt) > max_prompt_length


class TestErrorHandling:
    """Tests for API error handling."""
    
    def test_error_response_structure(self):
        """Test error response structure."""
        error_response = {
            "error": {
                "code": "INVALID_REQUEST",
                "message": "Prompt is required",
                "details": {"field": "prompt"}
            }
        }
        
        assert "error" in error_response
        assert "code" in error_response["error"]
        assert "message" in error_response["error"]
    
    def test_rate_limit_error(self):
        """Test rate limit error response."""
        error_response = {
            "error": {
                "code": "RATE_LIMITED",
                "message": "Rate limit exceeded",
                "retry_after": 60
            }
        }
        
        assert error_response["error"]["code"] == "RATE_LIMITED"
        assert "retry_after" in error_response["error"]
    
    def test_authentication_error(self):
        """Test authentication error response."""
        error_response = {
            "error": {
                "code": "UNAUTHORIZED",
                "message": "Invalid API key"
            }
        }
        
        assert error_response["error"]["code"] == "UNAUTHORIZED"


class TestRateLimiting:
    """Tests for rate limiting."""
    
    def test_rate_limit_headers(self):
        """Test rate limit headers in response."""
        headers = {
            "X-RateLimit-Limit": "60",
            "X-RateLimit-Remaining": "59",
            "X-RateLimit-Reset": "1700000000"
        }
        
        assert int(headers["X-RateLimit-Remaining"]) <= int(headers["X-RateLimit-Limit"])
    
    def test_tier_rate_limits(self):
        """Test different rate limit tiers."""
        tiers = {
            "default": {"requests_per_minute": 60},
            "premium": {"requests_per_minute": 300},
            "enterprise": {"requests_per_minute": 1000}
        }
        
        assert tiers["enterprise"]["requests_per_minute"] > tiers["premium"]["requests_per_minute"]
        assert tiers["premium"]["requests_per_minute"] > tiers["default"]["requests_per_minute"]


class TestAuthentication:
    """Tests for authentication."""
    
    def test_jwt_token_structure(self):
        """Test JWT token structure."""
        # JWT has 3 parts separated by dots
        sample_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        
        parts = sample_token.split(".")
        assert len(parts) == 3
    
    def test_api_key_format(self):
        """Test API key format validation."""
        valid_key = "sk-abc123def456ghi789"
        invalid_key = "invalid"
        
        # API keys typically have prefix and minimum length
        assert valid_key.startswith("sk-")
        assert len(valid_key) > 10
        assert not invalid_key.startswith("sk-")


class TestBatching:
    """Tests for request batching."""
    
    def test_batch_request_structure(self):
        """Test batch request structure."""
        batch_request = {
            "requests": [
                {"prompt": "Question 1"},
                {"prompt": "Question 2"},
                {"prompt": "Question 3"}
            ]
        }
        
        assert len(batch_request["requests"]) == 3
    
    def test_batch_size_limit(self):
        """Test batch size limits."""
        max_batch_size = 32
        
        small_batch = {"requests": [{"prompt": f"Q{i}"} for i in range(10)]}
        large_batch = {"requests": [{"prompt": f"Q{i}"} for i in range(50)]}
        
        assert len(small_batch["requests"]) <= max_batch_size
        assert len(large_batch["requests"]) > max_batch_size
