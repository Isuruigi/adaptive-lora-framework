#!/usr/bin/env python3
"""
API Documentation Generator

Generate OpenAPI documentation for the Adaptive LoRA API:
- Extract endpoints from FastAPI app
- Generate Markdown documentation
- Create example requests/responses
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class APIDocGenerator:
    """Generate API documentation."""
    
    def __init__(self, app):
        self.app = app
        self.openapi_schema = None
    
    def extract_openapi(self) -> Dict[str, Any]:
        """Extract OpenAPI schema from FastAPI app."""
        if self.app is None:
            return self._get_placeholder_schema()
        
        try:
            self.openapi_schema = self.app.openapi()
            return self.openapi_schema
        except Exception as e:
            logger.warning(f"Could not extract OpenAPI schema: {e}")
            return self._get_placeholder_schema()
    
    def _get_placeholder_schema(self) -> Dict[str, Any]:
        """Return placeholder OpenAPI schema."""
        return {
            "openapi": "3.0.0",
            "info": {
                "title": "Adaptive LoRA API",
                "version": "1.0.0",
                "description": "API for the Adaptive Multi-Agent LoRA Framework"
            },
            "servers": [
                {"url": "http://localhost:8000", "description": "Development"},
                {"url": "https://api.example.com", "description": "Production"}
            ],
            "paths": {
                "/health": {
                    "get": {
                        "summary": "Health Check",
                        "responses": {
                            "200": {"description": "Service is healthy"}
                        }
                    }
                },
                "/v1/generate": {
                    "post": {
                        "summary": "Generate Text",
                        "description": "Generate text using the appropriate adapter",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "prompt": {"type": "string"},
                                            "max_tokens": {"type": "integer", "default": 256},
                                            "temperature": {"type": "number", "default": 0.7},
                                            "adapter": {"type": "string", "nullable": True}
                                        },
                                        "required": ["prompt"]
                                    }
                                }
                            }
                        },
                        "responses": {
                            "200": {
                                "description": "Generated text",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "text": {"type": "string"},
                                                "adapter_used": {"type": "string"},
                                                "tokens_generated": {"type": "integer"}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "/v1/route": {
                    "post": {
                        "summary": "Route Request",
                        "description": "Determine which adapter to use for a prompt",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "prompt": {"type": "string"}
                                        },
                                        "required": ["prompt"]
                                    }
                                }
                            }
                        },
                        "responses": {
                            "200": {
                                "description": "Routing decision",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "adapter": {"type": "string"},
                                                "confidence": {"type": "number"},
                                                "probabilities": {
                                                    "type": "object",
                                                    "additionalProperties": {"type": "number"}
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "/v1/adapters": {
                    "get": {
                        "summary": "List Adapters",
                        "description": "Get list of available adapters",
                        "responses": {
                            "200": {
                                "description": "List of adapters",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "name": {"type": "string"},
                                                    "description": {"type": "string"},
                                                    "version": {"type": "string"}
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "components": {
                "securitySchemes": {
                    "bearerAuth": {
                        "type": "http",
                        "scheme": "bearer",
                        "bearerFormat": "JWT"
                    },
                    "apiKey": {
                        "type": "apiKey",
                        "in": "header",
                        "name": "X-API-Key"
                    }
                }
            }
        }
    
    def generate_markdown(self, schema: Dict[str, Any]) -> str:
        """Generate Markdown documentation from OpenAPI schema."""
        lines = []
        
        # Title and description
        info = schema.get('info', {})
        lines.append(f"# {info.get('title', 'API Documentation')}")
        lines.append("")
        lines.append(f"**Version:** {info.get('version', '1.0.0')}")
        lines.append("")
        lines.append(info.get('description', ''))
        lines.append("")
        
        # Servers
        servers = schema.get('servers', [])
        if servers:
            lines.append("## Servers")
            lines.append("")
            for server in servers:
                lines.append(f"- **{server.get('description', 'Server')}**: `{server.get('url')}`")
            lines.append("")
        
        # Authentication
        security = schema.get('components', {}).get('securitySchemes', {})
        if security:
            lines.append("## Authentication")
            lines.append("")
            for name, scheme in security.items():
                lines.append(f"### {name}")
                lines.append(f"- **Type**: {scheme.get('type')}")
                if scheme.get('scheme'):
                    lines.append(f"- **Scheme**: {scheme.get('scheme')}")
                if scheme.get('in'):
                    lines.append(f"- **In**: {scheme.get('in')}")
                if scheme.get('name'):
                    lines.append(f"- **Header Name**: {scheme.get('name')}")
                lines.append("")
        
        # Endpoints
        lines.append("## Endpoints")
        lines.append("")
        
        paths = schema.get('paths', {})
        for path, methods in paths.items():
            lines.append(f"### {path}")
            lines.append("")
            
            for method, details in methods.items():
                if method in ['get', 'post', 'put', 'delete', 'patch']:
                    lines.append(f"#### `{method.upper()} {path}`")
                    lines.append("")
                    lines.append(f"**{details.get('summary', 'No summary')}**")
                    lines.append("")
                    
                    if details.get('description'):
                        lines.append(details.get('description'))
                        lines.append("")
                    
                    # Request body
                    if 'requestBody' in details:
                        lines.append("**Request Body:**")
                        lines.append("")
                        content = details['requestBody'].get('content', {})
                        for content_type, content_schema in content.items():
                            lines.append(f"Content-Type: `{content_type}`")
                            lines.append("")
                            lines.append("```json")
                            example = self._generate_example(content_schema.get('schema', {}))
                            lines.append(json.dumps(example, indent=2))
                            lines.append("```")
                            lines.append("")
                    
                    # Responses
                    responses = details.get('responses', {})
                    if responses:
                        lines.append("**Responses:**")
                        lines.append("")
                        for status, response in responses.items():
                            lines.append(f"- **{status}**: {response.get('description', '')}")
                        lines.append("")
        
        return "\n".join(lines)
    
    def _generate_example(self, schema: Dict) -> Any:
        """Generate example from JSON schema."""
        schema_type = schema.get('type', 'object')
        
        if schema_type == 'object':
            result = {}
            for prop, prop_schema in schema.get('properties', {}).items():
                if prop_schema.get('default') is not None:
                    result[prop] = prop_schema['default']
                elif prop_schema.get('type') == 'string':
                    result[prop] = f"example_{prop}"
                elif prop_schema.get('type') == 'integer':
                    result[prop] = 0
                elif prop_schema.get('type') == 'number':
                    result[prop] = 0.0
                elif prop_schema.get('type') == 'boolean':
                    result[prop] = True
                elif prop_schema.get('type') == 'array':
                    result[prop] = []
                else:
                    result[prop] = None
            return result
        elif schema_type == 'array':
            return []
        elif schema_type == 'string':
            return "string"
        elif schema_type == 'integer':
            return 0
        elif schema_type == 'number':
            return 0.0
        elif schema_type == 'boolean':
            return True
        else:
            return None
    
    def save_openapi(self, output_path: Path, format: str = 'json') -> None:
        """Save OpenAPI schema to file."""
        schema = self.extract_openapi()
        
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(schema, f, indent=2)
        elif format == 'yaml':
            with open(output_path, 'w') as f:
                yaml.dump(schema, f, default_flow_style=False)
        
        logger.info(f"OpenAPI schema saved: {output_path}")
    
    def save_markdown(self, output_path: Path) -> None:
        """Save Markdown documentation to file."""
        schema = self.extract_openapi()
        markdown = self.generate_markdown(schema)
        
        with open(output_path, 'w') as f:
            f.write(markdown)
        
        logger.info(f"Markdown documentation saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate API Documentation')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='docs/api',
        help='Output directory'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'yaml', 'markdown', 'all'],
        default='all',
        help='Output format'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to import the FastAPI app
    app = None
    try:
        from src.serving.api import app
    except ImportError:
        logger.warning("Could not import FastAPI app, using placeholder schema")
    
    generator = APIDocGenerator(app)
    
    if args.format in ['json', 'all']:
        generator.save_openapi(output_dir / 'openapi.json', 'json')
    
    if args.format in ['yaml', 'all']:
        generator.save_openapi(output_dir / 'openapi.yaml', 'yaml')
    
    if args.format in ['markdown', 'all']:
        generator.save_markdown(output_dir / 'API.md')
    
    logger.info(f"Documentation generated in: {output_dir}")


if __name__ == '__main__':
    main()
