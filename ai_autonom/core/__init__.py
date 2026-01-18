"""
Core module - Configuration, Registry, Model Management, Guardrails
"""

from .config import Config, config, get_config
from .agent_registry import AgentRegistry, AgentDefinition, ToolDefinition

# Import Guardrails (ported from CAI)
try:
    from .guardrails import (
        Guardrails,
        GuardrailLevel,
        InjectionCheckResult,
        CommandValidationResult,
        validate_command,
        detect_injection_patterns,
        sanitize_external_content,
        redact_secrets,
        normalize_unicode_homographs,
        check_output_for_injection,
        get_guardrails,
        is_command_safe,
    )
    GUARDRAILS_AVAILABLE = True
except ImportError:
    GUARDRAILS_AVAILABLE = False
    Guardrails = None

__all__ = [
    'Config',
    'config',
    'get_config',
    'AgentRegistry',
    'AgentDefinition',
    'ToolDefinition',
    # Guardrails
    'Guardrails',
    'GuardrailLevel',
    'validate_command',
    'detect_injection_patterns',
    'sanitize_external_content',
    'redact_secrets',
    'get_guardrails',
    'is_command_safe',
    'GUARDRAILS_AVAILABLE',
]

