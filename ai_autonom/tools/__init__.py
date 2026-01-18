"""
Tools module - Tool Registry, Executors, Built-in Tools, CAI Security Tools, and Kill Chain Tools
"""

from .tool_registry import ToolRegistry, ToolDefinition
from .builtin_tools import BuiltinTools
from .tool_executor import ToolExecutor
from .code_executor import CodeExecutor

# Import Kill Chain Tools (ported from CAI)
try:
    from .kill_chain_tools import (
        KillChainTools,
        create_kill_chain_tools,
        run_command,
        ShellSession,
        create_shell_session,
        list_shell_sessions,
        get_session_output,
        terminate_session,
        # Guardrail functions
        validate_command,
        detect_unicode_homographs,
        sanitize_external_content,
        detect_injection_patterns,
        redact_secrets,
    )
    KILL_CHAIN_TOOLS_AVAILABLE = True
except ImportError:
    KILL_CHAIN_TOOLS_AVAILABLE = False
    KillChainTools = None

# Import CAI Security Tools
try:
    from .cai_security_tools import (
        CAI_SECURITY_TOOLS,
        AGENT_TOOLS,
        get_tools_for_agent,
        get_tool,
        get_tools_by_category,
        list_all_tools,
        # Core execution
        run_command as cai_run_command,
        generic_linux_command,
        execute_code,
        # Reconnaissance
        nmap_scan,
        curl_request,
        wget_download,
        netcat,
        netstat_info,
        shodan_search,
        shodan_host_info,
        # Filesystem
        filesystem_read,
        filesystem_write,
        filesystem_search,
        # Crypto
        hash_data,
        hash_file,
        base64_encode,
        base64_decode,
        # Reasoning
        think,
        thought_analysis,
        write_key_findings,
        read_key_findings,
        # Binary Analysis
        file_identify,
        strings_extract,
        hexdump,
        binwalk_analyze,
        readelf_info,
        objdump_disasm,
        # DFIR
        volatility_analyze,
        pcap_analyze,
        log_analyze,
        timeline_create,
        # Web
        http_request,
        google_search,
    )
    CAI_TOOLS_AVAILABLE = True
except ImportError:
    CAI_TOOLS_AVAILABLE = False
    CAI_SECURITY_TOOLS = {}
    AGENT_TOOLS = {}

__all__ = [
    'ToolRegistry',
    'ToolDefinition',
    'BuiltinTools',
    'ToolExecutor',
    'CodeExecutor',
    # Kill Chain Tools
    'KillChainTools',
    'create_kill_chain_tools',
    'KILL_CHAIN_TOOLS_AVAILABLE',
    # CAI Tools
    'CAI_SECURITY_TOOLS',
    'AGENT_TOOLS',
    'get_tools_for_agent',
    'CAI_TOOLS_AVAILABLE',
]

