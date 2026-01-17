
"""
CAI Security Tools Adapter for AI Autonom
Ported from the CAI framework (https://github.com/aliasrobotics/cai)
Includes:
- Guardrails
- Reconnaissance Tools
- Web Analysis Tools (Headers, JS Surface Mapper)
- Crypto Tools
"""

import os
import re
import json
import unicodedata
import requests
from urllib.parse import urlparse, urljoin
from typing import Tuple, Dict, Any, Optional, List, Set, Iterable
from dataclasses import dataclass, field
from html.parser import HTMLParser

# =========================================================================
# HELPER CLASSES (For JS Surface Mapper)
# =========================================================================

@dataclass
class _ExtractionResult:
    origins: Set[str] = field(default_factory=set)
    endpoints: Set[str] = field(default_factory=set)
    graphql_endpoints: Set[str] = field(default_factory=set)
    graphql_ops: Set[str] = field(default_factory=set)
    persisted_hashes: Set[str] = field(default_factory=set)
    ws_endpoints: Set[str] = field(default_factory=set)
    high_value: Set[str] = field(default_factory=set)

class _AssetHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.script_srcs: List[str] = []
        self.inline_scripts: List[str] = []
        self._in_script: bool = False
        self._current_inline: List[str] = []
        self.link_hrefs: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attrs_dict = {k.lower(): (v or "") for k, v in attrs}
        if tag.lower() == "script":
            src = attrs_dict.get("src", "").strip()
            if src:
                self.script_srcs.append(src)
            else:
                self._in_script = True
                self._current_inline = []
        elif tag.lower() == "link":
            rel = attrs_dict.get("rel", "").lower()
            href = attrs_dict.get("href", "").strip()
            as_attr = attrs_dict.get("as", "").lower()
            if href and (rel in ("modulepreload", "preload") or as_attr == "script"):
                self.link_hrefs.append(href)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "script" and self._in_script:
            content = "".join(self._current_inline).strip()
            if content:
                self.inline_scripts.append(content)
            self._in_script = False
            self._current_inline = []

    def handle_data(self, data: str) -> None:
        if self._in_script and data:
            self._current_inline.append(data)

# =========================================================================
# MAIN TOOLS CLASS
# =========================================================================

class CaiSecurityTools:
    """
    Security-focused tools ported from CAI framework.
    Includes guardrails and specialized security utilities.
    """
    
    def __init__(self, sandbox=None, workspace_dir: str = "outputs"):
        self.sandbox = sandbox
        self.workspace_dir = workspace_dir
    
    # =========================================================================
    # GUARDRAILS
    # =========================================================================
    
    def _detect_unicode_homographs(self, text: str) -> bool:
        """Detect potential unicode homograph attacks"""
        homograph_map = {
            '\u0430': 'a', '\u0435': 'e', '\u043e': 'o', '\u0440': 'p', '\u0441': 'c',
            '\u0443': 'y', '\u0445': 'x', '\u03b1': 'a', '\u03bf': 'o', '\u03c1': 'p'
        }
        return any(char in text for char in homograph_map)

    def _check_guardrails(self, command: str) -> Optional[str]:
        """Check command against security guardrails."""
        if self._detect_unicode_homographs(command):
            return "Error: Blocked Unicode homograph bypass attempt."
            
        dangerous_patterns = [
            r"(?i)rm\s+-rf\s+/", r"(?i):(){ :|:& };:", r"(?i)curl.*\|.*sh",
            r"(?i)wget.*\|.*bash", r"(?i)nc\s+[\d\.]+\s+\d+.*-e",
            r"(?i)bash.*-i.*>&.*tcp/", r"(?i)/dev/tcp/[\d\.]+/".strip() + "\d+",
            r"(?i)mkfs", r"(?i)dd\s+if=", r"(?i)chmod\s+-R\s+777\s+/"
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, command):
                return f"Error: Command blocked due to dangerous pattern: {pattern}"
                
        if "base64" in command and ("-d" in command or "--decode" in command):
            return "Error: Blocked potential encoded payload execution (base64 -d detected)."
            
        return None

    # =========================================================================
    # SYSTEM COMMANDS
    # =========================================================================

    def generic_linux_command(self, command: str) -> Tuple[bool, str]:
        """Execute any Linux command with CAI guardrails."""
        error = self._check_guardrails(command)
        if error:
            return False, error
        if self.sandbox:
            return self.sandbox.execute_command(command)
        return False, "Sandbox required"

    # =========================================================================
    # RECONNAISSANCE & CRYPTO
    # =========================================================================

    def nmap_scan(self, target: str, args: str = "-sV") -> Tuple[bool, str]:
        """Run Nmap scan on target."""
        if any(char in target for char in [';', '&', '|', '`', '$']):
            return False, "Invalid target format"
        command = f"nmap {args} {target}"
        return self.generic_linux_command(command)

    def shodan_search(self, query: str, limit: int = 5) -> Tuple[bool, str]:
        """Search Shodan API."""
        api_key = os.getenv("SHODAN_API_KEY")
        if not api_key:
            return False, "SHODAN_API_KEY not found in environment"
        try:
            response = requests.get(
                "https://api.shodan.io/shodan/host/search",
                params={"key": api_key, "query": query, "limit": limit},
                timeout=10
            )
            if response.status_code != 200:
                return False, f"Shodan API error: {response.text}"
            data = response.json()
            matches = data.get("matches", [])
            if not matches:
                return True, "No results found."
            formatted = ""
            for res in matches:
                formatted += f"IP: {res.get('ip_str')}\nOrg: {res.get('org', 'N/A')}\nPort: {res.get('port')}\nData: {res.get('data', '')[:100]}...\n\n"
            return True, formatted
        except Exception as e:
            return False, f"Shodan error: {str(e)}"

    def strings_command(self, file_path: str) -> Tuple[bool, str]:
        """Extract strings from binary."""
        return self.generic_linux_command(f"strings {file_path}")

    def decode64(self, input_data: str) -> Tuple[bool, str]:
        """Decode base64 string."""
        return self.generic_linux_command(f"echo '{input_data}' | base64 -d")

    # =========================================================================
    # WEB ANALYSIS TOOLS
    # =========================================================================

    def web_request_framework(
        self,
        url: str,
        method: str = "GET",
        headers: dict = None,
        data: dict = None,
        cookies: dict = None,
        params: dict = None
    ) -> Tuple[bool, str]:
        """
        Analyze HTTP requests/responses for security (headers, sensitive info).
        """
        try:
            analysis = ["\n=== HTTP Request Analysis ===\n"]
            parsed_url = urlparse(url)
            analysis.append(f"URL: {parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}")
            
            response = requests.request(
                method=method, url=url, headers=headers, data=data,
                cookies=cookies, params=params, verify=False, allow_redirects=True,
                timeout=10
            )
            
            analysis.append("\n=== HTTP Response Analysis ===")
            analysis.append(f"Status Code: {response.status_code}")
            analysis.append("\nResponse Headers:")
            for k, v in response.headers.items():
                analysis.append(f"- {k}: {v}")
                
            analysis.append("\n=== Security Analysis ===")
            security_headers = [
                'Strict-Transport-Security', 'Content-Security-Policy',
                'X-Frame-Options', 'X-XSS-Protection', 'X-Content-Type-Options'
            ]
            missing = [h for h in security_headers if h not in response.headers]
            if missing:
                analysis.append("Missing Security Headers:")
                for h in missing:
                    analysis.append(f"- {h}")
            
            sensitive_patterns = ['password', 'token', 'key', 'secret', 'admin', 'root']
            text_lower = response.text.lower()
            found_sensitive = [p for p in sensitive_patterns if p in text_lower]
            if found_sensitive:
                analysis.append(f"\nPotential sensitive info found: {', '.join(found_sensitive)}")
                
            return True, "\n".join(analysis)
            
        except Exception as e:
            return False, f"Error analyzing request: {str(e)}"

    def js_surface_mapper(
        self,
        base_url: str,
        max_assets: int = 15,
        timeout: int = 10
    ) -> Tuple[bool, str]:
        """
        Extract API endpoints and secrets from JS assets.
        """
        try:
            base_url = base_url.rstrip("/")
            if not base_url.startswith("http"):
                base_url = "http://" + base_url
                
            # Fetch entry page
            resp = requests.get(base_url, timeout=timeout, verify=False)
            parser = _AssetHTMLParser()
            parser.feed(resp.text)
            
            assets = []
            for src in parser.script_srcs:
                full = src if src.startswith("http") else urljoin(base_url, src)
                assets.append(full)
                
            # Extract hints
            endpoints = set()
            high_value = set()
            
            # Simple regex patterns from CAI
            _PATH_ENDPOINT_RE = re.compile(
                r"(?<![A-Za-z0-9_])/(?:" 
                r"api|graphql|gql|v\d+|admin|auth|login|token|users|account|"
                r"payment|order|flag|config|internal"
                r")(?:[A-Za-z0-9_\-./?=&%]*) "
            )
            
            for asset in assets[:max_assets]:
                try:
                    js_resp = requests.get(asset, timeout=5, verify=False)
                    text = js_resp.text
                    
                    # Find endpoints
                    for path in _PATH_ENDPOINT_RE.findall(text):
                        if path.startswith("/"):
                            endpoints.add(path)
                            
                    # Find high value strings
                    lowered = text.lower()
                    for s in ["admin", "api_key", "secret", "token", "password"]:
                        if s in lowered:
                            high_value.add(s)
                except:
                    continue
                    
            return True, json.dumps({
                "base_url": base_url,
                "assets_scanned": len(assets[:max_assets]),
                "endpoints_found": sorted(list(endpoints)),
                "high_value_keywords": sorted(list(high_value))
            }, indent=2)
            
        except Exception as e:
            return False, f"JS Mapper error: {str(e)}"

    # =========================================================================
    # NETWORK & UTILS
    # =========================================================================

    def shodan_host_info(self, host: str) -> Tuple[bool, str]:
        """Get Shodan host info."""
        api_key = os.getenv("SHODAN_API_KEY")
        if not api_key: return False, "SHODAN_API_KEY missing"
        try:
            r = requests.get(f"https://api.shodan.io/shodan/host/{host}", params={"key": api_key}, timeout=10)
            if r.status_code != 200: return False, f"Shodan API error: {r.text}"
            d = r.json()
            return True, f"IP: {d.get('ip_str')}\nOS: {d.get('os')}\nPorts: {d.get('ports')}\nVulns: {d.get('vulns', [])}"
        except Exception as e: return False, str(e)

    def web_spider(self, url: str, depth: int = 1) -> Tuple[bool, str]:
        return self.generic_linux_command(f"wget --spider --recursive --level={depth} --no-verbose {url}")

    def gobuster_dir(self, url: str, wordlist: str = "/usr/share/wordlists/dirb/common.txt") -> Tuple[bool, str]:
        return self.generic_linux_command(f"gobuster dir -u {url} -w {wordlist} --no-error -t 20 -z")
    
    def curl_request(self, url: str, args: str = "-I") -> Tuple[bool, str]:
        return self.generic_linux_command(f"curl {args} {url}")
        
    def wget_download(self, url: str, output: str = "") -> Tuple[bool, str]:
        cmd = f"wget {url}" + (f" -O {output}" if output else "")
        return self.generic_linux_command(cmd)

    def netcat(self, host: str = "", port: str = "", args: str = "") -> Tuple[bool, str]:
        return self.generic_linux_command(f"nc {args} {host} {port}".strip())

    def netstat_info(self, args: str = "-tuln") -> Tuple[bool, str]:
        return self.generic_linux_command(f"netstat {args}")
        
    def google_search(self, query: str) -> Tuple[bool, str]:
        return False, "Google search requires API key or browser automation"

    # =========================================================================
    # KNOWLEDGE BASE TOOLS
    # =========================================================================

    def record_finding(self, title: str, type: str, severity: str, details: str) -> Tuple[bool, str]:
        """Record a finding in the Operation Knowledge Base."""
        try:
            from ..memory.knowledge_base import KnowledgeBase
            kb = KnowledgeBase.get_instance()
            kb.add_finding(title, type, details, severity, "agent")
            return True, f"Finding recorded: [{severity.upper()}] {title}"
        except ImportError:
            return False, "Knowledge Base not available"

    def record_credential(self, username: str, password: str, service: str) -> Tuple[bool, str]:
        """Record a captured credential."""
        try:
            from ..memory.knowledge_base import KnowledgeBase
            kb = KnowledgeBase.get_instance()
            kb.add_credential(username, password, service=service)
            return True, f"Credential recorded for {username} @ {service}"
        except ImportError:
            return False, "Knowledge Base not available"

    def capture_remote_traffic(self, ip: str, username: str, password: str, interface: str = "eth0", duration: int = 10, filter: str = "") -> Tuple[bool, str]:
        """
        Capture network traffic from a remote host via SSH.
        Generates and runs a Python script using Paramiko inside the container.
        """
        if not self.sandbox:
            return False, "Sandbox required for traffic capture"
            
        # Create the capture script
        script_content = f'''
import paramiko
import time
import sys

def capture(ip, user, pwd, iface, duration, pcap_filter):
    print(f"Connecting to {{ip}}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        client.connect(ip, username=user, password=pwd, timeout=10)
        
        # Build command
        cmd = f"timeout {{duration}} tcpdump -U -i {{iface}} -w - {{pcap_filter}}"
        print(f"Running: {{cmd}}")
        
        stdin, stdout, stderr = client.exec_command(cmd)
        
        # Read pcap data
        output_file = f"/workspace/outputs/capture_{{int(time.time())}}.pcap"
        with open(output_file, "wb") as f:
            while True:
                data = stdout.read(4096)
                if not data:
                    break
                f.write(data)
                
        # Check for errors
        err = stderr.read().decode()
        if err and "permission denied" in err.lower():
            print(f"Error: {{err}}")
            sys.exit(1)
            
        print(f"SUCCESS: Traffic captured to {{output_file}}")
        
    except Exception as e:
        print(f"Failed: {{e}}")
        sys.exit(1)
    finally:
        client.close()

if __name__ == "__main__":
    capture("{ip}", "{username}", "{password}", "{interface}", {duration}, "{filter}")
'''
        # Write script to container
        script_path = "capture_tool.py"
        self.sandbox.write_file(script_path, script_content)
        
        # Execute
        return self.sandbox.run_python(script_path)

# =========================================================================
# EXPORTS
# =========================================================================

_TOOLS = CaiSecurityTools()

# Function exports
generic_linux_command = _TOOLS.generic_linux_command
nmap_scan = _TOOLS.nmap_scan
shodan_search = _TOOLS.shodan_search
shodan_host_info = _TOOLS.shodan_host_info
web_spider = _TOOLS.web_spider
gobuster_dir = _TOOLS.gobuster_dir
curl_request = _TOOLS.curl_request
wget_download = _TOOLS.wget_download
netcat = _TOOLS.netcat
netstat_info = _TOOLS.netstat_info
google_search = _TOOLS.google_search
strings_command = _TOOLS.strings_command
decode64 = _TOOLS.decode64
web_request_framework = _TOOLS.web_request_framework
js_surface_mapper = _TOOLS.js_surface_mapper
capture_remote_traffic = _TOOLS.capture_remote_traffic
record_finding = _TOOLS.record_finding
record_credential = _TOOLS.record_credential

# Placeholders for compatibility
execute_code = lambda code: _TOOLS.generic_linux_command(f"python3 -c '{code}'")
filesystem_read = lambda path: _TOOLS.generic_linux_command(f"cat {path}")
filesystem_write = lambda path, content: _TOOLS.generic_linux_command(f"cat > {path} << 'EOF'\n{content}\nEOF")
filesystem_search = lambda dir, pat: _TOOLS.generic_linux_command(f"find {dir} -name '{pat}'")
hash_data = lambda data: _TOOLS.generic_linux_command(f"echo -n '{data}' | sha256sum")
hash_file = lambda file: _TOOLS.generic_linux_command(f"sha256sum {file}")
base64_encode = lambda data: _TOOLS.generic_linux_command(f"echo -n '{data}' | base64")
base64_decode = lambda data: _TOOLS.generic_linux_command(f"echo '{data}' | base64 -d")
file_identify = lambda file: _TOOLS.generic_linux_command(f"file {file}")
strings_extract = lambda file: _TOOLS.generic_linux_command(f"strings {file}")
hexdump = lambda file: _TOOLS.generic_linux_command(f"hexdump -C {file}")
binwalk_analyze = lambda file: _TOOLS.generic_linux_command(f"binwalk {file}")
readelf_info = lambda file: _TOOLS.generic_linux_command(f"readelf -a {file}")
objdump_disasm = lambda file: _TOOLS.generic_linux_command(f"objdump -d {file}")
volatility_analyze = lambda file: False, "Volatility requires complex setup"
pcap_analyze = lambda file: _TOOLS.generic_linux_command(f"tcpdump -r {file}")
log_analyze = lambda file: _TOOLS.generic_linux_command(f"cat {file}")
timeline_create = lambda dir: _TOOLS.generic_linux_command(f"find {dir} -printf '%T@ %p\n' | sort -n")
http_request = curl_request
think = lambda thought: (True, f"Thought recorded: {thought}")
thought_analysis = lambda: (True, "Analysis complete")
write_key_findings = lambda findings: (True, "Findings recorded")
read_key_findings = lambda: (True, "No findings yet")

# Dictionary export
CAI_SECURITY_TOOLS = {
    "generic_linux_command": {
        "function": generic_linux_command,
        "description": "Execute Linux command with guardrails",
        "category": "system",
        "parameters": {"command": "Command to execute"}
    },
    "nmap_scan": {
        "function": nmap_scan,
        "description": "Run Nmap network scan",
        "category": "reconnaissance",
        "parameters": {"target": "Target IP/Host", "args": "Nmap arguments"}
    },
    "web_request_framework": {
        "function": web_request_framework,
        "description": "Analyze HTTP requests and responses for security",
        "category": "web",
        "parameters": {"url": "Target URL", "method": "GET/POST", "headers": "dict"}
    },
    "js_surface_mapper": {
        "function": js_surface_mapper,
        "description": "Extract API endpoints and secrets from JS assets",
        "category": "web",
        "parameters": {"base_url": "Target URL", "max_assets": "Limit"}
    },
    "shodan_search": {
        "function": shodan_search,
        "description": "Search Shodan API",
        "category": "reconnaissance",
        "parameters": {"query": "Search query", "limit": "Max results"}
    },
    "web_spider": {
        "function": web_spider,
        "description": "Spider a website",
        "category": "reconnaissance",
        "parameters": {"url": "URL to spider", "depth": "Recursion depth"}
    },
    "curl_request": {
        "function": curl_request,
        "description": "Make HTTP request with curl",
        "category": "web",
        "parameters": {"url": "URL", "args": "Curl arguments"}
    },
    "wget_download": {
        "function": wget_download,
        "description": "Download file",
        "category": "web",
        "parameters": {"url": "URL", "output": "Output filename"}
    },
    "netcat": {
        "function": netcat,
        "description": "Netcat utility",
        "category": "network",
        "parameters": {"host": "Host", "port": "Port", "args": "Args"}
    },
    "netstat_info": {
        "function": netstat_info,
        "description": "Network statistics",
        "category": "network",
        "parameters": {"args": "Arguments"}
    },
    "capture_remote_traffic": {
        "function": capture_remote_traffic,
        "description": "Capture traffic from remote host via SSH",
        "category": "network",
        "parameters": {
            "ip": "Remote IP", "username": "SSH User", "password": "SSH Password",
            "interface": "Interface (eth0)", "duration": "Duration (sec)", "filter": "BPF Filter"
        }
    },
    "strings_command": {
        "function": strings_command,
        "description": "Extract strings from binary",
        "category": "binary",
        "parameters": {"file_path": "Path to binary"}
    },
    "decode64": {
        "function": decode64,
        "description": "Decode base64 string",
        "category": "crypto",
        "parameters": {"input_data": "Base64 string"}
    },
    "record_finding": {
        "function": record_finding,
        "description": "Record a finding/vulnerability in the Knowledge Base",
        "category": "reporting",
        "parameters": {"title": "Short title", "type": "vulnerability/asset", "severity": "high/medium/low", "details": "Description"}
    },
    "record_credential": {
        "function": record_credential,
        "description": "Record a captured credential",
        "category": "reporting",
        "parameters": {"username": "User", "password": "Password", "service": "Service/IP"}
    }
}

def get_tools_for_agent(capabilities: List[str]) -> List[str]:
    """Get list of tool IDs for agent capabilities."""
    tools = []
    if "reconnaissance" in capabilities:
        tools.extend(["nmap_scan", "shodan_search", "web_spider", "strings_command"])
    if "web_hacking" in capabilities:
        tools.extend(["curl_request", "wget_download", "gobuster_dir", "web_request_framework", "js_surface_mapper"])
    if "network" in capabilities:
        tools.extend(["netcat", "netstat_info", "capture_remote_traffic"])
    return tools

def get_tool(tool_id: str):
    """Get tool definition by ID."""
    return CAI_SECURITY_TOOLS.get(tool_id)
