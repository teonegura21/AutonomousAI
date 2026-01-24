"""
Kali Linux Tools Reference for AI Autonom Agents
=================================================
This module documents all available Kali Linux tools organized by agent specialty.
Agents can reference this to know what tools they have access to in the Kali container.

Container: agent_kali (Kali Linux Rolling)
Web Terminal: http://localhost:7681
"""

from typing import Dict, List

# =============================================================================
# REVERSE ENGINEERING TOOLS
# For: reverse_engineering_agent, decompiling_analysis_agent
# =============================================================================

REVERSE_ENGINEERING_TOOLS = {
    "disassemblers": {
        "radare2": {
            "command": "r2",
            "description": "Advanced reverse engineering framework with disassembler",
            "usage": "r2 -A binary_file",
            "capabilities": ["disassembly", "debugging", "binary analysis", "scripting"]
        },
        "ghidra": {
            "command": "ghidra",
            "description": "NSA's software reverse engineering suite",
            "usage": "ghidraRun (GUI) or analyzeHeadless for CLI",
            "capabilities": ["decompilation", "disassembly", "scripting", "collaboration"]
        },
        "objdump": {
            "command": "objdump",
            "description": "Display information from object files",
            "usage": "objdump -d binary_file",
            "capabilities": ["disassembly", "header info", "symbol tables"]
        },
        "nm": {
            "command": "nm",
            "description": "List symbols from object files",
            "usage": "nm binary_file",
            "capabilities": ["symbol listing", "address lookup"]
        },
    },
    "debuggers": {
        "gdb": {
            "command": "gdb",
            "description": "GNU Debugger - powerful debugging tool",
            "usage": "gdb ./program",
            "capabilities": ["breakpoints", "memory inspection", "register view", "stepping"]
        },
        "gdb-peda": {
            "command": "gdb with peda",
            "description": "Python Exploit Development Assistance for GDB",
            "usage": "gdb -q ./program then 'source peda.py'",
            "capabilities": ["exploit development", "ROP gadgets", "pattern create"]
        },
        "ltrace": {
            "command": "ltrace",
            "description": "Library call tracer",
            "usage": "ltrace ./program",
            "capabilities": ["library call tracing", "function monitoring"]
        },
        "strace": {
            "command": "strace",
            "description": "System call tracer",
            "usage": "strace ./program",
            "capabilities": ["syscall tracing", "file access monitoring"]
        },
    },
    "binary_analysis": {
        "binwalk": {
            "command": "binwalk",
            "description": "Firmware analysis and extraction tool",
            "usage": "binwalk -e firmware.bin",
            "capabilities": ["firmware extraction", "signature scanning", "entropy analysis"]
        },
        "file": {
            "command": "file",
            "description": "Determine file type",
            "usage": "file unknown_file",
            "capabilities": ["file identification", "magic number detection"]
        },
        "strings": {
            "command": "strings",
            "description": "Extract printable strings from binary",
            "usage": "strings -a binary_file",
            "capabilities": ["string extraction", "encoding detection"]
        },
        "hexdump": {
            "command": "hexdump",
            "description": "Display file in hexadecimal",
            "usage": "hexdump -C file",
            "capabilities": ["hex view", "binary inspection"]
        },
        "xxd": {
            "command": "xxd",
            "description": "Make hex dump or reverse",
            "usage": "xxd file | xxd -r",
            "capabilities": ["hex dump", "binary patching"]
        },
        "readelf": {
            "command": "readelf",
            "description": "Display ELF file information",
            "usage": "readelf -a binary",
            "capabilities": ["ELF headers", "sections", "symbols", "relocations"]
        },
    },
    "decompilers": {
        "ghidra_decompiler": {
            "command": "ghidra",
            "description": "Ghidra's built-in decompiler (C-like output)",
            "usage": "analyzeHeadless with decompiler script",
            "capabilities": ["C pseudocode", "function recovery", "type inference"]
        },
        "retdec": {
            "command": "retdec-decompiler",
            "description": "Retargetable machine-code decompiler",
            "usage": "retdec-decompiler binary",
            "capabilities": ["C output", "multiple architectures"]
        },
    },
    "patching": {
        "patchelf": {
            "command": "patchelf",
            "description": "Modify ELF executables",
            "usage": "patchelf --set-interpreter /path binary",
            "capabilities": ["interpreter change", "rpath modification"]
        },
        "dd": {
            "command": "dd",
            "description": "Low-level data copying and conversion",
            "usage": "dd if=input of=output bs=1 seek=offset",
            "capabilities": ["binary patching", "data extraction"]
        },
    },
}


# =============================================================================
# FORENSICS / DFIR TOOLS
# For: dfir, memory_analysis_agent
# =============================================================================

FORENSICS_TOOLS = {
    "memory_forensics": {
        "volatility3": {
            "command": "vol3",
            "description": "Advanced memory forensics framework",
            "usage": "vol3 -f memory.raw windows.pslist",
            "plugins": [
                "windows.pslist - List processes",
                "windows.pstree - Process tree",
                "windows.dlllist - DLL listing",
                "windows.handles - Open handles",
                "windows.netscan - Network connections",
                "windows.filescan - File objects",
                "windows.cmdline - Command lines",
                "windows.malfind - Injected code",
                "windows.hashdump - Password hashes",
                "linux.pslist - Linux processes",
                "linux.bash - Bash history",
            ],
            "capabilities": ["process analysis", "malware detection", "credential extraction"]
        },
        "volatility2": {
            "command": "volatility",
            "description": "Legacy memory forensics (Python 2)",
            "usage": "volatility -f memory.raw --profile=Win7SP1x64 pslist",
            "capabilities": ["memory analysis", "profile-based"]
        },
    },
    "disk_forensics": {
        "autopsy": {
            "command": "autopsy",
            "description": "Digital forensics platform (GUI)",
            "usage": "autopsy",
            "capabilities": ["timeline analysis", "keyword search", "file recovery"]
        },
        "sleuthkit": {
            "command": "Various (fls, icat, mmls, etc.)",
            "description": "Collection of disk forensics tools",
            "tools": {
                "mmls": "Display partition layout",
                "fls": "List files and directories",
                "icat": "Extract file by inode",
                "blkcat": "Display data unit contents",
                "fsstat": "Filesystem statistics",
                "img_stat": "Image file statistics",
            },
            "capabilities": ["partition analysis", "file extraction", "timeline"]
        },
        "foremost": {
            "command": "foremost",
            "description": "File carving tool",
            "usage": "foremost -i disk.img -o output/",
            "capabilities": ["file recovery", "header-based carving"]
        },
        "scalpel": {
            "command": "scalpel",
            "description": "Fast file carver",
            "usage": "scalpel -c scalpel.conf disk.img",
            "capabilities": ["file carving", "configurable headers"]
        },
        "photorec": {
            "command": "photorec",
            "description": "File recovery from damaged media",
            "usage": "photorec /d output disk.img",
            "capabilities": ["photo recovery", "document recovery"]
        },
        "testdisk": {
            "command": "testdisk",
            "description": "Partition recovery tool",
            "usage": "testdisk disk.img",
            "capabilities": ["partition recovery", "boot sector repair"]
        },
    },
    "network_forensics": {
        "wireshark": {
            "command": "wireshark / tshark",
            "description": "Network protocol analyzer",
            "usage": "tshark -r capture.pcap -Y 'http'",
            "capabilities": ["packet analysis", "protocol dissection", "filtering"]
        },
        "tcpdump": {
            "command": "tcpdump",
            "description": "Command-line packet analyzer",
            "usage": "tcpdump -r capture.pcap -nn",
            "capabilities": ["packet capture", "BPF filtering"]
        },
        "zeek": {
            "command": "zeek",
            "description": "Network security monitor (formerly Bro)",
            "usage": "zeek -r capture.pcap",
            "capabilities": ["connection logs", "protocol analysis", "scripting"]
        },
        "networkMiner": {
            "command": "networkminer",
            "description": "Network forensic analysis tool",
            "usage": "networkminer",
            "capabilities": ["file extraction", "image reconstruction", "host analysis"]
        },
    },
    "log_analysis": {
        "grep": {
            "command": "grep",
            "description": "Pattern matching in files",
            "usage": "grep -rni 'pattern' /var/log/",
            "capabilities": ["pattern search", "regex support"]
        },
        "awk": {
            "command": "awk",
            "description": "Text processing and data extraction",
            "usage": "awk '{print $1, $4}' access.log",
            "capabilities": ["field extraction", "data transformation"]
        },
        "sed": {
            "command": "sed",
            "description": "Stream editor",
            "usage": "sed 's/old/new/g' file",
            "capabilities": ["text replacement", "line filtering"]
        },
        "jq": {
            "command": "jq",
            "description": "JSON processor",
            "usage": "jq '.field' file.json",
            "capabilities": ["JSON parsing", "data extraction"]
        },
        "logwatch": {
            "command": "logwatch",
            "description": "Log analysis and reporting",
            "usage": "logwatch --detail high",
            "capabilities": ["log summarization", "anomaly detection"]
        },
    },
    "file_analysis": {
        "exiftool": {
            "command": "exiftool",
            "description": "Read/write metadata in files",
            "usage": "exiftool image.jpg",
            "capabilities": ["metadata extraction", "EXIF data", "GPS info"]
        },
        "pdfparser": {
            "command": "pdf-parser",
            "description": "Parse PDF documents",
            "usage": "pdf-parser -a document.pdf",
            "capabilities": ["PDF analysis", "object extraction", "JavaScript detection"]
        },
        "olevba": {
            "command": "olevba",
            "description": "Analyze VBA macros in Office files",
            "usage": "olevba document.docm",
            "capabilities": ["macro extraction", "IOC detection"]
        },
    },
    "hashing": {
        "md5sum": {"command": "md5sum", "usage": "md5sum file"},
        "sha1sum": {"command": "sha1sum", "usage": "sha1sum file"},
        "sha256sum": {"command": "sha256sum", "usage": "sha256sum file"},
        "ssdeep": {
            "command": "ssdeep",
            "description": "Fuzzy hashing",
            "usage": "ssdeep -b file",
            "capabilities": ["similarity detection", "fuzzy matching"]
        },
    },
}


# =============================================================================
# VULNERABILITY / BUG BOUNTY TOOLS
# For: bug_bounty_agent
# =============================================================================

VULNERABILITY_TOOLS = {
    "reconnaissance": {
        "nmap": {
            "command": "nmap",
            "description": "Network scanner and security auditor",
            "usage": "nmap -sV -sC target",
            "capabilities": ["port scanning", "service detection", "scripting engine"]
        },
        "masscan": {
            "command": "masscan",
            "description": "Fast port scanner",
            "usage": "masscan -p1-65535 target --rate=1000",
            "capabilities": ["high-speed scanning", "banner grabbing"]
        },
        "amass": {
            "command": "amass",
            "description": "Subdomain enumeration",
            "usage": "amass enum -d domain.com",
            "capabilities": ["subdomain discovery", "DNS enumeration", "OSINT"]
        },
        "subfinder": {
            "command": "subfinder",
            "description": "Fast subdomain discovery",
            "usage": "subfinder -d domain.com",
            "capabilities": ["passive enumeration", "API integration"]
        },
        "theharvester": {
            "command": "theHarvester",
            "description": "Email and subdomain harvesting",
            "usage": "theHarvester -d domain.com -b all",
            "capabilities": ["email discovery", "subdomain enum", "OSINT"]
        },
        "recon-ng": {
            "command": "recon-ng",
            "description": "Web reconnaissance framework",
            "usage": "recon-ng",
            "capabilities": ["modular recon", "database storage", "reporting"]
        },
        "whatweb": {
            "command": "whatweb",
            "description": "Web technology fingerprinting",
            "usage": "whatweb target.com",
            "capabilities": ["CMS detection", "plugin identification"]
        },
        "wafw00f": {
            "command": "wafw00f",
            "description": "Web Application Firewall detector",
            "usage": "wafw00f target.com",
            "capabilities": ["WAF detection", "bypass hints"]
        },
    },
    "web_scanning": {
        "nikto": {
            "command": "nikto",
            "description": "Web server vulnerability scanner",
            "usage": "nikto -h target.com",
            "capabilities": ["vulnerability scanning", "misconfig detection"]
        },
        "gobuster": {
            "command": "gobuster",
            "description": "Directory/DNS brute-forcing",
            "usage": "gobuster dir -u http://target -w wordlist.txt",
            "capabilities": ["directory enumeration", "DNS brute force", "vhost discovery"]
        },
        "dirb": {
            "command": "dirb",
            "description": "Web content scanner",
            "usage": "dirb http://target wordlist.txt",
            "capabilities": ["directory scanning", "recursive search"]
        },
        "dirsearch": {
            "command": "dirsearch",
            "description": "Advanced web path scanner",
            "usage": "dirsearch -u http://target -e php,html",
            "capabilities": ["extension filtering", "recursive scanning"]
        },
        "ffuf": {
            "command": "ffuf",
            "description": "Fast web fuzzer",
            "usage": "ffuf -u http://target/FUZZ -w wordlist.txt",
            "capabilities": ["fuzzing", "parameter discovery", "vhost enumeration"]
        },
        "wpscan": {
            "command": "wpscan",
            "description": "WordPress vulnerability scanner",
            "usage": "wpscan --url http://target",
            "capabilities": ["WordPress scanning", "plugin enumeration", "user enum"]
        },
        "nuclei": {
            "command": "nuclei",
            "description": "Template-based vulnerability scanner",
            "usage": "nuclei -u target.com -t templates/",
            "capabilities": ["vulnerability scanning", "custom templates", "automation"]
        },
    },
    "exploitation": {
        "sqlmap": {
            "command": "sqlmap",
            "description": "Automatic SQL injection tool",
            "usage": "sqlmap -u 'http://target?id=1' --dbs",
            "capabilities": ["SQL injection", "database dump", "OS shell"]
        },
        "xsstrike": {
            "command": "xsstrike",
            "description": "XSS scanner and exploiter",
            "usage": "xsstrike -u 'http://target?param=test'",
            "capabilities": ["XSS detection", "payload generation", "WAF bypass"]
        },
        "commix": {
            "command": "commix",
            "description": "Command injection exploiter",
            "usage": "commix -u 'http://target?cmd=test'",
            "capabilities": ["command injection", "shell access"]
        },
        "metasploit": {
            "command": "msfconsole",
            "description": "Penetration testing framework",
            "usage": "msfconsole",
            "capabilities": ["exploit modules", "payload generation", "post-exploitation"]
        },
        "searchsploit": {
            "command": "searchsploit",
            "description": "Exploit-DB search tool",
            "usage": "searchsploit apache 2.4",
            "capabilities": ["exploit search", "offline database"]
        },
    },
    "password_attacks": {
        "hydra": {
            "command": "hydra",
            "description": "Network login cracker",
            "usage": "hydra -l admin -P wordlist.txt target ssh",
            "protocols": ["SSH", "FTP", "HTTP", "SMB", "MySQL", "RDP", "etc."],
            "capabilities": ["brute force", "dictionary attack"]
        },
        "john": {
            "command": "john",
            "description": "Password cracker",
            "usage": "john --wordlist=wordlist.txt hashes.txt",
            "capabilities": ["hash cracking", "multiple formats", "rules"]
        },
        "hashcat": {
            "command": "hashcat",
            "description": "Advanced password recovery",
            "usage": "hashcat -m 0 hashes.txt wordlist.txt",
            "capabilities": ["GPU cracking", "rule-based", "mask attack"]
        },
        "crackmapexec": {
            "command": "crackmapexec",
            "description": "Network attack toolkit",
            "usage": "crackmapexec smb target -u user -p pass",
            "capabilities": ["SMB/WinRM attacks", "credential spraying"]
        },
    },
    "api_testing": {
        "burpsuite": {
            "command": "burpsuite",
            "description": "Web security testing platform",
            "usage": "burpsuite (GUI)",
            "capabilities": ["proxy", "scanner", "intruder", "repeater"]
        },
        "postman": {
            "command": "postman",
            "description": "API development and testing",
            "usage": "postman (GUI)",
            "capabilities": ["API testing", "collection runner"]
        },
        "curl": {
            "command": "curl",
            "description": "Transfer data with URLs",
            "usage": "curl -X POST -d 'data' target",
            "capabilities": ["HTTP requests", "authentication", "headers"]
        },
        "httpx": {
            "command": "httpx",
            "description": "Fast HTTP toolkit",
            "usage": "httpx -l urls.txt -status-code",
            "capabilities": ["HTTP probing", "tech detection"]
        },
    },
}


# =============================================================================
# ORCHESTRATOR / CODE AGENT TOOLS
# For: thought, codeagent
# =============================================================================

ORCHESTRATOR_TOOLS = {
    "scripting": {
        "python": {
            "command": "python",
            "description": "Python interpreter",
            "usage": "python script.py",
            "libraries": ["requests", "pwntools", "scapy", "beautifulsoup4", "pandas"]
        },
        "perl": {
            "command": "perl",
            "description": "Perl interpreter",
            "usage": "perl script.pl"
        },
        "ruby": {
            "command": "ruby",
            "description": "Ruby interpreter",
            "usage": "ruby script.rb"
        },
        "bash": {
            "command": "bash",
            "description": "Bash shell scripting",
            "usage": "bash script.sh"
        },
    },
    "development": {
        "git": {
            "command": "git",
            "description": "Version control",
            "usage": "git clone/pull/push"
        },
        "gcc": {
            "command": "gcc",
            "description": "C compiler",
            "usage": "gcc -o output source.c"
        },
        "make": {
            "command": "make",
            "description": "Build automation",
            "usage": "make"
        },
    },
    "utilities": {
        "tmux": {
            "command": "tmux",
            "description": "Terminal multiplexer",
            "usage": "tmux new -s session"
        },
        "screen": {
            "command": "screen",
            "description": "Terminal session manager",
            "usage": "screen -S session"
        },
        "vim": {
            "command": "vim",
            "description": "Text editor",
            "usage": "vim file"
        },
        "nano": {
            "command": "nano",
            "description": "Simple text editor",
            "usage": "nano file"
        },
    },
}


# =============================================================================
# REPORTER TOOLS  
# For: reporter
# =============================================================================

REPORTER_TOOLS = {
    "documentation": {
        "pandoc": {
            "command": "pandoc",
            "description": "Document converter",
            "usage": "pandoc input.md -o output.pdf",
            "formats": ["markdown", "html", "pdf", "docx"]
        },
        "wkhtmltopdf": {
            "command": "wkhtmltopdf",
            "description": "HTML to PDF converter",
            "usage": "wkhtmltopdf input.html output.pdf"
        },
    },
    "screenshots": {
        "cutycapt": {
            "command": "cutycapt",
            "description": "Webpage screenshot tool",
            "usage": "cutycapt --url=http://target --out=screenshot.png"
        },
        "eyewitness": {
            "command": "eyewitness",
            "description": "Website screenshot and info gather",
            "usage": "eyewitness -f urls.txt --web"
        },
    },
    "reporting_frameworks": {
        "dradis": {
            "command": "dradis",
            "description": "Collaboration and reporting platform",
            "usage": "dradis (web interface)"
        },
        "faraday": {
            "command": "faraday",
            "description": "Collaborative penetration test IDE",
            "usage": "faraday-server"
        },
    },
}


# =============================================================================
# COMPLETE TOOL REGISTRY BY AGENT
# =============================================================================

AGENT_TOOL_REGISTRY = {
    # REVERSE category
    "reverse_engineering_agent": {
        "category": "REVERSE",
        "tools": REVERSE_ENGINEERING_TOOLS,
        "primary_tools": ["radare2", "ghidra", "gdb", "binwalk", "strings", "objdump"],
    },
    "decompiling_analysis_agent": {
        "category": "REVERSE", 
        "tools": REVERSE_ENGINEERING_TOOLS,
        "primary_tools": ["ghidra", "radare2", "retdec", "readelf", "objdump"],
    },
    
    # FORENSICS category
    "dfir": {
        "category": "FORENSICS",
        "tools": FORENSICS_TOOLS,
        "primary_tools": ["volatility3", "autopsy", "sleuthkit", "wireshark", "foremost"],
    },
    "memory_analysis_agent": {
        "category": "FORENSICS",
        "tools": FORENSICS_TOOLS,
        "primary_tools": ["volatility3", "strings", "grep", "hexdump"],
    },
    
    # VULNERABILITATI category
    "bug_bounty_agent": {
        "category": "VULNERABILITATI",
        "tools": VULNERABILITY_TOOLS,
        "primary_tools": ["nmap", "nikto", "sqlmap", "gobuster", "burpsuite", "nuclei"],
    },
    
    # Orchestrator category
    "thought": {
        "category": "Orchestrator",
        "tools": ORCHESTRATOR_TOOLS,
        "primary_tools": ["python3", "bash"],
        "note": "Planning agent - minimal tool execution"
    },
    "codeagent": {
        "category": "Orchestrator",
        "tools": ORCHESTRATOR_TOOLS,
        "primary_tools": ["python3", "gcc", "make", "git", "bash"],
    },
    
    # RAPORT category
    "reporter": {
        "category": "RAPORT",
        "tools": REPORTER_TOOLS,
        "primary_tools": ["pandoc", "wkhtmltopdf", "eyewitness"],
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_tools_for_agent(agent_name: str) -> Dict:
    """Get all tools available for a specific agent"""
    return AGENT_TOOL_REGISTRY.get(agent_name, {})


def get_tool_command(category: str, tool_name: str) -> str:
    """Get the command for a specific tool"""
    for tool_category in [REVERSE_ENGINEERING_TOOLS, FORENSICS_TOOLS, 
                          VULNERABILITY_TOOLS, ORCHESTRATOR_TOOLS, REPORTER_TOOLS]:
        for subcategory, tools in tool_category.items():
            if tool_name in tools:
                tool_info = tools[tool_name]
                if isinstance(tool_info, dict):
                    return tool_info.get("command", tool_name)
                return tool_name
    return tool_name


def list_all_tools() -> List[str]:
    """List all available tools across all categories"""
    all_tools = []
    for tool_category in [REVERSE_ENGINEERING_TOOLS, FORENSICS_TOOLS,
                          VULNERABILITY_TOOLS, ORCHESTRATOR_TOOLS, REPORTER_TOOLS]:
        for subcategory, tools in tool_category.items():
            all_tools.extend(tools.keys())
    return sorted(set(all_tools))


def print_tools_summary():
    """Print a summary of all tools by agent"""
    print("\n" + "=" * 70)
    print("KALI LINUX TOOLS BY AGENT CATEGORY")
    print("=" * 70)
    
    for agent, info in AGENT_TOOL_REGISTRY.items():
        print(f"\n[{info['category']}] {agent}")
        print(f"  Primary tools: {', '.join(info['primary_tools'])}")


if __name__ == "__main__":
    print_tools_summary()
    print(f"\nTotal unique tools available: {len(list_all_tools())}")
