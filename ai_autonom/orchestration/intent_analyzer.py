#!/usr/bin/env python3
"""
Enhanced Intent Analyzer
Deep analysis of user intent with LLM-powered understanding
Detects task type, complexity, requirements, and suggests optimal patterns
"""

import re
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum


class TaskType(Enum):
    """Types of tasks the system can handle"""
    CODING = "coding"
    SECURITY_ASSESSMENT = "security_assessment"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    REFACTORING = "refactoring"
    DEBUGGING = "debugging"
    DATA_ANALYSIS = "data_analysis"
    AUTOMATION = "automation"
    RESEARCH = "research"
    MIXED = "mixed"


class Complexity(Enum):
    """Task complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


@dataclass
class IntentAnalysisResult:
    """Result of intent analysis"""
    original_goal: str
    task_type: TaskType
    complexity: Complexity
    requirements: Dict[str, Any]
    entities: Dict[str, List[str]]
    constraints: List[str]
    ambiguities: List[Dict[str, str]]
    suggested_pattern: Optional[str]
    suggested_agents: List[str]
    estimated_tasks: int
    needs_clarification: bool
    confidence: float
    clarification_questions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_goal": self.original_goal,
            "task_type": self.task_type.value,
            "complexity": self.complexity.value,
            "requirements": self.requirements,
            "entities": self.entities,
            "constraints": self.constraints,
            "ambiguities": self.ambiguities,
            "suggested_pattern": self.suggested_pattern,
            "suggested_agents": self.suggested_agents,
            "estimated_tasks": self.estimated_tasks,
            "needs_clarification": self.needs_clarification,
            "confidence": self.confidence,
            "clarification_questions": self.clarification_questions,
            "metadata": self.metadata
        }


class IntentAnalyzer:
    """
    Enhanced intent analysis with LLM-powered understanding.
    
    Features:
    - Task type classification
    - Complexity estimation
    - Entity extraction (languages, frameworks, platforms)
    - Pattern recommendation
    - Agent selection
    - Ambiguity detection
    """
    
    # Task type keywords
    TASK_TYPE_KEYWORDS = {
        TaskType.CODING: ["write", "create", "build", "implement", "develop", "code", "program"],
        TaskType.SECURITY_ASSESSMENT: ["security", "pentest", "vulnerability", "exploit", "assess", "audit", "scan"],
        TaskType.DOCUMENTATION: ["document", "readme", "docs", "explain", "describe", "comment"],
        TaskType.TESTING: ["test", "pytest", "unittest", "validate", "verify", "check"],
        TaskType.REFACTORING: ["refactor", "improve", "optimize", "clean up", "restructure"],
        TaskType.DEBUGGING: ["debug", "fix", "bug", "error", "issue", "problem"],
        TaskType.DATA_ANALYSIS: ["analyze", "data", "csv", "json", "parse", "extract", "transform"],
        TaskType.AUTOMATION: ["automate", "script", "batch", "cron", "schedule"],
        TaskType.RESEARCH: ["research", "investigate", "explore", "find", "search", "learn"]
    }
    
    # Complexity indicators
    COMPLEXITY_INDICATORS = {
        "simple": ["simple", "basic", "quick", "small", "single", "one"],
        "complex": ["complex", "advanced", "multiple", "comprehensive", "full", "complete", "production"],
        "integrations": ["integrate", "connect", "combine", "multiple", "with"],
        "scale": ["scalable", "distributed", "microservice", "large-scale"]
    }
    
    # Language/Framework detection
    LANGUAGE_PATTERNS = {
        "python": ["python", "py", "pip", "django", "flask", "fastapi", "pandas", "numpy"],
        "javascript": ["javascript", "js", "node", "npm", "react", "vue", "angular", "express"],
        "typescript": ["typescript", "ts", "tsx"],
        "go": ["golang", "go", "goroutine"],
        "rust": ["rust", "cargo"],
        "java": ["java", "spring", "maven", "gradle"],
        "cpp": ["c++", "cpp"],
        "bash": ["bash", "shell", "sh"]
    }
    
    # Platform detection
    PLATFORM_PATTERNS = {
        "web": ["web", "http", "api", "rest", "graphql", "website"],
        "cli": ["cli", "command", "terminal", "console"],
        "desktop": ["desktop", "gui", "window"],
        "mobile": ["mobile", "android", "ios"],
        "cloud": ["aws", "azure", "gcp", "cloud", "kubernetes", "docker"]
    }
    
    # Security-specific patterns
    SECURITY_PATTERNS = {
        "web_security": ["web app", "xss", "sql injection", "csrf", "owasp"],
        "network_security": ["network", "port scan", "nmap", "firewall"],
        "api_security": ["api", "endpoint", "authentication", "authorization"],
        "code_security": ["static analysis", "sast", "code review", "vulnerability"]
    }
    
    def __init__(self, llm_provider=None):
        self.llm_provider = llm_provider
        self.analysis_history: List[IntentAnalysisResult] = []
    
    def analyze(self, user_goal: str, context: Optional[Dict[str, Any]] = None) -> IntentAnalysisResult:
        """
        Analyze user intent and extract structured information
        
        Args:
            user_goal: User's goal/request
            context: Optional context (previous results, etc.)
        
        Returns:
            IntentAnalysisResult with structured analysis
        """
        user_goal = user_goal.strip()
        goal_lower = user_goal.lower()
        
        # Extract basic information
        task_type = self._classify_task_type(goal_lower)
        complexity = self._estimate_complexity(goal_lower, user_goal)
        entities = self._extract_entities(goal_lower)
        requirements = self._extract_requirements(goal_lower, user_goal)
        constraints = self._extract_constraints(goal_lower)
        ambiguities = self._detect_ambiguities(goal_lower)
        
        # Suggest pattern and agents
        suggested_pattern = self._suggest_pattern(task_type, complexity, entities)
        suggested_agents = self._suggest_agents(task_type, entities)
        estimated_tasks = self._estimate_task_count(complexity, goal_lower)
        
        # Determine if clarification needed
        needs_clarification = len(ambiguities) > 0 or complexity == Complexity.VERY_COMPLEX
        clarification_questions = [amb["question"] for amb in ambiguities] if needs_clarification else []
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            task_type, entities, ambiguities, goal_lower
        )
        
        result = IntentAnalysisResult(
            original_goal=user_goal,
            task_type=task_type,
            complexity=complexity,
            requirements=requirements,
            entities=entities,
            constraints=constraints,
            ambiguities=ambiguities,
            suggested_pattern=suggested_pattern,
            suggested_agents=suggested_agents,
            estimated_tasks=estimated_tasks,
            needs_clarification=needs_clarification,
            confidence=confidence,
            clarification_questions=clarification_questions,
            metadata={
                "word_count": len(user_goal.split()),
                "has_code_blocks": "```" in user_goal,
                "has_urls": bool(re.search(r'https?://', user_goal))
            }
        )
        
        self.analysis_history.append(result)
        return result
    
    def _classify_task_type(self, goal_lower: str) -> TaskType:
        """Classify the primary task type"""
        scores = {task_type: 0 for task_type in TaskType}
        
        for task_type, keywords in self.TASK_TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in goal_lower:
                    scores[task_type] += 1
        
        # Get highest score
        max_score = max(scores.values())
        if max_score == 0:
            return TaskType.CODING  # Default
        
        # Check for mixed
        high_scorers = [t for t, s in scores.items() if s >= max_score * 0.7]
        if len(high_scorers) > 2:
            return TaskType.MIXED
        
        return max(scores, key=scores.get)
    
    def _estimate_complexity(self, goal_lower: str, full_goal: str) -> Complexity:
        """Estimate task complexity"""
        score = 0
        
        # Length indicators
        word_count = len(full_goal.split())
        if word_count > 100:
            score += 3
        elif word_count > 50:
            score += 2
        elif word_count > 20:
            score += 1
        
        # Complexity keywords
        for keyword in self.COMPLEXITY_INDICATORS["complex"]:
            if keyword in goal_lower:
                score += 2
        
        for keyword in self.COMPLEXITY_INDICATORS["integrations"]:
            if keyword in goal_lower:
                score += 1
        
        for keyword in self.COMPLEXITY_INDICATORS["scale"]:
            if keyword in goal_lower:
                score += 2
        
        # Simple keywords reduce score
        for keyword in self.COMPLEXITY_INDICATORS["simple"]:
            if keyword in goal_lower:
                score -= 1
        
        # Multiple technologies
        tech_count = sum(1 for entities in self._extract_entities(goal_lower).values() for _ in entities)
        if tech_count > 5:
            score += 2
        elif tech_count > 3:
            score += 1
        
        # Classify
        if score >= 6:
            return Complexity.VERY_COMPLEX
        elif score >= 4:
            return Complexity.COMPLEX
        elif score >= 2:
            return Complexity.MODERATE
        else:
            return Complexity.SIMPLE
    
    def _extract_entities(self, goal_lower: str) -> Dict[str, List[str]]:
        """Extract entities (languages, platforms, frameworks)"""
        entities = {
            "languages": [],
            "platforms": [],
            "frameworks": [],
            "tools": [],
            "security_domains": []
        }
        
        # Languages
        for lang, patterns in self.LANGUAGE_PATTERNS.items():
            if any(p in goal_lower for p in patterns):
                entities["languages"].append(lang)
        
        # Platforms
        for platform, patterns in self.PLATFORM_PATTERNS.items():
            if any(p in goal_lower for p in patterns):
                entities["platforms"].append(platform)
        
        # Security domains
        for domain, patterns in self.SECURITY_PATTERNS.items():
            if any(p in goal_lower for p in patterns):
                entities["security_domains"].append(domain)
        
        # Tools (extract from context)
        tool_keywords = ["docker", "kubernetes", "redis", "postgres", "mongodb", "nginx", "apache"]
        for tool in tool_keywords:
            if tool in goal_lower:
                entities["tools"].append(tool)
        
        return entities
    
    def _extract_requirements(self, goal_lower: str, full_goal: str) -> Dict[str, Any]:
        """Extract functional and non-functional requirements"""
        requirements = {
            "functional": [],
            "non_functional": {
                "performance": False,
                "security": False,
                "scalability": False,
                "maintainability": False
            },
            "testing": False,
            "documentation": False
        }
        
        # Non-functional requirements
        if any(w in goal_lower for w in ["fast", "quick", "performance", "optimize", "speed"]):
            requirements["non_functional"]["performance"] = True
        
        if any(w in goal_lower for w in ["secure", "security", "safe", "protect", "encrypt"]):
            requirements["non_functional"]["security"] = True
        
        if any(w in goal_lower for w in ["scalable", "scale", "distributed", "cluster"]):
            requirements["non_functional"]["scalability"] = True
        
        if any(w in goal_lower for w in ["clean", "maintainable", "readable", "well-structured"]):
            requirements["non_functional"]["maintainability"] = True
        
        # Testing required?
        if any(w in goal_lower for w in ["test", "testing", "tested", "unittest", "pytest"]):
            requirements["testing"] = True
        
        # Documentation required?
        if any(w in goal_lower for w in ["document", "documentation", "readme", "docs"]):
            requirements["documentation"] = True
        
        return requirements
    
    def _extract_constraints(self, goal_lower: str) -> List[str]:
        """Extract constraints from goal"""
        constraints = []
        
        # Time constraints
        if any(w in goal_lower for w in ["urgent", "asap", "quickly", "fast"]):
            constraints.append("time_sensitive")
        
        # Resource constraints
        if any(w in goal_lower for w in ["minimal", "lightweight", "small", "simple"]):
            constraints.append("resource_constrained")
        
        # Technology constraints
        if "only" in goal_lower or "must use" in goal_lower:
            constraints.append("technology_constrained")
        
        # Security constraints
        if any(w in goal_lower for w in ["production", "prod", "enterprise", "secure"]):
            constraints.append("production_grade")
        
        return constraints
    
    def _detect_ambiguities(self, goal_lower: str) -> List[Dict[str, str]]:
        """Detect ambiguous or unclear requirements"""
        ambiguities = []
        
        # Vague terms
        vague_terms = {
            "best": "What criteria define 'best'? (performance, readability, maintainability)",
            "optimal": "What should be optimized? (speed, memory, code clarity)",
            "good": "What makes it 'good'? Define success criteria",
            "nice": "What specific features are needed?",
            "proper": "What standards or requirements should be met?"
        }
        
        for term, question in vague_terms.items():
            if term in goal_lower and len(goal_lower.split()) < 20:
                ambiguities.append({
                    "issue": f"Vague term: '{term}'",
                    "question": question
                })
        
        # Conflicting requirements
        if "fast" in goal_lower and "comprehensive" in goal_lower:
            ambiguities.append({
                "issue": "Conflicting requirements: fast vs comprehensive",
                "question": "Priority: speed or completeness?"
            })
        
        # Security ambiguity
        if "hack" in goal_lower or "exploit" in goal_lower:
            if "ctf" not in goal_lower and "test" not in goal_lower:
                ambiguities.append({
                    "issue": "Security context unclear",
                    "question": "Purpose: (1) Authorized security testing, (2) CTF challenge, or (3) Educational?"
                })
        
        return ambiguities
    
    def _suggest_pattern(
        self,
        task_type: TaskType,
        complexity: Complexity,
        entities: Dict[str, List[str]]
    ) -> Optional[str]:
        """Suggest agentic pattern based on analysis"""
        # Security assessments -> security pipeline or CTF swarm
        if task_type == TaskType.SECURITY_ASSESSMENT:
            if "web_security" in entities.get("security_domains", []):
                return "security_pipeline"  # web_pentester -> retester -> report
            else:
                return "ctf_swarm"  # Decentralized security team
        
        # Complex coding with testing -> recursive
        if task_type == TaskType.CODING and complexity in [Complexity.COMPLEX, Complexity.VERY_COMPLEX]:
            return "code_review_recursive"
        
        # Multiple analysis types -> parallel
        if task_type == TaskType.MIXED or len(entities.get("languages", [])) > 2:
            return "parallel_analysis"
        
        # Simple linear tasks -> chain
        if complexity == Complexity.SIMPLE:
            return "chain"
        
        # Default hierarchical for moderate complexity
        return "hierarchical"
    
    def _suggest_agents(self, task_type: TaskType, entities: Dict[str, List[str]]) -> List[str]:
        """Suggest agents based on task type"""
        agents = []
        
        if task_type == TaskType.SECURITY_ASSESSMENT:
            agents.extend(["web_pentester", "retester", "report_agent"])
        elif task_type == TaskType.CODING:
            agents.append("coder_qwen")
            if entities.get("testing"):
                agents.append("test_runner")
        elif task_type == TaskType.DOCUMENTATION:
            agents.append("linguistic_dictalm")
        elif task_type == TaskType.TESTING:
            agents.append("test_runner")
        else:
            agents.append("coder_qwen")  # Default
        
        return agents
    
    def _estimate_task_count(self, complexity: Complexity, goal_lower: str) -> int:
        """Estimate number of tasks required"""
        base_count = {
            Complexity.SIMPLE: 1,
            Complexity.MODERATE: 3,
            Complexity.COMPLEX: 5,
            Complexity.VERY_COMPLEX: 8
        }
        
        count = base_count[complexity]
        
        # Adjust based on keywords
        if "and" in goal_lower:
            count += goal_lower.count(" and ")
        
        return min(count, 15)  # Cap at 15
    
    def _calculate_confidence(
        self,
        task_type: TaskType,
        entities: Dict[str, List[str]],
        ambiguities: List[Dict[str, str]],
        goal_lower: str
    ) -> float:
        """Calculate confidence score (0-1)"""
        confidence = 1.0
        
        # Reduce confidence for ambiguities
        confidence -= len(ambiguities) * 0.15
        
        # Reduce confidence for vague goals
        if len(goal_lower.split()) < 5:
            confidence -= 0.2
        
        # Increase confidence for specific entities
        total_entities = sum(len(v) for v in entities.values())
        if total_entities > 3:
            confidence += 0.1
        
        # Reduce confidence for mixed task types
        if task_type == TaskType.MIXED:
            confidence -= 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def get_prompt_enhancement(self, result: 'IntentAnalysisResult') -> str:
        """
        Generate prompt enhancement based on analysis results.
        Provides additional context to help Nemotron decompose better.
        
        Args:
            result: IntentAnalysisResult from analyze()
            
        Returns:
            Enhancement string to append to orchestrator prompt
        """
        enhancements = []
        
        # Add entity context
        if result.entities.get("languages"):
            enhancements.append(f"Languages: {', '.join(result.entities['languages'])}")
        
        if result.entities.get("platforms"):
            enhancements.append(f"Platforms: {', '.join(result.entities['platforms'])}")
        
        if result.entities.get("security_domains"):
            enhancements.append(f"Security focus: {', '.join(result.entities['security_domains'])}")
        
        # Add requirement hints
        if result.requirements.get("testing"):
            enhancements.append("Include testing tasks")
        
        if result.requirements.get("documentation"):
            enhancements.append("Include documentation")
        
        # Add non-functional requirements
        nfr = result.requirements.get("non_functional", {})
        active_nfr = [k for k, v in nfr.items() if v]
        if active_nfr:
            enhancements.append(f"NFR priorities: {', '.join(active_nfr)}")
        
        # Add constraint hints
        if result.constraints:
            enhancements.append(f"Constraints: {', '.join(result.constraints)}")
        
        # Add complexity hint
        enhancements.append(f"Complexity: {result.complexity.value}")
        
        # Add pattern suggestion
        if result.suggested_pattern:
            enhancements.append(f"Suggested pattern: {result.suggested_pattern}")
        
        return "\n".join(enhancements) if enhancements else ""


if __name__ == "__main__":
    analyzer = IntentAnalyzer()
    
    print("="*60)
    print("ENHANCED INTENT ANALYZER TEST")
    print("="*60)
    
    test_goals = [
        "Build a secure REST API with Python FastAPI",
        "Perform security assessment of web application",
        "Write tests for existing Python code",
        "Create a simple hello world script",
        "Develop comprehensive microservices architecture with Docker, Kubernetes, Redis, and PostgreSQL"
    ]
    
    for goal in test_goals:
        print(f"\n{'='*60}")
        print(f"GOAL: {goal}")
        print(f"{'='*60}")
        
        result = analyzer.analyze(goal)
        
        print(f"Task Type: {result.task_type.value}")
        print(f"Complexity: {result.complexity.value}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Entities: {result.entities}")
        print(f"Suggested Pattern: {result.suggested_pattern}")
        print(f"Suggested Agents: {result.suggested_agents}")
        print(f"Estimated Tasks: {result.estimated_tasks}")
        print(f"Needs Clarification: {result.needs_clarification}")
        
        if result.ambiguities:
            print(f"Ambiguities:")
            for amb in result.ambiguities:
                print(f"  - {amb['issue']}: {amb['question']}")
    
    print(f"\n{'='*60}")
    print("Intent analyzer working correctly!")
