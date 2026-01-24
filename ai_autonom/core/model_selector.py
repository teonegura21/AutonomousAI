#!/usr/bin/env python3
"""
Dynamic Model Selection
Select the best model for each task based on capabilities and constraints
"""

import sqlite3
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class ModelScore:
    """Score and metadata for a model"""
    model_name: str
    coding_score: float
    reasoning_score: float
    documentation_score: float
    speed_tokens_sec: float
    vram_gb: float
    composite_score: float = 0.0


class DynamicModelSelector:
    """
    Select optimal model for task based on capabilities and constraints
    The orchestrator uses this to dynamically assign the best available model
    """
    
    def __init__(self, db_path: str = ".runtime/data/agent_registry.db"):
        self.db_path = db_path
    
    def get_all_models(self) -> List[ModelScore]:
        """Get all active models with their scores"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT model_name, coding_score, reasoning_score, 
                       documentation_score, speed_tokens_sec, vram_gb
                FROM model_capabilities 
                WHERE is_active = 1
            """)
            rows = cursor.fetchall()
        except sqlite3.OperationalError:
            rows = []
        
        conn.close()
        
        return [
            ModelScore(
                model_name=row[0],
                coding_score=row[1] or 0,
                reasoning_score=row[2] or 0,
                documentation_score=row[3] or 0,
                speed_tokens_sec=row[4] or 0,
                vram_gb=row[5] or 0
            )
            for row in rows
        ]
    
    def select_best_model(
        self,
        task_type: str,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Select optimal model for task type and return its scoring details.
        
        Args:
            task_type: "coding", "reasoning", "documentation", "fast", "balanced"
            constraints: Optional constraints like {"max_vram": 4.0, "min_speed": 50}
        
        Returns:
            Dict with model_name, composite_score and related metrics, or None if no model found
        """
        models = self.get_all_models()
        
        if not models:
            return None
        
        constraints = constraints or {}
        
        # Apply constraints
        filtered_models = self._apply_constraints(models, constraints)
        
        if not filtered_models:
            return None
        
        # Calculate composite scores based on task type
        scored_models = self._score_for_task(filtered_models, task_type)
        scored_models.sort(key=lambda m: m.composite_score, reverse=True)
        
        best = scored_models[0] if scored_models else None
        if not best:
            return None
        
        return {
            "model_name": best.model_name,
            "composite_score": best.composite_score,
            "coding_score": best.coding_score,
            "reasoning_score": best.reasoning_score,
            "documentation_score": best.documentation_score,
            "speed_tokens_sec": best.speed_tokens_sec,
            "vram_gb": best.vram_gb,
        }
    
    def _apply_constraints(
        self,
        models: List[ModelScore],
        constraints: Dict[str, Any]
    ) -> List[ModelScore]:
        """Filter models based on constraints"""
        filtered = models.copy()
        
        if "max_vram" in constraints:
            filtered = [m for m in filtered if m.vram_gb <= constraints["max_vram"]]
        
        if "min_vram" in constraints:
            filtered = [m for m in filtered if m.vram_gb >= constraints["min_vram"]]
        
        if "min_speed" in constraints:
            filtered = [m for m in filtered if m.speed_tokens_sec >= constraints["min_speed"]]
        
        if "min_coding" in constraints:
            filtered = [m for m in filtered if m.coding_score >= constraints["min_coding"]]
        
        if "min_reasoning" in constraints:
            filtered = [m for m in filtered if m.reasoning_score >= constraints["min_reasoning"]]
        
        if "exclude_models" in constraints:
            excluded = set(constraints["exclude_models"])
            filtered = [m for m in filtered if m.model_name not in excluded]
        
        return filtered
    
    def _score_for_task(
        self,
        models: List[ModelScore],
        task_type: str
    ) -> List[ModelScore]:
        """Calculate composite scores based on task type"""
        
        # Weight profiles for different task types
        weights = {
            "coding": {
                "coding": 0.6,
                "reasoning": 0.2,
                "documentation": 0.1,
                "speed": 0.1
            },
            "reasoning": {
                "coding": 0.1,
                "reasoning": 0.6,
                "documentation": 0.2,
                "speed": 0.1
            },
            "documentation": {
                "coding": 0.1,
                "reasoning": 0.2,
                "documentation": 0.6,
                "speed": 0.1
            },
            "fast": {
                "coding": 0.1,
                "reasoning": 0.1,
                "documentation": 0.1,
                "speed": 0.7
            },
            "balanced": {
                "coding": 0.3,
                "reasoning": 0.3,
                "documentation": 0.2,
                "speed": 0.2
            }
        }
        
        w = weights.get(task_type, weights["balanced"])
        
        # Normalize speed scores (assuming max ~100 tokens/sec is excellent)
        max_speed = max((m.speed_tokens_sec for m in models), default=1) or 1
        
        for model in models:
            normalized_speed = (model.speed_tokens_sec / max_speed) * 100
            
            model.composite_score = (
                w["coding"] * model.coding_score +
                w["reasoning"] * model.reasoning_score +
                w["documentation"] * model.documentation_score +
                w["speed"] * normalized_speed
            )
        
        return models
    
    def get_model_for_capability(
        self,
        capability: str,
        max_vram: float = 20.0
    ) -> Optional[str]:
        """
        Get best model for a specific capability
        
        Args:
            capability: One of "code_generation", "debugging", "testing", 
                       "documentation", "summarization", "text_generation"
        
        Returns:
            Best model name
        """
        # Map capabilities to task types
        capability_map = {
            "code_generation": "coding",
            "debugging": "coding",
            "testing": "coding",
            "refactoring": "coding",
            "python": "coding",
            "technical_tasks": "coding",
            "documentation": "documentation",
            "summarization": "documentation",
            "text_generation": "documentation",
            "formatting": "documentation",
            "explanation": "documentation",
            "simple_tasks": "fast",
            "task_decomposition": "reasoning",
            "planning": "reasoning",
            "analysis": "reasoning",
            "reasoning": "reasoning",
            "math": "reasoning"
        }
        
        task_type = capability_map.get(capability, "balanced")
        best = self.select_best_model(task_type, {"max_vram": max_vram})
        return best["model_name"] if isinstance(best, dict) else best
    
    def should_replace(
        self,
        current_model: str,
        new_model: str,
        task_type: str
    ) -> bool:
        """
        Check if new model is better than current for the task type
        Used when a new model is discovered to see if it should replace existing
        """
        models = self.get_all_models()
        
        current = next((m for m in models if m.model_name == current_model), None)
        new = next((m for m in models if m.model_name == new_model), None)
        
        if not current or not new:
            return False
        
        # Score both for the task type
        scored = self._score_for_task([current, new], task_type)
        
        current_score = next((m.composite_score for m in scored if m.model_name == current_model), 0)
        new_score = next((m.composite_score for m in scored if m.model_name == new_model), 0)
        
        # New model should be significantly better (10%+) to warrant replacement
        return new_score > current_score * 1.1
    
    def get_model_rankings(self, task_type: str = "balanced") -> List[Dict[str, Any]]:
        """Get all models ranked for a specific task type"""
        models = self.get_all_models()
        
        if not models:
            return []
        
        scored = self._score_for_task(models, task_type)
        scored.sort(key=lambda m: m.composite_score, reverse=True)
        
        return [
            {
                "rank": i + 1,
                "model_name": m.model_name,
                "composite_score": round(m.composite_score, 1),
                "coding_score": m.coding_score,
                "reasoning_score": m.reasoning_score,
                "documentation_score": m.documentation_score,
                "speed_tokens_sec": round(m.speed_tokens_sec, 1),
                "vram_gb": m.vram_gb
            }
            for i, m in enumerate(scored)
        ]
    
    def recommend_agent_assignment(
        self,
        task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Recommend which model should handle a task based on its properties
        
        Args:
            task: Task dict with 'description', 'required_capability', etc.
        
        Returns:
            Recommendation with model_name, confidence, reasoning
        """
        capability = task.get('required_capability', 'balanced')
        constraints = task.get('constraints', {})
        
        # Get best model
        best_model = self.get_model_for_capability(capability, constraints.get('max_vram', 20.0))
        
        if not best_model:
            return {
                "model_name": None,
                "confidence": 0.0,
                "reasoning": "No suitable model found for the given constraints"
            }
        
        # Get rankings to determine confidence
        rankings = self.get_model_rankings(
            self._capability_to_task_type(capability)
        )
        
        top_model = rankings[0] if rankings else None
        
        confidence = 0.8  # Base confidence
        reasoning = []
        
        if top_model:
            if top_model['composite_score'] >= 80:
                confidence = 0.95
                reasoning.append(f"High composite score ({top_model['composite_score']:.1f})")
            elif top_model['composite_score'] >= 60:
                confidence = 0.85
                reasoning.append(f"Good composite score ({top_model['composite_score']:.1f})")
            else:
                confidence = 0.7
                reasoning.append(f"Moderate composite score ({top_model['composite_score']:.1f})")
            
            # Check if there's a clear winner
            if len(rankings) >= 2:
                score_diff = rankings[0]['composite_score'] - rankings[1]['composite_score']
                if score_diff > 10:
                    confidence += 0.05
                    reasoning.append(f"Clear winner by {score_diff:.1f} points")
        
        return {
            "model_name": best_model,
            "confidence": min(confidence, 1.0),
            "reasoning": "; ".join(reasoning) if reasoning else "Default selection"
        }
    
    def _capability_to_task_type(self, capability: str) -> str:
        """Convert capability to task type"""
        coding_caps = ["code_generation", "debugging", "testing", "refactoring", "python", "technical_tasks"]
        doc_caps = ["documentation", "summarization", "text_generation", "formatting", "explanation"]
        reason_caps = ["task_decomposition", "planning", "analysis", "meta_orchestration"]
        
        if capability in coding_caps:
            return "coding"
        elif capability in doc_caps:
            return "documentation"
        elif capability in reason_caps:
            return "reasoning"
        elif capability == "simple_tasks":
            return "fast"
        return "balanced"


if __name__ == "__main__":
    # Test model selector
    selector = DynamicModelSelector()
    
    print("\n" + "="*60)
    print("DYNAMIC MODEL SELECTOR TEST")
    print("="*60 + "\n")
    
    # Get all models
    models = selector.get_all_models()
    print(f"Found {len(models)} registered models\n")
    
    # Test selection for different task types
    task_types = ["coding", "reasoning", "documentation", "fast", "balanced"]
    
    for task_type in task_types:
        best = selector.select_best_model(task_type)
        print(f"Best model for '{task_type}': {best}")
    
    print("\n" + "-"*60)
    print("Model Rankings (Balanced):")
    print("-"*60 + "\n")
    
    rankings = selector.get_model_rankings("balanced")
    for r in rankings[:5]:
        print(f"  {r['rank']}. {r['model_name']}")
        print(f"     Score: {r['composite_score']} | VRAM: {r['vram_gb']}GB | Speed: {r['speed_tokens_sec']}t/s")
