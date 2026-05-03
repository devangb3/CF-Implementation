import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from pymongo import MongoClient, ASCENDING, DESCENDING
from dotenv import load_dotenv
import json

SNAPSHOT_MAX_CHARS = 300
TEXT_MAX_CHARS = 1000

class MongoDBStorage:

    def __init__(self):
        load_dotenv()
        db_name = os.getenv('MONGODB_NAME')
        self.mongo_uri = os.getenv('MONGODB_URI')

        if not self.mongo_uri:
            raise ValueError(
                "MONGODB_URI not found. Please set it in .env file or pass it as parameter."
            )

        self.client = self._get_client()
        db_name = db_name if db_name else "causalflow"
        self.db = self.client[db_name]
        self.runs = self.db['runs']
        self._setup_indexes()
        self.current_run_id: Optional[str] = None
    
    def _get_client(self) -> MongoClient:
        try:
            kwargs: Dict[str, str] = {}
            if os.getenv('MONGODB_AWS_ACCESS_KEY'):
                kwargs["username"] = os.getenv('MONGODB_AWS_ACCESS_KEY') or ""
                kwargs["password"] = os.getenv('MONGODB_AWS_SECRET_KEY') or ""
                kwargs["authMechanism"] = "MONGODB-AWS"

            client: MongoClient = MongoClient(
                self.mongo_uri,
                serverSelectionTimeoutMS=10000,
                connectTimeoutMS=10000,
                maxPoolSize=100,
                minPoolSize=10,
                **kwargs
            )
        except Exception as e:
            raise Exception(f"Failed to connect to MongoDB: {e}")
                
        return client

    def _setup_indexes(self) -> None:
        self.runs.create_index([("run_id", ASCENDING)], unique=True)
        self.runs.create_index([("timestamp", DESCENDING)])
        self.runs.create_index([("experiment_name", ASCENDING)])

    def _truncate(self, text: Optional[str], max_chars: int) -> Optional[str]:
        if text is None:
            return None
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."

    def _convert_keys_to_strings(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): self._convert_keys_to_strings(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_keys_to_strings(item) for item in obj]
        else:
            return obj

    def _parse_trace_json(self, trace_data: Any) -> Dict[str, Any]:
        if isinstance(trace_data, str):
            return json.loads(trace_data)
        return trace_data

    def _compact_trace(self, trace_obj: Dict[str, Any]) -> Dict[str, Any]:
        """Keep trace structure but truncate large string fields."""
        compact = {
            "success": trace_obj.get("success"),
            "final_answer": trace_obj.get("final_answer"),
            "gold_answer": trace_obj.get("gold_answer"),
            "problem_statement": trace_obj.get("problem_statement"),
            "num_steps": trace_obj.get("num_steps"),
        }
        
        steps = trace_obj.get("steps", [])
        compact_steps: List[Dict[str, Any]] = []
        
        for step in steps:
            compact_step: Dict[str, Any] = {
                "step_id": step.get("step_id"),
                "step_type": step.get("step_type"),
                "dependencies": step.get("dependencies", []),
            }
            
            # Keep tool_name and tool_args (small)
            if step.get("tool_name"):
                compact_step["tool_name"] = step["tool_name"]
            if step.get("tool_args"):
                compact_step["tool_args"] = step["tool_args"]
            if step.get("tool_call_result") is not None:
                compact_step["tool_call_result"] = step["tool_call_result"]
            
            # Truncate large text fields
            if step.get("text"):
                compact_step["text"] = self._truncate(step["text"], TEXT_MAX_CHARS)
            if step.get("tool_output"):
                compact_step["tool_output"] = self._truncate(str(step["tool_output"]), SNAPSHOT_MAX_CHARS)
            if step.get("observation"):
                compact_step["observation"] = self._truncate(step["observation"], SNAPSHOT_MAX_CHARS)
            if step.get("action"):
                compact_step["action"] = self._truncate(step["action"], TEXT_MAX_CHARS)
            if step.get("memory_key"):
                compact_step["memory_key"] = step["memory_key"]
            if step.get("memory_value"):
                compact_step["memory_value"] = self._truncate(str(step["memory_value"]), SNAPSHOT_MAX_CHARS)
            if step.get("state_snapshot"):
                snapshot_str = json.dumps(step["state_snapshot"]) if isinstance(step["state_snapshot"], dict) else str(step["state_snapshot"])
                compact_step["state_snapshot"] = self._truncate(snapshot_str, SNAPSHOT_MAX_CHARS)
            
            compact_steps.append(compact_step)
        
        compact["steps"] = compact_steps
        return compact

    def _compact_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Compact a single step dict, truncating large fields."""
        compact: Dict[str, Any] = {
            "step_id": step.get("step_id"),
            "step_type": step.get("step_type"),
        }
        if step.get("tool_name"):
            compact["tool_name"] = step["tool_name"]
        if step.get("tool_args"):
            compact["tool_args"] = step["tool_args"]
        if step.get("text"):
            compact["text"] = step["text"]
        if step.get("tool_output"):
            compact["tool_output"] = self._truncate(str(step["tool_output"]), SNAPSHOT_MAX_CHARS)
        return compact

    def _compact_analysis(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Keep analysis structure but truncate large string fields."""
        compact: Dict[str, Any] = {}
        
        # Keep causal_graph (usually small - just step relationships)
        if "causal_graph" in analysis_results:
            compact["causal_graph"] = analysis_results["causal_graph"]
        
        # Keep causal_attribution (step IDs and scores)
        if "causal_attribution" in analysis_results:
            compact["causal_attribution"] = analysis_results["causal_attribution"]
        
        # Compact the repairs - truncate large step data
        if "counterfactual_repair" in analysis_results:
            cf_repair = analysis_results["counterfactual_repair"]
            compact_cf: Dict[str, Any] = {
                "num_steps_repaired": cf_repair.get("num_steps_repaired"),
                "num_successful_repairs": cf_repair.get("num_successful_repairs"),
            }
            
            def _compact_proposals(proposals: Any) -> List[Dict[str, Any]]:
                out: List[Dict[str, Any]] = []
                if not isinstance(proposals, list):
                    return out
                for p in proposals:
                    if not isinstance(p, dict):
                        continue
                    out.append({
                        "proposal_idx": p.get("proposal_idx"),
                        "minimality_lex": p.get("minimality_lex"),
                        "minimality_edit": p.get("minimality_edit"),
                        "minimality_sem": p.get("minimality_sem"),
                        "success_predicted": p.get("success_predicted"),
                        "original_text": self._truncate(p.get("original_text"), TEXT_MAX_CHARS),
                        "repaired_text": self._truncate(p.get("repaired_text"), TEXT_MAX_CHARS),
                    })
                return out

            if "successful_repairs" in cf_repair:
                compact_repairs: Dict[str, Any] = {}
                for step_id, repair_data in cf_repair["successful_repairs"].items():
                    if isinstance(repair_data, dict):
                        repair_entry: Dict[str, Any] = {
                            "minimality_score": repair_data.get("minimality_score"),
                            "minimality_lex": repair_data.get("minimality_lex"),
                            "minimality_edit": repair_data.get("minimality_edit"),
                            "minimality_sem": repair_data.get("minimality_sem"),
                            "proposal_idx": repair_data.get("proposal_idx"),
                            "success_predicted": repair_data.get("success_predicted"),
                            "original_step": self._compact_step(repair_data.get("original_step", {})),
                            "repaired_step": self._compact_step(repair_data.get("repaired_step", {})),
                            "all_proposals": _compact_proposals(repair_data.get("all_proposals", [])),
                        }
                        # Include full repaired trace for successful repairs (not compacted)
                        if repair_data.get("success_predicted") and repair_data.get("repaired_trace"):
                            repair_entry["repaired_trace"] = repair_data["repaired_trace"]
                        compact_repairs[str(step_id)] = repair_entry
                compact_cf["successful_repairs"] = compact_repairs

            if "all_proposals_by_step" in cf_repair:
                compact_all_by_step: Dict[str, List[Dict[str, Any]]] = {}
                for step_id, proposals in cf_repair["all_proposals_by_step"].items():
                    compact_all_by_step[str(step_id)] = _compact_proposals(proposals)
                compact_cf["all_proposals_by_step"] = compact_all_by_step

            compact["counterfactual_repair"] = compact_cf
        
        # Keep multi_agent_critique with full judge ensemble output
        if "multi_agent_critique" in analysis_results:
            mac = analysis_results["multi_agent_critique"]
            compact_mac: Dict[str, Any] = {
                "skipped": mac.get("skipped", False),
            }
            
            if mac.get("skipped"):
                compact_mac["reason"] = mac.get("reason", "")
            else:
                compact_mac["num_steps_critiqued"] = mac.get("num_steps_critiqued", 0)
                
                # Store full critique details including judge ensemble reasoning
                if "critique_details" in mac:
                    compact_critiques: Dict[str, Any] = {}
                    for step_id, critique_data in mac["critique_details"].items():
                        if isinstance(critique_data, dict):
                            critique_entry: Dict[str, Any] = {
                                "step_id": critique_data.get("step_id"),
                                "proposed_by": critique_data.get("proposed_by"),
                                "consensus_score": critique_data.get("consensus_score"),
                                "final_verdict": critique_data.get("final_verdict"),
                                "num_critiques": critique_data.get("num_critiques"),
                                "judge_ensemble": []
                            }
                            # Include full judge ensemble output (reasoning from each agent)
                            for judge in critique_data.get("judge_ensemble", []):
                                judge_entry: Dict[str, Any] = {
                                    "agent": judge.get("agent"),
                                    "role": judge.get("role"),
                                    "agrees": judge.get("agrees"),
                                    "confidence": judge.get("confidence"),
                                    "reasoning": self._truncate(judge.get("reasoning", ""), TEXT_MAX_CHARS),
                                    "agreement": judge.get("agreement"),
                                    "evidence_strength": judge.get("evidence_strength"),
                                }
                                critique_entry["judge_ensemble"].append(judge_entry)
                            compact_critiques[str(step_id)] = critique_entry
                    compact_mac["critique_details"] = compact_critiques
            
            # Compact consensus_steps if present
            if "consensus_steps" in mac:
                compact_mac["consensus_steps"] = [
                    self._compact_step(step) for step in mac["consensus_steps"]
                ]
            
            compact["multi_agent_critique"] = compact_mac
        
        return compact

    def _build_metrics_document(self, metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not metrics:
            return {}
        
        minimality = metrics.get('minimality_metrics', {})
        attribution = metrics.get('causal_attribution_metrics', {})
        repairs = metrics.get('repair_metrics', {})
        
        return {
            "minimality": {
                "average": minimality.get("average_minimality"),
                "min": minimality.get("min_minimality"),
                "max": minimality.get("max_minimality"),
            },
            "attribution": {
                "num_identified_causal_steps": attribution.get("num_identified_causal_steps"),
                "identified_steps": attribution.get("identified_steps", []),
                "precision": attribution.get("precision"),
                "recall": attribution.get("recall"),
                "f1_score": attribution.get("f1_score"),
            },
            "repairs": {
                "total_repairs_attempted": repairs.get("total_repairs_attempted"),
                "successful_repairs": repairs.get("successful_repairs"),
                "failed_repairs": repairs.get("failed_repairs"),
                "success_rate": repairs.get("success_rate"),
            },
        }

    def create_run(
        self,
        experiment_name: str,
        num_problems: int,
        model_used: str,
    ) -> str:

        timestamp = datetime.utcnow().isoformat()
        run_id = f"run_{experiment_name}_{timestamp}"

        document: Dict[str, Any] = {
            "run_id": run_id,
            "experiment_name": experiment_name,
            "timestamp": timestamp,
            "num_problems": num_problems,
            "model_used": model_used,
            "passing_traces": [],
            "failing_traces": [],
            "stats": {
                "total": 0,
                "passing": 0,
                "failing": 0,
                "fixed": 0,
                "analyzed": 0,
                "accuracy": 0.0
            }
        }

        self.runs.insert_one(document)
        self.current_run_id = run_id
        print(f"Created new run: {run_id}")
        return run_id

    def add_passing_trace(
        self,
        run_id: str,
        trace_data: Any,
        problem_id: Any,
        problem_statement: str,
        gold_answer: Any,
        final_answer: Any,
        causal_flow_analysis_time_minutes: Optional[float] = None
    ) -> None:
        trace_obj = self._parse_trace_json(trace_data)
        trace_obj = self._convert_keys_to_strings(trace_obj)
        compact_trace = self._compact_trace(trace_obj)

        trace_document: Dict[str, Any] = {
            "problem_id": problem_id,
            "timestamp": datetime.utcnow().isoformat(),
            "success": True,
            "problem_statement": problem_statement,
            "gold_answer": gold_answer,
            "final_answer": final_answer,
            "trace": compact_trace,
            "causal_flow_analysis_time_minutes": causal_flow_analysis_time_minutes
        }

        self.runs.update_one(
            {"run_id": run_id},
            {
                "$push": {"passing_traces": trace_document},
                "$inc": {"stats.total": 1, "stats.passing": 1}
            }
        )

    def add_failing_trace(
        self,
        run_id: str,
        trace_data: Any,
        problem_id: Any,
        problem_statement: str,
        gold_answer: Any,
        final_answer: Any,
        analysis_results: Optional[Dict[str, Any]],
        metrics: Optional[Dict[str, Any]],
        causal_flow_analysis_time_minutes: Optional[float] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        trace_obj = self._parse_trace_json(trace_data)
        trace_obj = self._convert_keys_to_strings(trace_obj)
        compact_trace = self._compact_trace(trace_obj)

        if analysis_results is None:
            analysis_results = {}
        if metrics is None:
            metrics = {}

        analysis_results = self._convert_keys_to_strings(analysis_results)
        compact_analysis = self._compact_analysis(analysis_results)

        trace_document: Dict[str, Any] = {
            "problem_id": problem_id,
            "timestamp": datetime.utcnow().isoformat(),
            "success": False,
            "problem_statement": problem_statement,
            "gold_answer": gold_answer,
            "final_answer": final_answer,
            "causal_flow_analysis_time_minutes": causal_flow_analysis_time_minutes,
            "trace": compact_trace,
            "analysis": compact_analysis,
            "metrics": self._build_metrics_document(metrics),
        }
        if extra_metadata:
            trace_document.update(extra_metadata)

        self.runs.update_one(
            {"run_id": run_id},
            {
                "$push": {"failing_traces": trace_document},
                "$inc": {"stats.total": 1, "stats.failing": 1}
            }
        )

        print(f"Added failing trace for problem {problem_id}")

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        return self.runs.find_one({"run_id": run_id})

    def update_run_statistics(
        self,
        run_id: str,
        fixed: int,
        analyzed: int,
        accuracy: float,
        total_experiment_time_minutes: float = 0.0
    ) -> None:

        update_data: Dict[str, Any] = {
            "$set": {
                "stats.fixed": fixed,
                "stats.analyzed": analyzed,
                "stats.accuracy": accuracy,
                "stats.total_experiment_time_minutes": total_experiment_time_minutes
            }
        }
        
        self.runs.update_one(
            {"run_id": run_id},
            update_data
        )

    def get_run_statistics(self, run_id: str) -> Dict[str, Any]:
        run = self.get_run(run_id)
        if not run:
            return {}

        stats = run.get("stats", {})
        return {
            "run_id": run_id,
            "experiment_name": run.get("experiment_name"),
            "timestamp": run.get("timestamp"),
            "total_traces": stats.get("total", 0),
            "passing_traces": stats.get("passing", 0),
            "failing_traces": stats.get("failing", 0),
            "fixed": stats.get("fixed", 0),
            "analyzed": stats.get("analyzed", 0),
            "accuracy": stats.get("accuracy", 0)
        }

    def close(self) -> None:
        self.client.close()
        print("MongoDB connection closed")
