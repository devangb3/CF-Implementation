from typing import Dict, Any, Optional, List, Set
from trace_logger import TraceLogger, Step, StepType
from causal_graph import CausalGraph
from causal_attribution import CausalAttribution
from counterfactual_repair import CounterfactualRepair
from multi_agent_critique import MultiAgentCritique
from llm_client import LLMClient, MultiAgentLLM
from mongodb_storage import MongoDBStorage

class CausalFlow:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-4o-mini",
        num_critique_agents: int = 3, #Number of agents for multi-agent critique
        mongo_storage: Optional[MongoDBStorage] = None
    ):

        self.llm_client = LLMClient(api_key=api_key, model=model)
        self.multi_agent_llm = MultiAgentLLM(
            num_agents=num_critique_agents,
            api_key=api_key
        )

        self.trace: Optional[TraceLogger] = None
        self.causal_graph: Optional[CausalGraph] = None
        self.causal_attribution: Optional[CausalAttribution] = None
        self.counterfactual_repair: Optional[CounterfactualRepair] = None
        self.multi_agent_critique: Optional[MultiAgentCritique] = None
        self.mongo_storage = mongo_storage

    def analyze_trace(
        self,
        trace: TraceLogger,
        reexecutor: Optional[Any] = None,
        execution_context: Optional[Dict[str, Any]] = None,
        skip_critique: bool = False,
        intervene_step_types: Optional[Set[StepType]] = None,
        num_proposals: int = 3,
        compute_semantic_minimality: bool = False,
        use_gold_in_prompts: bool = True,
    ) -> Dict[str, Any]:

        self.trace = trace

        print(f"Constructing causal graph")
        self.causal_graph = CausalGraph(self.trace)

        print(f"Performing causal attribution")
        self.causal_attribution = CausalAttribution(
            trace=self.trace,
            causal_graph=self.causal_graph,
            llm_client=self.llm_client,
            re_executor=reexecutor
        )
        crs_scores = self.causal_attribution.compute_causal_responsibility(
            execution_context=execution_context,
            intervene_step_types=intervene_step_types,
        )
        causal_steps = self.causal_attribution.get_causal_steps()
        print(f"Attribution complete: {len(causal_steps)} causal steps identified")

        print(f"Generating counterfactual repairs (K={num_proposals}, sem={compute_semantic_minimality}, use_gold={use_gold_in_prompts})")
        self.counterfactual_repair = CounterfactualRepair(
            trace=self.trace,
            causal_attribution=self.causal_attribution,
            llm_client=self.llm_client,
            reexecutor=reexecutor,
            execution_context=execution_context,
            num_proposals=num_proposals,
            compute_semantic_minimality=compute_semantic_minimality,
            use_gold_in_prompts=use_gold_in_prompts,
        )
        repairs = self.counterfactual_repair.generate_repairs(step_ids=causal_steps)
        print(f"Repair complete: {sum(len(r) for r in repairs.values())} repairs proposed")
        
        critiques: Dict[int, Any] = {}
        consensus_steps: List[Step] = []
        
        if skip_critique:
            print(f"Skipping multi-agent critique (using deterministic reexecutor)")
            # When skipping critique, use causal steps directly as consensus
            consensus_steps = [self.trace.get_step(step_id) for step_id in causal_steps if self.trace.get_step(step_id)]
        else:
            print(f"Running multi-agent critique")
            self.multi_agent_critique = MultiAgentCritique(
                trace=self.trace,
                causal_attribution=self.causal_attribution,
                multi_agent_llm=self.multi_agent_llm
            )
            critiques = self.multi_agent_critique.critique_causal_attributions()
            consensus_steps = self.multi_agent_critique.get_consensus_causal_steps()
            print(f"Critique complete: {len(consensus_steps)} steps confirmed by consensus")

        print(f"Compiling results")
        results = self._compile_results(
            crs_scores,
            causal_steps,
            repairs,
            critiques,
            consensus_steps,
            skip_critique=skip_critique
        )

        print(f"Generating metrics")
        metrics = self.generate_metrics(consensus_steps, skip_critique=skip_critique)
        results['metrics'] = metrics

        return results

    def _compile_results(
        self,
        crs_scores: Dict[int, float],
        causal_steps: List[int],
        repairs: Dict[int, List],
        critiques: Dict[int, Any],
        consensus_steps: List[Step],
        skip_critique: bool = False
    ) -> Dict[str, Any]:

        results = {
            "trace_summary": {
                "total_steps": len(self.trace.steps) if self.trace else 0,
                "success": self.trace.success if self.trace else False,
                "final_answer": self.trace.final_answer if self.trace else "",
                "gold_answer": self.trace.gold_answer if self.trace else ""
            },
            "causal_graph": {
                "statistics": self.causal_graph.get_statistics() if self.causal_graph else {}
            },
            "causal_attribution": {
                "crs_scores": crs_scores if crs_scores else {},
                "causal_steps": causal_steps if causal_steps else []
            },
            "counterfactual_repair": {},
            "multi_agent_critique": {}
        }

        if self.counterfactual_repair:
            successful_repairs = self.counterfactual_repair.get_all_successful_repairs()
            all_repairs = self.counterfactual_repair.repairs
            compiled_repairs: Dict[str, Dict[str, Any]] = {}

            def _proposals_payload(step_id: int) -> List[Dict[str, Any]]:
                out: List[Dict[str, Any]] = []
                for r in all_repairs.get(step_id, []):
                    out.append({
                        "proposal_idx": r.proposal_idx,
                        "minimality_lex": r.minimality_lex,
                        "minimality_edit": r.minimality_edit,
                        "minimality_sem": r.minimality_sem,
                        "success_predicted": r.success_predicted,
                        "original_text": r.original_text,
                        "repaired_text": r.repaired_text,
                    })
                return out

            for step_id, repair_list in successful_repairs.items():
                if repair_list:
                    best_repair = repair_list[0]
                    repair_entry: Dict[str, Any] = {
                        "minimality_score": best_repair.minimality_lex,
                        "minimality_lex": best_repair.minimality_lex,
                        "minimality_edit": best_repair.minimality_edit,
                        "minimality_sem": best_repair.minimality_sem,
                        "success_predicted": best_repair.success_predicted,
                        "original_step": best_repair.original_step.to_dict(),
                        "repaired_step": best_repair.repaired_step.to_dict(),
                        "proposal_idx": best_repair.proposal_idx,
                        "all_proposals": _proposals_payload(step_id),
                    }
                    if best_repair.success_predicted and best_repair.repaired_trace is not None:
                        repair_entry["repaired_trace"] = best_repair.repaired_trace.to_dict()
                    compiled_repairs[str(step_id)] = repair_entry

            # Also persist proposals for causal steps where no proposal succeeded —
            # needed for the ablation so we can compute metric agreement on the full set.
            all_proposals_by_step: Dict[str, List[Dict[str, Any]]] = {
                str(step_id): _proposals_payload(step_id)
                for step_id in all_repairs.keys()
                if str(step_id) not in compiled_repairs
            }

            results["counterfactual_repair"] = {
                "num_steps_repaired": len(repairs) if repairs else 0,
                "num_successful_repairs": len(successful_repairs) if successful_repairs else 0,
                "successful_repairs": compiled_repairs,
                "all_proposals_by_step": all_proposals_by_step,
            }

        # Add critique results if available
        if skip_critique:
            results["multi_agent_critique"] = {
                "skipped": True,
                "reason": "Deterministic reexecutor used - critique not needed",
                "consensus_steps": [step.to_dict() for step in consensus_steps] if consensus_steps else []
            }
        elif self.multi_agent_critique:
            # Build full critique details including judge ensemble output
            critique_details: Dict[str, Dict[str, Any]] = {}
            for step_id, critique in critiques.items():
                if critique:
                    critique_entry: Dict[str, Any] = {
                        "step_id": critique.step_id,
                        "proposed_by": critique.proposed_by,
                        "consensus_score": critique.consensus_score,
                        "final_verdict": critique.final_verdict,
                        "num_critiques": len(critique.critiques),
                        "judge_ensemble": []  # Full output from each judge
                    }
                    # Store full critique from each judge agent
                    for agent_critique in critique.critiques:
                        judge_output: Dict[str, Any] = {
                            "agent": agent_critique.get("agent", "unknown"),
                            "role": agent_critique.get("role", "unknown"),
                            "agrees": agent_critique.get("agrees", False),
                            "confidence": agent_critique.get("confidence", 0.0),
                            "reasoning": agent_critique.get("response", ""),
                            "agreement": agent_critique.get("agreement", ""),
                            "evidence_strength": agent_critique.get("evidence_strength", ""),
                        }
                        critique_entry["judge_ensemble"].append(judge_output)
                    critique_details[str(step_id)] = critique_entry

            results["multi_agent_critique"] = {
                "skipped": False,
                "num_steps_critiqued": len(critiques) if critiques else 0,
                "consensus_steps": [step.to_dict() for step in consensus_steps] if consensus_steps else [],
                "critique_details": critique_details
            }

        return results

    def generate_metrics(
        self,
        consensus_steps: List[Step],
        skip_critique: bool = False
    ) -> Dict[str, Any]:

        if not self.causal_attribution:
            raise ValueError("No analysis has been performed yet. Call analyze_trace() first.")

        metrics = {}

        # 1. Causal Attribution Metrics
        initial_identified_step_ids = self.causal_attribution.get_causal_steps()
        initial_identified_steps = [self.trace.get_step(step_id) for step_id in initial_identified_step_ids]
        causal_metrics = {
            "num_identified_causal_steps": len(initial_identified_steps),
            "identified_steps": [step.to_dict() for step in initial_identified_steps if step],
        }

        if skip_critique:
            causal_metrics.update({
                "precision": 1.0,
                "recall": 1.0,
                "f1_score": 1.0,
                "num_ground_truth_causal_steps": len(consensus_steps),
                "ground_truth_steps": [step.to_dict() for step in consensus_steps],
                "true_positives": len(initial_identified_steps),
                "false_positives": 0,
                "false_negatives": 0,
                "note": "Critique skipped - using attribution results directly"
            })
        else:
            gt_set = set[int]([step.step_id for step in consensus_steps])
            id_set = set[int](initial_identified_step_ids)

            true_positives = len(gt_set & id_set)
            false_positives = len(id_set - gt_set)
            false_negatives = len(gt_set - id_set)

            precision = true_positives / len(id_set) if len(id_set) > 0 else 0.0
            recall = true_positives / len(gt_set) if len(gt_set) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            causal_metrics.update({
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1_score, 4),
                "num_ground_truth_causal_steps": len(consensus_steps),
                "ground_truth_steps": [step.to_dict() for step in consensus_steps],
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives
            })

        metrics["causal_attribution_metrics"] = causal_metrics

        # 2. Repair Metrics
        repair_metrics = {
            "total_repairs_attempted": 0,
            "successful_repairs": 0,
            "failed_repairs": 0,
            "success_rate": 0.0,
            "repairs_by_step": {}
        }

        if self.counterfactual_repair:
            successful_repairs = self.counterfactual_repair.get_all_successful_repairs()
            all_repairs = self.counterfactual_repair.repairs

            total = sum(len(repair_list) for repair_list in all_repairs.values())
            successful = sum(
                1 for repair_list in all_repairs.values()
                for repair in repair_list
                if repair.success_predicted
            )

            repair_metrics.update({
                "total_repairs_attempted": total,
                "successful_repairs": successful,
                "failed_repairs": total - successful,
                "success_rate": round(successful / total, 4) if total > 0 else 0.0,
                "repairs_by_step": {
                    step_id: {
                        "success": repair_list[0].success_predicted if repair_list else False,
                        "minimality_score": round(repair_list[0].minimality_score, 4) if repair_list else 0.0
                    }
                    for step_id, repair_list in successful_repairs.items()
                    if repair_list
                }
            })

        metrics["repair_metrics"] = repair_metrics

        # 3. Minimality Metrics
        minimality_metrics = {
            "average_minimality": None,
            "min_minimality": None,
            "max_minimality": None,
            "minimality_by_step": {}
        }

        if self.counterfactual_repair:
            successful_repairs = self.counterfactual_repair.get_all_successful_repairs()
            if successful_repairs:
                print(f"Successful repairs: {len(successful_repairs)}")
                minimality_scores = [r.minimality_score for repair in successful_repairs.values() for r in repair]
                if minimality_scores:
                    minimality_metrics.update({
                        "average_minimality": round(sum(minimality_scores) / len(minimality_scores), 4),
                        "min_minimality": round(min(minimality_scores), 4),
                        "max_minimality": round(max(minimality_scores), 4),
                        "minimality_by_step": {
                            step_id: round(repair_list[0].minimality_score, 4) if repair_list else 0.0
                            for step_id, repair_list in successful_repairs.items()
                            if repair_list
                        }
                    })

        metrics["minimality_metrics"] = minimality_metrics

        # 4. Multi-Agent Agreement
        agreement_data = {
            "average_consensus_score": None,
            "num_steps_critiqued": 0,
            "steps_with_agreement": []
        }

        if skip_critique:
            agreement_data["skipped"] = True
            agreement_data["reason"] = "Deterministic reexecutor used - critique not needed"
        elif self.multi_agent_critique:
            critique_results = self.multi_agent_critique.critique_results

            if critique_results:
                consensus_scores = [r.consensus_score for r in critique_results.values()]
                agreement_data["average_consensus_score"] = round(
                    sum(consensus_scores) / len(consensus_scores), 4
                )
                agreement_data["num_steps_critiqued"] = len(critique_results)

                for step_id, result in critique_results.items():
                    step_data = {
                        "step_id": step_id,
                        "consensus_score": round(result.consensus_score, 4),
                        "final_verdict": "CAUSAL" if result.final_verdict else "NOT CAUSAL",
                        "agent_a_score": self.causal_attribution.crs_scores.get(step_id, 0.0)
                    }

                    # Extract Agent B details
                    agent_b_critique = next(
                        (c for c in result.critiques if c["agent"] == "Agent_B"),
                        None
                    )
                    if agent_b_critique:
                        step_data["agent_b_agrees"] = agent_b_critique["agrees"]
                        step_data["agent_b_confidence"] = round(agent_b_critique["confidence"], 4)
                        step_data["agent_b_reasoning"] = self._extract_reasoning(
                            agent_b_critique["response"]
                        )

                    # Extract Agent C (final critic) details
                    agent_c_critique = next(
                        (c for c in result.critiques if c["agent"] == "Agent_C"),
                        None
                    )
                    if agent_c_critique:
                        step_data["agent_c_agrees"] = agent_c_critique["agrees"]
                        step_data["agent_c_confidence"] = round(agent_c_critique["confidence"], 4)
                        step_data["agent_c_reasoning"] = self._extract_reasoning(
                            agent_c_critique["response"]
                        )
                        step_data["final_critic_summary"] = self._extract_reasoning(
                            agent_c_critique["response"]
                        )

                    agreement_data["steps_with_agreement"].append(step_data)

        metrics["multi_agent_agreement"] = agreement_data

        return metrics

    def _extract_reasoning(self, response: str) -> str:
        if "REASONING:" in response:
            reasoning = response.split("REASONING:")[-1].strip()
            return reasoning
        return response