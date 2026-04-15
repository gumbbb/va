import os
import sys
import json
import time
import asyncio
import httpx
from pathlib import Path

# Config
ORCHESTRATOR_URL = "http://localhost:8000/v1/orchestrator/message_sync"
DATA_PATH = Path("/app/routing_benchmark_data.json")
RESULTS_PATH = Path("/app/benchmark_results_orin.json")

async def run_benchmark():
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}")
        return

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results = []
    total_queries = 0
    correct_queries = 0

    print(f"\nStarting Full Flow Benchmark (Orin)...")
    print("-" * 100)
    print(f"{'STATUS':<8} | {'QUERY':<40} | {'EXPECTED':<15} | {'ROUTED':<15} | {'LATENCY':<10}")
    print("-" * 100)

    async with httpx.AsyncClient(timeout=60.0) as client:
        for entry in dataset:
            expected_agent = entry["agent_id"]
            queries = entry["queries"]
            
            for query in queries:
                total_queries += 1
                start_time = time.perf_counter()
                
                try:
                    resp = await client.post(
                        ORCHESTRATOR_URL,
                        json={"message": query, "session_id": f"bench-{total_queries}"}
                    )
                    end_time = time.perf_counter()
                    latency = (end_time - start_time) * 1000 # ms
                    
                    if resp.status_code == 200:
                        res_data = resp.json()
                        selected_agents = res_data.get("agent_ids", [])
                        final_answer = res_data.get("response", "")
                        agent_answers = res_data.get("agent_responses", [])
                        
                        is_correct = expected_agent in selected_agents
                        if is_correct:
                            correct_queries += 1
                        
                        status = "PASS" if is_correct else "FAIL"
                        print(f"[{status:<4}] {query[:40]:<40} | {expected_agent:<15} | {str(selected_agents):<15} | {latency:.1f}ms")
                        
                        res_entry = {
                            "query": query,
                            "expected": expected_agent,
                            "selected": selected_agents,
                            "is_correct": is_correct,
                            "agent_response": agent_answers,
                            "final_answer": final_answer,
                            "latency_ms": round(latency, 2)
                        }
                        results.append(res_entry)
                    else:
                        print(f"[ERR ] {query[:40]:<40} | HTTP {resp.status_code}")
                        results.append({"query": query, "error": f"HTTP {resp.status_code}", "is_correct": False})
                        
                except Exception as e:
                    print(f"[EXC ] {query[:40]:<40} | {str(e)}")
                    results.append({"query": query, "error": str(e), "is_correct": False})

    accuracy = (correct_queries / total_queries) * 100 if total_queries > 0 else 0
    avg_latency = sum(r.get("latency_ms", 0) for r in results if "latency_ms" in r) / total_queries if total_queries > 0 else 0

    summary = {
        "platform": "Orin",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_queries": total_queries,
        "correct_queries": correct_queries,
        "accuracy": round(accuracy, 2),
        "avg_latency_ms": round(avg_latency, 2),
        "details": results
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("-" * 100)
    print(f"Benchmark Finished!")
    print(f"Accuracy: {accuracy:.2f}% ({correct_queries}/{total_queries})")
    print(f"Avg Latency: {avg_latency:.2f}ms")
    print(f"Results saved to: {RESULTS_PATH}")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
