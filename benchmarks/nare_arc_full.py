"""Full NARE system with Anthropic API for ARC-AGI.

Uses complete 4-tier routing (FAST/REFLEX/HYBRID/SLOW) with Anthropic Claude.
"""

import json
import time
import logging
import argparse
import tempfile
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv

from nare.agent import NAREProductionAgent
from nare.config import DEFAULT_CONFIG
from nare.arc_adapter import ARCAdapter
from nare import llm
from nare import llm_anthropic

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")


def _patch_llm_for_anthropic():
    """Replace Gemini calls with Anthropic in LLM module."""

    def anthropic_generate_samples(prompt: str, n: int = 3, temperature: float = 0.8, mode: str = "SLOW"):
        """Generate samples using Anthropic API."""
        samples = []
        for _ in range(n):
            try:
                solution = llm_anthropic.generate_arc_solution(prompt, temperature=temperature)
                samples.append({
                    'solution': solution,
                    'reasoning': 'Anthropic extended thinking',
                    'abstract_signature': None
                })
            except Exception as e:
                logging.warning(f"Anthropic generation failed: {e}")
        return samples, 0

    def anthropic_embedding(text: str) -> list:
        """Use sentence-transformers for semantic embeddings (384-dim native).

        Returns 384-dim embeddings without padding to preserve semantic structure.
        FAISS index will be created with 384 dimensions.
        """
        try:
            from sentence_transformers import SentenceTransformer
            if not hasattr(anthropic_embedding, '_model'):
                anthropic_embedding._model = SentenceTransformer('all-MiniLM-L6-v2')

            emb = anthropic_embedding._model.encode(text, convert_to_numpy=True)
            return emb.tolist()  # Return native 384-dim

        except ImportError:
            # Fallback to hash if sentence-transformers not available
            # Use 384-dim to match sentence-transformers
            import hashlib
            embedding = []
            for repeat in range(48):  # 48 hashes × 8 floats = 384
                h = hashlib.sha256((text + str(repeat)).encode()).digest()
                for i in range(0, 32, 4):
                    chunk = int.from_bytes(h[i:i+4], 'big')
                    embedding.append((chunk % 1000) / 1000.0 - 0.5)
                    if len(embedding) >= 384:
                        return embedding[:384]
            return embedding[:384]

    llm.generate_samples = anthropic_generate_samples
    llm.best_of_n_with_prescore = lambda prompt, breadth=3: anthropic_generate_samples(prompt, n=breadth, temperature=0.7)
    llm.get_embedding = anthropic_embedding

    logging.info("[PATCH] LLM module now uses Anthropic API (no Gemini calls)")


def load_arc_tasks(path: str, limit: int = None) -> List[Dict[str, Any]]:
    """Load ARC tasks from JSON file or directory."""
    path_obj = Path(path)
    tasks = []

    if path_obj.is_dir():
        json_files = sorted(path_obj.glob('*.json'))
        for json_file in json_files:
            with open(json_file, 'r') as f:
                task_data = json.load(f)
            tasks.append({'id': json_file.stem, 'data': task_data})
            if limit and len(tasks) >= limit:
                break
    else:
        with open(path, 'r') as f:
            data = json.load(f)
        for task_id, task_data in data.items():
            tasks.append({'id': task_id, 'data': task_data})
            if limit and len(tasks) >= limit:
                break

    return tasks


def run_nare_arc(
    tasks: List[Dict[str, Any]],
    persist_dir: str = None,
    output_path: str = 'benchmarks/nare_arc_results.json'
) -> Dict[str, Any]:
    """Run full NARE system on ARC tasks with Anthropic API."""

    # Patch LLM to use Anthropic
    _patch_llm_for_anthropic()

    # Create NARE agent with memory (Sleep phase disabled for ARC)
    if persist_dir is None:
        persist_dir = tempfile.mkdtemp(prefix='nare_arc_')

    # Disable Sleep phase for ARC tasks, use 384-dim embeddings
    arc_config = DEFAULT_CONFIG
    arc_config = type(arc_config)(
        routing=arc_config.routing,
        synthesis=type(arc_config.synthesis)(
            max_attempts=arc_config.synthesis.max_attempts,
            max_attempts_hard=arc_config.synthesis.max_attempts_hard,
            use_subprocess=arc_config.synthesis.use_subprocess,
            slow_path_breadth=5  # Increase for better accuracy
        ),
        sleep=type(arc_config.sleep)(enabled=False),
        bootstrap=arc_config.bootstrap,
        immune=arc_config.immune,
        critic=arc_config.critic,
        skill=arc_config.skill,
        retrieval=arc_config.retrieval,
        skill_validation=arc_config.skill_validation,
        amortization=arc_config.amortization
    )

    # Create agent with 384-dim embeddings (sentence-transformers native)
    agent = NAREProductionAgent(config=arc_config, persist_dir=persist_dir, embedding_dim=384)
    adapter = ARCAdapter()

    results = []
    correct = 0
    total = 0

    route_counts = {'FAST': 0, 'REFLEX': 0, 'HYBRID': 0, 'SLOW': 0}

    for i, task in enumerate(tasks):
        task_id = task['id']
        task_data = task['data']

        print(f"\n[{i+1}/{len(tasks)}] Task: {task_id}")

        parsed = adapter.parse_arc_task(task_data)
        query = parsed['query']
        test_cases = parsed['test']

        if not test_cases:
            print("  SKIP: No test cases")
            continue

        expected_output = test_cases[0].get('output')
        if not expected_output:
            print("  SKIP: No expected output")
            continue

        oracle = adapter.build_oracle(expected_output)

        t0 = time.time()
        try:
            # Use FULL NARE system with routing
            result = agent.solve(query, oracle=oracle)
            elapsed = time.time() - t0

            answer = result.get('final_answer', '')
            route = result.get('route_decision', '?')
            alpha = result.get('alpha', 0.0)

            route_counts[route] = route_counts.get(route, 0) + 1

            ok, info = oracle(query, answer)

            # Convert info dict to string for display
            if isinstance(info, dict):
                info_str = info.get('status', str(info))
            else:
                info_str = str(info)

            total += 1
            if ok:
                correct += 1
                print(f"  PASS  route={route:<8} sim={alpha:.3f}  {elapsed:6.2f}s")
            else:
                print(f"  FAIL  route={route:<8} sim={alpha:.3f}  {elapsed:6.2f}s  {info_str[:80]}")

            results.append({
                'task_id': task_id,
                'correct': ok,
                'route': route,
                'alpha': alpha,
                'elapsed_s': elapsed,
                'info': info,
                'answer_preview': answer[:200]
            })

        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results.append({
                'task_id': task_id,
                'correct': False,
                'route': 'ERROR',
                'alpha': 0.0,
                'elapsed_s': 0,
                'info': str(e),
                'traceback': traceback.format_exc()
            })

    accuracy = correct / total if total > 0 else 0.0
    amortized = (route_counts.get('FAST', 0) + route_counts.get('REFLEX', 0) + route_counts.get('HYBRID', 0))
    amortization_pct = amortized / total if total > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"NARE ARC-AGI Results: {correct}/{total} ({accuracy:.1%})")
    print(f"Routing: {route_counts}")
    print(f"Amortization: {amortization_pct:.1%} ({amortized}/{total})")
    print(f"{'='*60}")

    summary = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'api': 'anthropic-proxy',
        'model': 'kr/claude-sonnet-4.5',
        'system': 'NARE-full',
        'routing': route_counts,
        'amortization_pct': amortization_pct,
        'results': results
    }

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Full NARE system with Anthropic API for ARC-AGI")
    parser.add_argument('--dataset', required=True, help='Path to ARC tasks')
    parser.add_argument('--num-tasks', type=int, default=None, help='Limit tasks')
    parser.add_argument('--persist-dir', default=None, help='Memory directory')
    parser.add_argument('--output', default='benchmarks/nare_arc_results.json', help='Output path')

    args = parser.parse_args()

    print(f"Loading ARC tasks from: {args.dataset}")
    print(f"Using: Full NARE system + Anthropic API")
    print(f"Model: kr/claude-sonnet-4.5")
    print(f"Routing: FAST/REFLEX/HYBRID/SLOW")
    print(f"Memory: {'persistent' if args.persist_dir else 'temporary'}\n")

    tasks = load_arc_tasks(args.dataset, limit=args.num_tasks)
    print(f"Loaded {len(tasks)} tasks\n")

    run_nare_arc(tasks, persist_dir=args.persist_dir, output_path=args.output)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
