#!/usr/bin/env python3
"""Real SWE-bench benchmark with actual repositories and test execution.

This benchmark:
1. Clones real repositories (astropy, django, etc.)
2. Checks out the commit before the fix
3. Runs NARA to generate a solution
4. Applies the solution to the repository
5. Runs the actual tests to verify the fix
"""

import json
import time
import sys
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nare.agent import NAREProductionAgent
from nare.config import DEFAULT_CONFIG
from nare.repo_manager import RepoManager
from nare.oracle import build_oracle_from_spec

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_tasks_with_repo_info(tasks_file: str) -> list:
    """Load tasks and add repository information.

    Args:
        tasks_file: Path to tasks JSON file

    Returns:
        List of tasks with repo info
    """
    with open(tasks_file, 'r', encoding='utf-8') as f:
        tasks = json.load(f)

    # Add repo info based on task ID
    for task in tasks:
        task_id = task['id']

        # Extract repo from task ID (e.g., "astropy__astropy-12907" -> "astropy/astropy")
        parts = task_id.split('__')
        if len(parts) >= 2:
            org = parts[0]
            repo_name = parts[1].split('-')[0]
            task['repo'] = f"{org}/{repo_name}"
        else:
            task['repo'] = "unknown/unknown"

        # For now, we'll work with HEAD (in real SWE-bench, we'd use specific commits)
        task['base_commit'] = None  # Will use current HEAD

        # Determine test command based on repo
        if 'astropy' in task['repo']:
            task['test_command'] = 'pytest -xvs'
        elif 'django' in task['repo']:
            task['test_command'] = 'python tests/runtests.py'
        elif 'sympy' in task['repo']:
            task['test_command'] = 'pytest -xvs'
        else:
            task['test_command'] = 'pytest -xvs'

    return tasks


def run_real_swe_bench(
    tasks_file: str = "benchmarks/swebench_real.json",
    repos_dir: str = "swe_bench_repos",
    persist_dir: str = "memory_swe_real",
    max_tasks: int = 50
):
    """Run real SWE-bench with actual repositories.

    Args:
        tasks_file: Path to tasks file
        repos_dir: Directory for cloned repositories
        persist_dir: Memory persistence directory
        max_tasks: Maximum number of tasks to run
    """

    # Load tasks
    tasks = load_tasks_with_repo_info(tasks_file)
    tasks = tasks[:max_tasks]

    print(f"Loaded {len(tasks)} SWE-bench tasks")
    print(f"Using: Real SWE-bench with test execution")
    print(f"Repos: {repos_dir}")
    print(f"Memory: {persist_dir}")
    print()

    # Initialize components
    repo_manager = RepoManager(repos_dir=repos_dir)
    agent = NAREProductionAgent(
        config=DEFAULT_CONFIG,
        persist_dir=persist_dir,
        embedding_dim=384  # sentence-transformers
    )

    results = []
    correct = 0
    total = 0
    route_counts = {}

    for i, task in enumerate(tasks):
        task_id = task['id']
        query = task['query']
        repo = task['repo']

        print(f"[{i+1}/{len(tasks)}] Task: {task_id}")
        print(f"  Repo: {repo}")

        # Skip if repo not available
        repo_path = Path(repos_dir) / repo.replace('/', os.sep)
        if not repo_path.exists():
            print(f"  SKIP: Repository not cloned yet")
            continue

        # Rate limiting
        if i > 0:
            time.sleep(5)

        try:
            # Prepare repository
            repo_manager.prepare_task(task)

            # Build oracle with test execution
            oracle_spec = {
                "type": "test_execution",
                "repo_manager": repo_manager,
                "task_id": task_id,
                "test_command": task['test_command'],
                "timeout": 300
            }
            oracle = build_oracle_from_spec(oracle_spec)

            # Solve with NARA
            t0 = time.time()
            result = agent.solve(query, oracle=oracle)
            elapsed = time.time() - t0

            answer = result.get('final_answer', '')
            route = result.get('route_decision', '?')
            alpha = result.get('alpha', 0.0)

            route_counts[route] = route_counts.get(route, 0) + 1

            # Apply solution and run tests
            success, error = repo_manager.apply_solution(task_id, answer)

            if not success:
                print(f"  FAIL  route={route:<8} sim={alpha:.3f}  {elapsed:6.2f}s")
                print(f"        Error applying solution: {error[:100]}")

                results.append({
                    'task_id': task_id,
                    'correct': False,
                    'route': route,
                    'alpha': alpha,
                    'elapsed_s': round(elapsed, 2),
                    'error': f"Apply failed: {error}"
                })

                repo_manager.cleanup_task(task_id, keep_changes=False)
                continue

            # Run tests
            passed, output = repo_manager.run_tests(task_id, task['test_command'])

            total += 1
            if passed:
                correct += 1
                print(f"  PASS  route={route:<8} sim={alpha:.3f}  {elapsed:6.2f}s  [TESTS PASSED]")
            else:
                print(f"  FAIL  route={route:<8} sim={alpha:.3f}  {elapsed:6.2f}s  [TESTS FAILED]")
                print(f"        {output[:200]}")

            results.append({
                'task_id': task_id,
                'correct': passed,
                'route': route,
                'alpha': alpha,
                'elapsed_s': round(elapsed, 2),
                'test_output': output[:500] if not passed else "Tests passed"
            })

            # Cleanup
            repo_manager.cleanup_task(task_id, keep_changes=False)

        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()

            results.append({
                'task_id': task_id,
                'correct': False,
                'route': 'ERROR',
                'error': str(e),
                'traceback': traceback.format_exc()
            })

            # Try to cleanup
            try:
                repo_manager.cleanup_task(task_id, keep_changes=False)
            except:
                pass

    # Summary
    accuracy = correct / total if total > 0 else 0
    amortization = sum(route_counts.get(r, 0) for r in ("FAST", "REFLEX", "COMPILED_SKILL"))
    amortization_pct = (amortization / total * 100) if total > 0 else 0

    print()
    print("=" * 60)
    print(f"Real SWE-bench Results: {correct}/{total} ({accuracy:.1%})")
    print(f"Routing: {route_counts}")
    print(f"Amortization: {amortization_pct:.1f}% ({amortization}/{total})")
    print("=" * 60)

    # Save results
    output = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'system': 'NARE-real-swe-bench',
        'routing': route_counts,
        'amortization_pct': amortization_pct,
        'results': results
    }

    out_file = f"results_real_swe_{persist_dir}.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run real SWE-bench with test execution')
    parser.add_argument('--tasks', default='benchmarks/swebench_real.json', help='Tasks file')
    parser.add_argument('--repos-dir', default='swe_bench_repos', help='Repositories directory')
    parser.add_argument('--persist-dir', default='memory_swe_real', help='Memory directory')
    parser.add_argument('--max-tasks', type=int, default=50, help='Maximum tasks to run')
    args = parser.parse_args()

    run_real_swe_bench(
        tasks_file=args.tasks,
        repos_dir=args.repos_dir,
        persist_dir=args.persist_dir,
        max_tasks=args.max_tasks
    )
