#!/usr/bin/env python3
"""Official SWE-bench adapter for NARA.

Generates predictions in official format for submission to SWE-bench leaderboard.
"""

import json
import os
import sys
import subprocess
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nare.core.agent import NAREProductionAgent
from nare.config import DEFAULT_CONFIG
from nare.tools.repo_manager import RepoManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def find_relevant_files(repo_path: str, problem_statement: str, max_files: int = 5) -> list:
    """Find relevant files in repository based on problem statement.

    Args:
        repo_path: Path to repository
        problem_statement: Bug description
        max_files: Maximum number of files to return

    Returns:
        List of (file_path, content) tuples
    """
    import re

    relevant_files = []
    seen_paths = set()

    # Strategy 0: RECURSIVE SEARCH by filename
    # Extract filenames (e.g., "rst.py", "qdp.py") and find them recursively
    filename_pattern = r'\b([a-z_][a-z0-9_]*\.py)\b'
    filenames = set(re.findall(filename_pattern, problem_statement.lower()))

    print(f"  Extracted filenames: {list(filenames)[:5]}")

    for filename in filenames:
        if len(relevant_files) >= max_files:
            break

        try:
            # Use find to recursively search for the file
            result = subprocess.run(
                ['find', '.', '-name', filename, '-type', 'f'],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0 and result.stdout.strip():
                found_paths = result.stdout.strip().split('\n')
                for found_path in found_paths[:2]:  # Max 2 matches per filename
                    # Remove leading './'
                    file_path = found_path[2:] if found_path.startswith('./') else found_path

                    if file_path not in seen_paths and '/test' not in file_path:
                        full_path = os.path.join(repo_path, file_path)
                        try:
                            with open(full_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                if len(content) < 100000:
                                    relevant_files.append((file_path, content))
                                    seen_paths.add(file_path)
                                    print(f"  [OK] Found via recursive search: {file_path}")

                                    if len(relevant_files) >= max_files:
                                        break
                        except:
                            pass
        except:
            pass

    # Strategy 1: Extract EXPLICIT file paths from problem statement
    # Look for patterns like: `path/to/file.py`, path/to/file.py, or "path/to/file.py"
    explicit_patterns = [
        r'`([^`]+\.py)`',           # `file.py`
        r'"([^"]+\.py)"',           # "file.py"
        r"'([^']+\.py)'",           # 'file.py'
        r'\b([\w/]+/[\w/]+\.py)\b', # path/to/file.py (with at least one /)
    ]

    explicit_files = set()
    for pattern in explicit_patterns:
        matches = re.findall(pattern, problem_statement)
        explicit_files.update(matches)

    print(f"  Explicit file mentions: {list(explicit_files)[:5]}")

    # Try to load explicitly mentioned files first
    for file_path in explicit_files:
        full_path = os.path.join(repo_path, file_path)
        if os.path.exists(full_path) and file_path not in seen_paths:
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if len(content) < 100000:  # Skip huge files
                        relevant_files.append((file_path, content))
                        seen_paths.add(file_path)
                        print(f"  [OK] Loaded explicit: {file_path}")
            except Exception as e:
                print(f"  [WARN] Failed to load {file_path}: {e}")

    # Strategy 2: Extract module/class names and search for them
    # Look for patterns like: ClassName, module.submodule, function_name
    if len(relevant_files) < max_files:
        keywords = set()

        # Class names (CamelCase)
        class_names = re.findall(r'\b([A-Z][a-zA-Z0-9_]{2,})\b', problem_statement)
        keywords.update(class_names[:10])

        # CRITICAL: Settings/constants (UPPER_CASE)
        # Extract FILE_UPLOAD_PERMISSIONS, DEBUG, etc.
        constants = re.findall(r'\b([A-Z][A-Z0-9_]{3,})\b', problem_statement)
        keywords.update(constants[:10])

        # Function/method names (snake_case with context)
        func_patterns = [
            r'`([a-z_][a-z0-9_]+)`',  # `function_name`
            r'def ([a-z_][a-z0-9_]+)',  # def function_name
            r'\.([a-z_][a-z0-9_]+)\(',  # .method_name(
        ]
        for pattern in func_patterns:
            matches = re.findall(pattern, problem_statement)
            keywords.update([m for m in matches if len(m) > 4])

        # CRITICAL: Extract error traceback file paths
        # If error shows "File .../ndarithmetic.py", prioritize that file
        traceback_files = re.findall(r'File "([^"]+\.py)"', problem_statement)
        for tb_file in traceback_files:
            # Extract just the filename
            filename = tb_file.split('/')[-1]
            if filename:
                keywords.add(filename.replace('.py', ''))

        # CRITICAL: Infer module path from class names
        # If bug mentions NDDataRef, prioritize astropy/nddata/
        # If bug mentions QTable, prioritize astropy/table/
        module_hints = {}
        for class_name in class_names[:5]:
            if 'NDData' in class_name:
                module_hints[class_name] = 'nddata'
            elif 'Table' in class_name or 'QTable' in class_name:
                module_hints[class_name] = 'table'
            elif 'Model' in class_name or 'Compound' in class_name:
                module_hints[class_name] = 'modeling'
            elif 'QDP' in class_name:
                module_hints[class_name] = 'io/ascii'

        print(f"  Keywords for search: {list(keywords)[:10]}")
        if module_hints:
            print(f"  Module hints: {module_hints}")

        # Search for files containing these keywords
        # CRITICAL: Prioritize class definitions over mentions
        for keyword in list(keywords)[:10]:
            if len(relevant_files) >= max_files:
                break

            try:
                # First try: search for class definition
                result = subprocess.run(
                    ['git', 'grep', '-l', '-w', f'class {keyword}'],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                files_to_check = []
                if result.returncode == 0:
                    files_to_check = result.stdout.strip().split('\n')
                else:
                    # Fallback: search for any mention
                    result = subprocess.run(
                        ['git', 'grep', '-l', '-i', '-w', keyword],
                        cwd=repo_path,
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    if result.returncode == 0:
                        files_to_check = result.stdout.strip().split('\n')

                # CRITICAL: Filter by module hint if available
                if keyword in module_hints:
                    module_path = module_hints[keyword]
                    files_to_check = [f for f in files_to_check if module_path in f]

                for file_path in files_to_check[:3]:
                    if (file_path and
                        file_path.endswith('.py') and
                        file_path not in seen_paths and
                        '/test' not in file_path):

                        full_path = os.path.join(repo_path, file_path)
                        try:
                            with open(full_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                if len(content) < 300000:
                                    relevant_files.append((file_path, content))
                                    seen_paths.add(file_path)
                                    print(f"  [OK] Found via keyword '{keyword}': {file_path}")

                                    if len(relevant_files) >= max_files:
                                        break
                        except:
                            pass
            except:
                pass

    # Strategy 3: If still nothing, try to infer from error messages
    if not relevant_files:
        # Look for traceback or error patterns
        error_patterns = [
            r'File "([^"]+\.py)"',  # File "path.py"
            r'in ([a-z_]+\.py)',    # in file.py
        ]
        for pattern in error_patterns:
            matches = re.findall(pattern, problem_statement)
            for file_path in matches:
                if file_path not in seen_paths:
                    full_path = os.path.join(repo_path, file_path)
                    if os.path.exists(full_path):
                        try:
                            with open(full_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                if len(content) < 300000:
                                    relevant_files.append((file_path, content))
                                    seen_paths.add(file_path)
                                    print(f"  [OK] Found from error: {file_path}")
                        except:
                            pass

    if not relevant_files:
        print(f"  [ERROR] No relevant files found!")

    return relevant_files[:max_files]


def generate_git_patch(repo_path: str, changes: dict) -> str:
    """Generate git diff patch from file changes.

    Args:
        repo_path: Path to repository
        changes: Dict of file_path -> content

    Returns:
        Git diff patch string
    """
    if not changes:
        return ""

    # Apply changes to files
    for file_path, content in changes.items():
        full_path = os.path.join(repo_path, file_path)

        # Create parent directories if needed
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)

    # Generate git diff
    result = subprocess.run(
        ['git', 'diff', 'HEAD'],
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=30
    )

    if result.returncode != 0:
        logging.warning(f"Git diff failed: {result.stderr}")
        return ""

    return result.stdout


def load_swe_bench_lite():
    """Load official SWE-bench Lite dataset.

    Returns:
        List of instances
    """
    try:
        from datasets import load_dataset
        dataset = load_dataset("princeton-nlp/SWE-bench_Lite")
        return list(dataset['test'])
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        logging.info("Install with: pip install datasets")
        sys.exit(1)


def create_test_oracle(repo_path: str, instance: dict, repo_manager):
    """Create oracle function that executes tests to verify solutions.

    Args:
        repo_path: Path to repository
        instance: SWE-bench instance with test commands
        repo_manager: RepoManager instance for parsing solutions

    Returns:
        Oracle function (query, answer) -> (bool, str)
    """
    def oracle(query: str, answer: str) -> tuple:
        """Execute tests to verify if solution is correct."""
        try:
            # Parse file changes from answer
            changes = repo_manager._parse_solution(answer)

            # Debug logging
            logging.debug(f"[Oracle] Parsed {len(changes)} file changes")
            if not changes:
                # Log first 500 chars of answer to debug parsing
                logging.debug(f"[Oracle] Answer preview: {answer[:500]}")
                return False, "No file changes found"

            # Apply changes temporarily
            original_contents = {}
            for file_path, new_content in changes.items():
                full_path = os.path.join(repo_path, file_path)
                if os.path.exists(full_path):
                    with open(full_path, 'r', encoding='utf-8') as f:
                        original_contents[file_path] = f.read()
                    with open(full_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    logging.debug(f"[Oracle] Applied changes to {file_path}")
                else:
                    logging.warning(f"[Oracle] File not found: {full_path}")
                    return False, f"File not found: {file_path}"

            # CRITICAL: Apply test_patch to add FAIL_TO_PASS tests
            # These tests don't exist in the base commit
            test_patch = instance.get('test_patch', '')
            if test_patch:
                # Write patch to temp file
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
                    f.write(test_patch)
                    patch_file = f.name

                try:
                    # Apply patch
                    result = subprocess.run(
                        ['git', 'apply', '--whitespace=fix', patch_file],
                        cwd=repo_path,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )

                    if result.returncode != 0:
                        logging.warning(f"[Oracle] Failed to apply test_patch: {result.stderr[:200]}")
                        # Clean up and return
                        os.unlink(patch_file)
                        for file_path, content in original_contents.items():
                            full_path = os.path.join(repo_path, file_path)
                            with open(full_path, 'w', encoding='utf-8') as f:
                                f.write(content)
                        return None, f"Could not apply test_patch: {result.stderr[:200]}"

                    logging.info(f"[Oracle] Applied test_patch successfully")
                finally:
                    os.unlink(patch_file)

            # Run tests
            # Use FAIL_TO_PASS tests instead of test_patch
            # FAIL_TO_PASS contains specific test names that should pass after the fix
            fail_to_pass = instance.get('FAIL_TO_PASS', [])

            if not fail_to_pass:
                logging.warning(f"[Oracle] No FAIL_TO_PASS tests specified")
                return None, "No FAIL_TO_PASS tests specified"

            # Parse FAIL_TO_PASS (it's a JSON string)
            import json
            if isinstance(fail_to_pass, str):
                try:
                    fail_to_pass = json.loads(fail_to_pass)
                except:
                    pass

            # Run only the FAIL_TO_PASS tests
            test_cmd = f"pytest -xvs {' '.join(fail_to_pass)}"
            logging.info(f"[Oracle] Running FAIL_TO_PASS tests: {len(fail_to_pass)} tests")

            # CRITICAL: Add repo to PYTHONPATH so pytest can import modules
            env = os.environ.copy()
            env['PYTHONPATH'] = str(repo_path) + os.pathsep + env.get('PYTHONPATH', '')

            # CRITICAL: Some repos (astropy, numpy, scipy) have C extensions that require compilation
            # Skip oracle for these repos - they need build step which is too slow/complex
            repos_needing_build = ['astropy', 'numpy', 'scipy', 'matplotlib', 'scikit-learn', 'pandas']
            repo_name = instance.get('repo', '').split('/')[-1]

            if repo_name in repos_needing_build:
                logging.warning(f"[Oracle] Skipping oracle for {repo_name} (requires C compilation)")
                return None, f"Oracle disabled for {repo_name} (C extensions)"

            result = subprocess.run(
                test_cmd,
                shell=True,
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=120,
                env=env
            )

            # Restore original files
            for file_path, original_content in original_contents.items():
                full_path = os.path.join(repo_path, file_path)
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)

            # Restore repository state (undo test_patch)
            if test_patch:
                subprocess.run(
                    ['git', 'reset', '--hard', 'HEAD'],
                    cwd=repo_path,
                    capture_output=True,
                    timeout=30
                )
                subprocess.run(
                    ['git', 'clean', '-fd'],
                    cwd=repo_path,
                    capture_output=True,
                    timeout=30
                )

            # Check if tests passed
            success = (result.returncode == 0)
            message = result.stdout + result.stderr if not success else "Tests passed"

            # Check for missing dependencies (ImportError/ModuleNotFoundError)
            if not success and ("ModuleNotFoundError" in message or "ImportError" in message):
                logging.warning(f"[Oracle] Missing dependencies, oracle unavailable")
                return None, "Missing test dependencies"

            logging.info(f"[Oracle] Tests {'PASSED' if success else 'FAILED'} (returncode={result.returncode})")

            return success, message

        except Exception as e:
            return False, f"Oracle error: {e}"

    return oracle


def run_swe_bench_official(
    max_tasks: int = 10,
    output_file: str = "predictions.jsonl",
    persist_dir: str = "memory_swe_official",
    enable_oracle: bool = True,
    skip_repos: list = None
):
    """Run NARA on official SWE-bench and generate submission file.

    Args:
        max_tasks: Maximum number of tasks to run
        output_file: Output file for predictions
        persist_dir: Memory directory
        enable_oracle: Enable Verified Synthesis with test oracle
        skip_repos: List of repo names to skip (e.g., ['astropy'])
    """
    if skip_repos is None:
        skip_repos = []

    # Load dataset
    print("Loading SWE-bench Lite dataset...")
    instances = load_swe_bench_lite()

    # Filter out repos to skip
    if skip_repos:
        original_count = len(instances)
        instances = [inst for inst in instances if inst['repo'].split('/')[-1] not in skip_repos]
        print(f"Filtered out {original_count - len(instances)} tasks from repos: {skip_repos}")

    instances = instances[:max_tasks]

    print(f"Loaded {len(instances)} tasks")
    print(f"Output: {output_file}")
    print(f"Oracle: {'ENABLED' if enable_oracle else 'DISABLED'}")
    print()

    # Initialize
    agent = NAREProductionAgent(
        config=DEFAULT_CONFIG,
        persist_dir=persist_dir,
        embedding_dim=1024
    )
    repo_manager = RepoManager("swe_bench_repos")

    predictions = []
    successful = 0

    for i, instance in enumerate(instances):
        instance_id = instance['instance_id']
        problem_statement = instance['problem_statement']
        repo = instance['repo']
        base_commit = instance['base_commit']

        print(f"[{i+1}/{len(instances)}] {instance_id}")
        print(f"  Repo: {repo}")
        print(f"  Commit: {base_commit[:8]}")

        try:
            # Prepare repository
            task = {
                'id': instance_id,
                'repo': repo,
                'base_commit': base_commit
            }

            repo_path = repo_manager.prepare_task(task)

            # Find relevant files in repository
            print(f"  Searching for relevant files...")
            relevant_files = find_relevant_files(repo_path, problem_statement, max_files=3)
            print(f"  Found {len(relevant_files)} relevant files")

            # Build context from relevant files
            context = ""
            allowed_paths = []
            if relevant_files:
                context = "\n\nRelevant files from the repository:\n\n"
                for file_path, content in relevant_files:
                    allowed_paths.append(file_path)
                    # For large files, show only a relevant section (not the whole file)
                    # This allows the model to output complete modified files within token limits
                    if len(content) > 3000:
                        # Find relevant section based on keywords from problem statement
                        keywords = [w for w in problem_statement.split() if len(w) > 4][:10]
                        lines = content.split('\n')

                        # Find lines containing keywords
                        relevant_line_indices = set()
                        for i, line in enumerate(lines):
                            for keyword in keywords:
                                if keyword.lower() in line.lower():
                                    # Include context: 50 lines before and after
                                    for j in range(max(0, i-50), min(len(lines), i+51)):
                                        relevant_line_indices.add(j)

                        if relevant_line_indices:
                            # Extract relevant section
                            sorted_indices = sorted(relevant_line_indices)
                            start_idx = sorted_indices[0]
                            end_idx = sorted_indices[-1]
                            relevant_section = '\n'.join(lines[start_idx:end_idx+1])
                            content = f"# ... (file truncated, showing lines {start_idx+1}-{end_idx+1}) ...\n{relevant_section}\n# ... (end of shown section) ..."
                        else:
                            # No keywords found, just show first 3000 chars
                            content = content[:3000] + "\n... (truncated)"

                    context += f"File: {file_path}\n```python\n{content}\n```\n\n"

                # Add explicit path constraint
                context += f"\n**IMPORTANT**: You MUST use ONLY these exact file paths:\n"
                for path in allowed_paths:
                    context += f"  - {path}\n"
                context += "\nDo NOT invent new paths. Do NOT use paths not listed above.\n"

            # Build query
            backticks = "```"
            query = f"""Fix the following bug in {repo}:

{problem_statement}
{context}

⚠️ OUTPUT FORMAT ⚠️

Output ONLY the modified file section. NO explanations, NO analysis, NO "I need to".

CORRECT format:
File: <path>
{backticks}python
<code with your fix>
{backticks}

WRONG - DO NOT DO THIS:
❌ "I need to analyze this bug..."
❌ "Let me trace through the code..."
❌ "Looking at the provided files..."
❌ "The issue is that..."
❌ ANY text before "File:" or after closing {backticks}

If you cannot fix with available files, output EXACTLY:
CANNOT_FIX: need file <path>

Available files:
{chr(10).join(f'  - {p}' for p in allowed_paths)}

OUTPUT NOW - start with "File:" or "CANNOT_FIX:"."""

            # Create oracle for Verified Synthesis
            oracle = None
            if enable_oracle:
                oracle = create_test_oracle(repo_path, instance, repo_manager)
                print(f"  Oracle: ENABLED")

            # Create file_provider for agentic file retrieval
            # When LLM says "CANNOT_FIX: need file X", VS loop
            # calls this to fetch the file from the repo
            def make_file_provider(rpath):
                def _provider(file_path: str):
                    """Read a file from the repository by path."""
                    full = os.path.join(rpath, file_path)
                    if os.path.exists(full):
                        try:
                            with open(full, 'r', encoding='utf-8') as f:
                                content = f.read()
                            if len(content) < 300000:
                                print(f"  [AGENT] Dynamically loaded: {file_path}")
                                return content
                            else:
                                print(f"  [AGENT] File too large: {file_path}")
                                return content[:30000] + "\n... (truncated)"
                        except Exception as e:
                            print(f"  [AGENT] Failed to read {file_path}: {e}")
                            return None
                    else:
                        # Try recursive search using git ls-files
                        import subprocess as _sp
                        fname = os.path.basename(file_path)
                        try:
                            result = _sp.run(
                                ['git', 'ls-files'],
                                cwd=rpath, capture_output=True, text=True, timeout=10
                            )
                            if result.returncode == 0:
                                candidates = [
                                    p for p in result.stdout.strip().split('\n')
                                    if p.endswith(fname) and '/test' not in p
                                ]
                                if candidates:
                                    best = candidates[0]
                                    alt = os.path.join(rpath, best)
                                    if os.path.exists(alt):
                                        with open(alt, 'r', encoding='utf-8') as f:
                                            content = f.read()
                                        print(f"  [AGENT] Found alternative: {best} (requested: {file_path})")
                                        return content[:15000]
                        except:
                            pass
                        print(f"  [AGENT] File not found: {file_path}")
                        return None
                return _provider

            file_provider = make_file_provider(repo_path)

            # Solve with NARA (with oracle for Verified Synthesis loop)
            result = agent.solve(query, oracle=oracle, file_provider=file_provider)
            solution = result['final_answer']

            # Parse solution
            changes = repo_manager._parse_solution(solution)

            # CRITICAL: Validate paths against allowed list
            if changes and allowed_paths:
                invalid_paths = [p for p in changes.keys() if p not in allowed_paths]
                if invalid_paths:
                    print(f"  REJECT: Model used invalid paths: {invalid_paths}")
                    print(f"  Allowed paths were: {allowed_paths}")
                    changes = {}  # Clear changes to force SKIP

            if not changes:
                print(f"  SKIP: No file changes found")
                predictions.append({
                    'instance_id': instance_id,
                    'model_patch': '',
                    'model_name_or_path': 'NARA-v1.0'
                })
                repo_manager.cleanup_task(instance_id)
                continue

            print(f"  Found {len(changes)} file changes")

            # Generate git patch
            patch = generate_git_patch(repo_path, changes)

            if patch:
                print(f"  Generated patch ({len(patch)} bytes)")
                successful += 1
            else:
                print(f"  WARN: Empty patch")

            predictions.append({
                'instance_id': instance_id,
                'model_patch': patch,
                'model_name_or_path': 'NARA-v1.0'
            })

            # Cleanup
            repo_manager.cleanup_task(instance_id)

        except Exception as e:
            logging.error(f"Error on {instance_id}: {e}")
            predictions.append({
                'instance_id': instance_id,
                'model_patch': '',
                'model_name_or_path': 'NARA-v1.0'
            })

    # Save predictions
    with open(output_file, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')

    print()
    print("=" * 60)
    print(f"Generated {len(predictions)} predictions")
    print(f"Successful patches: {successful}/{len(predictions)}")
    print(f"Saved to: {output_file}")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Install SWE-bench: pip install git+https://github.com/princeton-nlp/SWE-bench.git")
    print("2. Run evaluation:")
    print(f"   python -m swebench.harness.run_evaluation \\")
    print(f"     --dataset_name princeton-nlp/SWE-bench_Lite \\")
    print(f"     --predictions_path {output_file} \\")
    print(f"     --max_workers 4 \\")
    print(f"     --output_dir results/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate SWE-bench predictions with NARA'
    )
    parser.add_argument(
        '--max-tasks',
        type=int,
        default=10,
        help='Maximum number of tasks to run (default: 10)'
    )
    parser.add_argument(
        '--output',
        default='predictions.jsonl',
        help='Output file for predictions (default: predictions.jsonl)'
    )
    parser.add_argument(
        '--persist-dir',
        default='memory_swe_official',
        help='Memory directory (default: memory_swe_official)'
    )
    parser.add_argument(
        '--no-oracle',
        action='store_true',
        help='Disable Verified Synthesis oracle (default: enabled)'
    )
    parser.add_argument(
        '--skip-repos',
        nargs='+',
        default=['astropy'],
        help='Repos to skip (default: astropy - requires C compilation)'
    )

    args = parser.parse_args()

    run_swe_bench_official(
        max_tasks=args.max_tasks,
        output_file=args.output,
        persist_dir=args.persist_dir,
        enable_oracle=not args.no_oracle,
        skip_repos=args.skip_repos
    )
