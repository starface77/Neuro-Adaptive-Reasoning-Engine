"""
Session Management Module

Component: NareSession
Purpose: Manages NARE agent lifecycle and query execution
Architecture: Thin wrapper over NARE core, handles CLI-specific concerns

Responsibilities:
- Initialize NARE agent with proper configuration
- Manage context files (loaded via /read command)
- Triage user queries before routing
- Provide file access to NARE core
- Track session metadata

Dependencies:
- NAREProductionAgent: Core reasoning engine
- TriageAgent: Intent classifier (QUESTION/EXPLORE/EDIT)
- NareConfig: Configuration management

Lifecycle:
1. __init__ - set repository path
2. init_agent - lazy initialize NARE + triage
3. solve - execute query through NARE pipeline
"""

import os
import time
import logging
import subprocess
from typing import Optional

log = logging.getLogger("nare.cli.session")

class NareSession:
    """Session manager for NARE CLI.

    Responsibilities:
    - Initialize NARE agent with proper config
    - Manage context files (loaded via /read)
    - Triage user queries before routing
    - Provide file access to NARE core

    Lifecycle:
    1. __init__ - set repo path
    2. init_agent - lazy init NARE + triage
    3. solve - execute query through NARE

    Attributes:
        repo_path: Working directory for file operations
        agent: NAREProductionAgent instance (lazy init)
        triage: TriageAgent instance (lazy init)
        context_files: Files loaded via /read command
    """

    def __init__(self, repo_path: str = "."):
        """Initialize session with repository path.

        Args:
            repo_path: Working directory for file operations
        """
        self.repo_path = os.path.abspath(repo_path)
        self.agent = None
        self.context_files: dict[str, str] = {}
        self.triage = None
        self.repo_map = None
        self.chat_history: list[dict] = []
        self._history: list[dict] = []
        self._total_tokens_in = 0
        self._total_tokens_out = 0

        env_flag = os.getenv("NARE_AGENT_LOOP", "1").strip().lower()
        self.use_agent_loop: bool = env_flag not in ("0", "false", "off", "no")
        self._agent_loop = None

    def _generate_repo_map(self) -> str:
        """Generate a compact tree representation of the repository."""
        import os

        excluded_dirs = {'.git', '__pycache__', 'node_modules', 'venv', 'env', '.env', '.venv', '.idea', '.vscode', 'build', 'dist', '.nare_memory', 'coverage'}
        excluded_exts = {'.pyc', '.pyo', '.pyd', '.so', '.dll', '.dylib', '.exe', '.png', '.jpg', '.jpeg', '.gif', '.ico', '.pdf', '.zip', '.tar', '.gz', '.mp4'}

        tree = []

        def walk_dir(current_dir, prefix=""):
            try:
                entries = sorted(os.listdir(current_dir))
            except PermissionError:
                return

            dirs = []
            files = []
            for e in entries:
                if e in excluded_dirs:
                    continue
                path = os.path.join(current_dir, e)
                if os.path.isdir(path):
                    dirs.append(e)
                else:
                    if any(e.endswith(ext) for ext in excluded_exts):
                        continue
                    files.append(e)

            for i, d in enumerate(dirs):
                is_last_dir = (i == len(dirs) - 1) and not files
                marker = "└── " if is_last_dir else "├── "
                tree.append(f"{prefix}{marker}{d}/")
                extension = "    " if is_last_dir else "│   "
                walk_dir(os.path.join(current_dir, d), prefix + extension)

            for i, f in enumerate(files):
                is_last_file = (i == len(files) - 1)
                marker = "└── " if is_last_file else "├── "
                tree.append(f"{prefix}{marker}{f}")

        tree.append(f"{os.path.basename(self.repo_path)}/")
        walk_dir(self.repo_path)

        if len(tree) > 1500:
            tree = tree[:1500] + ["... (truncated due to size)"]

        return "\n".join(tree)

    def init_agent(self):
        """Initialize NARE agent and triage classifier.

        Components initialized:
        - NAREProductionAgent: Core reasoning engine
        - TriageAgent: Intent classifier (QUESTION/EXPLORE/EDIT)

        Config:
        - Memory: .nare_memory/ in repo
        - Embeddings: 3072-dim (BGE-large)
        - Synthesis: max 8 attempts

        Performance: First call loads embedding model (~10s), then cached.
        """
        if self.agent is not None:
            return

        from nare.config import NareConfig, SynthesisConfig
        from nare.core.agent import NAREProductionAgent
        from nare.agents.roles.triage import TriageAgent

        log.info(f"[Session] Initializing NARE (embedding model will load on first query)...")

        config = NareConfig(synthesis=SynthesisConfig(max_attempts=8))
        persist_dir = os.path.join(self.repo_path, ".nare_memory")

        self.agent = NAREProductionAgent(
            config=config,
            persist_dir=persist_dir,
            embedding_dim=3072,
        )

        self.triage = TriageAgent()

        log.info(f"[Session] NARE initialized in {self.repo_path}")

    def _ensure_agent_loop(self):
        """Lazy-init the AgentLoop and wire its bus into the CLI renderer."""
        if self._agent_loop is not None:
            return self._agent_loop

        from nare.agents.loops.autonomous import build_loop
        from nare.cli.display.agent_renderer import attach_renderer
        from nare.cli.display import console as _shared_console

        loop = build_loop(working_dir=self.repo_path)
        attach_renderer(loop.bus, console=_shared_console)
        self._agent_loop = loop
        return loop

    def solve_agentic(self, query: str, thinking_display=None) -> dict:
        """Execute the query through the Phase-3 tool-calling AgentLoop.

        Now integrated with NARE:
        1. Check NARE memory for cached solutions (FAST route)
        2. Check compiled skills (REFLEX route)
        3. If no hit, fall back to AgentLoop tool-calling
        4. Save successful results to NARE memory
        5. Trigger crystallization when thresholds met

        Returns a dict shaped like `solve()` so callers don't need to
        special-case the path.

        ``thinking_display`` (optional): when provided, the agent will
        typewriter-stream its final answer through
        ``thinking_display.stream_token(...)`` so the user sees the
        reply appear letter-by-letter instead of all at once.
        """

        self.init_agent()

        history_text = ""
        if self.chat_history:
            recent = self.chat_history[-3:]
            for msg in recent:
                role = "USER" if msg["role"] == "user" else "ASSISTANT"
                content = msg["content"][:1000]
                history_text += f"\n{role}:\n{content}\n"

        repo_map = self._generate_repo_map()
        if repo_map and len(repo_map) > 5000:
            repo_map = repo_map[:5000] + "\n... (truncated)"

        enriched = query
        if repo_map:
            enriched += f"\n\nRepo Map:\n{repo_map}"
        if history_text:
            enriched += f"\n\nChat History:\n{history_text}"

        route_result = None
        try:
            route_result = self.agent.router.route(enriched)
        except ValueError as e:
            if "API_KEY" in str(e):
                log.info(f"[Session] NARE routing skipped (no API key) - using AgentLoop")
                route_result = {"route_decision": "AGENT"}
            else:
                raise

        if route_result["route_decision"] in ["FAST", "REFLEX", "COMPILED_SKILL"]:

            log.info(f"[Session] NARE hit: {route_result['route_decision']}")

            if thinking_display:
                thinking_display.start_waiting(f"Route: {route_result['route_decision']}")

            def file_provider(path: str) -> Optional[str]:
                full_path = os.path.join(self.repo_path, path)
                if os.path.exists(full_path):
                    try:
                        with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                            return f.read()[:100000]
                    except:
                        pass
                return None

            t0 = time.time()
            result = self.agent.solve(
                query=enriched,
                file_provider=file_provider,
                thinking_display=thinking_display,
                working_dir=self.repo_path,
                chat_history=history_text,
                repo_map=repo_map,
                intent="AGENTIC"
            )

            self.chat_history.append({"role": "user", "content": query})
            self.chat_history.append({
                "role": "assistant",
                "content": result.get("final_answer", "")[:4000],
            })
            if len(self.chat_history) > 10:
                self.chat_history = self.chat_history[-10:]

            return {
                "final_answer": result.get("final_answer", ""),
                "route_decision": route_result["route_decision"],
                "_elapsed": time.time() - t0,
                "_intent": "AGENTIC",
                "_nare_hit": True,
            }

        log.info(f"[Session] NARE miss ({route_result['route_decision']}) - using AgentLoop")

        if thinking_display:
            thinking_display.start_waiting("Route: AGENT (tool-calling)")

        loop = self._ensure_agent_loop()

        on_final_token = None
        if thinking_display is not None:
            try:
                thinking_display.switch_to_solution()
            except Exception:
                pass
            on_final_token = thinking_display.stream_token

        t0 = time.time()
        run = loop.run(
            query=query,
            chat_history=history_text or None,
            repo_map=repo_map,
            on_final_token=on_final_token,
        )

        if run.final_answer and run.stop_reason == "final_answer":
            try:
                from nare.reasoning import llm
                import numpy as np

                query_emb = llm.get_embedding(query)
                episode_data = {
                    "query": query,
                    "solution": run.final_answer,
                    "reasoning_trace": f"AgentLoop execution: {run.iterations} iterations",
                    "score": 0.85,
                    "metadata": {
                        "source": "agent_loop",
                        "iterations": run.iterations,
                        "tokens": run.tokens,
                        "elapsed": run.elapsed,
                    }
                }
                self.agent.memory.add_episode(episode_data, np.array([query_emb], dtype=np.float32))
                log.info(f"[Session] Saved AgentLoop result to NARE memory")

                if self.agent.config.sleep.enabled:
                    if self.agent.evolution.check_compilation_trigger():
                        log.info(f"[Session] Crystallization triggered")
                        self.agent.evolution.run_compilation_cycle()
            except Exception as e:
                log.warning(f"[Session] Failed to save to NARE memory: {e}")

        self.chat_history.append({"role": "user", "content": query})
        self.chat_history.append({
            "role": "assistant",
            "content": (run.final_answer or "")[:4000],
        })
        if len(self.chat_history) > 10:
            self.chat_history = self.chat_history[-10:]

        return {
            "final_answer": run.final_answer or "",
            "route_decision": "AGENT",
            "_elapsed": time.time() - t0,
            "_intent": "AGENTIC",
            "_iterations": run.iterations,
            "_tokens": run.tokens,
            "_stop_reason": run.stop_reason,
            "_nare_hit": False,
        }

    def solve(self, query: str, thinking_display=None) -> dict:
        """Execute query through NARE pipeline.

        Pipeline:
        1. Triage - classify intent (QUESTION/EXPLORE/EDIT)
        2. Route - NARE selects path (FAST/REFLEX/HYBRID/SLOW)
        3. Execute - run through reasoning engine
        4. Return - results with metadata

        Args:
            query: User query
            thinking_display: Optional display for streaming

        Returns:
            dict with keys:
            - final_answer: Solution text
            - route_decision: FAST/REFLEX/HYBRID/SLOW
            - _elapsed: Execution time
            - _intent: Classified intent
        """
        if self.use_agent_loop:
            return self.solve_agentic(query, thinking_display=thinking_display)

        self.init_agent()

        intent = self.triage.classify(query, use_llm_fallback=False)
        log.info(f"[Session] Intent: {intent}")

        if thinking_display:
            thinking_display.start_waiting(f"Classified Intent: {intent}")

        enriched = query
        if self.context_files:
            enriched += "\n\nContext files available:\n"
            for path in self.context_files.keys():
                enriched += f"- {path}\n"
            enriched += "\nUse <read_file><path>file.py</path></read_file> to read specific files.\n"

        repo_map = self._generate_repo_map()
        if repo_map and len(repo_map) > 5000:

            repo_map = repo_map[:5000] + "\n... (truncated, use read_file for specific files)"

        history_text = ""
        if self.chat_history:
            recent_history = self.chat_history[-3:]
            if len(self.chat_history) > 3:
                history_text = "--- RECENT CHAT HISTORY (last 3 messages) ---\n"
            else:
                history_text = "--- CHAT HISTORY ---\n"
            for msg in recent_history:
                role = "USER" if msg["role"] == "user" else "ASSISTANT"

                content = msg['content'][:1000]
                if len(msg['content']) > 1000:
                    content += "... (truncated)"
                history_text += f"\n{role}:\n{content}\n"
            history_text += "-----------------------------\n\n"

        def file_provider(path: str) -> Optional[str]:
            """Provide file content to NARE on demand."""
            full_path = os.path.join(self.repo_path, path)
            if os.path.exists(full_path):
                try:
                    with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                        return f.read()[:100000]
                except:
                    pass
            return None

        t0 = time.time()
        result = self.agent.solve(
            query=enriched,
            file_provider=file_provider,
            thinking_display=thinking_display,
            working_dir=self.repo_path,
            chat_history=history_text,
            repo_map=repo_map,
            intent=intent
        )
        result["_elapsed"] = time.time() - t0
        result["_intent"] = intent

        self.chat_history.append({"role": "user", "content": query})

        clean_ans = result.get("final_answer", "")
        import re
        clean_ans = re.sub(r'<reasoning>.*?</reasoning>', '', clean_ans, flags=re.DOTALL)
        clean_ans = re.sub(r'```[\s\S]*?```', '[CODE BLOCK REMOVED FOR BREVITY]', clean_ans)

        self.chat_history.append({"role": "assistant", "content": clean_ans})
        if len(self.chat_history) > 10:
            self.chat_history = self.chat_history[-10:]

        return result

    def read_file(self, path: str) -> Optional[str]:
        """Load file into context.

        Args:
            path: File path (relative to repo or absolute)

        Returns:
            File content or None if not found
        """
        full_path = os.path.join(self.repo_path, path)
        if not os.path.exists(full_path):

            import glob
            matches = glob.glob(os.path.join(self.repo_path, "**", path), recursive=True)
            if matches:
                full_path = matches[0]
            else:
                return None

        try:
            with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            rel_path = os.path.relpath(full_path, self.repo_path)
            self.context_files[rel_path] = content
            return content
        except Exception as e:
            log.warning(f"Failed to read {path}: {e}")
            return None

    def clear_context(self):
        """Clear loaded files from context."""
        self.context_files.clear()

    def set_repo(self, path: str) -> bool:
        """Change working directory.

        Args:
            path: New repository path

        Returns:
            True if successful, False otherwise
        """
        p = os.path.abspath(path)
        if os.path.isdir(p):
            self.repo_path = p
            self.context_files.clear()
            self.agent = None
            return True
        return False

    def get_status(self) -> dict:
        """Get session status.

        Returns:
            dict with keys:
            - repo: Repository path
            - context_files: Number of loaded files
            - agent_ready: Whether agent is initialized
            - episodes: Number of memory episodes (if agent ready)
            - skills: Number of compiled skills (if agent ready)
            - model: Model name
        """
        from nare.reasoning import llm as _llm

        info = {
            "repo": self.repo_path,
            "context_files": len(self.context_files),
            "agent_ready": self.agent is not None,
            "model": getattr(_llm, "ANTHROPIC_MODEL", "unknown"),
        }

        if self.agent:
            info["episodes"] = len(self.agent.memory.episodes)
            info["skills"] = len(self.agent.memory.compiled_skills)

        return info

    def git_commit(self, message: str, files: list[str]) -> Optional[str]:
        """Commit changes with message.

        Args:
            message: Commit message
            files: List of files to commit

        Returns:
            Commit hash or None if failed
        """
        try:

            subprocess.run(["git", "add"] + files, cwd=self.repo_path, check=True)

            subprocess.run(
                ["git", "commit", "-m", message],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )

            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            commit_hash = result.stdout.strip()

            self._history.append({
                "type": "commit",
                "commit": commit_hash,
                "message": message,
                "files": files,
                "timestamp": time.time()
            })

            return commit_hash
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def git_diff(self) -> str:
        """Get diff of uncommitted changes.

        Returns:
            Diff output or empty string if failed
        """
        try:
            result = subprocess.run(
                ["git", "diff"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""
