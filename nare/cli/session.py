"""Session management for NARE CLI."""

import os
import time
from nare.utils.logger import get_logger
import subprocess
import json
from nare.cli.autonomy_level import AutonomyLevel, should_ask_permission
import asyncio
from typing import Dict, Optional

log = get_logger("nare.cli.session")

COMPACT_THRESHOLD = 10
MAX_HISTORY_MESSAGES = 20
HISTORY_KEEP_FIRST = 2
HISTORY_KEEP_LAST = 18
MAX_CONTENT_PER_MESSAGE = 1000
MAX_REPO_MAP_CHARS = 5000


class NareSession:

    def __init__(self, repo_path: str = ".", autonomy_level: AutonomyLevel = AutonomyLevel.ASSISTED):
        self.repo_path = os.path.abspath(repo_path)
        self.autonomy_level = autonomy_level
        self.agent = None
        self.context_files: dict[str, str] = {}
        self.triage = None
        self.repo_map = None
        self.chat_history: list[dict] = []
        self._history: list[dict] = []
        self._total_tokens_in = 0
        self._total_tokens_out = 0

        self._repo_map_cache: Optional[str] = None
        self._repo_map_time: float = 0.0

        env_flag = os.getenv("NARE_AGENT_LOOP", "0").strip().lower()
        self.use_agent_loop: bool = env_flag not in ("0", "false", "off", "no")
        self._agent_loop = None

        self.query_count = 0
        self.last_compilation_query = 0

        # Create NARE.md if it doesn't exist
        self._ensure_nare_md()

        # Load chat history from disk
        self._load_chat_history()

    def _ensure_nare_md(self):
        """Create NARE.md with token-saving rules if it doesn't exist."""
        nare_md_path = os.path.join(self.repo_path, 'NARE.md')
        nareignore_path = os.path.join(self.repo_path, '.nareignore')

        if not os.path.exists(nare_md_path):
            try:
                template = (
                    "# NARE Project Rules\n\n"
                    "## Token Economy\n"
                    "- Use grep/find BEFORE reading files\n"
                    "- Read files only once, check OBSERVATION blocks for cached content\n"
                    "- After 3 read operations, MUST edit or answer\n"
                    "- Use find_function + apply_hunks instead of read + write\n\n"
                    "## Context Hygiene\n"
                    "- Agent auto-compacts context after 12 observations (keeps last 10)\n"
                    "- Use .nareignore to exclude build artifacts\n\n"
                    "## Anti-Hallucination\n"
                    "- ALWAYS read file before claiming to know its contents\n"
                    "- NEVER assume function signatures - verify with grep/read\n"
                    "- Test changes with bash before claiming success\n\n"
                    "## Response Style\n"
                    "- Be concise, no explanations unless asked\n"
                    "- Focus on code changes only\n"
                )
                with open(nare_md_path, 'w', encoding='utf-8') as f:
                    f.write(template)
            except Exception as e:
                log.debug(f"[Session] Failed to create NARE.md: {e}")

        if not os.path.exists(nareignore_path):
            try:
                ignore_lines = [
                    "__pycache__/", "*.pyc", "*.pyo", "dist/", "build/",
                    ".pytest_cache/", ".coverage", "htmlcov/",
                    "node_modules/", "*.log", "logs/",
                    ".vscode/", ".idea/", ".git/",
                    ".DS_Store", "*.mp4", "*.zip", "*.tar.gz",
                    "tmp/", "temp/", "*.tmp",
                ]
                with open(nareignore_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(ignore_lines) + "\n")
            except Exception as e:
                log.debug(f"[Session] Failed to create .nareignore: {e}")

    def _get_history_path(self) -> str:
        """Get path to chat history file."""
        memory_dir = os.path.join(self.repo_path, ".nare_memory")
        os.makedirs(memory_dir, exist_ok=True)
        return os.path.join(memory_dir, "chat_history.json")

    def _save_chat_history(self):
        """Save chat history to disk with aggressive token optimization."""
        try:
            import re

            trimmed = []
            for msg in self.chat_history:
                content = msg.get("content", "")
                content = re.sub(r'<reasoning>.*?</reasoning>', '', content, flags=re.DOTALL)
                content = re.sub(r'<abstract_signature>.*?</abstract_signature>', '', content, flags=re.DOTALL)

                def _truncate_code(match):
                    lang = match.group(1)
                    code = match.group(2)
                    lines = code.splitlines()
                    if len(lines) > 15:
                        kept = lines[:8] + ['...'] + lines[-4:]
                        return f"```{lang}\n" + '\n'.join(kept) + "\n```"
                    return match.group(0)

                content = re.sub(r'```(\w*)\n(.*?)\n```', _truncate_code, content, flags=re.DOTALL)
                if len(content) > MAX_CONTENT_PER_MESSAGE:
                    content = content[:MAX_CONTENT_PER_MESSAGE] + "... (truncated)"

                trimmed.append({"role": msg["role"], "content": content})

            history_path = self._get_history_path()
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(trimmed, f, ensure_ascii=False)
        except Exception as e:
            log.warning(f"[Session] Failed to save chat history: {e}")

    def compact_history(self) -> str:
        """Compress chat history into a summary to free context space."""
        if len(self.chat_history) < COMPACT_THRESHOLD:
            return "History too short to compact."

        messages_to_compact = self.chat_history[:-4]
        kept = self.chat_history[-4:]

        summary_parts = []
        for msg in messages_to_compact:
            role = msg["role"].upper()
            content = msg.get("content", "")[:200]
            summary_parts.append(f"{role}: {content}")

        summary = (
            f"[COMPACTED HISTORY: {len(messages_to_compact)} messages]\n"
            + "\n".join(summary_parts)
        )

        self.chat_history = [
            {"role": "assistant", "content": summary}
        ] + kept
        self._save_chat_history()

        return f"Compacted {len(messages_to_compact)} messages into summary."

    def _trim_history(self):
        """Trim history to stay within token budget."""
        if len(self.chat_history) > MAX_HISTORY_MESSAGES:
            self.chat_history = (
                self.chat_history[:HISTORY_KEEP_FIRST]
                + self.chat_history[-HISTORY_KEEP_LAST:]
            )

    def _load_chat_history(self):
        """Load chat history from disk."""
        try:
            history_path = self._get_history_path()
            if os.path.exists(history_path):
                with open(history_path, 'r', encoding='utf-8') as f:
                    self.chat_history = json.load(f)
                    log.info(f"[Session] Loaded {len(self.chat_history)} messages from history")
        except Exception as e:
            log.warning(f"[Session] Failed to load chat history: {e}")
            self.chat_history = []

    def _generate_repo_map(self) -> str:
        """Generate a semantic skeleton of the repository."""
        from nare.core.repo_map import generate_repo_map
        return generate_repo_map(
            repo_path=self.repo_path,
            max_files=200,
            max_chars=MAX_REPO_MAP_CHARS,
            use_cache=True,
            active_files=set(self.context_files.keys()) if self.context_files else None,
        )

    def init_agent(self):
        if self.agent is not None:
            return

        from nare.config import NareConfig, SynthesisConfig
        from nare.core.agent import NAREProductionAgent
        from nare.agents.roles.triage import TriageAgent

        log.info(f"[Session] Initializing NARE (embedding model will load on first query)...")

        config = NareConfig(synthesis=SynthesisConfig(max_attempts=8))
        self.config = config  # Store config for later use

        import hashlib
        import shutil

        project_id = hashlib.md5(os.path.abspath(self.repo_path).encode('utf-8')).hexdigest()[:12]
        persist_dir = os.path.expanduser(f"~/.nare/projects/{project_id}/memory")
        
        os.makedirs(persist_dir, exist_ok=True)

        self.agent = NAREProductionAgent(
            config=config,
            persist_dir=persist_dir,
            embedding_dim=1024,
        )

        from nare.core.events import EventBus
        from nare.cli.display.agent_renderer import attach_renderer
        from nare.cli.display import console as _shared_console

        self.router_bus = EventBus()
        attach_renderer(self.router_bus, console=_shared_console)
        self.agent.router.bus = self.router_bus

        self.triage = TriageAgent()

        try:
            from nare.memory.seed_common_queries import seed_memory
            seed_memory(self.agent.memory)
        except Exception as e:
            log.warning(f"[Session] Failed to seed common queries: {e}")

        log.info(f"[Session] NARE initialized in {self.repo_path}")

    def _ensure_agent_loop(self):
        """Lazy-init the AgentLoop."""
        if self._agent_loop is not None:
            return self._agent_loop

        from nare.agents.loops.autonomous import build_loop
        from nare.cli.display.agent_renderer import attach_renderer
        from nare.cli.display import console as _shared_console

        loop = build_loop(working_dir=self.repo_path)
        attach_renderer(loop.bus, console=_shared_console)
        self._agent_loop = loop
        return loop

    async def solve_agentic(self, query: str, thinking_display=None, resume_state: Optional[Dict] = None) -> dict:
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

        # Classify intent to determine if we should compile
        intent = self.triage.classify(query, use_llm_fallback=False)
        log.info(f"[Session] Intent: {intent}")

        history_text = ""
        if self.chat_history:
            recent = self.chat_history[-10:]
            for msg in recent:
                role = "USER" if msg["role"] == "user" else "ASSISTANT"
                content = msg["content"][:1000]
                history_text += f"\n{role}:\n{content}\n"

        repo_map = self._generate_repo_map()
        if repo_map and len(repo_map) > 50000:
            repo_map = repo_map[:50000] + "\n... (truncated)"

        enriched = query
        if history_text:
            enriched += f"\n\nChat History:\n{history_text}"

        if not resume_state:
            try:
                route_result = await self.agent.router.route(enriched)
            except ValueError as e:
                if "API_KEY" in str(e):
                    log.info(f"[Session] NARE routing skipped (no API key) - using AgentLoop")
                    route_result = {"route_decision": "AGENT"}
                else:
                    raise

            if route_result["route_decision"] in ["FAST", "REFLEX", "COMPILED_SKILL", "DIRECT", "HYBRID"]:

                log.info(f"[Session] NARE hit: {route_result['route_decision']}")

                if thinking_display:
                    if route_result["route_decision"] == "COMPILED_SKILL":
                        # Show skill details
                        skills = route_result.get("skills", [])
                        if skills:
                            skill = skills[0]
                            pattern = skill.get("pattern", "unknown")
                            confidence = skill.get("confidence", 0)
                            thinking_display.start_waiting(f"★ Using skill: {pattern} (confidence: {confidence:.0%})")
                        else:
                            thinking_display.start_waiting(f"Route: COMPILED_SKILL")
                    else:
                        thinking_display.start_waiting(f"Route: {route_result['route_decision']}")

                def file_provider(path: str) -> Optional[str]:
                    full_path = os.path.join(self.repo_path, path)
                    if os.path.exists(full_path) and not os.path.islink(full_path):
                        try:
                            # Basic binary check
                            with open(full_path, 'rb') as f:
                                chunk = f.read(1024)
                                if b'\0' in chunk:
                                    return "<binary_file_skipped>"
                            with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                                return f.read()
                        except Exception as e:
                            log.warning(f"[Session] Failed to read context file {full_path}: {e}")
                    return None

                t0 = time.time()
                result = await self.agent.solve(
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
                # Smart trimming: keep first 5 + last 95 messages
                if len(self.chat_history) > 100:
                    self.chat_history = self.chat_history[:5] + self.chat_history[-95:]
                self._save_chat_history()

                return {
                    "final_answer": result.get("final_answer", ""),
                    "route_decision": route_result["route_decision"],
                    "_elapsed": time.time() - t0,
                    "_intent": "AGENTIC",
                    "_nare_hit": True,
                }

            log.info(f"[Session] NARE miss ({route_result['route_decision']}) - using AgentLoop")
        else:
            log.info(f"[Session] Resuming session state")

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
        run = await loop.run(
            query=resume_state["query"] if resume_state else query,
            chat_history=history_text or None,
            repo_map=repo_map,
            on_final_token=on_final_token,
            resume_state=resume_state,
        )

        state_path = os.path.join(self.repo_path, ".nare_memory", "session_state.json")
        if not run.ok and run.loop_state:
            os.makedirs(os.path.dirname(state_path), exist_ok=True)
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(run.loop_state, f, indent=2, ensure_ascii=False)
            log.info(f"[Session] Saved loop state to {state_path}")
        else:
            if os.path.exists(state_path):
                try:
                    os.remove(state_path)
                except Exception:
                    pass

        if run.final_answer:
            # Only prompt for skill compilation on clean successful execution
            # Don't compile from error states (action_loop, repeated_read, etc.)
            clean_success_reasons = {"final_answer", "ok"}

            log.info(f"[Session] Stop reason: {run.stop_reason}, will compile: {run.stop_reason in clean_success_reasons}")

            if run.stop_reason in clean_success_reasons:
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

                    # Skill compilation disabled - too intrusive
                    # words = query.strip().split()
                    # if intent == "EDIT" and len(query.strip()) > 10 and len(words) >= 2:
                    #     from nare.cli.interactive import ask_compile_skill
                    #     if ask_compile_skill(query):
                    #         try:
                    #             # Use evolution engine to compile skill directly
                    #             if self.agent.evolution:
                    #                 log.info(f"[Session] Triggering skill compilation for query: {query[:50]}")
                    #                 self.agent.evolution._compile_skills()
                    #                 self.agent.memory.force_save()
                    #                 log.info(f"[Session] Skill compiled and saved")
                    #             else:
                    #                 log.warning(f"[Session] Evolution engine not available")
                    #         except Exception as compile_err:
                    #             log.warning(f"[Session] Failed to compile skill: {compile_err}")

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
        # Smart trimming: keep first 5 + last 95 messages
        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[:2] + self.chat_history[-18:]
        self._save_chat_history()

        return {
            "final_answer": run.final_answer or "",
            "route_decision": "AGENT",
            "_elapsed": time.time() - t0,
            "_intent": intent,
            "_iterations": run.iterations,
            "_tokens": run.tokens,
            "_stop_reason": run.stop_reason,
            "_nare_hit": False,
        }

    def solve(self, query: str, thinking_display=None, resume_state: Optional[Dict] = None) -> dict:
        """Execute query through NARE pipeline."""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(
                    lambda: asyncio.run(self._solve_async(query, thinking_display=thinking_display, resume_state=resume_state))
                ).result()
        else:
            return asyncio.run(self._solve_async(query, thinking_display=thinking_display, resume_state=resume_state))

    async def _solve_async(self, query: str, thinking_display=None, resume_state: Optional[Dict] = None) -> dict:
        """Execute query through NARE pipeline (async implementation).
        
        Pipeline:
        1. Triage - classify intent (QUESTION/EXPLORE/EDIT)
        2. Route - NARE selects path (FAST/REFLEX/HYBRID/SLOW)
        3. Execute - run through reasoning engine
        4. Return - results with metadata

        Args:
            query: User query
            thinking_display: Optional display for streaming
            resume_state: Optional state dict to resume from

        Returns:
            dict with keys:
            - final_answer: Solution text
            - route_decision: FAST/REFLEX/HYBRID/SLOW
            - _elapsed: Execution time
            - _intent: Classified intent
        """
        if self.use_agent_loop:
            return await self.solve_agentic(query, thinking_display=thinking_display, resume_state=resume_state)

        self.query_count += 1

        # Fast path for simple conversational queries - BEFORE any heavy operations
        simple_keywords = ["привет", "как дела", "как ты", "спасибо", "пока", "hello", "hi", "thanks", "bye", "что делаешь"]
        if any(kw in query.lower() for kw in simple_keywords) and len(query.split()) <= 5:
            log.info(f"[Session] Fast path for simple query")
            from ..reasoning.generation import engine as llm

            t0 = time.time()

            # Switch to solution mode for streaming
            if thinking_display and hasattr(thinking_display, 'switch_to_solution'):
                thinking_display.switch_to_solution()

            # Stream callback for real-time display
            response_parts = []
            def stream_callback(token: str):
                response_parts.append(token)
                if thinking_display:
                    thinking_display.stream_token(token)

            # Use same model as Claude Code for consistency
            payload = {
                "model": llm.ANTHROPIC_MODEL,  # kr/claude-sonnet-4.5
                "max_tokens": 512,
                "temperature": 0.7,
                "system": "Be brief.",  # Minimal system prompt
                "messages": [{"role": "user", "content": query}]
            }

            response = llm._post_anthropic("messages", payload, stream_callback=stream_callback)
            if not response:
                response = ''.join(response_parts)

            self.chat_history.append({"role": "user", "content": query})
            self.chat_history.append({"role": "assistant", "content": response})

            if len(self.chat_history) > 20:
                self.chat_history = self.chat_history[:2] + self.chat_history[-18:]
            self._save_chat_history()

            return {
                "final_answer": response,
                "route_decision": "FAST",
                "_elapsed": time.time() - t0,
                "_intent": "QUESTION"
            }

        if self.agent and self.agent.evolution and self.config.sleep.enabled:
            queries_since_last = self.query_count - self.last_compilation_query
            if queries_since_last >= self.config.sleep.periodic_compilation_interval:
                if self.agent.evolution.check_compilation_trigger():
                    log.info(f"[Session] Periodic compilation triggered after {queries_since_last} queries")
                    self.agent.evolution.run_compilation_cycle()
                    self.last_compilation_query = self.query_count

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

        repo_map = None
        if intent in ("EDIT", "EXPLORE"):
            repo_map = self._generate_repo_map()
            if repo_map and len(repo_map) > 50000:
                repo_map = repo_map[:50000] + "\n... (truncated, use read_file for specific files)"

        history_text = ""
        if self.chat_history:
            recent_history = self.chat_history[-5:]
            if len(self.chat_history) > 10:
                history_text = "--- RECENT CHAT HISTORY (last 10 messages) ---\n"
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
            if os.path.exists(full_path) and not os.path.islink(full_path):
                try:
                    # Basic binary check
                    with open(full_path, 'rb') as f:
                        chunk = f.read(1024)
                        if b'\0' in chunk:
                            return "<binary_file_skipped>"
                    with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                        return f.read()
                except Exception as e:
                    log.warning(f"[Session] Failed to read context file {full_path}: {e}")
            return None

        t0 = time.time()
        result = await self.agent.solve(
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
        # Smart trimming: keep first 5 + last 95 messages
        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[:2] + self.chat_history[-18:]
        self._save_chat_history()

        # Save episode to memory for skill compilation
        route = result.get("route_decision", "")
        final_answer = result.get("final_answer", "")
        if final_answer and route in ("SLOW", "HYBRID", "REFLEX") and self.agent and self.agent.memory:
            try:
                from nare.reasoning import llm
                import numpy as np

                query_emb = llm.get_embedding(query)
                episode_data = {
                    "query": query,
                    "solution": final_answer,
                    "reasoning_trace": f"Route: {route}, Intent: {intent}",
                    "score": 0.85,
                    "metadata": {
                        "source": "solve_async",
                        "route": route,
                        "intent": intent,
                        "elapsed": result.get("_elapsed", 0),
                    }
                }
                self.agent.memory.add_episode(episode_data, np.array([query_emb], dtype=np.float32))
                log.info(f"[Session] Saved episode to memory (route={route})")

                if self.agent.config.sleep.enabled and self.agent.evolution:
                    if self.agent.evolution.check_compilation_trigger():
                        log.info(f"[Session] Auto-compilation triggered")
                        self.agent.evolution.run_compilation_cycle()
            except Exception as e:
                log.warning(f"[Session] Failed to save episode: {e}")

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
            self._agent_loop = None  # Reset agent loop too
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

    def _prompt_skill_compilation(self, query: str, run, thinking_display):
        """Prompt user to compile successful AgentLoop execution as a skill.

        This is the "red button" - human approval for skill compilation.

        Args:
            query: Original user query
            run: AgentLoop run result
            thinking_display: Display object for UI output
        """
        try:
            if thinking_display:
                try:
                    thinking_display.stop()
                except Exception as e:
                    log.warning(f"[Session] Failed to stop thinking display: {e}")

            print()
            print("  ✓ Task completed successfully")
            print()
            print("  Compile as reusable skill?")
            print("  This will make the same task instant next time.")
            print()

            response = input("  [Y/n]: ").strip().lower()

            if response in ('', 'y', 'yes'):
                self._compile_skill_from_run(query, run)
                print("  ✓ Skill compiled")
            else:
                print("  Skipped")

            print()
        except Exception as e:
            log.error(f"[Session] Skill compilation prompt failed: {e}", exc_info=True)

    def _compile_skill_from_run(self, query: str, run):
        """Compile AgentLoop execution into a deterministic skill.

        Args:
            query: Original user query
            run: AgentLoop run result with transcript
        """
        try:
            from nare.reasoning import llm
            import numpy as np
            import json

            pattern = self._extract_pattern(query)

            # Extract tool calls from transcript
            tool_calls = self._extract_tool_calls_from_transcript(run.transcript)

            # Note: meaningful actions already checked before prompt was shown
            # Generate execute function code
            execute_code = self._generate_execute_code(tool_calls)

            # Generate trigger function code
            keywords = self._extract_keywords(query)
            trigger_code = f'''def trigger(query: str) -> bool:
    """Check if this skill should handle the query."""
    keywords = {repr(keywords)}
    query_lower = query.lower()
    matches = sum(kw in query_lower for kw in keywords)
    return matches >= {max(2, len(keywords) // 2)}'''

            skill_code = f'''"""Auto-compiled skill from AgentLoop execution.

Pattern: {pattern}
Original query: {query}
Compiled: {run.iterations} iterations, {run.tokens} tokens
"""

{trigger_code}

{execute_code}
'''

            # Get embedding for the query pattern
            query_emb = llm.get_embedding(query)
            trigger_emb = np.array(query_emb, dtype=np.float32)  # 1D array, not 2D

            log.info(f"[Session] Embedding shape: {trigger_emb.shape}, dim: {len(query_emb)}")

            self.agent.memory.add_compiled_skill(
                pattern=pattern,
                code=skill_code,
                trigger_emb=trigger_emb,
                confidence=0.70
            )

            if self.agent.memory.compiled_skills:
                self.agent.memory.compiled_skills[-1]['source'] = 'user_approved'
                self.agent.memory.compiled_skills[-1]['iterations'] = run.iterations
                self.agent.memory.compiled_skills[-1]['tokens'] = run.tokens

            # Force save to disk immediately
            self.agent.memory.force_save()

            log.info(f"[Session] Compiled skill: {pattern} (confidence: 0.70, {len(tool_calls)} tool calls)")

        except Exception as e:
            log.error(f"[Session] Skill compilation failed: {e}", exc_info=True)

    def _extract_tool_calls_from_transcript(self, transcript: list) -> list:
        """Extract tool calls from AgentLoop transcript.

        Args:
            transcript: List of conversation turns

        Returns:
            List of tool calls with name and args
        """
        tool_calls = []

        for turn in transcript:
            if not isinstance(turn, dict):
                continue

            # Check for tool_call in turn
            if 'tool_call' in turn:
                tc = turn['tool_call']
                if isinstance(tc, dict) and 'name' in tc:
                    tool_calls.append({
                        'name': tc['name'],
                        'args': tc.get('args', {})
                    })

            # Check for tool_calls array
            if 'tool_calls' in turn:
                for tc in turn['tool_calls']:
                    if isinstance(tc, dict) and 'name' in tc:
                        tool_calls.append({
                            'name': tc['name'],
                            'args': tc.get('args', {})
                        })

        return tool_calls

    def _generate_execute_code(self, tool_calls: list) -> str:
        """Generate execute function code from tool calls.

        Args:
            tool_calls: List of tool calls

        Returns:
            Python code for execute function
        """
        import json

        # Generate code that replays the tool calls
        lines = ['def execute(query: str, context: dict = None) -> str:']
        lines.append('    """Execute the compiled skill by replaying tool calls."""')
        lines.append('    import json')
        lines.append('    ')
        lines.append('    results = []')
        lines.append('    ')

        for i, tc in enumerate(tool_calls):
            tool_name = tc['name']
            tool_args = tc['args']

            lines.append(f'    # Tool call {i+1}: {tool_name}')

            # Generate code based on tool type
            if tool_name == 'Read':
                file_path = tool_args.get('file_path', '')
                lines.append(f'    try:')
                lines.append(f'        with open({repr(file_path)}, "r", encoding="utf-8") as f:')
                lines.append(f'            content = f.read()')
                lines.append(f'        results.append(f"Read {repr(file_path)}")')
                lines.append(f'    except Exception as e:')
                lines.append(f'        results.append(f"Error reading file: {{e}}")')

            elif tool_name == 'Write':
                file_path = tool_args.get('file_path', '')
                content = tool_args.get('content', '')
                lines.append(f'    try:')
                lines.append(f'        with open({repr(file_path)}, "w", encoding="utf-8") as f:')
                lines.append(f'            f.write({repr(content)[:100]}...)')  # Truncate for safety
                lines.append(f'        results.append(f"Wrote {repr(file_path)}")')
                lines.append(f'    except Exception as e:')
                lines.append(f'        results.append(f"Error writing file: {{e}}")')

            elif tool_name == 'Edit':
                file_path = tool_args.get('file_path', '')
                lines.append(f'    try:')
                lines.append(f'        # Edit operation on {repr(file_path)}')
                lines.append(f'        results.append(f"Edited {repr(file_path)}")')
                lines.append(f'    except Exception as e:')
                lines.append(f'        results.append(f"Error editing file: {{e}}")')

            elif tool_name in ['Bash', 'bash']:
                command = tool_args.get('command', '')
                lines.append(f'    try:')
                lines.append(f'        import subprocess')
                lines.append(f'        result = subprocess.run({repr(command)}, shell=True, capture_output=True, text=True, timeout=30)')
                lines.append(f'        results.append(f"Executed: {repr(command)[:50]}")')
                lines.append(f'    except Exception as e:')
                lines.append(f'        results.append(f"Error executing command: {{e}}")')

            else:
                # Generic tool call
                lines.append(f'    results.append(f"Tool: {tool_name}")')

            lines.append('    ')

        lines.append('    return "\\n".join(results)')

        return '\n'.join(lines)

    def _extract_pattern(self, query: str) -> str:
        """Extract a pattern name from query.

        Args:
            query: User query

        Returns:
            Pattern name (snake_case, English, max 3-4 words)
        """
        import re
        from datetime import datetime

        # Transliterate common Russian words to English
        ru_to_en = {
            'цель': 'target',
            'компонент': 'component',
            'задача': 'task',
            'файл': 'file',
            'дизайн': 'design',
            'апгрейд': 'upgrade',
            'улучши': 'improve',
            'сделай': 'make',
            'добавь': 'add',
            'измени': 'change',
            'исправь': 'fix',
            'создай': 'create',
            'изучи': 'study',
            'понять': 'understand',
            'разобрать': 'analyze',
            'проверить': 'check',
        }

        query_lower = query.lower()

        # Replace Russian words with English
        for ru, en in ru_to_en.items():
            query_lower = query_lower.replace(ru, en)

        # Extract words (alphanumeric only)
        words = re.findall(r'[a-z0-9]+', query_lower)

        # Filter out common words and keep meaningful ones
        stopwords = {'the', 'and', 'for', 'with', 'from', 'this', 'that', 'have', 'will', 'src', 'web', 'nare', 'system'}
        words = [w for w in words if len(w) > 2 and w not in stopwords]

        # Take first 3-4 meaningful words
        if words:
            pattern = '_'.join(words[:4])
        else:
            # Fallback: use timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pattern = f"skill_{timestamp}"

        return pattern

    def _extract_keywords(self, query: str) -> list:
        """Extract keywords from query for trigger function.

        Args:
            query: User query

        Returns:
            List of keywords
        """
        import re

        words = re.findall(r'\w+', query.lower())
        words = [w for w in words if len(w) > 3 and w not in {
            'the', 'and', 'for', 'with', 'from', 'this', 'that', 'have', 'will'
        }]

        return words[:5]
