"""Tool-calling agent loop for NARE.





This is a thin orchestration layer that sits *next to* the existing


`ReasoningRouter`. It does:





    triage → (plan when EDIT) → tool-call loop → final answer





Each iteration the LLM is given the conversation, the available tool


schemas, and the previous tool observations; it responds with either a


`<tool_call>` (executed locally; observation fed back) or a


`<final_answer>` (loop terminates).





The loop emits events on an `EventBus` so the CLI can render Phase-2


tool blocks live, without the loop knowing anything about Rich.





Honest scope:


- This loop deliberately does **not** replace `ReasoningRouter`. The


  existing router still owns 5-tier routing + verified synthesis +


  library learning. Phase 3 introduces a *parallel* path that the


  caller opts in to via `loop.run(...)`.


- Budgets are enforced (iterations / tokens / wall-clock). When a


  budget is exhausted the loop stops with a partial answer.


- The loop is single-threaded and synchronous. Phase 4 will layer


  resume / persistent task list on top.


"""





from __future__ import annotations





import json


from nare.utils.logger import get_logger


import re


import time


from dataclasses import dataclass, field


from typing import Any, Callable, Dict, List, Optional





from dotenv import load_dotenv


load_dotenv()





from ...core.events import (


    EventBus,


    IterationCompleted,


    PlanProposed,


    TaskFinished,


    TaskStarted,


    Thought,


    TokensConsumed,


    TokenStreamed,


    TodoUpdated,


    ToolEnd,


    ToolStart,


)


from ...tools.builtin import DEFAULT_REGISTRY, ToolRegistry, build_default_registry


from ...tools.builtin.base import ToolResult





log = get_logger("nare.agents.loop")





@dataclass


class Budget:


    """Hard caps on a single agent run."""





    max_iterations: int = 150


    max_tokens: int = 1_000_000


    max_wall_clock: float = 7200.0





    iterations: int = 0


    tokens_in: int = 0


    tokens_out: int = 0


    started: float = 0.0





    def start(self) -> None:


        self.started = time.time()


        self.iterations = 0


        self.tokens_in = 0


        self.tokens_out = 0





    @property


    def elapsed(self) -> float:


        return time.time() - self.started if self.started else 0.0





    @property


    def total_tokens(self) -> int:


        return self.tokens_in + self.tokens_out





    def exhausted(self) -> Optional[str]:


        """Return a reason string if the budget is exhausted, else None."""


        if self.iterations >= self.max_iterations:


            return f"max_iterations ({self.max_iterations})"


        if self.total_tokens >= self.max_tokens:


            return f"max_tokens ({self.max_tokens})"


        if self.elapsed >= self.max_wall_clock:


            return f"max_wall_clock ({self.max_wall_clock:.0f}s)"


        return None





@dataclass


class RunResult:


    ok: bool


    final_answer: Optional[str] = None


    iterations: int = 0


    tokens: int = 0


    elapsed: float = 0.0


    transcript: List[Dict[str, Any]] = field(default_factory=list)


    stop_reason: str = ""


    loop_state: Optional[Dict[str, Any]] = None





SYSTEM_PROMPT_HEAD = """


You are NARE, a senior software engineer operating in a CLI agent loop.


You have real filesystem access. You execute real commands.





IDENTITY:


- Your name is NARE (Neural Amortized Reasoning Engine).


- You are NOT Claude, GPT, Kiro, or any other assistant.


- Do NOT greet the user unless they greet you first.





OUTPUT FORMAT (strict):


Each turn MUST end with exactly ONE of these:





<tool_call>


{"name": "tool_name", "args": {"param": "value"}}


</tool_call>





OR





<final_answer>


Your response here.


</final_answer>





You may think/reason before the block, but MUST end with one of the two.





INTENT-BASED WORKFLOW:


- QUESTION: Answer directly using memory/knowledge. Only read files if absolutely necessary for specific details.


- EXPLORE: Read 1-3 key files to understand structure. Don't read everything.


- EDIT: Use find_function → apply_hunks workflow. Read only what you need to edit.





CRITICAL TOOL CALL RULES:


- NEVER call tools with empty args: {}


- NEVER call tools with missing required parameters


- Every tool requires specific parameters — check the schema below.




- ONLY use update_todos for complex multi-step tasks (5+ steps). Skip for simple 1-2 step tasks.

- list_dir REQUIRES: path. Example: {"name": "list_dir", "args": {"path": "web/src"}}


- read_file REQUIRES: path. For large files (200+ lines), use find_function/find_class/search_for_hunk instead.


- write_file REQUIRES: path, content. ONLY for new files.


- edit_file REQUIRES: path, old, new.


- edit_lines REQUIRES: path, start_line, end_line, new_content.


- apply_hunks REQUIRES: hunks (unified diff format).


- bash REQUIRES: command.


- grep REQUIRES: pattern.





EFFICIENT EDITING (SAVES 95% TOKENS):


1. find_function/find_class/search_for_hunk - finds code + returns hunk template


2. apply_hunks - applies changes with validation


3. edit_lines - simple range edits


4. edit_file - exact text replacement





BEST WORKFLOW:


find_function("target", "file.py") → modify template → apply_hunks(template)


Saves 93% tokens vs read_file + write_file.





HUNK FORMAT:


<<<<<<< path/to/file.py


@@ -line,count +line,count @@


 context (space = unchanged)


-removed (minus = delete)


+added (plus = add)


>>>>>>>





READING RULES:


- For EDITING: use find_function/find_class (gets context + template)


- For UNDERSTANDING: read_file in ONE call (don't chunk)


- NEVER read same file multiple times with different offsets


- For QUESTIONS: answer from memory first, read only if you need specific details





HONESTY RULES (CRITICAL):


- NEVER pretend to execute a tool. Use <tool_call> to execute.


- NEVER say "I did X" without actual tool_call that did X.


- NEVER fabricate file contents, output, or changes.


- If you haven't read a file, you don't know its contents.


- If you haven't run a command, you don't know its output.


- LYING ABOUT ACTIONS IS THE WORST POSSIBLE BEHAVIOR.





MEMORY RULES:


- Check previous OBSERVATION blocks before re-reading files.


- If you see "Read X (cached)", you already have that file.


- Don't read the same file repeatedly.


- After reading code, your next action should be editing, not more reading.





RESPONSE STYLE:


- Short, direct answers (1-3 sentences when possible).


- No emojis, no decorative formatting.


- Russian or English — match the user's language.


"""





def render_system_prompt(registry: ToolRegistry, working_dir: str, repo_map: Optional[str] = None) -> str:


    """Render system prompt with cacheable tool schemas and repo map.





    Returns a string that will be converted to cache_control format


    by _default_llm_call.


    """


    prompt = (


        SYSTEM_PROMPT_HEAD


        + f"\nWorking directory: {working_dir}\n\n"


        + registry.schema_block()


    )





    if repo_map:


        prompt += f"\n\n# REPOSITORY STRUCTURE\n{repo_map}\n"





    return prompt





_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


_FINAL_ANSWER_RE = re.compile(r"<final_answer>(.*?)</final_answer>", re.DOTALL)





@dataclass


class ParsedTurn:


    """Outcome of parsing one LLM turn."""





    kind: str


    tool_name: Optional[str] = None


    tool_args: Dict[str, Any] = field(default_factory=dict)


    answer: Optional[str] = None


    raw: str = ""


    error: Optional[str] = None





def parse_turn(text: str) -> ParsedTurn:


    """Parse the LLM's raw output for a tool call or final answer.





    Note: Ignores thinking blocks - they're handled separately in streaming.


    """





    final = _FINAL_ANSWER_RE.search(text)


    if final:


        return ParsedTurn(


            kind="final_answer",


            answer=final.group(1).strip(),


            raw=text,


        )





    call = _TOOL_CALL_RE.search(text)


    if call:


        body = call.group(1).strip()


        try:


            payload = json.loads(body)


        except json.JSONDecodeError as e:


            return ParsedTurn(


                kind="malformed",


                error=f"tool_call JSON parse error: {e}",


                raw=text,


            )


        name = payload.get("name")


        args = payload.get("args", {}) or {}


        if not name or not isinstance(name, str):


            return ParsedTurn(


                kind="malformed",


                error="tool_call missing 'name' string",


                raw=text,


            )


        if not isinstance(args, dict):


            return ParsedTurn(


                kind="malformed",


                error="tool_call 'args' must be an object",


                raw=text,


            )


        return ParsedTurn(


            kind="tool_call",


            tool_name=name,


            tool_args=args,


            raw=text,


        )





    if len(text.strip()) < 10:


        return ParsedTurn(


            kind="malformed",


            error="Response too short - waiting for more content",


            raw=text,


        )





    return ParsedTurn(


        kind="malformed",


        error=(


            "no <tool_call> or <final_answer> block found. "


            "Respond with exactly one of those two formats."


        ),


        raw=text,


    )








def _compress_observation(obs: str, tool_name: str) -> str:


    """Compress an observation to save context space.





    Keeps only summary for read_file, full content for errors.


    """


    if tool_name == "read_file":


        # Extract file path and line count


        lines = obs.split('\n')


        if len(lines) > 10:


            # Keep header + summary


            header = '\n'.join(lines[:3])


            line_count = len(lines) - 3


            return f"{header}\n... ({line_count} lines, content compressed)\n"


    elif "error" in obs.lower() or "failed" in obs.lower():


        # Keep errors in full


        return obs


    elif len(obs) > 500:


        # Truncate long observations


        return obs[:500] + "\n... (truncated)\n"





    return obs








def _apply_sliding_window(observations: List[str], window_size: int = 5) -> List[str]:


    """Apply sliding window to observations.





    Keeps first 3 (context) + last window_size observations in full.


    Compresses observations in the middle.


    """


    if len(observations) <= window_size + 3:


        return observations





    # Keep first 3 (CHAT HISTORY, PLAN, USER REQUEST)


    context = observations[:3]





    # Middle observations (compress)


    middle_start = 3


    middle_end = len(observations) - window_size


    compressed_middle = []





    for i in range(middle_start, middle_end):


        obs = observations[i]


        # Extract tool name from observation


        if "# OBSERVATION (" in obs:


            try:


                tool_name = obs.split("# OBSERVATION (")[1].split(")")[0]


                compressed = _compress_observation(obs, tool_name)


                compressed_middle.append(compressed)


            except (IndexError, AttributeError):


                compressed_middle.append(obs)


        else:


            compressed_middle.append(obs)





    # Keep last window_size in full


    recent = observations[-window_size:]





    return context + compressed_middle + recent








@dataclass


class AgentLoop:


    """Tool-calling loop driven by an LLM and an EventBus."""





    registry: ToolRegistry = field(default_factory=lambda: DEFAULT_REGISTRY)


    bus: EventBus = field(default_factory=EventBus)


    working_dir: str = "."


    budget: Budget = field(default_factory=Budget)





    llm_call: Optional[Callable[[str], str]] = None





    triage: Optional[Any] = None


    planner: Optional[Any] = None





    def __post_init__(self) -> None:


        if self.llm_call is None:


            self.llm_call = _default_llm_call


        if self.triage is None:


            try:


                from ..roles.triage import TriageAgent


                self.triage = TriageAgent()


            except Exception as e:


                log.warning(f"[Loop] TriageAgent unavailable: {e}")


        if self.planner is None:


            try:


                from ..planning import PlanningAgent


                self.planner = PlanningAgent()


            except Exception as e:


                log.warning(f"[Loop] PlanningAgent unavailable: {e}")





    async def run(


        self,


        query: str,


        *,


        chat_history: Optional[str] = None,


        repo_map: Optional[str] = None,


        on_final_token: Optional[Callable[[str], None]] = None,


        resume_state: Optional[Dict[str, Any]] = None,


    ) -> RunResult:


        """Execute the agent loop for `query`. Returns a `RunResult`.





        ``on_final_token`` (optional) is invoked with each text fragment


        the LLM streams while it is inside a ``<final_answer>`` block.


        The CLI uses this to typewriter the answer to the terminal in


        real time. Tokens emitted *outside* a ``<final_answer>`` block


        (i.e. inside ``<tool_call>`` JSON or model preambles) are not


        forwarded — those are still parsed and acted on, but never


        printed raw.


        """





        self.budget.start()





        intent = "EDIT"


        if self.triage is not None:


            try:


                intent = self.triage.classify(query)


            except Exception as e:


                log.warning(f"[Loop] triage failed: {e}")





        self.bus.emit(TaskStarted(query=query, intent=intent))





        transcript: List[Dict[str, Any]] = [


            {"role": "user", "content": query},


        ]





        plan_steps: List[str] = []


        # Skip planning for simple tasks (short queries, single action)
        skip_planning = (
            len(query.split()) < 10 or  # Short query
            any(word in query.lower() for word in ['начинай', 'продолжай', 'continue', 'go', 'start'])
        )
        
        if intent == "EDIT" and self.planner is not None and not skip_planning:


            try:


                plan = self.planner.generate_plan(query, repo_map=repo_map)


                plan_steps = plan.get("plan_steps") or []


                self.bus.emit(PlanProposed(


                    steps=plan_steps,


                    target_files=plan.get("target_files", []),


                    complexity=plan.get("complexity", "moderate"),


                ))


            except Exception as e:


                log.warning(f"[Loop] planning failed: {e}")





        system_prompt = render_system_prompt(self.registry, self.working_dir, repo_map)


        observations: List[str] = []


        if chat_history:


            observations.append(f"# CHAT HISTORY\n{chat_history}\n")


        if plan_steps:


            steps_block = "\n".join(f"  {i}. {s}" for i, s in enumerate(plan_steps, 1))


            observations.append(f"# PLAN\n{steps_block}\n")


        observations.append(f"# USER REQUEST\n{query}\n")





        final_answer: Optional[str] = None


        stop_reason = "ok"





        failed_tools: List[tuple] = []


        no_op_tools: List[tuple] = []


        all_reads: List[str] = []


        max_same_failure = 1  # Строже: 1 ошибка и стоп


        max_same_no_op = 1     # Строже: 1 no-op и стоп


        max_same_read = 4      # Allow up to 4 reads of the same file before stopping





        read_cache: Dict[str, str] = {}


        read_counts: Dict[str, int] = {}  # Track how many times each file was read





        recent_actions: List[str] = []


        max_action_history = 10





        empty_args_count = 0


        max_empty_args = 2  # Максимум 2 пустых вызова подряд





        if resume_state:


            observations = resume_state.get("observations", observations)


            transcript = resume_state.get("transcript", transcript)


            failed_tools = resume_state.get("failed_tools", failed_tools)


            no_op_tools = resume_state.get("no_op_tools", no_op_tools)


            all_reads = resume_state.get("all_reads", all_reads)


            read_cache = resume_state.get("read_cache", read_cache)


            read_counts = resume_state.get("read_counts", read_counts)


            recent_actions = resume_state.get("recent_actions", recent_actions)


            empty_args_count = resume_state.get("empty_args_count", empty_args_count)


            self.budget.iterations = resume_state.get("budget_iterations", 0)


            self.budget.tokens_in = resume_state.get("budget_tokens_in", 0)


            self.budget.tokens_out = resume_state.get("budget_tokens_out", 0)


            # Do not overwrite budget.started so we get fresh wall clock time


            log.info(f"[Loop] Resumed from state. Iterations: {self.budget.iterations}")





        while True:


            reason = self.budget.exhausted()


            if reason:


                stop_reason = f"budget_exhausted: {reason}"


                final_answer = (


                    final_answer


                    or "I ran out of budget before finishing. "


                    + f"Reason: {reason}. Use /resume in Phase 4 to continue."


                )


                break





            self.budget.iterations += 1





            # Apply sliding window to compress old observations


            compressed_observations = _apply_sliding_window(observations, window_size=5)


            user_body = "\n".join(compressed_observations) + "\n\nNext turn:"





            stream_cb: Optional[Callable[[str], None]] = None





            def emit_token(text: str) -> None:


                self.bus.emit(TokenStreamed(text=text))





            def emit_thinking(text: str) -> None:





                self.bus.emit(Thought(text=text))





            if on_final_token is not None:





                final_streamer = _make_final_answer_streamer(on_final_token)


                def combined_callback(text: str) -> None:





                    if text.startswith("[thinking] "):


                        thinking_text = text[len("[thinking] "):]


                        emit_thinking(thinking_text)


                    else:


                        final_streamer(text)


                stream_cb = combined_callback


            else:





                final_streamer = _make_final_answer_streamer(emit_token)


                def combined_callback(text: str) -> None:


                    if text.startswith("[thinking] "):


                        thinking_text = text[len("[thinking] "):]


                        emit_thinking(thinking_text)


                    else:


                        final_streamer(text)


                stream_cb = combined_callback





            try:





                import inspect as _inspect


                try:


                    sig = _inspect.signature(self.llm_call)


                    params = [


                        p for p in sig.parameters.values()


                        if p.kind in (


                            _inspect.Parameter.POSITIONAL_ONLY,


                            _inspect.Parameter.POSITIONAL_OR_KEYWORD,


                        )


                    ]


                    n_params = len(params)


                    accepts_stream = (


                        n_params >= 3


                        or any(


                            p.name == "stream_callback" for p in sig.parameters.values()


                        )


                    )


                except (TypeError, ValueError):


                    n_params = 1


                    accepts_stream = False





                if accepts_stream:


                    output = self.llm_call(system_prompt, user_body, stream_cb)


                elif n_params >= 2:


                    output = self.llm_call(system_prompt, user_body)


                else:


                    output = self.llm_call(_assemble_prompt(system_prompt, observations))


            except Exception as e:


                stop_reason = f"llm_error: {e}"


                final_answer = f"LLM call failed: {e}"


                break





            tok_in = max(1, (len(system_prompt) + len(user_body)) // 4)


            tok_out = max(1, len(output) // 4)


            self.budget.tokens_in += tok_in


            self.budget.tokens_out += tok_out


            self.bus.emit(TokensConsumed(delta_in=tok_in, delta_out=tok_out))





            parsed = parse_turn(output)


            transcript.append({"role": "assistant", "content": output})





            if parsed.kind == "final_answer":


                final_answer = parsed.answer or ""


                stop_reason = "final_answer"


                break





            if parsed.kind == "malformed":


                self.bus.emit(Thought(text=f"(parse retry: {parsed.error})"))


                observations.append(


                    "# OBSERVATION\n"


                    "Your last turn was malformed. "


                    f"{parsed.error}\n"


                    "Reply with EXACTLY ONE <tool_call>...</tool_call> or "


                    "<final_answer>...</final_answer> block.\n"


                )


                self.bus.emit(IterationCompleted(


                    n=self.budget.iterations,


                    budget_iterations=self.budget.max_iterations,


                    budget_tokens=self.budget.max_tokens,


                    used_tokens=self.budget.total_tokens,


                ))


                continue





            assert parsed.tool_name is not None





            # Extract thinking from LLM output (before <tool_call>)


            thinking_text = output.split('<tool_call>')[0].strip()


            if thinking_text and len(thinking_text) > 10:


                # LLM provided reasoning - emit it


                self.bus.emit(Thought(text=f"● {thinking_text[:200]}"))





            # Защита от зацикливания последовательностей


            action_sig = f"{parsed.tool_name}:{parsed.tool_args.get('path', '')}"


            recent_actions.append(action_sig)


            if len(recent_actions) > max_action_history:


                recent_actions.pop(0)





            if not parsed.tool_args:


                empty_args_count += 1


                self.bus.emit(Thought(text=f"  Invalid: {parsed.tool_name} called with no arguments ({empty_args_count}/{max_empty_args})"))





                example = ""


                if parsed.tool_name == "read_file":


                    example = '\n\nCorrect example:\n<tool_call>\n{"name": "read_file", "args": {"path": "nare/core/__init__.py"}}\n</tool_call>'


                elif parsed.tool_name == "write_file":


                    example = '\n\nCorrect example:\n<tool_call>\n{"name": "write_file", "args": {"path": "file.py", "content": "# code here"}}\n</tool_call>'


                elif parsed.tool_name == "bash":


                    example = '\n\nCorrect example:\n<tool_call>\n{"name": "bash", "args": {"command": "ls -la"}}\n</tool_call>'


                elif parsed.tool_name == "list_dir":


                    example = '\n\nCorrect example:\n<tool_call>\n{"name": "list_dir", "args": {"path": "web/src"}}\n</tool_call>'





                observations.append(


                    f"# OBSERVATION\n"


                    f"ERROR: Tool '{parsed.tool_name}' was called with empty args: {{}}\n"


                    f"This is INVALID. Every tool requires arguments.\n"


                    f"Check the TOOLS schema in the system prompt for required parameters.{example}\n"


                    f"Attempts remaining: {max_empty_args - empty_args_count}\n"


                )


                self.bus.emit(IterationCompleted(


                    n=self.budget.iterations,


                    budget_iterations=self.budget.max_iterations,


                    budget_tokens=self.budget.max_tokens,


                    used_tokens=self.budget.total_tokens,


                ))


                continue


            else:


                # Сброс счетчика при успешном вызове


                empty_args_count = 0





            required_params = {


                'read_file': ['path'],


                'write_file': ['path', 'content'],


                'edit_file': ['path', 'old', 'new'],


                'bash': ['command'],


                'grep': ['pattern'],


                'list_dir': [],


            }





            if parsed.tool_name in required_params:


                missing = [p for p in required_params[parsed.tool_name] if not parsed.tool_args.get(p)]


                if missing:


                    self.bus.emit(Thought(text=f"  Invalid: {parsed.tool_name} missing {', '.join(missing)}"))





                    example = ""


                    if parsed.tool_name == "read_file":


                        example = '\n\nCorrect:\n<tool_call>\n{"name": "read_file", "args": {"path": "path/to/file.py"}}\n</tool_call>'


                    elif parsed.tool_name == "write_file":


                        example = '\n\nCorrect:\n<tool_call>\n{"name": "write_file", "args": {"path": "file.py", "content": "code here"}}\n</tool_call>'


                    elif parsed.tool_name == "edit_file":


                        example = '\n\nCorrect:\n<tool_call>\n{"name": "edit_file", "args": {"path": "file.py", "old": "old text", "new": "new text"}}\n</tool_call>'


                    elif parsed.tool_name == "bash":


                        example = '\n\nCorrect:\n<tool_call>\n{"name": "bash", "args": {"command": "ls -la"}}\n</tool_call>'


                    elif parsed.tool_name == "grep":


                        example = '\n\nCorrect:\n<tool_call>\n{"name": "grep", "args": {"pattern": "search_term"}}\n</tool_call>'





                    observations.append(


                        f"# OBSERVATION\n"


                        f"ERROR: Tool '{parsed.tool_name}' missing required parameters: {', '.join(missing)}\n"


                        f"You provided: {parsed.tool_args}\n"


                        f"Required parameters: {', '.join(required_params[parsed.tool_name])}{example}\n"


                    )


                    self.bus.emit(IterationCompleted(


                        n=self.budget.iterations,


                        budget_iterations=self.budget.max_iterations,


                        budget_tokens=self.budget.max_tokens,


                        used_tokens=self.budget.total_tokens,


                    ))


                    continue





            verb = self._verb_for(parsed.tool_name)


            self.bus.emit(ToolStart(


                name=parsed.tool_name,


                display_verb=verb,


                args=parsed.tool_args,


            ))





            if parsed.tool_name == 'read_file':


                file_path = parsed.tool_args.get('path', '')





                # Track read count


                read_counts[file_path] = read_counts.get(file_path, 0) + 1





                # If file was read 3+ times, return error instead of cached result


                if read_counts[file_path] > 3:


                    from nare.tools.builtin.base import ToolResult


                    result = ToolResult(


                        ok=False,


                        error=(


                            f"STOP. You've read {file_path} {read_counts[file_path]-1} times already. "


                            f"The file contents are in your previous OBSERVATION blocks. "


                            f"DO NOT read this file again. "


                            f"Your next action MUST be write_file or edit_file to implement the code. "


                            f"If you need to read a different file, specify a different path."


                        )


                    )


                    self.bus.emit(Thought(text=f"  Blocked: {file_path} read {read_counts[file_path]} times"))


                elif file_path in read_cache:





                    from nare.tools.builtin.base import ToolResult


                    result = ToolResult(


                        ok=True,


                        summary=f"Read {file_path} (cached)",


                        body=read_cache[file_path],


                        meta={'lines': len(read_cache[file_path].splitlines()), 'cached': True}


                    )


                    self.bus.emit(Thought(text=f"  Using cached: {file_path}"))


                else:





                    result = await self.registry.call(parsed.tool_name, parsed.tool_args)


                    if result.ok and result.body:


                        read_cache[file_path] = result.body


            else:


                result = await self.registry.call(parsed.tool_name, parsed.tool_args)





            if result.ok and parsed.tool_name in ('edit_file', 'write_file'):


                file_path = parsed.tool_args.get('path')


                if file_path and file_path in read_cache:


                    del read_cache[file_path]


                    self.bus.emit(Thought(text=f"  Cache invalidated: {file_path}"))

                # Auto-commit changes for session persistence
                try:
                    import subprocess
                    import os
                    if os.path.exists(os.path.join(self.working_dir, '.git')):
                        subprocess.run(['git', 'add', file_path], cwd=self.working_dir, capture_output=True, timeout=2)
                        commit_msg = f"nare: {parsed.tool_name} {os.path.basename(file_path)}"
                        subprocess.run(['git', 'commit', '-m', commit_msg, '--no-verify'], cwd=self.working_dir, capture_output=True, timeout=2)
                        self.bus.emit(Thought(text=f"  Auto-committed: {os.path.basename(file_path)}"))
                except Exception as e:
                    log.debug(f"Auto-commit failed: {e}")





            if not result.ok:


                import hashlib


                args_hash = hashlib.md5(str(parsed.tool_args).encode()).hexdigest()[:8]


                failure_sig = (parsed.tool_name, args_hash, result.error or "error")





                failed_tools.append(failure_sig)





                # Detect repeated failed searches for same pattern


                if parsed.tool_name in ('find_function', 'find_class', 'search_for_hunk', 'grep'):


                    pattern = str(parsed.tool_args.get('pattern', '') or parsed.tool_args.get('name', ''))


                    recent_failed_searches = [


                        sig for sig in failed_tools[-5:]


                        if sig[0] == parsed.tool_name and pattern in str(sig)


                    ]





                    if len(recent_failed_searches) >= 3:


                        self.bus.emit(Thought(text=f"  LOOP DETECTED: {parsed.tool_name} failed 3+ times for '{pattern[:30]}'"))


                        self.bus.emit(Thought(text=f"  This pattern does not exist. Try a different approach."))





                        # Inject strong feedback into conversation


                        conversation.append({


                            "role": "user",


                            "content": f"STOP. You've tried {parsed.tool_name} for '{pattern}' {len(recent_failed_searches)} times and it failed every time. This pattern DOES NOT EXIST in the codebase. You MUST try a completely different approach or ask for clarification. DO NOT search for this pattern again."


                        })


                        continue


            else:





                import hashlib


                args_hash = hashlib.md5(str(parsed.tool_args).encode()).hexdigest()[:8]





                is_no_op = False


                if parsed.tool_name == 'edit_file':





                    if result.summary and ('+0' in result.summary and '-0' in result.summary):


                        is_no_op = True


                        self.bus.emit(Thought(text=f"  Warning: edit_file made no changes"))


                elif parsed.tool_name == 'read_file':





                    file_path = parsed.tool_args.get('path', '')


                    all_reads.append(file_path)





                    # Disabled: stuck reading check removed per user request


                    # The agent should be allowed to read files as many times as needed


                elif parsed.tool_name == 'update_todos':





                    recent_updates = [op for op in no_op_tools[-3:] if op[0] == 'update_todos']


                    if len(recent_updates) >= 1:


                        is_no_op = True


                        self.bus.emit(Thought(text=f"  Warning: repeatedly updating todos"))





                if is_no_op:


                    no_op_sig = (parsed.tool_name, args_hash, parsed.tool_args.get('path', ''))


                    same_no_ops = sum(1 for op in no_op_tools if op[:2] == no_op_sig[:2])





                    no_op_tools.append(no_op_sig)


                    self.bus.emit(Thought(text=f"  Warning: {parsed.tool_name} made no changes ({same_no_ops + 1})"))


                else:





                    no_op_tools.clear()





            self.bus.emit(ToolEnd(


                name=parsed.tool_name,


                display_verb=verb,


                args=parsed.tool_args,


                ok=result.ok,


                summary=result.summary,


                body=result.body,


                body_lang=result.body_lang,


                meta=dict(result.meta),


                error=result.error,


            ))





            todo_items = result.meta.get("_todo_items") if result.meta else None


            if todo_items:


                self.bus.emit(TodoUpdated(items=list(todo_items)))





            obs = (


                f"# OBSERVATION ({parsed.tool_name})\n"


                f"{result.to_llm_observation()}\n"


            )


            observations.append(obs)


            transcript.append({"role": "tool", "content": obs})





            self.bus.emit(IterationCompleted(


                n=self.budget.iterations,


                budget_iterations=self.budget.max_iterations,


                budget_tokens=self.budget.max_tokens,


                used_tokens=self.budget.total_tokens,


            ))





            # Check for general looping pattern (similar actions repeated)


            if self.budget.iterations >= 5:


                recent_actions = []


                for i in range(min(5, len(transcript))):


                    msg = transcript[-(i+1)]


                    if msg.get('role') == 'assistant':


                        content = msg.get('content', '')


                        # Extract tool name from <tool_call>


                        import re


                        match = re.search(r'"name":\s*"([^"]+)"', content)


                        if match:


                            recent_actions.append(match.group(1))





                if len(recent_actions) >= 4:


                    # Check if same action repeated 3+ times


                    from collections import Counter


                    action_counts = Counter(recent_actions)


                    most_common = action_counts.most_common(1)


                    if most_common and most_common[0][1] >= 3:


                        repeated_action = most_common[0][0]


                        self.bus.emit(Thought(text=f"  LOOP DETECTED: {repeated_action} repeated {most_common[0][1]} times"))


                        conversation.append({


                            "role": "user",


                            "content": f"STOP. You're stuck in a loop - you've called {repeated_action} {most_common[0][1]} times in the last 5 iterations. This approach is not working. You MUST try a completely different strategy or provide a final answer with what you've learned so far."


                        })


                        # Force next iteration to be final answer or different approach


                        continue





        summary = await self._generate_summary(query, transcript, final_answer, stop_reason)





        elapsed = self.budget.elapsed


        # Save transcript for session persistence
        try:
            import json
            import os
            from datetime import datetime
            session_dir = os.path.join(self.working_dir, '.nare_cache', 'sessions')
            os.makedirs(session_dir, exist_ok=True)
            session_file = os.path.join(session_dir, f'last_session.json')
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'query': query,
                    'transcript': transcript[-20:],  # Last 20 messages
                    'final_answer': final_answer,
                    'stop_reason': stop_reason,
                    'iterations': self.budget.iterations,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            log.debug(f"Failed to save transcript: {e}")

        self.bus.emit(TaskFinished(


            ok=stop_reason in {"final_answer", "ok"},


            final_answer=final_answer,


            iterations=self.budget.iterations,


            tokens=self.budget.total_tokens,


            elapsed=elapsed,


        ))





        if summary:


            self.bus.emit(Thought(text=summary))





        current_state = {


            "observations": observations,


            "transcript": transcript,


            "failed_tools": failed_tools,


            "no_op_tools": no_op_tools,


            "all_reads": all_reads,


            "read_cache": read_cache,


            "read_counts": read_counts,


            "recent_actions": recent_actions,


            "empty_args_count": empty_args_count,


            "budget_iterations": self.budget.iterations,


            "budget_tokens_in": self.budget.tokens_in,


            "budget_tokens_out": self.budget.tokens_out,


            "query": query,


        }





        return RunResult(


            ok=stop_reason in {"final_answer", "ok"},


            final_answer=final_answer,


            iterations=self.budget.iterations,


            tokens=self.budget.total_tokens,


            elapsed=elapsed,


            transcript=transcript,


            stop_reason=stop_reason,


            loop_state=current_state,


        )





    async def _generate_summary(


        self,


        query: str,


        transcript: List[Dict[str, Any]],


        final_answer: Optional[str],


        stop_reason: str,


    ) -> str:


        """Generate a concise summary using LLM.





        CRITICAL: Only generate summary if work was actually done.


        If stopped by loop/error, return empty string to avoid hallucination.


        """





        # Don't generate summary if stopped by error/loop


        if stop_reason in ("action_loop", "repeated_empty_args", "repeated_failure",


                          "repeated_no_op", "repeated_read", "budget_exhausted"):


            return ""  # No summary - avoid hallucination





        if len(transcript) < 3:


            return ""





        # Count actual successful actions (not just reads)


        successful_writes = 0


        successful_edits = 0


        actions_summary = []





        for turn in transcript:


            if turn.get("role") == "tool":


                content = turn.get("content", "")


                if "edit_file" in content and "+0 -0" not in content:


                    successful_edits += 1


                    actions_summary.append("edit_file")


                elif "write_file" in content:


                    successful_writes += 1


                    actions_summary.append("write_file")


                elif "bash" in content and "exit 0" in content:


                    actions_summary.append("bash")





        # If no actual changes were made, don't generate summary


        if successful_writes == 0 and successful_edits == 0:


            return ""





        from ...reasoning import llm





        summary_prompt = f"""Task: {query}


Actions taken: {', '.join(actions_summary[:10])}


Files modified: {successful_edits}


Files created: {successful_writes}


Result: {stop_reason}





Generate a 1-sentence summary of what was ACTUALLY done (not planned).


Only mention actions that were successfully executed.


Be concise and factual. No bullet points, just plain text."""





        try:


            samples, _ = llm.generate_samples(


                summary_prompt,


                n=1,


                temperature=0.3,


                mode="DIRECT"


            )


            if samples and len(samples) > 0 and isinstance(samples[0], dict):


                return samples[0].get('solution', '').strip()


        except Exception:


            pass





        return ""





    def _verb_for(self, name: str) -> str:


        try:


            return self.registry.get(name).display_verb or name


        except Exception:


            return name





def _assemble_prompt(system_prompt: str, observations: List[str]) -> str:


    """Concatenate the system prompt and rolling observation log.





    Used as a single combined string only when the caller-supplied


    ``llm_call`` accepts one positional argument (the legacy contract


    used by tests and stub LLMs). The default LLM caller bypasses this


    function and talks to the provider directly with a proper


    system/user split.


    """


    body = "\n".join(observations)


    return f"{system_prompt}\n\n---\n\n{body}\n\nNext turn:"





def _make_final_answer_streamer(


    on_text: Callable[[str], None],


) -> Callable[[str], None]:


    """Filter LLM token deltas to only what's inside ``<final_answer>``.





    The model emits a mix of prose, ``<tool_call>{...}</tool_call>`` JSON


    and ``<final_answer>...</final_answer>`` Markdown. We only want to


    typewriter-stream the answer to the user; everything else stays


    buffered until the loop's parser handles it.





    The returned callable maintains a small rolling buffer so that


    opening/closing tags split across multiple deltas (``<final_`` then


    ``answer>``) are still detected.


    """


    state = {


        "buf": "",


        "in_final": False,


        "done": False,


        "open_tag": "<final_answer>",


        "close_tag": "</final_answer>",


    }





    TAIL = 24





    def _push(delta: str) -> None:


        if state["done"]:


            return


        state["buf"] += delta





        if not state["in_final"]:


            idx = state["buf"].find(state["open_tag"])


            if idx >= 0:





                rest = state["buf"][idx + len(state["open_tag"]):]





                if rest.startswith("\n"):


                    rest = rest[1:]


                state["buf"] = rest


                state["in_final"] = True


            else:





                if len(state["buf"]) > TAIL:


                    state["buf"] = state["buf"][-TAIL:]


                return





        idx = state["buf"].find(state["close_tag"])


        if idx >= 0:


            chunk = state["buf"][:idx]


            if chunk:


                on_text(chunk)


            state["buf"] = ""


            state["done"] = True


            return





        if len(state["buf"]) > TAIL:


            emit = state["buf"][:-TAIL]


            state["buf"] = state["buf"][-TAIL:]


            on_text(emit)





    return _push








# ─── Dynamic Thinking Budget ─────────────────────────────────────────





def _estimate_thinking_budget(user_prompt: str) -> int:


    """Estimate optimal thinking budget based on prompt complexity.





    Returns a budget in tokens (1024–4000) that scales with complexity,


    or 0 to disable thinking for simple conversational queries.


    """


    length = len(user_prompt)





    # Complexity signals


    edit_signals = [


        "apply_hunks", "write_file", "edit_file", "refactor",


        "migrate", "implement", "create", "build", "fix all",


        "<<<<<<", ">>>>>>>", "@@ -", "plan", "план"


    ]


    multi_step_signals = [


        "step 1", "step 2", "first", "then", "after that",


        "finally", "next", "шаг 1", "сначала", "потом",


    ]





    prompt_lower = user_prompt.lower()


    has_edits = any(s in prompt_lower for s in edit_signals)


    has_multi_step = any(s in prompt_lower for s in multi_step_signals)





    # Simple queries get no thinking overhead


    if not has_edits and not has_multi_step and length < 1000:


        return 0





    # Base budget for complex queries (min 1024 for Claude 3.7)


    budget = 1024


    


    if length >= 5000:


        budget = 2000


    if length >= 15000:


        budget = 4000





    # Boost for complex operations


    if has_multi_step:


        budget = max(budget, 2048)


    if has_edits and has_multi_step:


        budget = max(budget, 3000)





    return min(budget, 8192)





def _default_llm_call(


    system_prompt: str,


    user_prompt: str,


    stream_callback: Optional[Callable[[str], None]] = None,


) -> str:


    """Default LLM caller — talks to Anthropic (or a compatible proxy).





    The user's proxy returns SSE (``event: ...\\ndata: {...}``) even for


    non-streaming requests, so we always parse the body as SSE first


    and fall back to plain JSON. When ``stream_callback`` is provided


    we also enable ``stream=True`` server-side and forward each


    ``content_block_delta`` token to the callback in real time.





    The endpoint shape varies between providers:


      - Anthropic public API:    ``/v1/messages``


      - Most reverse proxies:    ``/messages``


    We try ``/messages`` first (matches the user's proxy and the


    legacy ``nare/reasoning/llm.py``) and fall back to ``/v1/messages``


    on 404.


    """


    import json


    import os


    import urllib.error


    import urllib.request





    base = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")


    key = os.getenv("ANTHROPIC_API_KEY", "")


    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")


    if not key:


        raise RuntimeError(


            "ANTHROPIC_API_KEY is not set. The agent loop needs an LLM key. "


            "Add it to your .env or use /agent off to fall back to the legacy router."


        )





    thinking_budget = _estimate_thinking_budget(user_prompt)





    payload: Dict[str, Any] = {


        "model": model,


        "max_tokens": 4096,


        "temperature": 0.2,


        "thinking": {


            "type": "enabled",


            "budget_tokens": thinking_budget


        },


        "system": [


            {


                "type": "text",


                "text": system_prompt,


                "cache_control": {"type": "ephemeral"}


            }


        ],


        "messages": [{"role": "user", "content": user_prompt}],


    }


    if stream_callback is not None:


        payload["stream"] = True





    headers = {


        "content-type": "application/json",


        "x-api-key": key,


        "anthropic-version": "2023-06-01",


        "anthropic-beta": "prompt-caching-2024-07-31,extended-thinking-2024-12-12",


        "accept": "text/event-stream" if stream_callback is not None else "application/json",


    }


    data = json.dumps(payload).encode("utf-8")





    def _extract_sse_text(body_str: str) -> str:


        """Walk an SSE body and concatenate all content_block_delta texts."""


        out: List[str] = []


        for line in body_str.splitlines():


            line = line.strip()


            if not line.startswith("data:"):


                continue


            chunk = line[len("data:"):].strip()


            if not chunk or chunk == "[DONE]":


                continue


            try:


                ev = json.loads(chunk)


            except json.JSONDecodeError:


                continue


            if ev.get("type") == "content_block_delta":


                delta = ev.get("delta", {}) or {}


                t = delta.get("text", "")


                if t:


                    out.append(t)


        return "".join(out)





    def _post(path: str) -> str:


        req = urllib.request.Request(f"{base}{path}", data=data, headers=headers)


        with urllib.request.urlopen(req, timeout=120) as r:


            if stream_callback is not None:





                parts: List[str] = []


                thinking_parts: List[str] = []


                current_block_type = None





                while True:


                    raw_line = r.readline()


                    if not raw_line:


                        break


                    line = raw_line.decode("utf-8", errors="ignore").strip()


                    if not line.startswith("data:"):


                        continue


                    chunk = line[len("data:"):].strip()


                    if not chunk or chunk == "[DONE]":


                        continue


                    try:


                        ev = json.loads(chunk)


                    except json.JSONDecodeError:


                        continue





                    if ev.get("type") == "content_block_start":


                        block = ev.get("content_block", {})


                        current_block_type = block.get("type")





                    if ev.get("type") == "content_block_delta":


                        delta = ev.get("delta", {}) or {}


                        t = delta.get("text", "")


                        if t:





                            if current_block_type == "thinking":


                                thinking_parts.append(t)





                                try:


                                    stream_callback(f"[thinking] {t}")


                                except Exception:


                                    pass


                            else:


                                parts.append(t)


                                try:


                                    stream_callback(t)


                                except Exception:


                                    pass





                result = "".join(parts)





                if not result and thinking_parts:





                    return ""





                return result


            else:


                raw = r.read().decode("utf-8", errors="ignore")


                if not raw:


                    raise RuntimeError(f"empty response from {base}{path}")





                stripped = raw.lstrip()


                if stripped.startswith("event:") or stripped.startswith("data:"):


                    return _extract_sse_text(raw)


                obj = json.loads(raw)


                parts = obj.get("content") or []


                if isinstance(parts, list):


                    text = "".join(


                        p.get("text", "")


                        for p in parts


                        if isinstance(p, dict) and p.get("type") == "text"


                    )


                    if text:


                        return text





                choices = obj.get("choices") or []


                if choices:


                    msg = choices[0].get("message", {}) or {}


                    text = msg.get("content", "") or ""


                    if text:


                        return text


                raise RuntimeError(f"LLM returned no text content; raw body: {raw[:300]}")





    import time


    import random





    max_retries = 5


    for attempt in range(max_retries):


        try:


            return _post("/messages")


        except urllib.error.HTTPError as e:


            if e.code == 404:


                try:


                    return _post("/v1/messages")


                except urllib.error.HTTPError as e_v1:


                    e = e_v1  # Fall through to retry logic using the v1 error





            # 429 Too Many Requests, or 5xx Server Errors


            if e.code in (429, 500, 502, 503, 504):


                if attempt < max_retries - 1:


                    base_wait = min(5 * (2 ** attempt), 60)


                    jitter = random.uniform(0, base_wait * 0.1)


                    wait_time = base_wait + jitter


                    log.warning(f"[LLM] HTTP {e.code} (attempt {attempt+1}/{max_retries}). Retrying in {wait_time:.1f}s...")


                    time.sleep(wait_time)


                    continue


                


            try:


                err_body = e.read().decode("utf-8", errors="replace")[:300]


            except Exception:


                err_body = ""


            raise RuntimeError(f"LLM HTTP {e.code}: {err_body}") from None


        except urllib.error.URLError as e:


            # Handle connection errors (DNS failure, timeout, etc.)


            if attempt < max_retries - 1:


                wait_time = min(5 * (2 ** attempt), 30)


                log.warning(f"[LLM] Connection error: {e} (attempt {attempt+1}/{max_retries}). Retrying in {wait_time}s...")


                time.sleep(wait_time)


                continue


            raise RuntimeError(f"LLM Connection failed: {e}") from None





    raise RuntimeError("LLM max retries exceeded")





def build_loop(


    working_dir: str = ".",


    *,


    bus: Optional[EventBus] = None,


    budget: Optional[Budget] = None,


) -> AgentLoop:


    return AgentLoop(


        registry=build_default_registry(working_dir=working_dir),


        bus=bus or EventBus(),


        working_dir=working_dir,


        budget=budget or Budget(),


    )


