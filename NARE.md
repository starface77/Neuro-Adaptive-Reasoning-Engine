# NARE Project Rules

## Token Economy
- Use grep/find BEFORE reading files
- Read files only once, check OBSERVATION blocks for cached content
- After 3 read operations, MUST edit or answer
- Use find_function + apply_hunks instead of read + write

## Context Hygiene
- Agent auto-compacts context after 12 observations (keeps last 10)
- Use .nareignore to exclude build artifacts

## Anti-Hallucination
- ALWAYS read file before claiming to know its contents
- NEVER assume function signatures - verify with grep/read
- Test changes with bash before claiming success

## Response Style
- Be concise, no explanations unless asked
- Focus on code changes only
