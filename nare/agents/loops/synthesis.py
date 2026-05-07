"""
Verified Synthesis Loop — System 2 reasoning with formal verification.

MDP formulation:
- State S: (query, error_history)
- Action A: generate candidate y_k ~ G_θ
- Transition T: execute in V_sandbox
- Reward R: R(y_k) = 1 if verified, 0 otherwise
- Horizon H: max_retries

Iteratively refines solution until formal verification passes.
"""

import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from ...core.synthesis.verifier import FormalVerifier, VerificationResult

log = logging.getLogger("nare.agents.loops.synthesis")

@dataclass
class SynthesisState:
    """MDP state for synthesis loop."""
    query: str
    error_history: List[str]
    attempt: int

class VerifiedSynthesisLoop:
    """Iterative synthesis with formal verification feedback."""

    def __init__(self, generator, verifier: FormalVerifier, max_retries: int = 8):
        """
        Args:
            generator: LLM generator (G_θ)
            verifier: Formal verifier (V_sandbox)
            max_retries: Maximum synthesis attempts (horizon H)
        """
        self.generator = generator
        self.verifier = verifier
        self.max_retries = max_retries

    def synthesize(
        self,
        query: str,
        oracle_fn=None,
        thinking_display=None
    ) -> Dict[str, Any]:
        """Run verified synthesis loop until success or max_retries.

        Returns:
            {
                'solution': str,
                'verified': bool,
                'attempts': int,
                'verification_result': VerificationResult
            }
        """

        state = SynthesisState(
            query=query,
            error_history=[],
            attempt=0
        )

        log.info(f"[Synthesis] Starting loop for query: {query[:100]}")

        while state.attempt < self.max_retries:
            state.attempt += 1

            if thinking_display:
                thinking_display.stream_token(
                    f"| Synthesis attempt {state.attempt}/{self.max_retries}\n"
                )

            candidate = self._generate_candidate(state, thinking_display)

            if thinking_display:
                thinking_display.stream_token("| Verifying candidate\n")

            if oracle_fn:
                verification = self.verifier.verify_with_oracle(candidate, oracle_fn)
            else:
                verification = self.verifier.verify(candidate)

            if verification.success:
                log.info(f"[Synthesis] SUCCESS on attempt {state.attempt}")

                if thinking_display:
                    thinking_display.stream_token(
                        f"| Verified in {state.attempt} attempts\n"
                    )

                return {
                    'solution': candidate,
                    'verified': True,
                    'attempts': state.attempt,
                    'verification_result': verification,
                    'final_score': 1.0
                }

            error_msg = verification.error_trace or "Unknown error"
            state.error_history.append(error_msg)

            log.warning(f"[Synthesis] Attempt {state.attempt} failed: {error_msg[:100]}")

            if thinking_display:
                thinking_display.stream_token(
                    f"| Failed: {error_msg[:80]}\n"
                )

        log.error(f"[Synthesis] FAILED after {self.max_retries} attempts")

        return {
            'solution': candidate if 'candidate' in locals() else "",
            'verified': False,
            'attempts': self.max_retries,
            'verification_result': verification if 'verification' in locals() else None,
            'final_score': 0.0,
            'error': f"Failed to synthesize solution after {self.max_retries} attempts"
        }

    def _generate_candidate(
        self,
        state: SynthesisState,
        thinking_display=None
    ) -> str:
        """Generate next candidate using G_θ with error feedback."""

        prompt = self._build_refinement_prompt(state)

        from ...reasoning import llm

        candidates, _ = llm.generate_samples(
            prompt,
            n=1,
            temperature=0.2,
            mode="SYNTHESIS",
            thinking_display=thinking_display
        )

        if not candidates:
            raise RuntimeError("Generator returned no candidates")

        return candidates[0]['solution']

    def _build_refinement_prompt(self, state: SynthesisState) -> str:
        """Build prompt with error feedback for self-refinement."""

        if not state.error_history:

            prompt = f"""
Generate Python code that solves this task.
The code will be formally verified - it must:
1. Compile without syntax errors
2. Execute without exceptions
3. Pass all test cases

Provide only the code, no explanations."""

        prompt += f"""
PREVIOUS ATTEMPTS FAILED VERIFICATION:
"""

        for i, error in enumerate(state.error_history[-2:], 1):
            prompt += f"\nAttempt {state.attempt - len(state.error_history) + i} error:\n{error}\n"

        prompt += f"""
CURRENT ATTEMPT {state.attempt}:
Analyze the errors above and generate CORRECTED code.
Focus on fixing the specific issues mentioned.

The code will be formally verified - ensure it:
1. Compiles without syntax errors
2. Executes without exceptions
3. Passes all test cases

Provide only the corrected code, no explanations."""

        return prompt
