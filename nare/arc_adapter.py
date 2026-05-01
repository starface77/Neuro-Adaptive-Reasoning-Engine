import logging
from typing import Dict, Any, Optional, Callable
from .sandbox import extract_python_block, safe_execute_freeform

# Use Anthropic API for ARC tasks
try:
    from . import llm_anthropic as arc_llm
    USE_ANTHROPIC = True
except ImportError:
    from . import llm as arc_llm
    USE_ANTHROPIC = False


class ARCAdapter:
    """ARC-AGI task adapter for NARE."""

    def __init__(self):
        self.task_cache = {}

    def parse_arc_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        train = task_data.get('train', [])
        test = task_data.get('test', [])

        query_parts = ["ARC-AGI grid transformation task. Analyze the pattern and write Python code."]
        query_parts.append("")
        query_parts.append("Training examples:")
        for i, example in enumerate(train):
            inp = example['input']
            out = example['output']
            query_parts.append(f"  Input {i+1}: {inp}")
            query_parts.append(f"  Output {i+1}: {out}")

        if test:
            query_parts.append("")
            query_parts.append(f"Test input: {test[0]['input']}")
            query_parts.append("")
            query_parts.append("Write a Python function that transforms the test input to match the pattern.")
            query_parts.append("Output ONLY a ```python``` code block with:")
            query_parts.append("1. A function `transform(grid)` that applies the transformation")
            query_parts.append("2. Call `print(transform(test_input))` at the end")
            query_parts.append(f"3. Use test_input = {test[0]['input']}")

        return {
            'query': '\n'.join(query_parts),
            'train': train,
            'test': test
        }

    def build_oracle(self, expected_output, fuzzy_threshold: float = 1.0) -> Callable:
        """Build oracle with exact match requirement (theory.md compliance).

        Args:
            expected_output: Expected grid output
            fuzzy_threshold: IoU threshold (1.0 = exact match only, as per theory.md)
        """
        def grid_iou(grid1, grid2) -> float:
            """Calculate Intersection over Union for grids."""
            try:
                if not isinstance(grid1, list) or not isinstance(grid2, list):
                    return 0.0
                if len(grid1) != len(grid2):
                    return 0.0

                total_cells = 0
                matching_cells = 0

                for row1, row2 in zip(grid1, grid2):
                    if not isinstance(row1, list) or not isinstance(row2, list):
                        return 0.0
                    if len(row1) != len(row2):
                        return 0.0

                    for c1, c2 in zip(row1, row2):
                        total_cells += 1
                        if c1 == c2:
                            matching_cells += 1

                return matching_cells / total_cells if total_cells > 0 else 0.0
            except:
                return 0.0

        def oracle(query: str, answer: str) -> tuple:
            answer_clean = answer.strip()

            # Try to parse as direct array output
            try:
                import ast
                parsed = ast.literal_eval(answer_clean)
                if parsed == expected_output:
                    return True, {"status": "exact match", "iou": 1.0}

                # Fuzzy match with IoU
                iou = grid_iou(parsed, expected_output)
                if iou >= fuzzy_threshold:
                    return True, {"status": "fuzzy match", "iou": iou}
                elif iou > 0.5:
                    return False, {"status": "partial match", "iou": iou}
                else:
                    return False, {"status": "low match", "iou": iou}
            except:
                pass

            # Try to extract from code block
            code = extract_python_block(answer)
            if code:
                try:
                    result = safe_execute_freeform(code)
                    result_clean = result.strip()

                    # Parse executed output
                    try:
                        parsed = ast.literal_eval(result_clean)
                        if parsed == expected_output:
                            return True, {"status": "code match", "iou": 1.0}

                        # Fuzzy match with IoU
                        iou = grid_iou(parsed, expected_output)
                        if iou >= fuzzy_threshold:
                            return True, {"status": "code fuzzy match", "iou": iou}
                        elif iou > 0.5:
                            return False, {"status": "code partial match", "iou": iou}
                        else:
                            return False, {"status": "code low match", "iou": iou}
                    except:
                        pass

                    if str(expected_output) in result_clean:
                        return True, {"status": "substring match", "iou": 1.0}

                    return False, {"status": "wrong output", "expected": str(expected_output)[:100], "got": result_clean[:100], "iou": 0.0}
                except Exception as e:
                    return False, {"status": "execution error", "error": str(e), "iou": 0.0}

            # Check if answer contains expected output as string
            if str(expected_output) in answer_clean:
                return True, {"status": "contains expected", "iou": 1.0}

            return False, {"status": "no match", "iou": 0.0}

        return oracle

    def extract_grid_transform(self, solution: str) -> Optional[Callable]:
        code = extract_python_block(solution)
        if not code:
            return None

        try:
            namespace = {}
            exec(code, namespace)
            if 'transform' in namespace:
                return namespace['transform']
        except:
            pass

        return None
