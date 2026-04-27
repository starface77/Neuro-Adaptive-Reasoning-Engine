import re
import json

def test_rule():
    with open('memory_store/rules.json', 'r') as f:
        rules = json.load(f)
    
    rule = rules[0]
    code = rule['python_code']
    
    query = "Sequence: 1, 3, 5, 7. Next?"
    
    import re as _re, math as _math
    # Use a single dictionary for both globals and locals
    env = {"re": _re, "math": _math, "__builtins__": __builtins__}
    exec(code, env)
    
    is_triggered = env['trigger'](query)
    print(f"Triggered: {is_triggered}")
    
    if is_triggered:
        try:
            res = env['execute'](query)
            print(f"Result: {res}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_rule()
