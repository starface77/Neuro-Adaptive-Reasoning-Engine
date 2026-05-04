#!/usr/bin/env python3
"""
NARE Showcase Demo — "See it to believe it"

Runs the full cognitive cycle:
  Phase 1: Novel queries → SLOW path (expensive LLM reasoning)
  Phase 2: Exact repeats → FAST path (cached, 0 tokens)
  Phase 3: Sleep consolidation → Skills crystallized into Python
  Phase 4: Variations → REFLEX path (compiled code, 0 tokens, instant)

Generates a beautiful standalone HTML report with:
  - Real timing data & token counts
  - Route distribution chart (SVG)
  - Before/After cost comparison
  - Actual generated Python skill code
  - Full audit trail for every query

Usage:
    python demo_showcase.py
    python demo_showcase.py --output my_report.html
    python demo_showcase.py --skip-slow   # use cached episodes if available
"""

import os
import sys
import time
import json
import argparse
import html as html_mod
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ── Preflight ────────────────────────────────────────────────────────
if not os.getenv("GEMINI_API_KEY"):
    print("✗ GEMINI_API_KEY not set. Add it to .env file.")
    print("  Get a free key: https://aistudio.google.com/apikey")
    sys.exit(1)

from nare.agent import NAREProductionAgent

# ── Test Scenarios ───────────────────────────────────────────────────
# Each group: (domain_label, [query1, query2_exact_repeat, query3_variation])
SCENARIOS = [
    (
        "Arithmetic Series",
        [
            "What is the sum of all integers from 1 to 100?",
            "What is the sum of all integers from 1 to 100?",        # exact repeat
            "What is the sum of all integers from 1 to 50?",          # variation
        ],
    ),
    (
        "Sequence Prediction",
        [
            "Find the next number: 2, 6, 12, 20, 30, ?",
            "Find the next number: 2, 6, 12, 20, 30, ?",             # exact repeat
            "Find the next number: 3, 8, 15, 24, 35, ?",             # variation
        ],
    ),
    (
        "Algorithmic Reasoning",
        [
            "What is the time complexity of binary search on a sorted array of n elements?",
            "What is the time complexity of binary search on a sorted array of n elements?",
            "What is the time complexity of linear search on an unsorted array of n elements?",
        ],
    ),
]


# ── Data Collection ──────────────────────────────────────────────────

def run_query(agent, query, phase_label):
    """Run a single query, collect timing and route info."""
    t0 = time.perf_counter()
    result = agent.solve(query)
    elapsed = time.perf_counter() - t0

    route = result.get("route_decision", result.get("route", "UNKNOWN"))
    answer = result.get("final_answer", "")
    logs = result.get("memory_update_log", [])

    # token count from metrics (last recorded entry)
    tokens = 0
    if agent.metrics.history:
        tokens = agent.metrics.history[-1].get("tokens_used", 0)

    return {
        "query": query,
        "phase": phase_label,
        "route": route,
        "elapsed": round(elapsed, 4),
        "tokens": tokens,
        "answer": answer[:500],
        "logs": logs,
    }


def run_full_demo(agent, skip_slow=False):
    """Execute all phases and return structured results."""
    results = []
    print("\n" + "=" * 64)
    print("  NARE SHOWCASE — Full Cognitive Cycle Demo")
    print("=" * 64)

    # ── Phase 1: Novel queries (SLOW) ────────────────────────────────
    print("\n◆ PHASE 1: Novel Queries (expecting SLOW path)")
    print("  The agent has never seen these. Full LLM reasoning required.\n")
    for domain, queries in SCENARIOS:
        q = queries[0]
        print(f"  [{domain}] Solving: {q[:60]}...")
        r = run_query(agent, q, "1-Novel")
        results.append(r)
        print(f"    → Route: {r['route']}  |  Time: {r['elapsed']:.2f}s  |  Tokens: {r['tokens']}")

    # ── Phase 2: Exact repeats (FAST) ────────────────────────────────
    print("\n◆ PHASE 2: Exact Repeats (expecting FAST path)")
    print("  Same queries again. Should be instant, 0 tokens.\n")
    for domain, queries in SCENARIOS:
        q = queries[1]
        print(f"  [{domain}] Re-asking: {q[:60]}...")
        r = run_query(agent, q, "2-Repeat")
        results.append(r)
        print(f"    → Route: {r['route']}  |  Time: {r['elapsed']:.4f}s  |  Tokens: {r['tokens']}")

    # ── Phase 3: Sleep Consolidation ─────────────────────────────────
    print("\n◆ PHASE 3: Sleep Consolidation")
    print("  Crystallizing reasoning traces into executable Python skills...\n")
    t0 = time.perf_counter()
    agent._sleep_phase()
    agent._rem_sleep_phase()
    sleep_time = time.perf_counter() - t0
    skills_count = len(agent.memory.semantic_rules)
    print(f"  ✓ Sleep complete in {sleep_time:.2f}s — {skills_count} skill(s) in registry")

    # ── Phase 4: Variations (REFLEX or HYBRID) ───────────────────────
    print("\n◆ PHASE 4: Query Variations (expecting REFLEX or HYBRID)")
    print("  Similar but different queries. Skills should fire.\n")
    for domain, queries in SCENARIOS:
        q = queries[2]
        print(f"  [{domain}] Variation: {q[:60]}...")
        r = run_query(agent, q, "4-Variation")
        results.append(r)
        print(f"    → Route: {r['route']}  |  Time: {r['elapsed']:.4f}s  |  Tokens: {r['tokens']}")

    # Collect skill code
    skills = []
    for rule in agent.memory.semantic_rules:
        skills.append({
            "pattern": rule.get("pattern", "Unknown"),
            "confidence": round(rule.get("confidence", 0), 3),
            "maturity": rule.get("maturity", 0),
            "code": rule.get("python_code", "# No code"),
        })

    return results, skills, sleep_time


# ── HTML Report Generator ────────────────────────────────────────────

def generate_html_report(results, skills, sleep_time, output_path):
    """Generate a self-contained, beautiful HTML report."""

    # Aggregate stats
    routes = {}
    total_tokens_novel = 0
    total_time_novel = 0
    total_tokens_cached = 0
    total_time_cached = 0

    for r in results:
        route = r["route"]
        routes[route] = routes.get(route, 0) + 1
        if r["phase"] == "1-Novel":
            total_tokens_novel += r["tokens"]
            total_time_novel += r["elapsed"]
        else:
            total_tokens_cached += r["tokens"]
            total_time_cached += r["elapsed"]

    total_queries = len(results)
    amortized = sum(routes.get(r, 0) for r in ("FAST", "REFLEX", "REFLEX_PROVISIONAL"))
    amort_pct = round(amortized / max(total_queries, 1) * 100, 1)

    # Speedup: average novel time vs average cached time
    n_novel = sum(1 for r in results if r["phase"] == "1-Novel")
    n_cached = sum(1 for r in results if r["phase"] != "1-Novel")
    avg_novel = total_time_novel / max(n_novel, 1)
    avg_cached = total_time_cached / max(n_cached, 1)
    speedup = avg_novel / max(avg_cached, 0.0001)

    token_saving = round((1 - total_tokens_cached / max(total_tokens_novel, 1)) * 100, 1)

    # Build query rows
    query_rows = ""
    for i, r in enumerate(results):
        route_class = r["route"].lower().replace("_", "-")
        query_rows += f"""
        <tr class="route-{route_class}">
          <td>{i+1}</td>
          <td class="phase">{r['phase']}</td>
          <td class="query">{html_mod.escape(r['query'][:80])}</td>
          <td><span class="badge badge-{route_class}">{r['route']}</span></td>
          <td class="num">{r['elapsed']:.4f}s</td>
          <td class="num">{r['tokens']}</td>
          <td class="answer">{html_mod.escape(r['answer'][:120])}...</td>
        </tr>"""

    # Build skill cards
    skill_cards = ""
    for s in skills:
        conf_color = "#10b981" if s["confidence"] >= 0.7 else "#f59e0b" if s["confidence"] >= 0.4 else "#ef4444"
        skill_cards += f"""
        <div class="skill-card">
          <div class="skill-header">
            <span class="skill-pattern">{html_mod.escape(s['pattern'])}</span>
            <span class="skill-conf" style="color:{conf_color}">
              conf: {s['confidence']:.2f} | maturity: {s['maturity']}
            </span>
          </div>
          <pre class="skill-code"><code>{html_mod.escape(s['code'])}</code></pre>
        </div>"""

    if not skills:
        skill_cards = '<p class="no-skills">No skills crystallized yet. Try adding more similar queries to trigger consolidation.</p>'

    # Route distribution SVG bar chart
    route_colors = {
        "SLOW": "#ef4444", "HYBRID": "#f59e0b",
        "FAST": "#3b82f6", "REFLEX": "#10b981", "REFLEX_PROVISIONAL": "#06b6d4",
    }
    max_count = max(routes.values()) if routes else 1
    bars_svg = ""
    bar_y = 30
    for route_name, count in sorted(routes.items(), key=lambda x: -x[1]):
        color = route_colors.get(route_name, "#6b7280")
        bar_w = max(count / max_count * 300, 2)
        pct = round(count / total_queries * 100, 1)
        bars_svg += f"""
          <rect x="120" y="{bar_y}" width="{bar_w}" height="28" rx="4" fill="{color}" opacity="0.85"/>
          <text x="110" y="{bar_y+19}" text-anchor="end" class="bar-label">{route_name}</text>
          <text x="{125+bar_w}" y="{bar_y+19}" class="bar-value">{count} ({pct}%)</text>"""
        bar_y += 42
    svg_height = bar_y + 10

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NARE Showcase Report</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
  :root {{
    --bg: #0f0f13; --bg2: #18181f; --bg3: #1e1e28;
    --border: #2a2a3a; --text: #e4e4ec; --text2: #9999b0;
    --accent: #7c5cfc; --accent2: #a78bfa;
    --green: #10b981; --red: #ef4444; --yellow: #f59e0b; --blue: #3b82f6; --cyan: #06b6d4;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Inter',sans-serif; background:var(--bg); color:var(--text); line-height:1.6; }}

  .container {{ max-width:1100px; margin:0 auto; padding:40px 24px; }}

  /* Hero */
  .hero {{ text-align:center; padding:60px 0 40px; }}
  .hero h1 {{ font-size:2.8rem; font-weight:800;
    background:linear-gradient(135deg, var(--accent), var(--cyan), var(--green));
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    margin-bottom:8px; }}
  .hero .sub {{ color:var(--text2); font-size:1.1rem; font-weight:300; }}
  .hero .timestamp {{ color:var(--text2); font-size:0.85rem; margin-top:12px; opacity:0.6; }}

  /* KPI Cards */
  .kpi-grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(200px,1fr)); gap:16px; margin:32px 0; }}
  .kpi {{ background:var(--bg2); border:1px solid var(--border); border-radius:12px; padding:24px; text-align:center; }}
  .kpi .value {{ font-size:2.2rem; font-weight:700; }}
  .kpi .label {{ font-size:0.85rem; color:var(--text2); margin-top:4px; }}
  .kpi.green .value {{ color:var(--green); }}
  .kpi.blue .value {{ color:var(--blue); }}
  .kpi.yellow .value {{ color:var(--yellow); }}
  .kpi.accent .value {{ color:var(--accent2); }}
  .kpi.cyan .value {{ color:var(--cyan); }}

  /* Section */
  .section {{ margin:48px 0; }}
  .section h2 {{ font-size:1.5rem; font-weight:700; margin-bottom:16px;
    padding-bottom:8px; border-bottom:2px solid var(--border); }}

  /* Chart */
  .chart-box {{ background:var(--bg2); border:1px solid var(--border); border-radius:12px; padding:24px; }}
  .bar-label {{ fill:var(--text); font-family:'Inter'; font-size:13px; font-weight:500; }}
  .bar-value {{ fill:var(--text2); font-family:'Inter'; font-size:12px; }}

  /* Table */
  .table-wrap {{ overflow-x:auto; border-radius:12px; border:1px solid var(--border); }}
  table {{ width:100%; border-collapse:collapse; font-size:0.85rem; }}
  th {{ background:var(--bg3); color:var(--text2); font-weight:600; text-transform:uppercase;
    font-size:0.75rem; letter-spacing:0.5px; padding:12px 14px; text-align:left; }}
  td {{ padding:10px 14px; border-top:1px solid var(--border); }}
  tr:hover {{ background:rgba(124,92,252,0.04); }}
  .num {{ font-family:'JetBrains Mono',monospace; text-align:right; }}
  .query {{ max-width:260px; word-break:break-word; }}
  .answer {{ max-width:200px; color:var(--text2); font-size:0.8rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  .phase {{ font-weight:600; font-size:0.8rem; }}

  /* Badges */
  .badge {{ display:inline-block; padding:3px 10px; border-radius:6px; font-size:0.75rem;
    font-weight:600; font-family:'JetBrains Mono'; letter-spacing:0.3px; }}
  .badge-slow {{ background:rgba(239,68,68,0.15); color:var(--red); }}
  .badge-hybrid {{ background:rgba(245,158,11,0.15); color:var(--yellow); }}
  .badge-fast {{ background:rgba(59,130,246,0.15); color:var(--blue); }}
  .badge-reflex {{ background:rgba(16,185,129,0.15); color:var(--green); }}
  .badge-reflex-provisional {{ background:rgba(6,182,212,0.15); color:var(--cyan); }}

  /* Skills */
  .skill-card {{ background:var(--bg2); border:1px solid var(--border); border-radius:12px; padding:20px; margin-bottom:16px; }}
  .skill-header {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; flex-wrap:wrap; gap:8px; }}
  .skill-pattern {{ font-weight:600; font-size:1rem; }}
  .skill-conf {{ font-family:'JetBrains Mono'; font-size:0.85rem; }}
  .skill-code {{ background:var(--bg); border:1px solid var(--border); border-radius:8px; padding:16px;
    overflow-x:auto; font-size:0.82rem; line-height:1.5; }}
  .skill-code code {{ font-family:'JetBrains Mono',monospace; color:var(--green); }}
  .no-skills {{ color:var(--text2); font-style:italic; padding:20px; }}

  /* How it works */
  .flow {{ display:flex; align-items:center; gap:0; justify-content:center; flex-wrap:wrap; margin:24px 0; }}
  .flow-step {{ background:var(--bg2); border:1px solid var(--border); border-radius:10px; padding:16px 20px;
    text-align:center; min-width:140px; }}
  .flow-step .icon {{ font-size:1.8rem; margin-bottom:4px; }}
  .flow-step .name {{ font-weight:600; font-size:0.9rem; }}
  .flow-step .desc {{ font-size:0.75rem; color:var(--text2); }}
  .flow-arrow {{ font-size:1.5rem; color:var(--accent); padding:0 8px; }}

  /* Footer */
  .footer {{ text-align:center; margin-top:60px; padding:24px; color:var(--text2); font-size:0.8rem; }}
  .footer a {{ color:var(--accent2); text-decoration:none; }}
</style>
</head>
<body>
<div class="container">

  <div class="hero">
    <h1>🧠 NARE Showcase</h1>
    <div class="sub">Compile expensive LLM thoughts into zero-cost Python reflexes</div>
    <div class="timestamp">Generated: {now}</div>
  </div>

  <!-- How It Works -->
  <div class="flow">
    <div class="flow-step"><div class="icon">🐌</div><div class="name">SLOW</div><div class="desc">Full LLM reasoning<br>~700 tokens, ~15s</div></div>
    <div class="flow-arrow">→</div>
    <div class="flow-step"><div class="icon">⚡</div><div class="name">FAST</div><div class="desc">Cached answer<br>0 tokens, ~0.01s</div></div>
    <div class="flow-arrow">→</div>
    <div class="flow-step"><div class="icon">🌙</div><div class="name">SLEEP</div><div class="desc">Crystallize skills<br>into Python code</div></div>
    <div class="flow-arrow">→</div>
    <div class="flow-step"><div class="icon">🧬</div><div class="name">REFLEX</div><div class="desc">Execute Python<br>0 tokens, ~0.001s</div></div>
  </div>

  <!-- KPI Cards -->
  <div class="kpi-grid">
    <div class="kpi green"><div class="value">{amort_pct}%</div><div class="label">Queries Amortized<br>(FAST + REFLEX)</div></div>
    <div class="kpi blue"><div class="value">{speedup:.0f}×</div><div class="label">Avg Speedup<br>(Novel → Cached)</div></div>
    <div class="kpi yellow"><div class="value">{token_saving}%</div><div class="label">Token Savings<br>(Repeat + Variation)</div></div>
    <div class="kpi accent"><div class="value">{len(skills)}</div><div class="label">Skills Crystallized<br>(Executable Python)</div></div>
    <div class="kpi cyan"><div class="value">{total_queries}</div><div class="label">Total Queries<br>Processed</div></div>
  </div>

  <!-- Route Distribution -->
  <div class="section">
    <h2>📊 Route Distribution</h2>
    <div class="chart-box">
      <svg width="100%" viewBox="0 0 520 {svg_height}" xmlns="http://www.w3.org/2000/svg">
        <text x="260" y="18" text-anchor="middle" style="fill:var(--text2);font-family:Inter;font-size:12px">
          How queries were routed across the cognitive hierarchy
        </text>
        {bars_svg}
      </svg>
    </div>
  </div>

  <!-- Full Audit Trail -->
  <div class="section">
    <h2>🔍 Full Audit Trail — Every Query, Every Route, Every Token</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr><th>#</th><th>Phase</th><th>Query</th><th>Route</th><th>Time</th><th>Tokens</th><th>Answer</th></tr>
        </thead>
        <tbody>{query_rows}</tbody>
      </table>
    </div>
  </div>

  <!-- Crystallized Skills -->
  <div class="section">
    <h2>🧬 Crystallized Skills — Real Generated Python Code</h2>
    <p style="color:var(--text2);margin-bottom:16px;font-size:0.9rem;">
      These Python functions were <strong>automatically written by the LLM</strong> during sleep consolidation,
      validated through stress tests, and are now executed <strong>directly</strong> — bypassing the LLM entirely.
    </p>
    {skill_cards}
  </div>

  <div class="footer">
    Built with <strong>NARE</strong> — Neuro-Adaptive Reasoning Engine<br>
    <a href="https://github.com/starface77/Neuro-Adaptive-Reasoning-Engine" target="_blank">
      github.com/starface77/Neuro-Adaptive-Reasoning-Engine
    </a>
  </div>

</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path


# ── CLI Entry Point ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NARE Showcase Demo")
    parser.add_argument("--output", "-o", default="nare_showcase_report.html",
                        help="Output HTML report path")
    parser.add_argument("--skip-slow", action="store_true",
                        help="Skip Phase 1 if episodes already exist")
    args = parser.parse_args()

    print("Initializing NARE agent...")
    agent = NAREProductionAgent()

    # Provide sleep_consolidate shortcut (call both NREM + REM phases)
    def do_sleep(a):
        a._sleep_phase()
        a._rem_sleep_phase()

    results, skills, sleep_time = run_full_demo(agent, skip_slow=args.skip_slow)

    print(f"\n◆ Generating report...")
    path = generate_html_report(results, skills, sleep_time, args.output)
    abs_path = os.path.abspath(path)
    print(f"\n{'=' * 64}")
    print(f"  ✓ Report saved: {abs_path}")
    print(f"  Open it in your browser to see the results!")
    print(f"{'=' * 64}\n")

    # Auto-open on Windows
    if sys.platform == "win32":
        os.startfile(abs_path)


if __name__ == "__main__":
    main()
