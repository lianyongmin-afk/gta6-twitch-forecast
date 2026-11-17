"""Local scorer for gta6-twitch-ccu-forecast-01 practice cases.
Replicates RULES.txt Scoring: mean pinball loss on log scale.
Answers from SullyGnome (verified: hour-1 matches each case's `known` exactly).
Usage: python3 score.py <output_dir>  (reads <id>.out for p00..p03, prints per-case + overall loss)
"""
import json, math, sys
from pathlib import Path

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
ANSWERS = json.load(open(Path(__file__).parent / 'practice_answers.json'))

def pinball(y, q, tau):
    y = max(1.0, float(y))
    q = float(q)
    d = math.log(y) - math.log(q)
    return 2.0 * max(tau * d, (tau - 1.0) * d)

def score_case(out_path, pid):
    out = json.loads(Path(out_path).read_text())
    rows = {r['hour']: r['q'] for r in out['hours']}
    ans = ANSWERS[pid]
    losses = []
    for h in range(2, 25):
        y = ans[str(h)]
        if y is None:
            continue
        for tau, q in zip(LEVELS, rows[h]):
            losses.append(pinball(y, q, tau))
    return sum(losses) / len(losses), len(losses)

def main():
    outdir = Path(sys.argv[1])
    total, n = 0.0, 0
    for pid in ['p00', 'p01', 'p02', 'p03']:
        loss, cnt = score_case(outdir / f'{pid}.out', pid)
        print(f'{pid}: loss={loss:.6f} (items={cnt})')
        total += loss * cnt
        n += cnt
    print(f'OVERALL: loss={total/n:.6f} (items={n})')

if __name__ == '__main__':
    main()
