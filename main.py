"""GTA VI trophy earn-rate forecaster v1 scaffold."""
import json, math, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.64485, -0.67449, 0.0, 0.67449, 1.64485]

def clamp(x): return min(99.99, max(0.01, float(x)))
def logit(p):
    p = clamp(p)/100; return math.log(p/(1-p))
def inv_logit(z): return 100/(1+math.exp(-z))

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        last = case["history"][-1]
        hd = case["horizon_minutes"]/1440
        rows = []
        for i, tr in enumerate(case["trophies"]):
            r = last["rates"][i] or 0.0
            sd = 0.5 * math.sqrt(hd)
            q = []
            prev = 0.0
            for z in Z:
                v = round(min(100, max(0, inv_logit(logit(r)+z*sd))), 4)
                v = max(v, prev); prev = v
                q.append(int(v) if abs(v-round(v))<1e-12 else v)
            rows.append({"id": tr["id"], "q": q})
        (OUTPUT_DIR/f"{path.stem}.out").write_text(json.dumps({"trophies": rows}, separators=(",",":"))+"\n", encoding="utf-8")
if __name__ == "__main__":
    main()
