"""Cohen's kappa per v2 field between two labelings of the same people.

Input: per-person CSVs from parse_v2.py.
  - Double labeling:  kappa_v2.py people.csv --pass2           (pass 1 vs pass 2 tasks of the same global_id)
  - Two annotators:   kappa_v2.py A.csv --other B.csv          (e.g. Claude/Codex vs Luciano's human check)
People are matched within an image by box IoU >= 0.5 (Hungarian-free greedy, best IoU first).
'unannotated' on either side -> the pair is skipped for that field (reported as n_missing), never counted as a value.
Fields with kappa < 0.6 should not be used as training targets (C_relation_v2.yaml).
"""
import argparse, csv, collections

FIELDS = ["support_surface", "on_scaffold", "feet_on", "feet_visible", "proximity",
          "helmet_state", "harness_state", "anchor_visible"]


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0, min(ax2, bx2) - max(a[0], b[0])), max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih; u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0


def box(r): return tuple(float(r[k]) for k in ("x", "y", "w", "h"))


def match(A, B):
    pairs = sorted(((iou(box(a), box(b)), i, j) for i, a in enumerate(A) for j, b in enumerate(B)), reverse=True)
    ui, uj, out = set(), set(), []
    for s, i, j in pairs:
        if s < 0.5: break
        if i in ui or j in uj: continue
        ui.add(i); uj.add(j); out.append((A[i], B[j]))
    return out, len(A) - len(ui), len(B) - len(uj)


def kappa(pairs):
    n = len(pairs)
    if n == 0: return None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = collections.Counter(a for a, _ in pairs), collections.Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("--other"); ap.add_argument("--pass2", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv)))
    if a.pass2:
        A, B = collections.defaultdict(list), collections.defaultdict(list)
        for r in rows: (A if str(r["pass"]) == "1" else B)[r["global_id"]].append(r)
    else:
        A, B = collections.defaultdict(list), collections.defaultdict(list)
        for r in rows: A[r["global_id"]].append(r)
        for r in csv.DictReader(open(a.other)): B[r["global_id"]].append(r)
    matched, ua, ub = [], 0, 0
    for g in set(A) & set(B):
        m, x, y = match(A[g], B[g]); matched += m; ua += x; ub += y
    out = []
    for f in FIELDS:
        pairs = [(p[f], q[f]) for p, q in matched if p.get(f, "unannotated") != "unannotated" and q.get(f, "unannotated") != "unannotated"]
        out.append(dict(field=f, n=len(pairs), n_missing=len(matched) - len(pairs),
                        agreement=round(sum(x == y for x, y in pairs) / len(pairs), 3) if pairs else None,
                        kappa=kappa(pairs), usable_as_target=(kappa(pairs) or 0) >= 0.6))
    print(f"images both sides: {len(set(A) & set(B))} | matched people: {len(matched)} | unmatched A/B: {ua}/{ub}")
    for r in out: print(r)
    if a.out:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)


if __name__ == "__main__":
    main()
