import inspect
import numpy as np
from river import datasets
from river.datasets import synth

def labels(ds, n):
    out = []
    for i, (x, y) in enumerate(ds):
        if i >= n:
            break
        out.append(y)
    return np.array(out)

streams = {
    "Agrawal_synth": (synth.Agrawal(seed=42), 20000),
    "ConceptDrift_Agrawal": (synth.ConceptDriftStream(
        stream=synth.Agrawal(seed=1), drift_stream=synth.Agrawal(seed=2),
        seed=42, position=2500, width=1000), 20000),
    "Elec2": (datasets.Elec2(), 10000),
    "Sine": (synth.Sine(seed=42), 40000),
    "Bananas": (datasets.Bananas(), 5000),
    "CreditCard": (datasets.CreditCard(), 5000),
    "Phishing": (datasets.Phishing(), 1250),
    "RandomRBFDrift": (synth.RandomRBFDrift(), 20000),
}
print(f"{'stream':22s} {'n':>6s} {'majority':>9s} {'no-change':>10s}")
for name, (ds, n) in streams.items():
    y = labels(ds, n)
    _, counts = np.unique(y, return_counts=True)
    print(f"{name:22s} {len(y):6d} {counts.max()/len(y):9.3f} {np.mean(y[1:] == y[:-1]):10.3f}")

print()
print(inspect.signature(synth.Agrawal.__init__))