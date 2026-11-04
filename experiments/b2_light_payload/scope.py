"""User-authorized nominal payload scope: only0kg and5kg."""
LOADS=(0.,5.)
def basic_cases():return [(kg,1.,1.,False) for kg in LOADS]
def full_cases():
    basic=basic_cases()
    return basic+[(kg,fr,m,asym) for kg in LOADS for fr in [.4,.7,1.,1.3] for m in [.9,1.,1.1] for asym in [False,True] if (kg,fr,m,asym) not in basic]
