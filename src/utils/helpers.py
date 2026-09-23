import pandas as pd

def max_pain(strikes, ce_oi, pe_oi):
    """Max pain calculation"""
    pains = []
    for s in strikes:
        pain = sum(abs(s - k) * oi for k, oi in zip(strikes, ce_oi) if k > s) + \
               sum(abs(s - k) * oi for k, oi in zip(strikes, pe_oi) if k < s)
        pains.append(pain)
    idx = min(range(len(pains)), key=lambda i: pains[i])
    return strikes[idx], pains[idx]

def pcr(ce_oi_total, pe_oi_total):
    return pe_oi_total / ce_oi_total if ce_oi_total else 0
