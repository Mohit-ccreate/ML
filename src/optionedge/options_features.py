"""Options-chain microstructure features.

From NSE F&O bhavcopy we have per-contract OI, ΔOI, OHLC prices (no IV column),
so we:

1. Aggregate OI / PCR / walls / max-pain / OI-weighted strikes (all at close t).
2. Recover per-contract implied volatility by inverting Black–Scholes on the
   closing option premium (European, r from config, q from index dividend yield),
   then derive IV skew, term structure and portfolio Greeks.

Everything is computed from data available at time t only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Black–Scholes (European) with vectorised IV inversion
# ---------------------------------------------------------------------------

def bs_price(s, k, t, r, q, sigma, cp):
    t = np.maximum(t, 1e-6)
    sigma = np.maximum(sigma, 1e-6)
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    if cp == "CE":
        return s * np.exp(-q * t) * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)
    return k * np.exp(-r * t) * norm.cdf(-d2) - s * np.exp(-q * t) * norm.cdf(-d1)


def bs_greeks(s, k, t, r, q, sigma, cp):
    t = np.maximum(t, 1e-6)
    sigma = np.maximum(sigma, 1e-6)
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    pdf = np.exp(-0.5 * d1 ** 2) / np.sqrt(2 * np.pi)
    if cp == "CE":
        delta = np.exp(-q * t) * norm.cdf(d1)
    else:
        delta = -np.exp(-q * t) * norm.cdf(-d1)
    gamma = np.exp(-q * t) * pdf / (s * sigma * np.sqrt(t))
    vega = s * np.exp(-q * t) * pdf * np.sqrt(t) / 100.0          # per 1 vol pt
    theta = (-(s * np.exp(-q * t) * pdf * sigma) / (2 * np.sqrt(t))
             + q * s * np.exp(-q * t) * (norm.cdf(d1) if cp == "CE" else -norm.cdf(-d1))
             - (-r * k * np.exp(-r * t) * (norm.cdf(d2) if cp == "CE" else -norm.cdf(-d2)))
             ) / 365.0
    if cp != "CE":
        theta = (-(s * np.exp(-q * t) * pdf * sigma) / (2 * np.sqrt(t))
                 - q * s * np.exp(-q * t) * norm.cdf(-d1)
                 + r * k * np.exp(-r * t) * norm.cdf(-d2)) / 365.0
    return delta, gamma, theta, vega


def implied_vol(price, s, k, t, r, q, cp, lo=0.005, hi=3.0):
    """Bisection IV inversion; returns NaN where inversion is unreliable."""
    price = np.atleast_1d(np.asarray(price, dtype=float))
    shape = price.shape
    s = np.broadcast_to(np.asarray(s, dtype=float), shape).copy()
    k = np.broadcast_to(np.asarray(k, dtype=float), shape).copy()
    t = np.broadcast_to(np.asarray(t, dtype=float), shape).copy()
    cp_arr = np.broadcast_to(np.asarray(cp), shape)
    out = np.full(price.shape, np.nan)
    # intrinsic sanity band
    intrinsic = np.where(cp_arr == "CE",
                         s * np.exp(-q * t) - k * np.exp(-r * t),
                         k * np.exp(-r * t) - s * np.exp(-q * t))
    intrinsic = np.maximum(intrinsic, 0.0)
    valid = (price > intrinsic + 0.02) & (price > 0.05) & (t > 1e-4) & np.isfinite(price)
    # cap by no-arb upper bound loosely (undiscounted spot for CE)
    valid &= price < np.maximum(s, k) * 1.05
    idx = np.where(valid)
    if len(idx[0]) == 0:
        return out
    a = np.full(len(idx[0]), lo)
    b = np.full(len(idx[0]), hi)
    p = price[idx]
    Sv, Kv, Tv, cpv = s[idx], k[idx], t[idx], cp_arr[idx]
    for _ in range(40):
        mid = 0.5 * (a + b)
        model = np.empty_like(mid)
        ce_mask = cpv == "CE"
        if ce_mask.any():
            model[ce_mask] = bs_price(Sv[ce_mask], Kv[ce_mask], Tv[ce_mask],
                                      r, q, mid[ce_mask], "CE")
        pe_mask = ~ce_mask
        if pe_mask.any():
            model[pe_mask] = bs_price(Sv[pe_mask], Kv[pe_mask], Tv[pe_mask],
                                      r, q, mid[pe_mask], "PE")
        too_high = model > p
        b = np.where(too_high, mid, b)
        a = np.where(too_high, a, mid)
    out[idx] = 0.5 * (a + b)
    # reject converged-to-bound results (bad fit)
    final = 0.5 * (a + b)
    lo_hits = np.isclose(final, lo, atol=1e-3)
    hi_hits = np.isclose(final, hi, atol=1e-3)
    out[idx] = np.where(lo_hits | hi_hits, np.nan, final)
    return out


def _safe(x, default=np.nan):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _max_pain(strikes, ce_oi, pe_oi):
    """Strike minimizing total option-writer payout at expiry."""
    strikes = np.asarray(strikes, float)
    if strikes.size == 0:
        return np.nan, np.nan
    payout = np.zeros_like(strikes)
    for i, s_eval in enumerate(strikes):
        ce = np.maximum(s_eval - strikes, 0) * ce_oi
        pe = np.maximum(strikes - s_eval, 0) * pe_oi
        payout[i] = ce.sum() + pe.sum()
    j = int(np.argmin(payout))
    return float(strikes[j]), float(payout[j])


# ---------------------------------------------------------------------------
# Per-day chain feature builder
# ---------------------------------------------------------------------------

def chain_features_for_symbol(chain_sym: pd.DataFrame,
                              prices_sym: pd.DataFrame,
                              r: float = 0.065) -> pd.DataFrame:
    """chain_sym: F&O rows for one underlying. prices_sym: index OHLCV for symbol."""
    rows = []
    px = prices_sym.set_index("date")
    # group once (O(n)) instead of scanning the full chain per date (O(n²))
    day_groups = {d: g for d, g in chain_sym.groupby("date", sort=True)}
    for d, day in day_groups.items():
        if d not in px.index:
            continue
        spot = float(px.loc[d, "close"])
        q = (_safe(px.loc[d, "div_yield"], 1.0) or 1.0) / 100.0
        if not np.isfinite(spot) or spot <= 0:
            continue
        feats: dict = {"date": d}

        opts = day[day["instrument"] == "IDX_OPT"]
        futs = day[day["instrument"] == "IDX_FUT"]

        # ---- futures basis / futs OI ---------------------------------------
        if len(futs):
            futs = futs.sort_values("expiry")
            nf = futs.iloc[0]
            feats["fut_basis"] = float(nf["close"] - spot)
            feats["fut_basis_ann"] = feats["fut_basis"] / spot * 365.0 / max(
                (nf["expiry"] - d).days, 1)
            feats["fut_oi"] = _safe(futs["oi"].sum(), 0.0)
            feats["fut_oi_chg"] = _safe(futs["chg_oi"].sum(), 0.0)
            feats["fut_oi_total_base"] = feats["fut_oi"]
        else:
            feats.update(fut_basis=np.nan, fut_basis_ann=np.nan,
                         fut_oi=np.nan, fut_oi_chg=np.nan, fut_oi_total_base=np.nan)

        if opts.empty:
            rows.append({"date": d, **feats})
            continue

        # ---- aggregate OI across ALL expiries ------------------------------
        ce_all = opts[opts["option_type"] == "CE"]
        pe_all = opts[opts["option_type"] == "PE"]
        ce_oi = _safe(ce_all["oi"].sum(), 0.0)
        pe_oi = _safe(pe_all["oi"].sum(), 0.0)
        feats["ce_oi"] = ce_oi
        feats["pe_oi"] = pe_oi
        feats["pcr_oi"] = pe_oi / ce_oi if ce_oi > 0 else np.nan
        feats["pcr_oi_log"] = np.log1p(pe_oi) - np.log1p(ce_oi)
        ce_vol = _safe(ce_all["contracts"].sum(), np.nan)
        pe_vol = _safe(pe_all["contracts"].sum(), np.nan)
        feats["pcr_vol"] = pe_vol / ce_vol if (ce_vol or 0) > 0 else np.nan
        feats["ce_oi_chg"] = _safe(ce_all["chg_oi"].sum(), 0.0)
        feats["pe_oi_chg"] = _safe(pe_all["chg_oi"].sum(), 0.0)
        feats["net_oi_chg"] = feats["ce_oi_chg"] + feats["pe_oi_chg"]
        feats["pe_oi_chg_minus_ce"] = feats["pe_oi_chg"] - feats["ce_oi_chg"]

        # nearest expiry ≥ today (weekly/monthly)
        expiries = sorted(opts["expiry"].dropna().unique())
        fut_exp = [e for e in expiries if e >= d]
        nearest = fut_exp[0] if fut_exp else (expiries[-1] if expiries else None)
        feats["days_to_expiry"] = float((nearest - d).days) if nearest is not None else np.nan
        nx = opts[opts["expiry"] == nearest] if nearest is not None else opts.iloc[0:0]

        # expiring-this-week share of total OI (roll risk)
        feats["near_oi_share"] = (nx["oi"].sum() / opts["oi"].sum()
                                  if opts["oi"].sum() > 0 else np.nan)

        # ---- OI profile on nearest expiry ----------------------------------
        if len(nx):
            step = 50.0 if spot < 40000 else 100.0
            atm = round(spot / step) * step
            feats["atm_strike"] = atm
            nx_ce = nx[nx["option_type"] == "CE"]
            nx_pe = nx[nx["option_type"] == "PE"]

            def _wall(df):
                if df.empty:
                    return np.nan, np.nan
                g = df.groupby("strike")["oi"].sum()
                k = float(g.idxmax())
                return k, float(g.max())
            k_ce, oi_ce_wall = _wall(nx_ce)
            k_pe, oi_pe_wall = _wall(nx_pe)
            feats["ce_wall_strike"] = k_ce
            feats["pe_wall_strike"] = k_pe
            feats["ce_wall_dist"] = (k_ce - spot) / spot if np.isfinite(k_ce) else np.nan
            feats["pe_wall_dist"] = (k_pe - spot) / spot if np.isfinite(k_pe) else np.nan
            feats["ce_wall_oi"] = oi_ce_wall
            feats["pe_wall_oi"] = oi_pe_wall
            feats["wall_balance"] = ((oi_pe_wall - oi_ce_wall) /
                                     (oi_pe_wall + oi_ce_wall)
                                     if np.isfinite(oi_pe_wall) and (oi_pe_wall + oi_ce_wall) > 0
                                     else np.nan)

            # OI-weighted mean strikes (positioning)
            if nx_ce["oi"].sum() > 0:
                feats["ce_oi_wavg_strike"] = float(
                    (nx_ce["strike"] * nx_ce["oi"]).sum() / nx_ce["oi"].sum())
            else:
                feats["ce_oi_wavg_strike"] = np.nan
            if nx_pe["oi"].sum() > 0:
                feats["pe_oi_wavg_strike"] = float(
                    (nx_pe["strike"] * nx_pe["oi"]).sum() / nx_pe["oi"].sum())
            else:
                feats["pe_oi_wavg_strike"] = np.nan
            feats["ce_oi_wavg_dist"] = (feats.get("ce_oi_wavg_strike", np.nan) - spot) / spot
            feats["pe_oi_wavg_dist"] = (feats.get("pe_oi_wavg_strike", np.nan) - spot) / spot

            # strikes above vs below spot (call wall above / put wall below typical)
            ce_above = nx_ce.loc[nx_ce["strike"] > spot, "oi"].sum()
            ce_below = nx_ce.loc[nx_ce["strike"] <= spot, "oi"].sum()
            pe_above = nx_pe.loc[nx_pe["strike"] > spot, "oi"].sum()
            pe_below = nx_pe.loc[nx_pe["strike"] <= spot, "oi"].sum()
            feats["ce_oi_call_zone"] = ce_above / nx_ce["oi"].sum() if nx_ce["oi"].sum() else np.nan
            feats["pe_oi_put_zone"] = pe_below / nx_pe["oi"].sum() if nx_pe["oi"].sum() else np.nan

            uniq = np.sort(nx["strike"].unique())
            ce_by = nx_ce.groupby("strike")["oi"].sum().reindex(uniq).fillna(0).to_numpy()
            pe_by = nx_pe.groupby("strike")["oi"].sum().reindex(uniq).fillna(0).to_numpy()
            mp, mp_pay = _max_pain(uniq, ce_by, pe_by)
            feats["max_pain"] = mp
            feats["max_pain_dist"] = (mp - spot) / spot if np.isfinite(mp) else np.nan

            # ---- IV inversion on liquid near-the-money contracts ------------
            T = max((nearest - d).days, 0) / 365.0
            liquid = nx[(nx["close"] >= 0.05) & (nx["strike"] > 0)].copy()
            # focus ±4% moneyness for stability
            liquid = liquid[(liquid["strike"] / spot >= 0.96) &
                            (liquid["strike"] / spot <= 1.04)]
            if len(liquid) >= 4:
                ivs = implied_vol(liquid["close"].to_numpy(), spot,
                                  liquid["strike"].to_numpy(), T, r, q,
                                  liquid["option_type"].to_numpy())
                liquid = liquid.assign(iv=ivs)
                liq = liquid.dropna(subset=["iv"])
                liq = liq[(liq["iv"] > 0.02) & (liq["iv"] < 2.0)]
                if len(liq) >= 4:
                    atm_ix = (liq["strike"] - spot).abs().idxmin()
                    feats["iv_atm"] = float(liq.loc[atm_ix, "iv"])
                    feats["iv_mean"] = float(liq["iv"].mean())
                    feats["iv_std"] = float(liq["iv"].std())

                    # skew: avg IV of OTM puts (K<S) vs OTM calls (K>S) ~ 25Δ proxies
                    otm_p = liq[(liq["option_type"] == "PE") & (liq["strike"] < spot)]
                    otm_c = liq[(liq["option_type"] == "CE") & (liq["strike"] > spot)]
                    iv_put = float(otm_p.sort_values("strike").iloc[-1]["iv"]) if len(otm_p) else np.nan
                    # deepest still-liquid OTM put ~ closest below spot from put side
                    if len(otm_p):
                        iv_put = float(otm_p.loc[(otm_p["strike"] - spot).abs().idxmin(), "iv"])
                    iv_call = float(otm_c.loc[(otm_c["strike"] - spot).abs().idxmin(), "iv"]) if len(otm_c) else np.nan
                    feats["iv_otm_put"] = iv_put
                    feats["iv_otm_call"] = iv_call
                    feats["iv_skew"] = iv_put - iv_call if np.isfinite(iv_put) and np.isfinite(iv_call) else np.nan

                    # regression slope of IV vs log-moneyness (all liquid)
                    x = np.log(liq["strike"].to_numpy() / spot)
                    y = liq["iv"].to_numpy()
                    if np.std(x) > 1e-6:
                        slope = np.polyfit(x, y, 1)[0]
                        feats["iv_slope"] = float(slope)
                    else:
                        feats["iv_slope"] = np.nan

                    # term structure: IV ATM at nearest vs next expiry
                    if len(fut_exp) >= 2:
                        n2 = fut_exp[1]
                        nx2 = opts[opts["expiry"] == n2]
                        liq2 = nx2[(nx2["close"] >= 0.05) &
                                   (nx2["strike"] / spot >= 0.97) &
                                   (nx2["strike"] / spot <= 1.03)]
                        if len(liq2) >= 3:
                            T2 = max((n2 - d).days, 0) / 365.0
                            iv2 = implied_vol(liq2["close"].to_numpy(), spot,
                                              liq2["strike"].to_numpy(), T2, r, q,
                                              liq2["option_type"].to_numpy())
                            iv2 = iv2[np.isfinite(iv2) & (iv2 > 0.02) & (iv2 < 2.0)]
                            feats["iv_next_expiry"] = float(np.nanmean(iv2)) if len(iv2) else np.nan
                        else:
                            feats["iv_next_expiry"] = np.nan
                        if np.isfinite(feats.get("iv_atm", np.nan)) and np.isfinite(feats.get("iv_next_expiry", np.nan)):
                            feats["iv_term_slope"] = feats["iv_next_expiry"] - feats["iv_atm"]
                        else:
                            feats["iv_term_slope"] = np.nan

                    # Greeks aggregated with per-option IV
                    liq2g = liquid.dropna(subset=["iv"])
                    liq2g = liq2g[(liq2g["iv"] > 0.02) & (liq2g["iv"] < 2.0)]
                    if len(liq2g) >= 4:
                        deltas, thetas, gammas, vegas = [], [], [], []
                        for _, rr in liq2g.iterrows():
                            dd, gg, tt, vv = bs_greeks(spot, rr["strike"], T, r, q,
                                                       rr["iv"], rr["option_type"])
                            deltas.append(dd); gammas.append(gg)
                            thetas.append(tt); vegas.append(vv)
                        oi_w = liq2g["oi"].to_numpy(dtype=float)
                        oi_w = np.where(np.isfinite(oi_w), oi_w, 0.0)
                        wsum = oi_w.sum() or 1.0
                        deltas = np.asarray(deltas)
                        # net portfolio delta (long CE = +, long PE = −) weighted by OI
                        signs = np.where(liq2g["option_type"].to_numpy() == "CE", 1.0, -1.0)
                        feats["oi_w_delta"] = float((deltas * oi_w).sum() / wsum)
                        feats["net_delta_pos"] = float((deltas * signs * oi_w).sum() / wsum)
                        feats["oi_w_theta"] = float((np.asarray(thetas) * oi_w).sum() / wsum)
                        feats["oi_w_gamma"] = float((np.asarray(gammas) * oi_w).sum() / wsum)
                        feats["oi_w_vega"] = float((np.asarray(vegas) * oi_w).sum() / wsum)
                    else:
                        feats.update(iv_atm=feats.get("iv_atm", np.nan),
                                     iv_mean=np.nan, iv_std=np.nan, iv_skew=np.nan,
                                     iv_slope=np.nan, iv_otm_put=np.nan, iv_otm_call=np.nan,
                                     iv_next_expiry=np.nan, iv_term_slope=np.nan,
                                     oi_w_delta=np.nan, net_delta_pos=np.nan,
                                     oi_w_theta=np.nan, oi_w_gamma=np.nan, oi_w_vega=np.nan)
                else:
                    feats.update(iv_atm=np.nan, iv_mean=np.nan, iv_std=np.nan,
                                 iv_skew=np.nan, iv_slope=np.nan, iv_otm_put=np.nan,
                                 iv_otm_call=np.nan, iv_next_expiry=np.nan,
                                 iv_term_slope=np.nan, oi_w_delta=np.nan,
                                 net_delta_pos=np.nan, oi_w_theta=np.nan,
                                 oi_w_gamma=np.nan, oi_w_vega=np.nan)
            else:
                feats.update(iv_atm=np.nan, iv_mean=np.nan, iv_std=np.nan,
                             iv_skew=np.nan, iv_slope=np.nan, iv_otm_put=np.nan,
                             iv_otm_call=np.nan, iv_next_expiry=np.nan,
                             iv_term_slope=np.nan, oi_w_delta=np.nan,
                             net_delta_pos=np.nan, oi_w_theta=np.nan,
                             oi_w_gamma=np.nan, oi_w_vega=np.nan)
        rows.append({"date": d, **feats})

    feats_df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    # ---- temporal changes / rolling normalisations (still causal) ---------
    def dc(col, n=1):
        return feats_df[col].diff(n)
    for col in ("pcr_oi", "ce_oi", "pe_oi", "iv_atm", "iv_skew", "iv_slope",
                "max_pain_dist", "net_delta_pos", "fut_basis", "near_oi_share"):
        if col in feats_df.columns:
            feats_df[f"d1_{col}"] = dc(col, 1)
            feats_df[f"d5_{col}"] = dc(col, 5)
    # IV rank over trailing year
    if "iv_atm" in feats_df.columns:
        roll_min = feats_df["iv_atm"].rolling(252, min_periods=60).min()
        roll_max = feats_df["iv_atm"].rolling(252, min_periods=60).max()
        rng = (roll_max - roll_min).replace(0, np.nan)
        feats_df["iv_rank"] = (feats_df["iv_atm"] - roll_min) / rng
        feats_df["pcr_oi_z_63"] = (feats_df["pcr_oi"] - feats_df["pcr_oi"].rolling(63, min_periods=20).mean()) \
            / feats_df["pcr_oi"].rolling(63, min_periods=20).std().replace(0, np.nan)
    return feats_df


def build_options_features(chain: pd.DataFrame, prices: pd.DataFrame,
                           r: float = 0.065) -> pd.DataFrame:
    frames = []
    for sym, g in chain.groupby("symbol"):
        ps = prices[prices["symbol"] == sym]
        f = chain_features_for_symbol(g, ps, r=r)
        f["symbol"] = sym
        frames.append(f)
    return pd.concat(frames, ignore_index=True)
