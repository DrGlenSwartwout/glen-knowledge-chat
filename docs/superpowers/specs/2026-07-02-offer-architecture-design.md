# Remedy Match — Offer Architecture (Healing-First)

**Date:** 2026-07-02 · **Status:** design for review · **Author:** Glen + Claude
**Supersedes the pricing assumptions in:** PR #487 (prepay ladder + $1 deposit) — see "Re-scope" below.

---

## 1. The thesis (the whole model in three words)

**Service = acquisition. Product = engine. Support = retention.**

- **Service acquires.** The Biofield Analysis (scan → matched protocol → report) is the low-cost front door. Its job is reach and a *personalized reason to buy the right remedies* — not profit.
- **Product is the engine.** Revenue is the *flow* of matched remedies the client buys each cycle. The basket **rotates** as they heal (most clients change most remedies as they progress) — so it is a moving basket fed by re-matching, **not** a static autoship of one bottle.
- **Support retains.** Continuous care — periodic re-scan + re-match + live group coaching + AI ally + Terrain Restore — is the *fuel line* that keeps the protocol evolving and the product engine turning. When support lapses, the basket goes stale and the engine stalls.

**Healing-first corollary — LTV tapers by design.** Success means the client needs us *less* and eventually graduates. We do **not** build for retention-milking. Growth comes from **reach + referral**: healed clients become advocates. This is why the affiliate/referral layer exists — it is the flywheel that refills a practice that "loses" clients to their own recovery.

*Lifecycle:* acquire cheap → evolving-protocol product flow is the revenue → support keeps it flowing → they heal & refer → repeat with more people.

---

## 2. What the client sees — one path, simple enough to choose at a glance

The prospect (often exhausted, in real need) sees a **path**, not a pricing matrix.

**Step 1 — See what's really going on.**
Your Biofield Analysis: a scan, a matched protocol, and a plain-language report.
→ **$1 unlocks your analysis** (a credited deposit, not a trial that turns into a bill — Model #2). Applies to your first program.

**Step 2 — Get your remedies.**
The exact remedies matched to your protocol, at honest pricing: **you save when you stock up**, because bigger orders cost us less to pack and we pass that on.

**Step 3 — Choose how you want to heal:**

| | **On your own** | **With me — Continuous Care** *(recommended)* |
|---|---|---|
| | Take your remedies. Come back for a fresh analysis whenever you want. | I re-scan & re-match your protocol as you progress + live weekly group coaching + your AI ally + Terrain Restore. |
| Price | pay per analysis + remedies | **~$99/mo** (annual: 2 months free) |
| Best for | self-directed / maintenance | anyone who wants to get well fastest — your protocol keeps up with your healing |

Underneath, invisible unless they want it: **free membership** (portal, see your analysis, referral tracking) and the **advocate layer** (share and earn referral points) for those moved to refer others.

That is the entire visible choice: **get seen → get your remedies → solo or guided.** Two doors at the decision point.

---

## 3. Pricing mechanics (internal)

> **Design principle — every lever is console-tunable, not hardcoded.** The quantity-discount curve, prepay depth, Continuous Care price, and points/referral reward rates all live as **console-editable settings** (extend the existing pricing-settings console + points engine), shipped with sensible defaults. Glen/Rae tune them live as real margins run. No pricing number below is a constant in code — each is a default in a settings row.


**The program (acquisition):** scalable group-supported version **~$100** (standardized design + group coaching, includes a 30-day support window post-delivery). Premium 1:1 hand-designed version remains **~$300** for those who want it. $1 deposit unlocks the analysis and credits toward the program.

**Products (engine) — cost-based quantity curve, open to everyone:**
- Keyed on **total bottles in the order** (mixed SKUs count together), because packing efficiency is per *shipment* and clients order a *basket* of different matched remedies. A 5-remedy protocol = 5 bottles = the qty-5 bracket on every line.
- **Cost basis (shape) + volume floor:** the curve *shape* is cost-driven — ordering the protocol together amortizes the fixed per-order cost (box, label, base postage, pick/pack, handling ≈ $9–10/bottle), so the biggest drop is 1→2 and it flattens by a case. The **$50 floor at 12+ is Glen's volume price** (blends packing savings + volume loyalty), not strict cost pass-through.
- **Default curve (console-tunable — base price + floor + each bracket). Base $70, floor $50:**

  | Total bottles | Discount off single price | at $70 base |
  |---|---|---|
  | 1 | 0% (protects the common order) | $70 |
  | 2–3 | 14% | $60 |
  | 4–6 | 21% | $55 |
  | 7–11 | 26% | $52 |
  | **12+** | **~29% (flat floor)** | **$50** |

  Steep early, flat by a case. The $40/mo single-large-bottle (360-cap) remains a **separate format SKU**, not part of this curve.
- **Open to all**, not member-gated. Members earn value elsewhere (support, credit, Terrain Restore, referral points), not a hidden cheaper price. *(Reverses today's `_is_paid_member` volume-gate — see Re-scope.)*

**Continuous Care (retention/recurring):** **~$99/mo**, delivering the periodic re-match + group support + ally + Terrain Restore. **Prepay:** graduated, capped at **2 months free (~17%) annually** (6-mo ≈ 8%); the earlier 50% is dropped (trains discount-waiting; deep-discount cohorts churn ~2×). Products billed separately per current protocol.

**Maintenance tail:** once stable (e.g., wet-AMD on AngiogenX 3/day ≈ a 4-month 360-cap bottle), graduate to low-touch product auto-refill — cheap to serve, sticky.

**Affiliate / advocate layer:** 2-tier referral **tracking available to everyone**; rewards **sales-tied and non-cashable** (FTC discipline, mirrors Pay-It-Forward). Kept in the background of the healing UX; front-and-center only for those who choose to advocate.

---

## 4. The scalability assumption (the one number that gates the model)

Continuous Care only scales if the **monthly re-match is delivered by system, not by hand**: E4L scan → AI-assisted match → standardized protocol adjustment → group support, with Glen **spot-checking exceptions** rather than hand-designing each protocol.

**RESOLVED (2026-07-02, Glen):** AI Match is performing well and is **virtually hands-free today** — Glen continues to spot-check and train it, but it does not gate throughput. Therefore **Continuous Care scales at ~$99/mo** and is the true recurring engine (not a capacity-capped concierge tier). The model is confirmed scalable; growth is bounded by reach + referral, not by Glen's time.

---

## 5. Re-scope of in-flight work (PR #487)

- **Survives:** the **$1 deposit → unlock analysis, no silent auto-charge** (PR2a) is the acquisition front door — keep as built.
- **Changes:**
  - The **prepay ladder** (PR1) is repurposed: "1–12 months of membership" → **Continuous Care prepay** (annual = 2 months free, not 50%). The tier/term machinery is reusable; the numbers and framing change.
  - **Product volume pricing** moves from a member-gated "1–12" to a **cost-based per-order quantity curve open to all** → revisit the `_is_paid_member` volume-gate.
  - "Membership" as a standalone $0/$9/$99 tier stack is **replaced** by the two-door choice (Free vs Continuous Care). *Recommend cutting the standalone $9 tier for simplicity — folded into Free (tracking) + Continuous Care.*

---

## 6. Open decisions (to lock before implementation)

1. ~~**Automation crux (§4)**~~ — **RESOLVED 2026-07-02:** AI Match is virtually hands-free; Continuous Care scales at ~$99/mo.
2. ~~**$9 "Active" tier**~~ — **RESOLVED: cut.** Two visible doors only: Free (portal + referral tracking) and Continuous Care ~$99/mo.
3. ~~**Product quantity curve**~~ — **RESOLVED: open to everyone** (cost-based, honest; membership earns value elsewhere). Reverses the current `_is_paid_member` volume-gate.
4. ~~**Program price**~~ — **RESOLVED: $100 scalable headline + $300 premium 1:1 option.**
5. ~~**Real product cost inputs**~~ — **RESOLVED by design:** ship sensible defaults; all discount/points levers are **console-tunable** (§3), so Glen/Rae dial them in live as margins run. No blocking input remains.

**→ All design decisions resolved. Ready for implementation planning.**

---

## 7. Success criteria

- A prospect can understand and choose in **one screen** (get seen → remedies → solo/guided).
- **Program→Continuous-Care (or →repeat-product) conversion rate** is instrumented — it is the number the whole model lives or dies on.
- No client is ever shown a comp plan; advocates opt in.
- Pricing is defensible as cost-based / value-based, never arbitrary.
