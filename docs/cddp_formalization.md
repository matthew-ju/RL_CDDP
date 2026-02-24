# Contextual Deontic Decision Process (CDDP): Formal Framework

---

## 1. Preliminaries

### 1.1 Constrained Markov Decision Process (CMDP)

A CMDP is a tuple **(S, A, T, R, C, d)** where:
- **S** is a finite state space
- **A** is a finite action space
- **T**: S × A × S → [0, 1] is the transition kernel, T(s'|s, a)
- **R**: S × A → ℝ is the reward function, bounded |R(s, a)| ≤ R_max
- **C**: S × A → {0, 1} is the constraint indicator (1 = violation)
- **d** ≥ 0 is the expected cost budget

The CMDP objective is:

```
max_π  J_R(π) = E_π [ Σ_{t=0}^{H-1} γ^t R(s_t, a_t) ]
s.t.   J_C(π) = E_π [ Σ_{t=0}^{H-1} γ^t C(s_t, a_t) ] ≤ d
```

The Lagrangian relaxation is L(π, λ) = J_R(π) − λ(J_C(π) − d), λ ≥ 0, solved via primal-dual updates.

**Key limitation:** Lagrangian methods satisfy constraints *in expectation*, not pointwise. The converged policy may violate C(s,a) = 1 at individual steps.

### 1.2 Distributional Reinforcement Learning

In distributional RL, we learn the full return distribution Z^π(s, a), not just Q^π(s, a) = E[Z^π(s, a)].

**Quantile Regression (QR-DQN):** Approximate Z^π(s, a) by N quantile values {θ_i(s, a)}_{i=1}^{N} at quantile fractions τ_i = (2i−1)/2N, trained via:

```
L_QR(θ) = Σ_{i=1}^{N} E[ ρ_{τ_i}(r + γ θ_j(s', a') − θ_i(s, a)) ]
where  ρ_τ(u) = u · (τ − 𝟙{u < 0})
```

**CVaR at level α:** CVaR_α(s, a) = (1/⌈αN⌉) Σ_{i=1}^{⌈αN⌉} θ_i(s, a), the expected return in the worst α-fraction of outcomes.

### 1.3 Safety Shields

A **safety shield** (Alshiekh et al., 2018) is a runtime enforcement mechanism that restricts the agent's action space to guarantee constraint satisfaction regardless of the learned policy.

Given a constraint predicate C: S × A → {0, 1}, the **safe action set** at state s is:

```
A_safe(s) = { a ∈ A : C(s, a) = 0 }
```

The **shielded policy** is:

```
π_shield(a | s) = π_θ(a | s) · 𝟙{a ∈ A_safe(s)} / Z(s)
where  Z(s) = Σ_{a ∈ A_safe(s)} π_θ(a | s)
```

**Assumption (Shield Feasibility):** A_safe(s) ≠ ∅ for all reachable s. There always exists at least one non-violating action. (In the gridworld: the agent can always stay in place or move away from civilians.)

**Assumption (Full Support — FS).** The policy π_θ satisfies π_θ(a | s) > 0 for all a ∈ A and all s ∈ S. This is standard in policy gradient methods (e.g., softmax parameterization with finite temperature) and ensures that the shield normalization constant Z(s) = Σ_{a ∈ A_safe(s)} π_θ(a | s) > 0 whenever A_safe(s) ≠ ∅.

---

## 2. The Contextual Deontic Decision Process

### 2.1 Definition

> **Definition 1 (CDDP).** A *Contextual Deontic Decision Process* is a tuple
>
> **(S, A, T, R, C, S(·), τ, h, d)**
>
> where (S, A, T, R, C, d) is a CMDP, and additionally:
>
> - **S: S → [0, 1]** is the *stakes functional*, mapping each state to a scalar measuring situation severity.
> - **τ ∈ (0, 1)** is the *stakes threshold*.
> - **h: [0, 1] × (0, 1) → [0, 1]** is the *constraint modulation function*.

### 2.2 Constraint Modulation

> **Definition 2a (Hard Switching).**
> ```
> h_hard(S(s), τ) = 𝟙{S(s) < τ}
> ```

> **Definition 2b (Soft Switching).**
> ```
> h_soft(S(s), τ) = 1 − σ_β(S(s) − τ),   σ_β(x) = 1/(1 + exp(−βx))
> ```

Note: h_hard = lim_{β→∞} h_soft. We use hard switching throughout; soft switching is reserved for future work.

### 2.3 The CDDP Policy: Shield + Lagrangian

The CDDP uses a **two-layer architecture** to achieve both learnability (Lagrangian) and hard safety (shield):

> **Definition 3 (Shielded CDDP Policy).** The CDDP policy is:
>
> ```
> π_CDDP(a | s) = 
>   ⎧ π_θ(a | s) · 𝟙{a ∈ A_safe(s)} / Z(s)    if S(s) < τ   (deontic mode)
>   ⎨
>   ⎩ π_θ(a | s)                                  if S(s) ≥ τ   (utilitarian mode)
> ```
>
> where Z(s) = Σ_{a ∈ A_safe(s)} π_θ(a|s) is the normalization constant.

**Interpretation:** When stakes are low, a safety shield masks out all constraint-violating actions — the agent *cannot* trample a civilian regardless of what the learned policy π_θ suggests. When stakes exceed the threshold, the shield is lifted and the agent may violate constraints to prevent catastrophe.

### 2.4 Stakes-Modulated Lagrangian (Training Objective)

During training, π_θ is optimized via a stakes-modulated CMDP:

> **Definition 4 (Modulated Cost).**
> ```
> C̃(s, a) = h(S(s), τ) · C(s, a)
> ```

The CDDP Lagrangian is:

```
L_CDDP(π, λ) = J_R(π) − λ(J_{C̃}(π) − d)
```

**Note on the two-layer separation:** The Lagrangian trains a policy that *learns* to avoid violations in low-stakes states (improving sample efficiency). The shield *guarantees* it at execution time. Neither alone is sufficient: the Lagrangian alone gives only expected-value satisfaction; the shield alone gives safety but without learning pressure toward good constrained behavior.

---

## 3. The Stakes Functional

### 3.1 Analytical Stakes (Domain-Specific)

> **Definition 5 (Analytical Stakes).**
>
> ```
> S_analytic(s) = clip(D_min(s) / T_rem(s), 0, 1)
> ```
>
> where:
> - T_rem(s) = remaining time before bomb detonation (∞ if no bomb)
> - D_min(s) = minimum steps to reach hostage from s (Manhattan distance)
> - clip(x, 0, 1) = max(0, min(x, 1))

Properties:
- S(s) = 0 when T_rem = ∞ (no bomb → no time pressure)
- S(s) ≈ 0 when T_rem ≫ D_min (plenty of time)
- S(s) = 1 when T_rem ≤ D_min (cannot reach hostage without shortcuts)
- S(s) ∈ [0, 1] for all s by construction (clip enforces bounds)

### 3.2 Distributional Stakes (General, Learned)

> **Definition 6 (Distributional Stakes).** Let Z^π(s) be the return distribution from state s under policy π, approximated by QR-DQN with quantile estimates {θ_i(s)}_{i=1}^{N} sorted in ascending order. Let c_cat ∈ ℝ be the catastrophe return threshold. Define:
>
> ```
> P_cat(s) = (1/N) · |{ i ∈ {1,...,N} : θ_i(s) < c_cat }|
> ```
>
> the estimated probability that the return falls below the catastrophe level.
>
> The distributional stakes functional is:
>
> ```
> S_dist(s) = clip(P_cat(s) / P_ref, 0, 1)
> ```
>
> where P_ref ∈ (0, 1] is a normalizing reference probability.

**Alternative (CVaR-based):**

```
S_CVaR(s) = clip((c_safe − CVaR_α(s)) / (c_safe − c_cat), 0, 1)
```

where c_safe is the expected return under safe conditions.

---

## 4. Theoretical Results

### Theorem 1 (Hard Deontic Safety via Shield)

> **Theorem 1.** Let π_CDDP be the shielded CDDP policy (Definition 3) with hard switching (Definition 2a). Assume:
> - **(Shield Feasibility):** A_safe(s) ≠ ∅ for all reachable states s ∈ S.
> - **(Full Support — FS):** π_θ(a | s) > 0 for all a ∈ A and all s ∈ S.
>
> Then for *any* trajectory ξ = (s_0, a_0, s_1, a_1, ...) sampled under π_CDDP:
>
> ```
> S(s_t) < τ  ⟹  C(s_t, a_t) = 0    almost surely, for all t.
> ```
>
> That is, whenever the stakes are below the threshold, the agent commits zero constraint violations.

**Proof.**

Fix an arbitrary time step t and suppose S(s_t) < τ.

By Definition 3, when S(s_t) < τ, the policy is:

```
π_CDDP(a | s_t) = π_θ(a | s_t) · 𝟙{a ∈ A_safe(s_t)} / Z(s_t)
```

where Z(s_t) = Σ_{a ∈ A_safe(s_t)} π_θ(a | s_t).

By **(Shield Feasibility)**, A_safe(s_t) ≠ ∅, so there exists at least one a' ∈ A_safe(s_t). By **(FS)**, π_θ(a' | s_t) > 0. Therefore Z(s_t) ≥ π_θ(a' | s_t) > 0, and the normalization is well-defined.

For any a ∉ A_safe(s_t): π_CDDP(a | s_t) = π_θ(a | s_t) · 0 / Z(s_t) = 0.

Therefore P(a_t ∉ A_safe(s_t)) = Σ_{a ∉ A_safe(s_t)} π_CDDP(a | s_t) = 0. So a_t ∈ A_safe(s_t) almost surely, and by definition C(s_t, a_t) = 0.

Since t was arbitrary, this holds for all t where S(s_t) < τ. ∎

**Remark 1 (Necessity of Full Support).** Assumption (FS) is substantive, not cosmetic. A deterministic or near-deterministic policy (e.g., softmax with temperature → 0) may assign π_θ(a | s) ≈ 0 to all safe actions, making Z(s) ≈ 0 and the shielded distribution numerically degenerate. In practice, (FS) is ensured by standard softmax parameterization with bounded temperature or by entropy regularization, both of which are standard in PPO.

**Remark 2 (Comparison to Lagrangian-only approaches).** Standard CMDP methods (PPO-Lagrangian, RCPO) satisfy constraints only in expectation: E[Σ_t C(s_t, a_t)] ≤ d. This permits individual violations even at convergence. The CDDP shield provides a strictly stronger guarantee — *pointwise* constraint satisfaction — without requiring the Lagrangian to solve the safety problem alone.

**Remark 3 (Shield assumptions).** The shield requires (i) a known constraint function C(s,a) so that A_safe(s) can be computed, and (ii) feasibility of A_safe(s). These are standard assumptions in the shielding literature (Alshiekh et al., 2018; Dalal et al., 2018). In our gridworld domain, both hold trivially: C(s,a) = 𝟙{next cell is civilian}, and the agent can always stay in place.

---

### Proposition 1 (Stakes Monotonicity under Stochastic Dynamics)

> **Proposition 1.** Consider the hostage gridworld with stochastic bomb timer T ~ F_T, where T has support on [T_min, T_max] ⊂ ℤ₊ and F_T is the CDF. At state s at time step t, let T_rem = T − t be the remaining time (a random variable). Let D_path(s, π) denote the number of steps policy π takes to reach the hostage from s.
>
> Define the catastrophe probability:
>
> ```
> P_cat(s, t) = P(T_rem < D_path(s, π) | s_t = s) = P(T < t + D_path(s, π))
>             = F_T(t + D_path(s, π) − 1)
> ```
>
> Then S_dist(s, t) = clip(P_cat(s, t) / P_ref, 0, 1) satisfies:
>
> **(i)** S_dist is non-decreasing in the elapsed time t.
>
> **(ii)** S_dist is non-decreasing in D_min(s) (minimum distance to hostage).

**Proof.**

**(i) Monotonicity in t.** Fix s and π. Let D = D_path(s, π). Then:

```
P_cat(s, t) = F_T(t + D − 1)
```

Since F_T is a CDF, it is non-decreasing. The argument t + D − 1 is strictly increasing in t. Therefore F_T(t + D − 1) is non-decreasing in t, and so is S_dist(s, t) = clip(F_T(t + D − 1) / P_ref, 0, 1), since clip preserves monotonicity. ∎(i)

**(ii) Monotonicity in D_min(s).** Fix t and π.

> **Assumption (Grid Topology — GT).** The MDP has deterministic transitions T(s'|s,a) ∈ {0, 1} and the state space has grid topology such that for any policy π and states s₁, s₂ with D_min(s₁) < D_min(s₂), we have D_path(s₁, π) ≤ D_path(s₂, π). That is, being closer to the goal (in grid distance) implies a weakly shorter path under any fixed policy.

**Remark on (GT).** In a general MDP, this assumption can fail: a policy might wander from s₁ but teleport from s₂. In deterministic gridworlds with standard movement (Up/Down/Left/Right, no teleportation), (GT) holds because any path from s₂ can be decomposed into a path from s₂ to s₁ (of length D_min(s₂) − D_min(s₁) by triangle inequality in L₁) followed by the path from s₁, so D_path(s₂, π) ≥ D_path(s₁, π). For stochastic or non-grid environments, (GT) must be verified separately.

Proceeding under (GT): For any policy π, D_path(s, π) ≥ D_min(s) (actual path length ≥ shortest path). Consider states s₁, s₂ with D_min(s₁) < D_min(s₂). By (GT), D_path(s₂, π) ≥ D_path(s₁, π). Moreover, D_path(s₂, π) ≥ D_min(s₂) > D_min(s₁). Thus:

```
P_cat(s₂, t) = F_T(t + D_path(s₂, π) − 1) ≥ F_T(t + D_path(s₁, π) − 1) = P_cat(s₁, t)
```

where the inequality uses D_path(s₂, π) ≥ D_path(s₁, π) and monotonicity of F_T (as a CDF). Therefore S_dist(s₂, t) ≥ S_dist(s₁, t). ∎(ii)

**Remark (Strict monotonicity).** For strict inequality, we additionally require F_T to be strictly increasing on the interval [t + D_path(s₁, π) − 1, t + D_path(s₂, π) − 1]. This holds for continuous distributions (Uniform, Gaussian) and for discrete distributions with full support on [T_min, T_max]. ∎

---

### Theorem 2 (Utilitarian Recovery under High Stakes)

> **Theorem 2.** Consider a CDDP with hard switching. If S(s_t) ≥ τ for all t in trajectory ξ, then:
>
> ```
> J_R(π_CDDP | ξ) = J_R(π_θ | ξ)
> ```
>
> where π_θ is the unshielded base policy, trained to maximize J_R with modulated cost C̃(s,a) = 0 for all (s,a) in ξ (since h = 0). Up to the approximation quality of π_θ, this equals J_R(π*), the unconstrained optimal return.

**Proof.** When S(s_t) ≥ τ for all t, Definition 3 gives π_CDDP(a|s) = π_θ(a|s) — the shield is inactive. The trajectory distribution under π_CDDP is identical to that under π_θ. Therefore J_R(π_CDDP | ξ) = J_R(π_θ | ξ).

Furthermore, in the training Lagrangian, C̃(s_t, a_t) = h(S(s_t), τ) · C(s_t, a_t) = 0 · C(s_t, a_t) = 0 for all (s_t, a_t) in these states. The constraint J_{C̃} ≤ d is trivially satisfied, so the Lagrangian optimizes pure reward, producing a policy that approximates the unconstrained optimum π*. ∎

---

### Theorem 3 (Impossibility of State-Independent Strategies)

The core theoretical claim of CDDP is that state-dependent constraint modulation achieves a combination of properties that no state-independent strategy can replicate. We formalize this as a structural impossibility result that does not depend on a specific utility function.

> **Definition (State-Independent Strategy).** A strategy is *state-independent* if it applies the same constraint enforcement mechanism in every state. Formally, a state-independent shielded policy has shield activation function h(s) = h₀ ∈ {0, 1} for all s (constant). This encompasses:
> - *Permanent shield* (h₀ = 1): constraints enforced everywhere (pure deontological)
> - *No shield* (h₀ = 0): constraints never enforced (pure utilitarian)
> - *Fixed-λ CMDP*: a single scalar Lagrange multiplier λ applied uniformly

> **Theorem 3.** Consider a CDDP environment satisfying:
>
> **(E1)** The state space partitions into S_low = {s : S(s) < τ} and S_high = {s : S(s) ≥ τ}, both non-empty and visited with positive probability: p_L = P(s ∈ S_low) > 0, p_H = P(s ∈ S_high) > 0.
>
> **(E2)** In S_low, there exists at least one safe path (through A_safe actions only) that reaches the goal.
>
> **(E3)** In S_high, every path that reaches the goal before catastrophe requires at least one constraint-violating action. That is, for any policy π with C(s_t, a_t) = 0 for all t, the goal is not reached before catastrophe.
>
> **(E4, Full Support — FS)** π_θ(a|s) > 0 for all a ∈ A, s ∈ S.
>
> Then no state-independent strategy can simultaneously achieve both:
>
> **(P1)** Pointwise low-stakes safety: C(s_t, a_t) = 0 a.s. for all t where s_t ∈ S_low
>
> **(P2)** High-stakes goal achievement: the goal is reached before catastrophe in S_high with positive probability
>
> The CDDP policy π_CDDP (Definition 3) achieves both (P1) and (P2) simultaneously.

**Proof.**

**Part I: No state-independent strategy achieves (P1) ∧ (P2).**

We consider each class of state-independent strategy:

**Case 1: Permanent shield (h₀ = 1, constraints enforced everywhere).**

The shield enforces C(s_t, a_t) = 0 for all t and all s, so (P1) holds trivially. However, in S_high, the shield blocks all violating actions. By (E3), every path to the goal from S_high requires at least one violation. Since violations are blocked, the goal cannot be reached. Therefore (P2) fails. ∎(Case 1)

**Case 2: No shield (h₀ = 0, no constraint enforcement).**

The policy π_θ is unconstrained and may reach the goal in S_high, satisfying (P2). However, by (FS), π_θ(a | s) > 0 for all a, including constraint-violating actions. For any s ∈ S_low where A_unsafe(s) = A \ A_safe(s) ≠ ∅:

```
P(C(s_t, a_t) = 1 | s_t = s) = Σ_{a ∈ A_unsafe(s)} π_θ(a | s) > 0
```

Therefore (P1) fails: there is positive probability of violation in low-stakes states. ∎(Case 2)

**Case 3: Fixed-λ CMDP (single scalar multiplier).**

A CMDP with scalar λ produces a policy π_λ that solves max_π J_R(π) − λ J_C(π). The key structural constraint: the same λ creates the same violation incentive in every state.

- If λ is large enough that π_λ avoids violations in S_low (approaching P1 in expectation), then in S_high the same λ also discourages violations. By (E3), violating is necessary for goal achievement. With high λ, the agent avoids violations in S_high too, and (P2) fails.
- If λ is small enough that π_λ commits violations in S_high to reach the goal (achieving P2), then in states s ∈ S_low where violating has any reward advantage (even marginal), the same small λ permits violations and (P1) fails.
- More precisely: under (FS), for any finite λ, π_λ(a|s) assigns positive probability to every action (since the Lagrangian-penalized reward is finite for all actions, and the softmax over finite values gives positive probabilities). Therefore P(C(s_t, a_t) = 1 | s_t ∈ S_low) > 0 for any finite λ, and (P1) — which requires *almost sure* zero violations — cannot hold.

In all three cases, (P1) ∧ (P2) cannot be simultaneously achieved. ∎(Part I)

**Part II: The CDDP achieves (P1) ∧ (P2).**

**(P1):** By Theorem 1, for any s_t ∈ S_low, the shield is active and C(s_t, a_t) = 0 a.s. ✓

**(P2):** For s_t ∈ S_high, the shield is lifted (h = 0), and π_θ acts without constraint. By (FS) and the fact that the Lagrangian with C̃ = 0 in S_high places no penalty on violations, the policy can learn to reach the goal via violating paths. The probability of reaching the goal is determined by the quality of π_θ, which can be made positive (and in practice high) through training. ✓

Therefore π_CDDP achieves (P1) ∧ (P2). ∎

**Remark 1 (Scope of the result).** This theorem is purely structural: it characterizes what the CDDP *can* achieve that no state-independent strategy can, without reference to any specific utility function. The impossibility arises from the fundamental conflict between pointwise safety (which requires blocking violating actions) and goal achievement under constraint-violating-only paths (which requires permitting violating actions).

**Remark 2 (Relation to optimal CMDP policies).** A standard CMDP with carefully tuned λ can approximate the desired behavior in *expectation* — mostly avoiding violations in S_low and mostly committing them in S_high. But "mostly" is not "always": under (FS), any stochastic policy with finite λ has positive violation probability in every state. The CDDP's shield converts "mostly safe" into "always safe" in S_low, which no purely Lagrangian method can achieve.

**Remark 3 (Not a claim about expected reward).** This theorem does not claim the CDDP achieves higher expected reward than any CMDP policy — a well-tuned CMDP can achieve high expected reward. The claim is about the *achievable property set*: simultaneous pointwise safety in low-stakes AND goal achievement in high-stakes is a property combination outside the reach of state-independent strategies.

---

### Corollary 1 (Convergence of Stakes-Modulated Lagrangian)

> **Corollary 1.** *(Follows from Paternain et al. (2019) and Tessler et al. (2018).)*
>
> Under the following assumptions:
> - (A1) |R| ≤ R_max, |C| ≤ C_max (bounded reward and cost)
> - (A2) Slater's condition: ∃ π̃ s.t. J_{C̃}(π̃) < d (strict feasibility for modulated cost)
> - (A3) S(·) is a fixed function of state (not learned jointly with the policy)
>
> The modulated cost C̃(s,a) = h(S(s), τ) · C(s,a) satisfies |C̃| ≤ C_max (since h ∈ [0,1]). The CDDP Lagrangian L_CDDP(π, λ) = J_R(π) − λ(J_{C̃}(π) − d) is structurally a standard CMDP Lagrangian with cost function C̃.
>
> By Theorem 1 of Paternain et al. (2019), the CMDP with cost C̃ has zero duality gap. The primal-dual algorithm:
>
> ```
> π_{k+1} = π_k + α_k ∇_π L_CDDP(π_k, λ_k)
> λ_{k+1} = [λ_k + β_k (J_{C̃}(π_k) − d)]₊
> ```
>
> with step sizes α_k, β_k = O(1/√K) converges to an ε-saddle point in O(1/ε²) iterations.

**Proof.** The modulated cost C̃(s,a) = h(S(s), τ) · C(s,a) is a fixed, bounded, state-dependent function (by A1, A3, and h ∈ [0,1]). The CDDP Lagrangian is identical in structure to a CMDP Lagrangian with cost function C̃. All conditions of Theorem 1 in Paternain et al. (2019) — bounded costs, Slater's condition (A2), linearity of J_{C̃} in the occupancy measure — are satisfied. The convergence rate follows directly. ∎

**Remark (Novelty).** This corollary is not a new convergence result. The contribution of CDDP is not in the optimization algorithm but in the *formulation*: the definition of C̃ via stakes modulation, the safety shield for hard guarantees, and the resulting performance dominance (Theorem 3). The Lagrangian is a tool; the framework is the contribution.

---

### Proposition 2 (Relationship to Existing Frameworks)

> **Proposition 2.** The CDDP Lagrangian component (C̃ = h(s)·C) is a state-dependent cost CMDP, a structure that exists in the literature. The genuine novelty of CDDP lies in the specific combination of:
> - (a) a distributional-RL-derived stakes functional S(s) determining h(s),
> - (b) a mode-switched safety shield providing pointwise guarantees (Theorem 1), and
> - (c) the explicit goal of implementing threshold deontology.
>
> No prior framework combines these three elements simultaneously.

We compare against CVaR-constrained MDPs (Chow et al., 2017) and state-dependent cost CMDPs to clarify the relationship:

**Relationship to CVaR-constrained MDPs.**

A CVaR-CMDP has objective max_π E[R^π] s.t. CVaR_α(Z_C^π) ≤ d, applying a single global risk measure.

**(1) State-dependent vs. global modulation.** The CDDP modulates constraints per-state via h(S(s), τ). This is a state-dependent cost CMDP — replacing C(s,a) with h(s)·C(s,a) is a known construction. However, in the CDDP, h(s) is specifically derived from distributional tail risk (Definition 6), creating a closed loop between the return distribution and constraint enforcement. This specific derivation of h from distributional RL is novel.

**(2) Hard safety via shield.** The CDDP provides pointwise constraint satisfaction (Theorem 1) via the safety shield. CVaR-constrained MDPs, like all Lagrangian-based methods, bound costs in a distributional sense — even CVaR_α with α → 0 does not guarantee C(s_t, a_t) = 0 at individual steps. The shield is not part of any CVaR-CMDP framework.

**(3) Bimodal structure.** The CDDP explicitly switches between two qualitative behavioral modes (shield on/off), controlled by the stakes functional. CVaR-CMDPs produce a single policy with a smooth risk-reward tradeoff. The discrete mode switch formalizes the philosophical concept of threshold deontology, which has no analogue in risk-constrained optimization.

**Relationship to state-dependent cost CMDPs.** Replacing C with h(s)·C in a standard CMDP yields a state-dependent cost CMDP, and the Lagrangian convergence (Corollary 1) follows directly. The CDDP's contribution is not the cost-modulation mechanism itself but: (i) deriving h(s) from distributional tail risk, (ii) the shield for hard guarantees, and (iii) the threshold-deontological motivation that determines the specific form of h.

| Property | CVaR-CMDP | State-dep. cost CMDP | CDDP |
|---|---|---|---|
| Constraint modulation | Global (fixed α) | State-dependent h(s) | State-dependent h(S(s), τ) |
| Source of h(s) | N/A | Arbitrary / hand-designed | Distributional RL (CVaR/tail risk) |
| Safety guarantee | Distributional | In expectation | Pointwise (shield, Thm 1) |
| Policy structure | Single smooth | Single smooth | Bimodal (mode-switched) |
| Philosophical grounding | Risk aversion | None specific | Threshold deontology |

∎

---

## 5. CDDP-PPO Algorithm

### 5.1 Training Stability: The Warmup Problem

The distributional stakes functional S_dist(s; Z_ψ) depends on the distributional critic Z_ψ, which is initialized randomly and trained concurrently with the policy. In early iterations, the quantile estimates {θ_i(s)} are meaningless, so P_cat(s) is noise, and the shield activates/deactivates randomly. This corrupts training trajectories, which in turn produces badly trained Z_ψ — a feedback loop.

**Solution: Two-phase training with analytical warmup.**

> **Phase 1 (Warmup, iterations 1 to K_w):** Use the analytical stakes functional S_analytic(s) = clip(D_min/T_rem, 0, 1) instead of S_dist. This provides a reliable, domain-specific stakes signal while Z_ψ is trained passively from collected trajectories. During this phase, Z_ψ is updated but not used for shield gating.
>
> **Phase 2 (Full CDDP, iterations K_w+1 to K):** Switch to the distributional stakes functional S_dist(s; Z_ψ). By this point, Z_ψ has been trained on K_w iterations of data and provides meaningful quantile estimates. Optionally, blend: S(s) = (1−α_k) · S_analytic(s) + α_k · S_dist(s) with α_k ramping from 0 to 1 over a transition window.

This two-phase approach ensures stable early training while allowing the agent to eventually rely on the learned, general-purpose distributional stakes.

### 5.2 Algorithm

```
Input: CDDP = (S, A, T, R, C, S(·), τ, h, d), warmup iterations K_w
Initialize: policy π_θ, value critic V_φ, distributional critic Z_ψ (QR-DQN, N quantiles),
            Lagrange multiplier λ ≥ 0

For each iteration k = 1, ..., K:
  1. Collect trajectories using the shielded policy π_CDDP (Definition 3)

  2. Update distributional critic Z_ψ via quantile regression loss

  3. Compute stakes for each visited state:
       if k ≤ K_w:   (warmup phase)
           S(s_t) ← S_analytic(s_t)      (domain-specific, reliable)
       else:          (full CDDP phase)
           S(s_t) ← S_dist(s_t; Z_ψ, c_cat, P_ref)   (learned, general)

  4. Compute modulated costs:
       c̃_t ← h(S(s_t), τ) · c_t

  5. Compute advantage estimates:
       Â_t via GAE from V_φ
       Â^{C̃}_t via cost advantage from modulated costs

  6. Policy gradient step (PPO with clipping):
       ∇_θ L ≈ E_t [ ∇_θ log π_θ(a_t|s_t) · (Â_t − λ · Â^{C̃}_t) ]
       θ ← θ + α_k · ∇_θ L  (with PPO clipping)

  7. Dual variable update:
       λ ← [λ + β_k (Ĵ_{C̃} − d)]₊

  8. Value critic update:
       φ ← φ − η_φ ∇_φ E_t [(V_φ(s_t) − R̂_t)²]

Return: π_θ, Z_ψ
```

**Remark (Generality of warmup).** The analytical warmup is domain-specific (requires knowing T_rem and D_min). In domains without a natural analytical stakes function, alternative warmup strategies include: (i) conservative warmup with h ≡ 1 (full shield) for K_w iterations, then transitioning; (ii) pre-training Z_ψ on random-policy rollouts before joint optimization; or (iii) using a high P_ref in S_dist to make the stakes signal conservative during early unreliable estimates.

---

## 6. Connection to Existing Frameworks

| Framework | Constraint mechanism | Safety guarantee | Modes |
|---|---|---|---|
| MDP (PPO) | None (h ≡ 0) | None | Utilitarian only |
| CMDP (PPO-Lag) | Fixed Lagrangian (h ≡ 1) | In expectation | Deontological only |
| CVaR-CMDP | Global tail risk | Distributional | Single smooth |
| Shielded MDP | Runtime filter | Pointwise | Constrained only |
| **CDDP (ours)** | **Stakes-modulated shield + Lagrangian** | **Pointwise when S(s) < τ** | **Bimodal** |

> **Proposition 3 (Generalization).** Setting h ≡ 0 (τ = 0) recovers the unconstrained MDP. Setting h ≡ 1 (τ = 1) with the shield always active recovers a shielded CMDP. For τ ∈ (0, 1), the CDDP represents a novel interpolation.

---

## 7. Hostage Gridworld Instantiation

| CDDP component | Gridworld mapping |
|---|---|
| **S** | (agent_pos, hostage_pos, civilian_pos[], bomb_timer, elapsed_time) |
| **A** | {Up, Down, Left, Right} |
| **R(s,a)** | Reward shaping − civilian penalty + hostage bonus |
| **C(s,a)** | 𝟙{action moves agent onto civilian tile} |
| **A_safe(s)** | {a : next_pos(s, a) ∉ civilian_positions} |
| **S(s)** | clip(D_min / T_rem, 0, 1) (analytical) or CVaR-based (distributional) |
| **c_cat** | −100 (return level associated with bomb detonation) |
| **τ** | Tuned via grid search; expected range [0.3, 0.7] |
| **d** | 0 (zero violations tolerated in low-stakes) |

---

## 8. References

1. Leike et al. (2017). "AI Safety Gridworlds." arXiv:1711.09883.
2. Altman (1999). *Constrained Markov Decision Processes.* Chapman & Hall.
3. Bellemare et al. (2017). "A Distributional Perspective on RL." ICML.
4. Dabney et al. (2018). "Distributional RL with Quantile Regression." AAAI.
5. Tessler et al. (2018). "Reward Constrained Policy Optimization." ICLR.
6. Paternain et al. (2019). "Constrained RL Has Zero Duality Gap." NeurIPS.
7. Chow et al. (2017). "Risk-Constrained RL with CVaR." NeurIPS.
8. Alshiekh et al. (2018). "Safe RL via Shielding." AAAI.
9. Dalal et al. (2018). "Safe Exploration in Continuous Action Spaces." arXiv.
10. Alexander (2000). "Deontology at the Threshold." San Diego Law Review.
