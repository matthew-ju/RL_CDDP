# Contextual Deontic Decision Process (CDDP): Formal Framework

---

## 1. Preliminaries

### 1.1 Constrained Markov Decision Process (CMDP)

A CMDP is a tuple $(\mathcal{S}, \mathcal{A}, T, R, C, d)$ where:

- $\mathcal{S}$ is a finite state space
- $\mathcal{A}$ is a finite action space
- $T: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to [0, 1]$ is the transition kernel, $T(s'|s, a)$
- $R: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ is the reward function, bounded $|R(s, a)| \leq R_{\max}$
- $C: \mathcal{S} \times \mathcal{A} \to \{0, 1\}$ is the constraint indicator (1 = violation)
- $d \geq 0$ is the expected cost budget

The CMDP objective is:

$$\max_{\pi} \; J_R(\pi) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{H-1} \gamma^t R(s_t, a_t) \right]$$

$$\text{s.t.} \quad J_C(\pi) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{H-1} \gamma^t C(s_t, a_t) \right] \leq d$$

The Lagrangian relaxation is $L(\pi, \lambda) = J_R(\pi) - \lambda(J_C(\pi) - d)$, $\lambda \geq 0$, solved via primal-dual updates.

**Key limitation:** Lagrangian methods satisfy constraints _in expectation_, not pointwise. The converged policy may violate $C(s,a) = 1$ at individual steps.

### 1.2 Distributional Reinforcement Learning

In distributional RL, we learn the full return distribution $Z^\pi(s, a)$, not just $Q^\pi(s, a) = \mathbb{E}[Z^\pi(s, a)]$.

**Quantile Regression (QR-DQN):** Approximate $Z^\pi(s, a)$ by $N$ quantile values $\{\theta_i(s, a)\}_{i=1}^{N}$ at quantile fractions $\tau_i = (2i-1)/2N$, trained via:

$$\mathcal{L}_{QR}(\theta) = \sum_{i=1}^{N} \mathbb{E}\left[ \rho_{\tau_i}\left(r + \gamma \theta_j(s', a') - \theta_i(s, a)\right) \right]$$

$$\text{where} \quad \rho_\tau(u) = u \cdot (\tau - \mathbb{1}\{u < 0\})$$

**CVaR at level $\alpha$:** $\text{CVaR}_\alpha(s, a) = \frac{1}{\lceil \alpha N \rceil} \sum_{i=1}^{\lceil \alpha N \rceil} \theta_i(s, a)$, the expected return in the worst $\alpha$-fraction of outcomes.

### 1.3 Safety Shields

A **safety shield** (Alshiekh et al., 2018) is a runtime enforcement mechanism that restricts the agent's action space to guarantee constraint satisfaction regardless of the learned policy.

Given a constraint predicate $C: \mathcal{S} \times \mathcal{A} \to \{0, 1\}$, the **safe action set** at state $s$ is:

$$\mathcal{A}_{\text{safe}}(s) = \{ a \in \mathcal{A} : C(s, a) = 0 \}$$

The **shielded policy** is:

$$\pi_{\text{shield}}(a \mid s) = \frac{\pi_\theta(a \mid s) \cdot \mathbb{1}\{a \in \mathcal{A}_{\text{safe}}(s)\}}{Z(s)}$$

$$\text{where} \quad Z(s) = \sum_{a \in \mathcal{A}_{\text{safe}}(s)} \pi_\theta(a \mid s)$$

**Assumption (Shield Feasibility):** $\mathcal{A}_{\text{safe}}(s) \neq \emptyset$ for all reachable $s$. There always exists at least one non-violating action. (In the gridworld: the agent can always stay in place or move away from civilians.)

**Assumption (Full Support — FS).** The policy $\pi_\theta$ satisfies $\pi_\theta(a \mid s) > 0$ for all $a \in \mathcal{A}$ and all $s \in \mathcal{S}$. This is standard in policy gradient methods (e.g., softmax parameterization with finite temperature) and ensures that the shield normalization constant $Z(s) = \sum_{a \in \mathcal{A}_{\text{safe}}(s)} \pi_\theta(a \mid s) > 0$ whenever $\mathcal{A}_{\text{safe}}(s) \neq \emptyset$.

---

## 2. The Contextual Deontic Decision Process

### 2.1 Definition

> **Definition 1 (CDDP).** A _Contextual Deontic Decision Process_ is a tuple
>
> $$(\mathcal{S}, \mathcal{A}, T, R, C, \mathcal{S}(\cdot), \tau, h, d)$$
>
> where $(\mathcal{S}, \mathcal{A}, T, R, C, d)$ is a CMDP, and additionally:
>
> - $\mathcal{S}: \mathcal{S} \to [0, 1]$ is the _stakes functional_, mapping each state to a scalar measuring situation severity.
> - $\tau \in (0, 1)$ is the _stakes threshold_.
> - $h: [0, 1] \times (0, 1) \to [0, 1]$ is the _constraint modulation function_.

### 2.2 Constraint Modulation

> **Definition 2a (Hard Switching).**
>
> $$h_{\text{hard}}(\mathcal{S}(s), \tau) = \mathbb{1}\{\mathcal{S}(s) < \tau\}$$

> **Definition 2b (Soft Switching).**
>
> $$h_{\text{soft}}(\mathcal{S}(s), \tau) = 1 - \sigma_\beta(\mathcal{S}(s) - \tau), \quad \sigma_\beta(x) = \frac{1}{1 + \exp(-\beta x)}$$

Note: $h_{\text{hard}} = \lim_{\beta \to \infty} h_{\text{soft}}$. We use hard switching throughout; soft switching is reserved for future work.

### 2.3 The CDDP Policy: Shield + Lagrangian

The CDDP uses a **two-layer architecture** to achieve both learnability (Lagrangian) and hard safety (shield):

> **Definition 3 (Shielded CDDP Policy).** The CDDP policy is:
>
> $$\pi_{\text{CDDP}}(a \mid s) = \begin{cases} \dfrac{\pi_\theta(a \mid s) \cdot \mathbb{1}\{a \in \mathcal{A}_{\text{safe}}(s)\}}{Z(s)} & \text{if } \mathcal{S}(s) < \tau \quad \text{(deontic mode)} \\[8pt] \pi_\theta(a \mid s) & \text{if } \mathcal{S}(s) \geq \tau \quad \text{(utilitarian mode)} \end{cases}$$
>
> where $Z(s) = \sum_{a \in \mathcal{A}_{\text{safe}}(s)} \pi_\theta(a \mid s)$ is the normalization constant.

**Interpretation:** When stakes are low, a safety shield masks out all constraint-violating actions — the agent _cannot_ trample a civilian regardless of what the learned policy $\pi_\theta$ suggests. When stakes exceed the threshold, the shield is lifted and the agent may violate constraints to prevent catastrophe.

### 2.4 Stakes-Modulated Lagrangian (Training Objective)

During training, $\pi_\theta$ is optimized via a stakes-modulated CMDP:

> **Definition 4 (Modulated Cost).**
>
> $$\tilde{C}(s, a) = h(\mathcal{S}(s), \tau) \cdot C(s, a)$$

The CDDP Lagrangian is:

$$L_{\text{CDDP}}(\pi, \lambda) = J_R(\pi) - \lambda\left(J_{\tilde{C}}(\pi) - d\right)$$

**Note on the two-layer separation:** The Lagrangian trains a policy that _learns_ to avoid violations in low-stakes states (improving sample efficiency). The shield _guarantees_ it at execution time. Neither alone is sufficient: the Lagrangian alone gives only expected-value satisfaction; the shield alone gives safety but without learning pressure toward good constrained behavior.

---

## 3. The Stakes Functional

The stakes functional $\mathcal{S}: \mathcal{S} \to [0,1]$ is the central object that governs mode switching in CDDP. This section develops the stakes concept in three stages: a domain-specific closed-form version (§3.1), a general learned version that reveals a fundamental limitation (§3.2), and the canonical prospective formulation that resolves it (§3.3).

### 3.0 Observability Assumption

The analytical stakes functional (Definition 5, below) requires access to quantities such as $T_{\text{rem}}$ (remaining time before catastrophe) and $D_{\min}$ (minimum distance to goal). In many realistic deployments, these quantities are **hidden variables**: the agent observes only partial state information, and the environment is effectively a POMDP.

When $T_{\text{rem}}$ and $D_{\min}$ are unobservable, the stakes functional cannot be computed in closed form. Learned stakes must therefore _estimate_ counterfactual risk from the observable state, rather than compute it from privileged information.

This motivates a key architectural principle:

> **CDDP separates stakes _definition_ from stakes _estimation_.** The stakes functional specifies the _ideal_ switching criterion — what quantity the agent should threshold on. The estimation method specifies _how_ that quantity is approximated in practice. Sections 3.1–3.2 provide two estimation strategies; Section 3.3 defines the conceptually correct switching criterion that both approximate.

### 3.1 Analytical Stakes (Domain-Specific Closed-Form Approximation)

The analytical stakes functional provides a closed-form approximation of the prospective stakes concept (Definition 7, §3.3) for domains where the relevant hidden variables are observable.

> **Definition 5 (Analytical Stakes).**
>
> $$\mathcal{S}_{\text{analytic}}(s) = \text{clip}\left(\frac{D_{\min}(s)}{T_{\text{rem}}(s)},\; 0,\; 1\right)$$
>
> where:
>
> - $T_{\text{rem}}(s)$ = remaining time before bomb detonation ($\infty$ if no bomb)
> - $D_{\min}(s)$ = minimum steps to reach hostage from $s$ (Manhattan distance)
> - $\text{clip}(x, 0, 1) = \max(0, \min(x, 1))$

Properties:

- $\mathcal{S}_{\text{analytic}}(s) = 0$ when $T_{\text{rem}} = \infty$ (no bomb → no time pressure)
- $\mathcal{S}_{\text{analytic}}(s) \approx 0$ when $T_{\text{rem}} \gg D_{\min}$ (plenty of time)
- $\mathcal{S}_{\text{analytic}}(s) = 1$ when $T_{\text{rem}} \leq D_{\min}$ (cannot reach hostage without shortcuts)
- $\mathcal{S}_{\text{analytic}}(s) \in [0, 1]$ for all $s$ by construction (clip enforces bounds)

This formulation implicitly performs **lookahead feasibility reasoning**: $D_{\min} / T_{\text{rem}} \geq 1$ precisely when no constraint-respecting path to the goal exists within the remaining time horizon. In this sense, the analytical stakes answer a prospective question — _can the agent still succeed while obeying constraints?_ — via a geometric shortcut.

**Limitation.** The analytical stakes require **privileged knowledge** of $T_{\text{rem}}$, which is a hidden environment variable in the general case. They also assume deterministic dynamics (the Manhattan distance $D_{\min}$ is meaningful only when the agent moves one cell per step with certainty).

**Remark (Relationship to prospective stakes).** When dynamics are deterministic and $T_{\text{rem}}$ is observable, $\mathcal{S}_{\text{analytic}}(s) \approx \mathcal{S}_{\text{pros}}(s)$ (Definition 7). The analytical stakes are a closed-form realization of the prospective stakes concept in the full-information deterministic setting. See §3.3 for the general definition.

### 3.2 Distributional Stakes (General, Learned)

The distributional stakes functional removes the dependence on privileged environment variables by learning the catastrophe probability from the return distribution.

> **Definition 6 (Distributional Stakes).** Let $Z^\pi(s)$ be the return distribution from state $s$ under policy $\pi$, approximated by QR-DQN with quantile estimates $\{\theta_i(s)\}_{i=1}^{N}$ sorted in ascending order. Let $c_{\text{cat}} \in \mathbb{R}$ be the catastrophe return threshold. Define:
>
> $$P_{\text{cat}}(s;\, \pi) = \frac{1}{N} \cdot \left|\{ i \in \{1,\ldots,N\} : \theta_i(s) < c_{\text{cat}} \}\right|$$
>
> the estimated probability that the return falls below the catastrophe level under policy $\pi$.
>
> The distributional stakes functional is:
>
> $$\mathcal{S}_{\text{dist}}(s;\, \pi) = \text{clip}\left(\frac{P_{\text{cat}}(s;\, \pi)}{P_{\text{ref}}},\; 0,\; 1\right)$$
>
> where $P_{\text{ref}} \in (0, 1]$ is a normalizing reference probability.

**Alternative (CVaR-based):**

$$\mathcal{S}_{\text{CVaR}}(s;\, \pi) = \text{clip}\left(\frac{c_{\text{safe}} - \text{CVaR}_\alpha(s;\, \pi)}{c_{\text{safe}} - c_{\text{cat}}},\; 0,\; 1\right)$$

where $c_{\text{safe}}$ is the expected return under safe conditions.

#### Limitation: Policy-Mismatch Bias

As stated in Definition 6, the distributional stakes $\mathcal{S}_{\text{dist}}(s;\, \pi_\theta)$ measure catastrophe probability under the **learned blended policy** $\pi_\theta$. This introduces a systematic bias when $\pi_\theta$ has already learned to violate constraints:

- If $\pi_\theta$ occasionally takes shortcuts through constraint-violating actions (e.g., trampling civilians to save time), these violations _reduce_ the time-to-goal, which in turn _lowers_ the estimated catastrophe probability $P_{\text{cat}}(s;\, \pi_\theta)$.
- The distributional stakes therefore **underestimate urgency**: the agent appears to have more time than it actually does under constraint-respecting behavior.
- This can cause **late switching**: the stakes remain below $\tau$ in states where constraint-respecting paths are already infeasible, because $\pi_\theta$'s return distribution reflects a faster (but violation-dependent) trajectory.

The fundamental issue is a **counterfactual mismatch**: the switching decision asks whether constraints should be relaxed, but the evidence used to make this decision is contaminated by prior relaxation. The relevant counterfactual for deontic switching is catastrophe risk under _continued constraint adherence_, not under the current blended policy.

### 3.3 Prospective (Lookahead) Stakes

The policy-mismatch bias of §3.2 motivates the canonical stakes definition: stakes should measure catastrophe risk under the assumption that the agent _continues to follow constraints_.

#### Motivation

The deontic switching decision asks a precise counterfactual question:

> _If the agent continues obeying constraints from this state forward, will catastrophe become inevitable?_

If the answer is "yes" (with sufficiently high probability), the agent should switch to utilitarian mode. If "no," the agent should remain deontic. The correct input to this decision is not the return distribution under the current policy $\pi_\theta$, but under the **deontic policy** — the policy that permanently respects all constraints.

#### Definition

> **Definition 7 (Deontic Policy).** The _deontic policy_ $\pi_d$ is the CDDP policy (Definition 3) restricted to permanent deontic mode:
>
> $$\pi_d(a \mid s) = \frac{\pi_\theta(a \mid s) \cdot \mathbb{1}\{a \in \mathcal{A}_{\text{safe}}(s)\}}{Z(s)} \quad \text{for all } s$$
>
> That is, $\pi_d$ is the shielded policy applied unconditionally, regardless of stakes.

> **Definition 8 (Prospective Stakes).** The _prospective stakes functional_ is:
>
> $$\mathcal{S}_{\text{pros}}(s) = \text{clip}\left(\frac{P_{\text{cat}}(s;\, \pi_d)}{P_{\text{ref}}},\; 0,\; 1\right)$$
>
> where:
>
> - $P_{\text{cat}}(s;\, \pi_d) = \mathbb{P}(Z^{\pi_d}(s) < c_{\text{cat}})$ is the catastrophe probability under the deontic policy $\pi_d$
> - $c_{\text{cat}} \in \mathbb{R}$ is the catastrophe return threshold
> - $P_{\text{ref}} \in (0, 1]$ is a normalizing reference probability

#### Interpretation

The prospective stakes evaluate **future inevitability under rule-following**. When $\mathcal{S}_{\text{pros}}(s) \geq \tau$, the agent has determined — via its learned model of deontic-mode outcomes — that continued constraint adherence leads to catastrophe with probability exceeding $\tau \cdot P_{\text{ref}}$. This triggers a switch to utilitarian mode _before_ constraint-respecting behavior becomes infeasible, rather than waiting until the blended policy's tail risk crosses the threshold.

#### Late-Switch Failure Band

The policy-mismatch bias of §3.2 manifests most acutely in a specific region of the state space. Define $L_{\text{safe}}(s)$ as the minimum path length to the goal using only $\mathcal{A}_{\text{safe}}$ actions (i.e., the shortest constraint-respecting path). The **late-switch failure band** is the set of states satisfying:

$$D_{\min}(s) < T_{\text{rem}}(s) < L_{\text{safe}}(s)$$

In this region:

- $D_{\min} < T_{\text{rem}}$: a path to the goal exists within the time limit (geometrically feasible)
- $T_{\text{rem}} < L_{\text{safe}}$: no _constraint-respecting_ path reaches the goal in time (deontically infeasible)

Standard distributional stakes $\mathcal{S}_{\text{dist}}(s;\, \pi_\theta)$ underestimate urgency in this band because the blended policy's return distribution reflects shortcut-taking trajectories that make the deadline. Prospective stakes $\mathcal{S}_{\text{pros}}(s)$ correctly identify these states as high-stakes, since under $\pi_d$ the agent cannot reach the goal in time.

#### Two-Critic Architecture

Computing $\mathcal{S}_{\text{pros}}(s)$ requires estimating the return distribution under $\pi_d$, which differs from the return distribution under $\pi_\theta$ used for reward optimization. CDDP therefore requires **two distributional value estimators**:

| Critic                   | Evaluates policy                         | Role                                                                                  |
| ------------------------ | ---------------------------------------- | ------------------------------------------------------------------------------------- |
| $Z_{\psi}^{\text{free}}$ | $\pi_\theta$ (unconstrained base policy) | Utility optimization: provides distributional value estimates for reward maximization |
| $Z_{\psi}^{\text{deon}}$ | $\pi_d$ (shielded deontic policy)        | Stakes estimation: provides $P_{\text{cat}}(s;\, \pi_d)$ for prospective switching    |

Both critics are trained via quantile regression (§1.2) on trajectories generated by their respective policies. During data collection, the CDDP policy $\pi_{\text{CDDP}}$ generates trajectories that contain both deontic-mode and utilitarian-mode segments; $Z_{\psi}^{\text{deon}}$ is trained on deontic-mode transitions (where the shield was active), and $Z_{\psi}^{\text{free}}$ is trained on all transitions (or utilitarian-mode transitions, depending on the implementation).

> **Remark (Analytical stakes as special case).** When dynamics are deterministic and $T_{\text{rem}}$ is observable, the deontic policy $\pi_d$ follows the shortest safe path of length $L_{\text{safe}}(s)$, and catastrophe occurs if and only if $L_{\text{safe}}(s) > T_{\text{rem}}(s)$, i.e., $D_{\min}(s) / T_{\text{rem}}(s) \geq 1$ (since $L_{\text{safe}} \geq D_{\min}$). Thus $\mathcal{S}_{\text{analytic}}(s) \approx \mathcal{S}_{\text{pros}}(s)$, confirming that the analytical stakes are a closed-form realization of the prospective stakes in the full-information deterministic setting.

### 3.4 Canonical Stakes Convention

Throughout the remainder of this manuscript, the unqualified stakes functional refers to the prospective formulation:

$$\mathcal{S}(s) := \mathcal{S}_{\text{pros}}(s) \quad \text{(default)}$$

The analytical stakes $\mathcal{S}_{\text{analytic}}(s)$ serve as a domain-specific closed-form approximation (used during warmup training, §5.1). The standard distributional stakes $\mathcal{S}_{\text{dist}}(s;\, \pi_\theta)$ serve as a policy-dependent approximation that may be used when only a single critic is available, with the understanding that this introduces the policy-mismatch bias described in §3.2.

---

## 4. Theoretical Results

### Theorem 1 (Hard Deontic Safety via Shield)

> **Theorem 1.** Let $\pi_{\text{CDDP}}$ be the shielded CDDP policy (Definition 3) with hard switching (Definition 2a). Assume:
>
> - **(Shield Feasibility):** $\mathcal{A}_{\text{safe}}(s) \neq \emptyset$ for all reachable states $s \in \mathcal{S}$.
> - **(Full Support — FS):** $\pi_\theta(a \mid s) > 0$ for all $a \in \mathcal{A}$ and all $s \in \mathcal{S}$.
>
> Then for _any_ trajectory $\xi = (s_0, a_0, s_1, a_1, \ldots)$ sampled under $\pi_{\text{CDDP}}$:
>
> $$\mathcal{S}(s_t) < \tau \;\Longrightarrow\; C(s_t, a_t) = 0 \quad \text{almost surely, for all } t.$$
>
> That is, whenever the stakes are below the threshold, the agent commits zero constraint violations.

**Proof.**

Fix an arbitrary time step $t$ and suppose $\mathcal{S}(s_t) < \tau$.

By Definition 3, when $\mathcal{S}(s_t) < \tau$, the policy is:

$$\pi_{\text{CDDP}}(a \mid s_t) = \frac{\pi_\theta(a \mid s_t) \cdot \mathbb{1}\{a \in \mathcal{A}_{\text{safe}}(s_t)\}}{Z(s_t)}$$

where $Z(s_t) = \sum_{a \in \mathcal{A}_{\text{safe}}(s_t)} \pi_\theta(a \mid s_t)$.

By **(Shield Feasibility)**, $\mathcal{A}_{\text{safe}}(s_t) \neq \emptyset$, so there exists at least one $a' \in \mathcal{A}_{\text{safe}}(s_t)$. By **(FS)**, $\pi_\theta(a' \mid s_t) > 0$. Therefore $Z(s_t) \geq \pi_\theta(a' \mid s_t) > 0$, and the normalization is well-defined.

For any $a \notin \mathcal{A}_{\text{safe}}(s_t)$: $\pi_{\text{CDDP}}(a \mid s_t) = \pi_\theta(a \mid s_t) \cdot 0 / Z(s_t) = 0$.

Therefore $\mathbb{P}(a_t \notin \mathcal{A}_{\text{safe}}(s_t)) = \sum_{a \notin \mathcal{A}_{\text{safe}}(s_t)} \pi_{\text{CDDP}}(a \mid s_t) = 0$. So $a_t \in \mathcal{A}_{\text{safe}}(s_t)$ almost surely, and by definition $C(s_t, a_t) = 0$.

Since $t$ was arbitrary, this holds for all $t$ where $\mathcal{S}(s_t) < \tau$. $\blacksquare$

**Remark 1 (Necessity of Full Support).** Assumption (FS) is substantive, not cosmetic. A deterministic or near-deterministic policy (e.g., softmax with temperature $\to 0$) may assign $\pi_\theta(a \mid s) \approx 0$ to all safe actions, making $Z(s) \approx 0$ and the shielded distribution numerically degenerate. In practice, (FS) is ensured by standard softmax parameterization with bounded temperature or by entropy regularization, both of which are standard in PPO.

**Remark 2 (Comparison to Lagrangian-only approaches).** Standard CMDP methods (PPO-Lagrangian, RCPO) satisfy constraints only in expectation: $\mathbb{E}[\sum_t C(s_t, a_t)] \leq d$. This permits individual violations even at convergence. The CDDP shield provides a strictly stronger guarantee — _pointwise_ constraint satisfaction — without requiring the Lagrangian to solve the safety problem alone.

**Remark 3 (Shield assumptions).** The shield requires (i) a known constraint function $C(s,a)$ so that $\mathcal{A}_{\text{safe}}(s)$ can be computed, and (ii) feasibility of $\mathcal{A}_{\text{safe}}(s)$. These are standard assumptions in the shielding literature (Alshiekh et al., 2018; Dalal et al., 2018). In our gridworld domain, both hold trivially: $C(s,a) = \mathbb{1}\{\text{next cell is civilian}\}$, and the agent can always stay in place.

---

### Proposition 1 (Stakes Monotonicity under Stochastic Dynamics)

> **Proposition 1.** Consider the hostage gridworld with stochastic bomb timer $T \sim F_T$, where $T$ has support on $[T_{\min}, T_{\max}] \subset \mathbb{Z}_+$ and $F_T$ is the CDF. At state $s$ at time step $t$, let $T_{\text{rem}} = T - t$ be the remaining time (a random variable). Let $D_{\text{path}}(s, \pi)$ denote the number of steps policy $\pi$ takes to reach the hostage from $s$.
>
> Define the catastrophe probability:
>
> $$P_{\text{cat}}(s, t) = \mathbb{P}(T_{\text{rem}} < D_{\text{path}}(s, \pi) \mid s_t = s) = \mathbb{P}(T < t + D_{\text{path}}(s, \pi)) = F_T(t + D_{\text{path}}(s, \pi) - 1)$$
>
> Then $\mathcal{S}_{\text{dist}}(s, t) = \text{clip}(P_{\text{cat}}(s, t) / P_{\text{ref}},\; 0,\; 1)$ satisfies:
>
> **(i)** $\mathcal{S}_{\text{dist}}$ is non-decreasing in the elapsed time $t$.
>
> **(ii)** $\mathcal{S}_{\text{dist}}$ is non-decreasing in $D_{\min}(s)$ (minimum distance to hostage).

**Proof.**

**(i) Monotonicity in $t$.** Fix $s$ and $\pi$. Let $D = D_{\text{path}}(s, \pi)$. Then:

$$P_{\text{cat}}(s, t) = F_T(t + D - 1)$$

Since $F_T$ is a CDF, it is non-decreasing. The argument $t + D - 1$ is strictly increasing in $t$. Therefore $F_T(t + D - 1)$ is non-decreasing in $t$, and so is $\mathcal{S}_{\text{dist}}(s, t) = \text{clip}(F_T(t + D - 1) / P_{\text{ref}},\; 0,\; 1)$, since clip preserves monotonicity. $\blacksquare$(i)

**(ii) Monotonicity in $D_{\min}(s)$.** Fix $t$ and $\pi$.

> **Assumption (Grid Topology — GT).** The MDP has deterministic transitions $T(s'|s,a) \in \{0, 1\}$ and the state space has grid topology such that for any policy $\pi$ and states $s_1, s_2$ with $D_{\min}(s_1) < D_{\min}(s_2)$, we have $D_{\text{path}}(s_1, \pi) \leq D_{\text{path}}(s_2, \pi)$. That is, being closer to the goal (in grid distance) implies a weakly shorter path under any fixed policy.

**Remark on (GT).** In a general MDP, this assumption can fail: a policy might wander from $s_1$ but teleport from $s_2$. In deterministic gridworlds with standard movement (Up/Down/Left/Right, no teleportation), (GT) holds because any path from $s_2$ can be decomposed into a path from $s_2$ to $s_1$ (of length $D_{\min}(s_2) - D_{\min}(s_1)$ by triangle inequality in $L_1$) followed by the path from $s_1$, so $D_{\text{path}}(s_2, \pi) \geq D_{\text{path}}(s_1, \pi)$. For stochastic or non-grid environments, (GT) must be verified separately.

Proceeding under (GT): For any policy $\pi$, $D_{\text{path}}(s, \pi) \geq D_{\min}(s)$ (actual path length $\geq$ shortest path). Consider states $s_1, s_2$ with $D_{\min}(s_1) < D_{\min}(s_2)$. By (GT), $D_{\text{path}}(s_2, \pi) \geq D_{\text{path}}(s_1, \pi)$. Moreover, $D_{\text{path}}(s_2, \pi) \geq D_{\min}(s_2) > D_{\min}(s_1)$. Thus:

$$P_{\text{cat}}(s_2, t) = F_T(t + D_{\text{path}}(s_2, \pi) - 1) \geq F_T(t + D_{\text{path}}(s_1, \pi) - 1) = P_{\text{cat}}(s_1, t)$$

where the inequality uses $D_{\text{path}}(s_2, \pi) \geq D_{\text{path}}(s_1, \pi)$ and monotonicity of $F_T$ (as a CDF). Therefore $\mathcal{S}_{\text{dist}}(s_2, t) \geq \mathcal{S}_{\text{dist}}(s_1, t)$. $\blacksquare$(ii)

**Remark (Strict monotonicity).** For strict inequality, we additionally require $F_T$ to be strictly increasing on the interval $[t + D_{\text{path}}(s_1, \pi) - 1,\; t + D_{\text{path}}(s_2, \pi) - 1]$. This holds for continuous distributions (Uniform, Gaussian) and for discrete distributions with full support on $[T_{\min}, T_{\max}]$. $\blacksquare$

---

### Proposition 2 (Policy-Mismatch Bias and Stochastic Dominance)

The following result formalizes the claim that distributional stakes under the blended policy $\pi_\theta$ systematically underestimate urgency relative to prospective stakes under the deontic policy $\pi_d$. The proof proceeds via stochastic dominance of time-to-goal distributions and a coupling argument.

#### Setup and Assumptions

> **Proposition 2.** Consider a CDDP environment with finite horizon $H$ and the following properties:
>
> **(B1) Time-critical catastrophe.** The return decomposes as $Z^\pi(s) = G^\pi(s) + \Delta_{\text{cat}}^\pi(s)$, where $G^\pi(s)$ is the non-catastrophic component and $\Delta_{\text{cat}}^\pi(s) \leq 0$ is the catastrophe penalty. Catastrophic outcomes correspond to failing to reach the goal before a deadline: $Z^\pi(s) < c_{\text{cat}}$ if and only if the time-to-goal $D_{\text{path}}(s, \pi) > T_{\text{rem}}(s)$.
>
> **(B2) Path-shortening violations.** There exist states where constraint-violating actions provide shortcuts. Formally, define the **safe path length** $L_{\text{safe}}(s) = \min\{|\xi| : \xi \text{ is a path from } s \text{ to goal using only } \mathcal{A}_{\text{safe}} \text{ actions}\}$ and the **unrestricted path length** $D_{\min}(s) = \min\{|\xi| : \xi \text{ is a path from } s \text{ to goal}\}$. There exist states $s$ with:
>
> $$L_{\text{safe}}(s) > D_{\min}(s)$$
>
> **(B3) Learned exploitation.** The blended policy $\pi_\theta$ assigns positive probability to constraint-violating actions that reduce time-to-goal. That is, for states $s$ satisfying (B2), there exist $a \in \mathcal{A}_{\text{unsafe}}(s)$ with $\pi_\theta(a \mid s) > 0$ and $\mathbb{E}[D_{\text{path}}(s', \pi_\theta)] < \mathbb{E}[D_{\text{path}}(s'', \pi_\theta)]$ where $s' = T(s, a)$ is the successor under the violating action and $s'' = T(s, a_s)$ for some $a_s \in \mathcal{A}_{\text{safe}}(s)$.
>
> Define the **late-switch failure band**:
>
> $$\mathcal{F} = \{s \in \mathcal{S} : D_{\min}(s) < T_{\text{rem}}(s) < L_{\text{safe}}(s)\}$$
>
> Then:
>
> **(a) Weak inequality (general).** For all states $s$:
>
> $$P_{\text{cat}}(s;\, \pi_d) \geq P_{\text{cat}}(s;\, \pi_\theta)$$
>
> **(b) Strict inequality (failure band).** For all $s \in \mathcal{F}$:
>
> $$P_{\text{cat}}(s;\, \pi_d) > P_{\text{cat}}(s;\, \pi_\theta)$$
>
> Consequently, $\mathcal{S}_{\text{dist}}(s;\, \pi_\theta) < \mathcal{S}_{\text{pros}}(s)$ for all $s \in \mathcal{F}$.
>
> **(c) Stochastic dominance.** Under (B1)–(B3), the time-to-goal under $\pi_d$ first-order stochastically dominates that under $\pi_\theta$ in the failure band:
>
> $$D_{\text{path}}(s, \pi_d) \succeq_{\text{FSD}} D_{\text{path}}(s, \pi_\theta) \quad \text{for all } s \in \mathcal{F}$$
>
> That is, $\mathbb{P}(D_{\text{path}}(s, \pi_d) > k) \geq \mathbb{P}(D_{\text{path}}(s, \pi_\theta) > k)$ for all $k \in \mathbb{Z}_+$.

#### Proof

**Step 1: Action-space restriction and trajectory coupling.**

The deontic policy $\pi_d$ is obtained from $\pi_\theta$ by restricting to $\mathcal{A}_{\text{safe}}(s)$ at every state (Definition 7). We construct an explicit **coupling** of trajectories under the two policies.

Let $\{U_t\}_{t=0}^{H-1}$ be a sequence of i.i.d. uniform random variables on $[0,1]$ that serve as the shared source of randomness for action selection. At each time step $t$, both policies use the same $U_t$ to select an action via the inverse-CDF method:

- Under $\pi_\theta$: $a_t^\theta = F_{\pi_\theta(\cdot | s_t)}^{-1}(U_t)$, where $F_{\pi_\theta}$ is the CDF of the action distribution
- Under $\pi_d$: $a_t^d = F_{\pi_d(\cdot | s_t)}^{-1}(U_t)$, where $F_{\pi_d}$ is the CDF of the restricted distribution

This coupling produces paired trajectories $\xi^\theta = (s_0, a_0^\theta, s_1^\theta, \ldots)$ and $\xi^d = (s_0, a_0^d, s_1^d, \ldots)$ that start from the same initial state $s_0 = s$ and share the same random seed.

**Step 2: Time-to-goal ordering via path structure.**

We show that $D_{\text{path}}(s, \pi_d) \succeq_{\text{FSD}} D_{\text{path}}(s, \pi_\theta)$, i.e., the deontic policy is stochastically slower.

The deontic policy samples from $\mathcal{A}_{\text{safe}}(s) \subseteq \mathcal{A}$ at every state. The support of the trajectory distribution under $\pi_d$ is therefore a _subset_ of the trajectories available under $\pi_\theta$. In particular, any trajectory that uses a constraint-violating action at any step has probability zero under $\pi_d$ but potentially positive probability under $\pi_\theta$.

Under **(B3)**, the violating actions that $\pi_\theta$ can take lead to states with shorter expected time-to-goal. Consider the one-step decomposition of time-to-goal from state $s$:

$$\mathbb{E}[D_{\text{path}}(s, \pi)] = 1 + \sum_{a \in \mathcal{A}} \pi(a \mid s) \cdot \mathbb{E}[D_{\text{path}}(T(s,a), \pi)]$$

For the deontic policy, the sum ranges only over $\mathcal{A}_{\text{safe}}(s)$:

$$\mathbb{E}[D_{\text{path}}(s, \pi_d)] = 1 + \sum_{a \in \mathcal{A}_{\text{safe}}(s)} \frac{\pi_\theta(a \mid s)}{Z(s)} \cdot \mathbb{E}[D_{\text{path}}(T(s,a), \pi_d)]$$

For the blended policy:

$$\mathbb{E}[D_{\text{path}}(s, \pi_\theta)] = 1 + \sum_{a \in \mathcal{A}_{\text{safe}}(s)} \pi_\theta(a \mid s) \cdot \mathbb{E}[D_{\text{path}}(T(s,a), \pi_\theta)] + \sum_{a \in \mathcal{A}_{\text{unsafe}}(s)} \pi_\theta(a \mid s) \cdot \mathbb{E}[D_{\text{path}}(T(s,a), \pi_\theta)]$$

The key observation: $\pi_\theta$ allocates probability mass $1 - Z(s) = \sum_{a \in \mathcal{A}_{\text{unsafe}}} \pi_\theta(a \mid s) > 0$ to unsafe actions. By **(B3)**, these unsafe actions lead to states $s' = T(s, a)$ with $\mathbb{E}[D_{\text{path}}(s', \pi_\theta)] < \mathbb{E}[D_{\text{path}}(s'', \pi_\theta)]$ for some safe successor $s''$. The deontic policy redistributes this mass to safe actions (via normalization by $Z(s)$), which lead to weakly longer paths. Therefore:

$$\mathbb{E}[D_{\text{path}}(s, \pi_d)] \geq \mathbb{E}[D_{\text{path}}(s, \pi_\theta)]$$

The first-order stochastic dominance (not just mean ordering) follows from a standard inductive argument over the remaining horizon: at every step, $\pi_\theta$ has access to all actions in $\mathcal{A}$ while $\pi_d$ is restricted to $\mathcal{A}_{\text{safe}} \subseteq \mathcal{A}$. By the principle that enlarging the feasible set in a stochastic optimization problem can only improve (weakly) the distribution of outcomes, $D_{\text{path}}(s, \pi_d) \succeq_{\text{FSD}} D_{\text{path}}(s, \pi_\theta)$.

**Step 3: Catastrophe probability ordering.**

By **(B1)**, $P_{\text{cat}}(s;\, \pi) = \mathbb{P}(D_{\text{path}}(s, \pi) > T_{\text{rem}}(s))$. Since $D_{\text{path}}(s, \pi_d) \succeq_{\text{FSD}} D_{\text{path}}(s, \pi_\theta)$:

$$P_{\text{cat}}(s;\, \pi_d) = \mathbb{P}(D_{\text{path}}(s, \pi_d) > T_{\text{rem}}) \geq \mathbb{P}(D_{\text{path}}(s, \pi_\theta) > T_{\text{rem}}) = P_{\text{cat}}(s;\, \pi_\theta)$$

This proves **(a)**.

**Step 4: Strict inequality in the failure band.**

For $s \in \mathcal{F}$, i.e., $D_{\min}(s) < T_{\text{rem}}(s) < L_{\text{safe}}(s)$:

- **Under $\pi_d$:** Every trajectory uses only safe actions, so $D_{\text{path}}(s, \pi_d) \geq L_{\text{safe}}(s) > T_{\text{rem}}(s)$. Therefore $P_{\text{cat}}(s;\, \pi_d) = 1$.

- **Under $\pi_\theta$:** Since $D_{\min}(s) < T_{\text{rem}}(s)$, there exists at least one path to the goal within the deadline (necessarily using violations). By **(B3)**, $\pi_\theta$ follows such a path with positive probability. Therefore $P_{\text{cat}}(s;\, \pi_\theta) < 1$.

Combining: $P_{\text{cat}}(s;\, \pi_d) = 1 > P_{\text{cat}}(s;\, \pi_\theta)$. This proves **(b)**.

The stakes inequality follows by monotonicity of $\text{clip}(\cdot / P_{\text{ref}}, 0, 1)$:

$$\mathcal{S}_{\text{dist}}(s;\, \pi_\theta) = \text{clip}\!\left(\frac{P_{\text{cat}}(s;\, \pi_\theta)}{P_{\text{ref}}}\right) < \text{clip}\!\left(\frac{1}{P_{\text{ref}}}\right) = \mathcal{S}_{\text{pros}}(s)$$

Part **(c)** was established in Step 2. $\blacksquare$

#### Quantile Ordering (Connection to QR-DQN)

The stochastic dominance result has a direct implication for the quantile-based distributional critics used in practice. Let $\{\theta_i^d(s)\}_{i=1}^N$ and $\{\theta_i^\theta(s)\}_{i=1}^N$ denote the sorted quantile estimates from $Z_\psi^{\text{deon}}$ and $Z_\psi^{\text{free}}$ respectively.

> **Corollary 2a (Quantile Ordering).** Under (B1)–(B3), for all $s \in \mathcal{F}$ and for all quantile indices $i \in \{1, \ldots, N\}$:
>
> $$\theta_i^d(s) \leq \theta_i^\theta(s)$$
>
> That is, every quantile of the deontic return distribution is weakly below the corresponding quantile of the blended return distribution.

**Proof.** First-order stochastic dominance of $D_{\text{path}}(s, \pi_d)$ over $D_{\text{path}}(s, \pi_\theta)$ (Part (c) of Proposition 2) implies, via (B1), that $Z^{\pi_d}(s) \preceq_{\text{FSD}} Z^{\pi_\theta}(s)$ — the return distribution under the deontic policy is first-order stochastically dominated by the return under the blended policy (longer paths yield lower returns). First-order stochastic dominance is equivalent to pointwise quantile ordering: if $F \preceq_{\text{FSD}} G$, then $F^{-1}(\tau) \leq G^{-1}(\tau)$ for all $\tau \in (0,1)$. In the QR-DQN approximation with quantile fractions $\tau_i = (2i-1)/2N$, this yields $\theta_i^d(s) \leq \theta_i^\theta(s)$ for all $i$. $\blacksquare$

This quantile ordering means that the number of quantiles falling below $c_{\text{cat}}$ is weakly larger for the deontic critic than for the blended critic:

$$|\{i : \theta_i^d(s) < c_{\text{cat}}\}| \geq |\{i : \theta_i^\theta(s) < c_{\text{cat}}\}|$$

which directly implies $P_{\text{cat}}(s;\, \pi_d) \geq P_{\text{cat}}(s;\, \pi_\theta)$ in the quantile approximation, confirming consistency between the theoretical result and the finite-sample QR-DQN estimator.

#### Corollary 2b (Late-Switch Failure Mode)

> **Corollary 2b.** Under the conditions of Proposition 2, there exist threshold values $\tau$ such that for states $s \in \mathcal{F}$:
>
> $$\mathcal{S}_{\text{dist}}(s;\, \pi_\theta) < \tau \leq \mathcal{S}_{\text{pros}}(s)$$
>
> For such $\tau$, a CDDP using $\mathcal{S}_{\text{dist}}$ remains in deontic mode at $s$ (shield active), but since $L_{\text{safe}}(s) > T_{\text{rem}}(s)$, the goal cannot be reached under the shield. Catastrophe occurs with probability 1.
>
> A CDDP using $\mathcal{S}_{\text{pros}}$ correctly switches to utilitarian mode at $s$, lifting the shield and allowing the agent to reach the goal via constraint-violating shortcuts. Catastrophe occurs with probability strictly less than 1.

**Proof.** By Proposition 2(b), $P_{\text{cat}}(s;\, \pi_\theta) < 1 = P_{\text{cat}}(s;\, \pi_d)$ for $s \in \mathcal{F}$. Therefore $\mathcal{S}_{\text{dist}}(s;\, \pi_\theta) < \mathcal{S}_{\text{pros}}(s)$, and there exists $\tau$ in the interval $(\mathcal{S}_{\text{dist}}(s;\, \pi_\theta),\; \mathcal{S}_{\text{pros}}(s)]$. For any such $\tau$:

- Under $\mathcal{S}_{\text{dist}}$: $\mathcal{S}(s) = \mathcal{S}_{\text{dist}}(s;\, \pi_\theta) < \tau$, so the shield remains active. But $L_{\text{safe}}(s) > T_{\text{rem}}(s)$ (by definition of $\mathcal{F}$), so the shielded agent cannot reach the goal before the deadline. By (B1), $Z^{\pi_d}(s) < c_{\text{cat}}$ with certainty.
- Under $\mathcal{S}_{\text{pros}}$: $\mathcal{S}(s) = \mathcal{S}_{\text{pros}}(s) \geq \tau$, so the shield is lifted. Since $D_{\min}(s) < T_{\text{rem}}(s)$, a goal-reaching path exists, and by (B3), $\pi_\theta$ follows it with positive probability. $\blacksquare$

**Remark 1 (Magnitude of the bias).** In the failure band, the bias is maximal: $\mathcal{S}_{\text{pros}}(s) = \text{clip}(1/P_{\text{ref}}, 0, 1) = 1$ (for $P_{\text{ref}} \leq 1$), while $\mathcal{S}_{\text{dist}}(s;\, \pi_\theta)$ depends on the success probability of $\pi_\theta$'s shortcut-taking behavior. For a well-trained $\pi_\theta$ that reliably exploits violations, $P_{\text{cat}}(s;\, \pi_\theta)$ can be very small, making the gap $\mathcal{S}_{\text{pros}}(s) - \mathcal{S}_{\text{dist}}(s;\, \pi_\theta) \approx 1$.

**Remark 2 (Beyond the failure band).** Outside $\mathcal{F}$, the weak inequality (Part (a)) still holds, but may not be strict. In states where $T_{\text{rem}} \geq L_{\text{safe}}$ (ample time), both policies reach the goal with high probability and $P_{\text{cat}}(s;\, \pi_d) \approx P_{\text{cat}}(s;\, \pi_\theta) \approx 0$. In states where $T_{\text{rem}} \leq D_{\min}$ (insufficient time regardless), both policies fail and $P_{\text{cat}}(s;\, \pi_d) \approx P_{\text{cat}}(s;\, \pi_\theta) \approx 1$. The bias is concentrated in the failure band $\mathcal{F}$, which is precisely where it matters for switching correctness.

**Remark 3 (Generality beyond gridworlds).** The proof structure extends to any CDDP environment where (i) catastrophe is time-critical, (ii) constraint-violating actions provide path-shortening opportunities, and (iii) the learned policy exploits them. The stochastic dominance argument requires only that enlarging the action set (from $\mathcal{A}_{\text{safe}}$ to $\mathcal{A}$) weakly improves the time-to-goal distribution, which holds whenever unsafe actions are not strictly dominated in terms of goal-reaching speed.

### Theorem 2 (Utilitarian Recovery under High Stakes)

> **Theorem 2.** Consider a CDDP with hard switching. If $\mathcal{S}(s_t) \geq \tau$ for all $t$ in trajectory $\xi$, then:
>
> $$J_R(\pi_{\text{CDDP}} \mid \xi) = J_R(\pi_\theta \mid \xi)$$
>
> where $\pi_\theta$ is the unshielded base policy, trained to maximize $J_R$ with modulated cost $\tilde{C}(s,a) = 0$ for all $(s,a)$ in $\xi$ (since $h = 0$). Up to the approximation quality of $\pi_\theta$, this equals $J_R(\pi^*)$, the unconstrained optimal return.

**Proof.** When $\mathcal{S}(s_t) \geq \tau$ for all $t$, Definition 3 gives $\pi_{\text{CDDP}}(a|s) = \pi_\theta(a|s)$ — the shield is inactive. The trajectory distribution under $\pi_{\text{CDDP}}$ is identical to that under $\pi_\theta$. Therefore $J_R(\pi_{\text{CDDP}} \mid \xi) = J_R(\pi_\theta \mid \xi)$.

Furthermore, in the training Lagrangian, $\tilde{C}(s_t, a_t) = h(\mathcal{S}(s_t), \tau) \cdot C(s_t, a_t) = 0 \cdot C(s_t, a_t) = 0$ for all $(s_t, a_t)$ in these states. The constraint $J_{\tilde{C}} \leq d$ is trivially satisfied, so the Lagrangian optimizes pure reward, producing a policy that approximates the unconstrained optimum $\pi^*$. $\blacksquare$

---

### Theorem 3 (Impossibility of State-Independent Strategies)

The core theoretical claim of CDDP is that state-dependent constraint modulation achieves a combination of properties that no state-independent strategy can replicate. We formalize this as a structural impossibility result that does not depend on a specific utility function.

> **Definition (State-Independent Strategy).** A strategy is _state-independent_ if it applies the same constraint enforcement mechanism in every state. Formally, a state-independent shielded policy has shield activation function $h(s) = h_0 \in \{0, 1\}$ for all $s$ (constant). This encompasses:
>
> - _Permanent shield_ ($h_0 = 1$): constraints enforced everywhere (pure deontological)
> - _No shield_ ($h_0 = 0$): constraints never enforced (pure utilitarian)
> - _Fixed-$\lambda$ CMDP_: a single scalar Lagrange multiplier $\lambda$ applied uniformly

> **Theorem 3.** Consider a CDDP environment satisfying:
>
> **(E1)** The state space partitions into $\mathcal{S}_{\text{low}} = \{s : \mathcal{S}(s) < \tau\}$ and $\mathcal{S}_{\text{high}} = \{s : \mathcal{S}(s) \geq \tau\}$, both non-empty and visited with positive probability: $p_L = \mathbb{P}(s \in \mathcal{S}_{\text{low}}) > 0$, $p_H = \mathbb{P}(s \in \mathcal{S}_{\text{high}}) > 0$.
>
> **(E2)** In $\mathcal{S}_{\text{low}}$, there exists at least one safe path (through $\mathcal{A}_{\text{safe}}$ actions only) that reaches the goal.
>
> **(E3)** In $\mathcal{S}_{\text{high}}$, every path that reaches the goal before catastrophe requires at least one constraint-violating action. That is, for any policy $\pi$ with $C(s_t, a_t) = 0$ for all $t$, the goal is not reached before catastrophe.
>
> **(E4, Full Support — FS)** $\pi_\theta(a|s) > 0$ for all $a \in \mathcal{A}$, $s \in \mathcal{S}$.
>
> Then no state-independent strategy can simultaneously achieve both:
>
> **(P1)** Pointwise low-stakes safety: $C(s_t, a_t) = 0$ a.s. for all $t$ where $s_t \in \mathcal{S}_{\text{low}}$
>
> **(P2)** High-stakes goal achievement: the goal is reached before catastrophe in $\mathcal{S}_{\text{high}}$ with positive probability
>
> The CDDP policy $\pi_{\text{CDDP}}$ (Definition 3) achieves both (P1) and (P2) simultaneously.

**Proof.**

**Part I: No state-independent strategy achieves (P1) $\wedge$ (P2).**

We consider each class of state-independent strategy:

**Case 1: Permanent shield ($h_0 = 1$, constraints enforced everywhere).**

The shield enforces $C(s_t, a_t) = 0$ for all $t$ and all $s$, so (P1) holds trivially. However, in $\mathcal{S}_{\text{high}}$, the shield blocks all violating actions. By (E3), every path to the goal from $\mathcal{S}_{\text{high}}$ requires at least one violation. Since violations are blocked, the goal cannot be reached. Therefore (P2) fails. $\blacksquare$(Case 1)

**Case 2: No shield ($h_0 = 0$, no constraint enforcement).**

The policy $\pi_\theta$ is unconstrained and may reach the goal in $\mathcal{S}_{\text{high}}$, satisfying (P2). However, by (FS), $\pi_\theta(a \mid s) > 0$ for all $a$, including constraint-violating actions. For any $s \in \mathcal{S}_{\text{low}}$ where $\mathcal{A}_{\text{unsafe}}(s) = \mathcal{A} \setminus \mathcal{A}_{\text{safe}}(s) \neq \emptyset$:

$$\mathbb{P}(C(s_t, a_t) = 1 \mid s_t = s) = \sum_{a \in \mathcal{A}_{\text{unsafe}}(s)} \pi_\theta(a \mid s) > 0$$

Therefore (P1) fails: there is positive probability of violation in low-stakes states. $\blacksquare$(Case 2)

**Case 3: Fixed-$\lambda$ CMDP (single scalar multiplier).**

A CMDP with scalar $\lambda$ produces a policy $\pi_\lambda$ that solves $\max_\pi J_R(\pi) - \lambda J_C(\pi)$. The key structural constraint: the same $\lambda$ creates the same violation incentive in every state.

- If $\lambda$ is large enough that $\pi_\lambda$ avoids violations in $\mathcal{S}_{\text{low}}$ (approaching P1 in expectation), then in $\mathcal{S}_{\text{high}}$ the same $\lambda$ also discourages violations. By (E3), violating is necessary for goal achievement. With high $\lambda$, the agent avoids violations in $\mathcal{S}_{\text{high}}$ too, and (P2) fails.
- If $\lambda$ is small enough that $\pi_\lambda$ commits violations in $\mathcal{S}_{\text{high}}$ to reach the goal (achieving P2), then in states $s \in \mathcal{S}_{\text{low}}$ where violating has any reward advantage (even marginal), the same small $\lambda$ permits violations and (P1) fails.
- More precisely: under (FS), for any finite $\lambda$, $\pi_\lambda(a|s)$ assigns positive probability to every action (since the Lagrangian-penalized reward is finite for all actions, and the softmax over finite values gives positive probabilities). Therefore $\mathbb{P}(C(s_t, a_t) = 1 \mid s_t \in \mathcal{S}_{\text{low}}) > 0$ for any finite $\lambda$, and (P1) — which requires _almost sure_ zero violations — cannot hold.

In all three cases, (P1) $\wedge$ (P2) cannot be simultaneously achieved. $\blacksquare$(Part I)

**Part II: The CDDP achieves (P1) $\wedge$ (P2).**

**(P1):** By Theorem 1, for any $s_t \in \mathcal{S}_{\text{low}}$, the shield is active and $C(s_t, a_t) = 0$ a.s. ✓

**(P2):** For $s_t \in \mathcal{S}_{\text{high}}$, the shield is lifted ($h = 0$), and $\pi_\theta$ acts without constraint. By (FS) and the fact that the Lagrangian with $\tilde{C} = 0$ in $\mathcal{S}_{\text{high}}$ places no penalty on violations, the policy can learn to reach the goal via violating paths. The probability of reaching the goal is determined by the quality of $\pi_\theta$, which can be made positive (and in practice high) through training. ✓

Therefore $\pi_{\text{CDDP}}$ achieves (P1) $\wedge$ (P2). $\blacksquare$

**Remark 1 (Scope of the result).** This theorem is purely structural: it characterizes what the CDDP _can_ achieve that no state-independent strategy can, without reference to any specific utility function. The impossibility arises from the fundamental conflict between pointwise safety (which requires blocking violating actions) and goal achievement under constraint-violating-only paths (which requires permitting violating actions).

**Remark 2 (Relation to optimal CMDP policies).** A standard CMDP with carefully tuned $\lambda$ can approximate the desired behavior in _expectation_ — mostly avoiding violations in $\mathcal{S}_{\text{low}}$ and mostly committing them in $\mathcal{S}_{\text{high}}$. But "mostly" is not "always": under (FS), any stochastic policy with finite $\lambda$ has positive violation probability in every state. The CDDP's shield converts "mostly safe" into "always safe" in $\mathcal{S}_{\text{low}}$, which no purely Lagrangian method can achieve.

**Remark 3 (Not a claim about expected reward).** This theorem does not claim the CDDP achieves higher expected reward than any CMDP policy — a well-tuned CMDP can achieve high expected reward. The claim is about the _achievable property set_: simultaneous pointwise safety in low-stakes AND goal achievement in high-stakes is a property combination outside the reach of state-independent strategies.

---

### Corollary 1 (Convergence of Stakes-Modulated Lagrangian)

> **Corollary 1.** _(Follows from Paternain et al. (2019) and Tessler et al. (2018).)_
>
> Under the following assumptions:
>
> - (A1) $|R| \leq R_{\max}$, $|C| \leq C_{\max}$ (bounded reward and cost)
> - (A2) Slater's condition: $\exists\; \tilde{\pi}$ s.t. $J_{\tilde{C}}(\tilde{\pi}) < d$ (strict feasibility for modulated cost)
> - (A3) $\mathcal{S}(\cdot)$ is a fixed function of state (not learned jointly with the policy)
>
> The modulated cost $\tilde{C}(s,a) = h(\mathcal{S}(s), \tau) \cdot C(s,a)$ satisfies $|\tilde{C}| \leq C_{\max}$ (since $h \in [0,1]$). The CDDP Lagrangian $L_{\text{CDDP}}(\pi, \lambda) = J_R(\pi) - \lambda(J_{\tilde{C}}(\pi) - d)$ is structurally a standard CMDP Lagrangian with cost function $\tilde{C}$.
>
> By Theorem 1 of Paternain et al. (2019), the CMDP with cost $\tilde{C}$ has zero duality gap. The primal-dual algorithm:
>
> $$\pi_{k+1} = \pi_k + \alpha_k \nabla_\pi L_{\text{CDDP}}(\pi_k, \lambda_k)$$
>
> $$\lambda_{k+1} = \left[\lambda_k + \beta_k \left(J_{\tilde{C}}(\pi_k) - d\right)\right]_+$$
>
> with step sizes $\alpha_k, \beta_k = O(1/\sqrt{K})$ converges to an $\varepsilon$-saddle point in $O(1/\varepsilon^2)$ iterations.

**Proof.** The modulated cost $\tilde{C}(s,a) = h(\mathcal{S}(s), \tau) \cdot C(s,a)$ is a fixed, bounded, state-dependent function (by A1, A3, and $h \in [0,1]$). The CDDP Lagrangian is identical in structure to a CMDP Lagrangian with cost function $\tilde{C}$. All conditions of Theorem 1 in Paternain et al. (2019) — bounded costs, Slater's condition (A2), linearity of $J_{\tilde{C}}$ in the occupancy measure — are satisfied. The convergence rate follows directly. $\blacksquare$

**Remark (Novelty).** This corollary is not a new convergence result. The contribution of CDDP is not in the optimization algorithm but in the _formulation_: the definition of $\tilde{C}$ via stakes modulation, the safety shield for hard guarantees, and the resulting performance dominance (Theorem 3). The Lagrangian is a tool; the framework is the contribution.

---

### Proposition 2 (Relationship to Existing Frameworks)

> **Proposition 2.** The CDDP Lagrangian component ($\tilde{C} = h(s) \cdot C$) is a state-dependent cost CMDP, a structure that exists in the literature. The genuine novelty of CDDP lies in the specific combination of:
>
> - (a) a distributional-RL-derived stakes functional $\mathcal{S}(s)$ determining $h(s)$,
> - (b) a mode-switched safety shield providing pointwise guarantees (Theorem 1), and
> - (c) the explicit goal of implementing threshold deontology.
>
> No prior framework combines these three elements simultaneously.

We compare against CVaR-constrained MDPs (Chow et al., 2017) and state-dependent cost CMDPs to clarify the relationship:

**Relationship to CVaR-constrained MDPs.**

A CVaR-CMDP has objective $\max_\pi \mathbb{E}[R^\pi]$ s.t. $\text{CVaR}_\alpha(Z_C^\pi) \leq d$, applying a single global risk measure.

**(1) State-dependent vs. global modulation.** The CDDP modulates constraints per-state via $h(\mathcal{S}(s), \tau)$. This is a state-dependent cost CMDP — replacing $C(s,a)$ with $h(s) \cdot C(s,a)$ is a known construction. However, in the CDDP, $h(s)$ is specifically derived from distributional tail risk (Definition 6), creating a closed loop between the return distribution and constraint enforcement. This specific derivation of $h$ from distributional RL is novel.

**(2) Hard safety via shield.** The CDDP provides pointwise constraint satisfaction (Theorem 1) via the safety shield. CVaR-constrained MDPs, like all Lagrangian-based methods, bound costs in a distributional sense — even $\text{CVaR}_\alpha$ with $\alpha \to 0$ does not guarantee $C(s_t, a_t) = 0$ at individual steps. The shield is not part of any CVaR-CMDP framework.

**(3) Bimodal structure.** The CDDP explicitly switches between two qualitative behavioral modes (shield on/off), controlled by the stakes functional. CVaR-CMDPs produce a single policy with a smooth risk-reward tradeoff. The discrete mode switch formalizes the philosophical concept of threshold deontology, which has no analogue in risk-constrained optimization.

**Relationship to state-dependent cost CMDPs.** Replacing $C$ with $h(s) \cdot C$ in a standard CMDP yields a state-dependent cost CMDP, and the Lagrangian convergence (Corollary 1) follows directly. The CDDP's contribution is not the cost-modulation mechanism itself but: (i) deriving $h(s)$ from distributional tail risk, (ii) the shield for hard guarantees, and (iii) the threshold-deontological motivation that determines the specific form of $h$.

| Property                | CVaR-CMDP               | State-dep. cost CMDP      | CDDP                                      |
| ----------------------- | ----------------------- | ------------------------- | ----------------------------------------- |
| Constraint modulation   | Global (fixed $\alpha$) | State-dependent $h(s)$    | State-dependent $h(\mathcal{S}(s), \tau)$ |
| Source of $h(s)$        | N/A                     | Arbitrary / hand-designed | Distributional RL (CVaR/tail risk)        |
| Safety guarantee        | Distributional          | In expectation            | Pointwise (shield, Thm 1)                 |
| Policy structure        | Single smooth           | Single smooth             | Bimodal (mode-switched)                   |
| Philosophical grounding | Risk aversion           | None specific             | Threshold deontology                      |

$\blacksquare$

---

## 5. CDDP-PPO Algorithm

### 5.1 Training Stability: The Warmup Problem

The prospective stakes functional $\mathcal{S}_{\text{pros}}(s)$ depends on the deontic distributional critic $Z_{\psi}^{\text{deon}}$, which is initialized randomly and trained concurrently with the policy. In early iterations, the quantile estimates $\{\theta_i(s)\}$ are meaningless, so $P_{\text{cat}}(s;\, \pi_d)$ is noise, and the shield activates/deactivates randomly. This corrupts training trajectories, which in turn produces badly trained critics — a feedback loop.

**Solution: Two-phase training with analytical warmup.**

> **Phase 1 (Warmup, iterations $1$ to $K_w$):** Use the analytical stakes functional $\mathcal{S}_{\text{analytic}}(s) = \text{clip}(D_{\min}/T_{\text{rem}},\; 0,\; 1)$ instead of $\mathcal{S}_{\text{pros}}$. This provides a reliable, domain-specific stakes signal while both distributional critics ($Z_{\psi}^{\text{free}}$, $Z_{\psi}^{\text{deon}}$) are trained passively from collected trajectories. During this phase, the critics are updated but not used for shield gating.
>
> **Phase 2 (Full CDDP, iterations $K_w+1$ to $K$):** Switch to the prospective stakes functional $\mathcal{S}_{\text{pros}}(s)$ computed from $Z_{\psi}^{\text{deon}}$. By this point, $Z_{\psi}^{\text{deon}}$ has been trained on $K_w$ iterations of deontic-mode data and provides meaningful quantile estimates. Optionally, blend: $\mathcal{S}(s) = (1-\alpha_k) \cdot \mathcal{S}_{\text{analytic}}(s) + \alpha_k \cdot \mathcal{S}_{\text{pros}}(s)$ with $\alpha_k$ ramping from $0$ to $1$ over a transition window.

This two-phase approach ensures stable early training while allowing the agent to eventually rely on the learned, general-purpose prospective stakes.

### 5.2 Algorithm

```
Input: CDDP = (S, A, T, R, C, S(·), τ, h, d), warmup iterations K_w
Initialize: policy π_θ, value critic V_φ,
            distributional critic Z_ψ^free (QR-DQN, N quantiles, evaluates π_θ),
            distributional critic Z_ψ^deon (QR-DQN, N quantiles, evaluates π_d),
            Lagrange multiplier λ ≥ 0

For each iteration k = 1, ..., K:
  1. Collect trajectories using the shielded policy π_CDDP (Definition 3)
     - Tag each transition with its mode: deontic (shield active) or utilitarian

  2. Update distributional critics:
       Z_ψ^free ← quantile regression on all transitions
       Z_ψ^deon ← quantile regression on deontic-mode transitions

  3. Compute stakes for each visited state:
       if k ≤ K_w:   (warmup phase)
           S(s_t) ← S_analytic(s_t)      (domain-specific, reliable)
       else:          (full CDDP phase)
           S(s_t) ← S_pros(s_t; Z_ψ^deon, c_cat, P_ref)   (prospective, learned)

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

Return: π_θ, Z_ψ^free, Z_ψ^deon
```

**Remark (Generality of warmup).** The analytical warmup is domain-specific (requires knowing $T_{\text{rem}}$ and $D_{\min}$). In domains without a natural analytical stakes function, alternative warmup strategies include: (i) conservative warmup with $h \equiv 1$ (full shield) for $K_w$ iterations, then transitioning; (ii) pre-training $Z_{\psi}^{\text{deon}}$ on random-policy rollouts before joint optimization; or (iii) using a high $P_{\text{ref}}$ in $\mathcal{S}_{\text{pros}}$ to make the stakes signal conservative during early unreliable estimates.

---

## 6. Connection to Existing Frameworks

| Framework       | Constraint mechanism                     | Safety guarantee                           | Modes              |
| --------------- | ---------------------------------------- | ------------------------------------------ | ------------------ |
| MDP (PPO)       | None ($h \equiv 0$)                      | None                                       | Utilitarian only   |
| CMDP (PPO-Lag)  | Fixed Lagrangian ($h \equiv 1$)          | In expectation                             | Deontological only |
| CVaR-CMDP       | Global tail risk                         | Distributional                             | Single smooth      |
| Shielded MDP    | Runtime filter                           | Pointwise                                  | Constrained only   |
| **CDDP (ours)** | **Stakes-modulated shield + Lagrangian** | **Pointwise when $\mathcal{S}(s) < \tau$** | **Bimodal**        |

> **Proposition 3 (Generalization).** Setting $h \equiv 0$ ($\tau = 0$) recovers the unconstrained MDP. Setting $h \equiv 1$ ($\tau = 1$) with the shield always active recovers a shielded CMDP. For $\tau \in (0, 1)$, the CDDP represents a novel interpolation.

---

## 7. Hostage Gridworld Instantiation

| CDDP component                 | Gridworld mapping                                                                                                                                                                                    |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| $\mathcal{S}$                  | (agent_pos, hostage_pos, civilian_pos[], bomb_timer, elapsed_time)                                                                                                                                   |
| $\mathcal{A}$                  | $\{\text{Up, Down, Left, Right}\}$                                                                                                                                                                   |
| $R(s,a)$                       | Reward shaping $-$ civilian penalty $+$ hostage bonus                                                                                                                                                |
| $C(s,a)$                       | $\mathbb{1}\{\text{action moves agent onto civilian tile}\}$                                                                                                                                         |
| $\mathcal{A}_{\text{safe}}(s)$ | $\{a : \text{next\_pos}(s, a) \notin \text{civilian\_positions}\}$                                                                                                                                   |
| $\mathcal{S}(s)$               | $\mathcal{S}_{\text{pros}}(s) = \text{clip}(P_{\text{cat}}(s;\, \pi_d) / P_{\text{ref}},\; 0,\; 1)$ (prospective, canonical); $\text{clip}(D_{\min} / T_{\text{rem}},\; 0,\; 1)$ (analytical warmup) |
| $c_{\text{cat}}$               | $-100$ (return level associated with bomb detonation)                                                                                                                                                |
| $\tau$                         | Tuned via grid search; expected range $[0.3, 0.7]$                                                                                                                                                   |
| $d$                            | $0$ (zero violations tolerated in low-stakes)                                                                                                                                                        |

---

## 8. References

1. Leike et al. (2017). "AI Safety Gridworlds." arXiv:1711.09883.
2. Altman (1999). _Constrained Markov Decision Processes._ Chapman & Hall.
3. Bellemare et al. (2017). "A Distributional Perspective on RL." ICML.
4. Dabney et al. (2018). "Distributional RL with Quantile Regression." AAAI.
5. Tessler et al. (2018). "Reward Constrained Policy Optimization." ICLR.
6. Paternain et al. (2019). "Constrained RL Has Zero Duality Gap." NeurIPS.
7. Chow et al. (2017). "Risk-Constrained RL with CVaR." NeurIPS.
8. Alshiekh et al. (2018). "Safe RL via Shielding." AAAI.
9. Dalal et al. (2018). "Safe Exploration in Continuous Action Spaces." arXiv.
10. Alexander (2000). "Deontology at the Threshold." San Diego Law Review.
