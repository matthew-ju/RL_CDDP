# Contextual Deontic Decision Process (CDDP): Formal Framework

---

## 1. Preliminaries & Unifying Theory

### 1.1 Constrained Markov Decision Process (CMDP)

A CMDP is defined as a tuple $(\mathcal{S}, \mathcal{A}, T, R, C, d)$ where:

- $\mathcal{S}$ is a state space
- $\mathcal{A}$ is an action space
- $T: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to [0, 1]$ is the transition kernel, $T(s'|s, a)$
- $R: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ is the reward function
- $C: \mathcal{S} \to \mathbb{R}$ is a differentiable state-dependent cost function
- $d \geq 0$ is the static cost budget

The standard CMDP objective is to maximize expected reward subject to static cost bounds:

$$\max_{\pi} \; J_R(\pi) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{H-1} \gamma^t R(s_t, a_t) \right]$$

$$\text{s.t.} \quad J_C(\pi) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{H-1} \gamma^t C(s_t, a_t) \right] \leq d$$

**Key limitation:** Standard CMDPs enforce a static bound $d$. This forces a global, rigid compromise across the entire state space, preventing the agent from exhibiting dynamic, context-aware behaviors where safety limits adjust based on the urgency of the situation. 

### 1.2 Distributional Reinforcement Learning

In distributional RL, instead of learning only the expected return $Q^\pi(s, a) = \mathbb{E}[Z^\pi(s, a)]$, we learn the full return distribution $Z^\pi(s, a)$. This allows us to quantify risk and measure the explicit probability of catastrophic failure under a policy $\pi$: 

$$P_{\text{cat}}(s; \pi) = \mathbb{P}(Z^\pi(s) < c_{\text{cat}})$$

where $c_{\text{cat}}$ is a defined catastrophe threshold.

### 1.3 The Stakes Functional and Dynamic Boundary

To resolve the limitation of static CMDPs, we introduce the *Contextual Deontic Decision Process* (CDDP). The core mechanism is parameterized by a bounded stakes functional $\mathcal{S}(s) \in [0, 1]$ and a dynamic boundary $d(\mathcal{S}(s))$. The stakes functional evaluates the situational urgency or catastrophe probability.

The unifying mathematical philosophy of CDDP is to formulate Threshold Deontology as a smoothly expanding continuous constraint boundary. The safety rules are never "turned off." Instead, the mathematical boundary of the constraint physically expands in exact proportion to the situational stakes:

$$d(\mathcal{S}(s)) = \max(0, \alpha(\mathcal{S}(s) - \tau))$$

*(Note: this formulation is $C^0$ continuous but has a non-differentiable kink at $\mathcal{S}(s) = \tau$. The discrete instantiation in Section 3 replaces this with a $C^\infty$ softplus approximation to enable gradient-based training.)*

where $\tau \in (0, 1)$ is the threshold at which catastrophic pressure overrides nominal bounds, and $\alpha > 0$ defines the expansion rate. When $\mathcal{S}(s) < \tau$, the boundary remains a strict zero-tolerance wall ($d=0$). As $\mathcal{S}(s)$ exceeds $\tau$, the safe manifold expands up to a maximum boundary of $\alpha(1-\tau)$, permitting precisely the minimum necessary violation to circumvent catastrophe.

---

## 2. Proposed Theoretical Extension for Continuous Control (Future Work)

> **Disclaimer:** The continuous formulation described in this section represents a proposed theoretical extension. The actual empirical results and current architecture in this specific repository are strictly evaluated using the **Discrete Instantiation** (Section 3). Reviewers should not expect to find continuous physics engines (e.g., MuJoCo or PyBullet) or continuous QP solvers within the current codebase.

For real-world continuous control, the CDDP is formulated as a Stakes-Conditioned Differentiable Safe RL Model leveraging Control Barrier Functions (CBFs) and Quadratic Programs (QPs).

The CDDP continuous policy is defined as the end-to-end differentiable composition of a nominal neural network $\pi_\theta(a \mid s, \mathcal{S}(s))$ and a continuous QP projection layer. The network outputs a nominal action $a_{\text{nom}}$, which is projected onto the dynamically expanding safe manifold to obtain the executed action $a^*$:

$$a^* = \arg\min_a \frac{1}{2} ||a - a_{\text{nom}}||^2$$

$$\text{s.t.} \quad \nabla C(s) \cdot a \le d(\mathcal{S}(s))$$

During training, the objective is pure reward maximization. Gradients flow backward through the Karush-Kuhn-Tucker (KKT) conditions of the active QP constraints via implicit differentiation (Amos & Kolter, 2017), enabling stable optimization without the inner-loop instability of iterative Lagrangian methods. See Corollary 1 for a precise characterization of the gradient flow, including its behaviour at the boundary transition point $\mathcal{S}(s) = \tau$.

---

## 3. Discrete Instantiation (The Empirical Bridge)

To empirically isolate the Mission-Feasibility Gap without the confounding dynamics of continuous physics simulators, we derive a discrete analogue of CDDP suitable for gridworld environments.

> **Definition (Mission-Feasibility Gap).** The *Mission-Feasibility Gap* occurs when achieving a critical goal strictly requires violating a nominal safety constraint. Formally, it is the set of states $\mathcal{S}_{\text{high}}$ in which every path to the goal within the deadline passes through at least one action $a \in \mathcal{A}_{\text{unsafe}}(s)$.

In discrete action spaces, continuous QP projections are inapplicable. Instead, we implement the expanding dynamic boundary via **Differentiable Logit Masking**. The core neural network outputs continuous action logits $l_a$ for each discrete action $a \in \mathcal{A}$. Let $\mathcal{A}_{\text{unsafe}}(s)$ be the set of constraint-violating actions. We apply a stakes-modulated penalty to the logits before the softmax operator:

$$\tilde{l}_a = \begin{cases} l_a - \text{penalty}(a, \mathcal{S}(s)) & \text{if } a \in \mathcal{A}_{\text{unsafe}}(s) \\ l_a & \text{otherwise} \end{cases}$$

Rather than inheriting the $\max(0, \cdot)$ formulation from the continuous case — which is $C^0$ but not $C^1$ at $\tau$ — the discrete instantiation replaces $d(\mathcal{S}(s))$ with a Softplus approximation to guarantee $C^\infty$ differentiability throughout training:

$$d_{\text{soft}}(\mathcal{S}(s)) = \frac{1}{\alpha'} \ln\!\left(1 + \exp\!\left(\alpha'(\mathcal{S}(s) - \tau)\right)\right)$$

where $\alpha' > 0$ controls the sharpness of the transition. As $\alpha' \to \infty$, $d_{\text{soft}}$ converges pointwise to $\max(0, \alpha(\mathcal{S}(s) - \tau))$. For practical values of $\alpha'$, $d_{\text{soft}} > 0$ everywhere, meaning the penalty

$$\text{penalty}(a, \mathcal{S}(s)) = \frac{\beta}{d_{\text{soft}}(\mathcal{S}(s)) + \epsilon}$$

is finite and smooth at all $\mathcal{S}(s) \in [0,1]$. The penalty remains large (effectively infinite in the softmax) for $\mathcal{S}(s) \ll \tau$, suppressing unsafe action probability to approximately zero. As $\mathcal{S}(s)$ exceeds $\tau$, the penalty decays continuously, allowing the softmax to differentiably assign nonzero probability to unsafe actions.

---

## 4. Estimating Stakes via Distributional RL

To accurately estimate the stakes functional $\mathcal{S}(s)$, CDDP employs a two-critic Distributional RL architecture.

- **Utilitarian Critic $Z_\psi^{\text{free}}(s, a)$:** Evaluates the distributional action-value under the unconstrained nominal policy. The expected value $Q^{\text{free}}(s,a) = \mathbb{E}[Z_\psi^{\text{free}}(s,a)]$ is used to construct the advantage estimate for actor training.

Concretely, $Z_\psi^{\text{free}}(s, a)$ provides the distributional baseline for policy gradient computation, where we extend the utilitarian critic to be action-conditional. Define the expected action-value as:

$$Q^{\text{free}}(s, a) = \mathbb{E}\!\left[Z_\psi^{\text{free}}(s, a)\right]$$

and the state-value baseline as:

$$V^{\text{free}}(s) = \mathbb{E}\!\left[Z_\psi^{\text{free}}(s, \pi_\theta(s))\right]$$

The actor's training objective is then:

$$\mathcal{L}_{\text{actor}}(\theta) = -\mathbb{E}_{s, a \sim \pi_\theta}\!\left[A^{\text{free}}(s, a)\right]$$

where $A^{\text{free}}(s, a) = Q^{\text{free}}(s, a) - V^{\text{free}}(s)$ is the advantage estimated from the utilitarian critic. During environment interaction, the executed action is either projected through the QP layer (continuous instantiation) or passed through Differentiable Logit Masking (discrete instantiation) before the environment step. The deontic critic $Z_\psi^{\text{deon}}$ does not appear in the actor loss; it is used exclusively for stakes estimation.

**Action semantics in the actor loss.** In both instantiations, $a$ in the expectation $\mathbb{E}_{s, a \sim \pi_\theta}[\cdot]$ refers to the **executed action** $a^*$ — the action received by the environment after projection (continuous) or logit masking (discrete). The advantage $A^{\text{free}}(s, a^*)$ therefore evaluates the actual environmental consequence of the policy's output, not the unconstrained nominal action. Policy gradients flow backward from $\mathcal{L}_{\text{actor}}$ through $a^*$ and into $\theta$ via:

- **Discrete:** Direct backpropagation through the differentiable softmax and logit masking operations, which are smooth in $\theta$ by Theorem 2 and Corollary 1.
- **Continuous:** Implicit differentiation of the KKT conditions of the QP projection (Amos \& Kolter, 2017), as described in Corollary 1. The gradient $\frac{\partial a^*}{\partial a_{\text{nom}}}$ is well-defined except at the measure-zero boundary $\mathcal{S}(s) = \tau$, where the $C^0$-but-not-$C^1$ nature of $d(\mathcal{S}(s))$ introduces a discontinuity in the Jacobian, as noted in Corollary 1.

This explicit gradient path ensures that the actor learns to output nominal actions $a_{\text{nom}}$ that, after projection, maximize task reward while the safety layer enforces the dynamic boundary.

- **Deontic Critic $Z_\psi^{\text{deon}}(s)$:** Evaluates the distributional value under the assumption of strict constraint adherence ($d=0$).

The stakes are defined as **Prospective Stakes** $\mathcal{S}_{\text{pros}}(s)$, which evaluate the future inevitability of catastrophe if the agent were to strictly adhere to the constraints:

$$\mathcal{S}_{\text{pros}}(s) = \text{clip}\left(\frac{P_{\text{cat}}(s;\, \pi_d)}{P_{\text{ref}}},\; 0,\; 1\right)$$

where $P_{\text{cat}}(s;\, \pi_d)$ is derived directly from the deontic critic $Z_\psi^{\text{deon}}(s)$, and $P_{\text{ref}} \in (0, 1]$ is a hyperparameter representing the baseline catastrophe probability tolerance — the level of catastrophic risk deemed acceptable under strict constraint adherence. In all experiments reported in this paper, we set $P_{\text{ref}} = 0.05$, representing a 5% catastrophic failure tolerance under strict constraint adherence. Sensitivity to this value is discussed in Appendix~X. This explicitly measures the catastrophic risk of rule-following, bounded strictly within $[0, 1]$.

---

## 5. Theoretical Guarantees

### Theorem 1 (Continuous Adherence)

**Theorem 1.** Assuming a bounded action space where QP feasibility holds — i.e., for all $s \in \mathcal{S}$ there exists at least one feasible $a$ satisfying $\nabla C(s) \cdot a \le d(\mathcal{S}(s))$ — let $a^*$ be the action produced by the differentiable QP projection layer. Then the agent exactly bounds its constraint violations to the dynamic boundary $d(\mathcal{S}(s))$.

**Proof.**
The executed action $a^*$ is the exact optimum of the continuous QP:
$$a^* = \arg\min_a \frac{1}{2} ||a - a_{\text{nom}}||^2 \quad \text{s.t.} \quad \nabla C(s) \cdot a \le d(\mathcal{S}(s))$$
By the feasibility assumption, such an $a^*$ exists. The hard constraint ensures $\nabla C(s) \cdot a^* \le d(\mathcal{S}(s))$. When $\mathcal{S}(s) < \tau$, $d(\mathcal{S}(s)) = \max(0, \alpha(\mathcal{S}(s) - \tau)) = 0$, guaranteeing exactly zero violations. When $\mathcal{S}(s) \ge \tau$, the violation is bounded exactly by $\alpha(\mathcal{S}(s) - \tau)$. $\blacksquare$

### Theorem 2 (Discrete Probability Smoothing)

**Theorem 2.** Under Differentiable Logit Masking with the smooth penalty $\text{penalty}(a, \mathcal{S}(s)) = \frac{\beta}{d_{\text{soft}}(\mathcal{S}(s)) + \epsilon}$, the probability of selecting an unsafe action $a_u \in \mathcal{A}_{\text{unsafe}}(s)$ is everywhere continuous and differentiable in $\mathcal{S}(s)$, transitions from effectively zero at low stakes to a viable nonzero probability at high stakes, and retains a strictly positive finite minimum penalty throughout.

**Proof.**
The probability of action $a_u$ under softmax is:
$$P(a_u \mid s) = \frac{\exp\!\left(l_{a_u} - \frac{\beta}{d_{\text{soft}}(\mathcal{S}(s)) + \epsilon}\right)}{\sum_{a} \exp(\tilde{l}_a)}$$
For all $\mathcal{S}(s) \in [0,1]$, the penalty $\frac{\beta}{d_{\text{soft}}(\mathcal{S}(s)) + \epsilon}$ is finite and smooth. Because $d_{\text{soft}} > 0$ everywhere, there is no discontinuity anywhere on $[0,1]$. For $\mathcal{S}(s) \ll \tau$, $d_{\text{soft}} \approx 0^+$ and the penalty is effectively $\beta/\epsilon$; by choosing $\epsilon \ll \beta$, this suppresses $P(a_u \mid s) \approx 0$. As $\mathcal{S}(s)$ increases past $\tau$, $d_{\text{soft}}$ grows and the penalty decays. Because $\mathcal{S}(s)$ is bounded by $1$, the minimum attainable penalty is $\frac{\beta}{d_{\text{soft}}(1) + \epsilon} > 0$. The logit $\tilde{l}_{a_u}$ becomes meaningfully finite, making $a_u$ selectable, but the permanent positive penalization margin relative to $l_{a_u}$ ensures unsafe actions remain continuously discouraged. Differentiability everywhere on $[0,1]$ follows from the $C^\infty$ smoothness of $d_{\text{soft}}(\mathcal{S}(s))$ — unlike the $\max(0,\cdot)$ formulation used in the continuous case, the softplus approximation has no kink at $\tau$ — composed with the smooth $\exp(\cdot)$ and softmax operations. $\blacksquare$

### Remark 1 (Necessity of Dynamic Boundaries)

**Remark 1.** Consider a strictly Shielded CMDP (or Pointwise CMDP) where the boundary is enforced statically at every state: $d(s) = 0$. Let $\mathcal{S}_{\text{low}} = \{s \mid \mathcal{S}(s) < \tau\}$ and $\mathcal{S}_{\text{high}} = \{s \mid \mathcal{S}(s) \ge \tau\}$. Assume reaching the goal from $\mathcal{S}_{\text{high}}$ strictly requires constraint-violating actions. A strictly Shielded CMDP cannot simultaneously achieve zero violations in $\mathcal{S}_{\text{low}}$ AND goal achievement in $\mathcal{S}_{\text{high}}$. The dynamic-boundary CDDP inherently achieves both.

**Argument.**
This follows directly from the definition of the Mission-Feasibility Gap and is stated here to make explicit the structural motivation for CDDP's dynamic boundary. To guarantee exactly zero violations in $\mathcal{S}_{\text{low}}$, a strictly Shielded CMDP must rigidly enforce $d(s) = 0$ globally at every state. Under this static constraint, the policy within $\mathcal{S}_{\text{high}}$ is similarly restricted to zero violations. By the assumption that goal achievement from $\mathcal{S}_{\text{high}}$ requires violation, the goal is rendered mathematically unreachable.
In contrast, CDDP evaluates $d(\mathcal{S}(s))$ dynamically. In $\mathcal{S}_{\text{low}}$, $d = 0$, enforcing strict zero violations. In $\mathcal{S}_{\text{high}}$, the manifold expands to $d(\mathcal{S}(s)) > 0$. By utilizing continuous QP expansion or Differentiable Logit Masking, the permissible violation scales proportionally to accommodate goal-reaching trajectories, naturally resolving the impossibility.

### Corollary 1 (Gradient Flow through KKT Conditions — Discrete Instantiation)

**Corollary 1 (Gradient Flow through KKT Conditions — Discrete Instantiation).**
In the discrete instantiation of CDDP, the gradient of the task reward objective $J_R$ with respect to the nominal policy parameters $\theta$ flows continuously and differentiably through the Differentiable Logit Masking layer at all $\mathcal{S}(s) \in [0, 1]$.

**Explanation.**
Because the discrete instantiation uses $d_{\text{soft}}(\mathcal{S}(s))$ — a $C^\infty$ softplus function — the penalty $\frac{\beta}{d_{\text{soft}}(\mathcal{S}(s)) + \epsilon}$ and the resulting masked logits $\tilde{l}_a$ are smooth everywhere. Gradients flow differentiably from the task loss backward through the softmax, the logit masking, and into $\theta$ without any discontinuity.

For the **continuous instantiation** (Section 2), the QP projection layer admits gradient flow via implicit differentiation of the KKT conditions (following Amos \& Kolter, 2017). The Jacobian of $a^*$ with respect to $a_{\text{nom}}$ is:

- **Inactive constraint** ($\nabla C(s) \cdot a_{\text{nom}} < d(\mathcal{S}(s))$): $\lambda^* = 0$ and $\frac{\partial a^*}{\partial a_{\text{nom}}} = I$.
- **Active constraint** ($\nabla C(s) \cdot a_{\text{nom}} > d(\mathcal{S}(s))$): $\lambda^* > 0$ and the gradient is projected onto the null space of the constraint Jacobian.

Note that because the continuous boundary $d(\mathcal{S}(s)) = \max(0, \alpha(\mathcal{S}(s) - \tau))$ is $C^0$ but not $C^1$ at $\mathcal{S}(s) = \tau$, the gradient $\frac{\partial d}{\partial \mathcal{S}(s)}$ is discontinuous at the boundary transition point. Smooth gradient flow through the KKT conditions therefore holds away from $\tau$ but is not guaranteed at the exact transition. In practice, the measure-zero set where $\mathcal{S}(s) = \tau$ exactly is rarely encountered during training, and the $C^0$ continuity of $d$ ensures the QP solution $a^*$ itself remains continuous. $\blacksquare$

