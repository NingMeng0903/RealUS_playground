# Energy tank literature audit and narrow model options — 2026-09-11

Scope: all eight supplied local PDFs were extracted with `pdftotext -layout`; relevant controller, passivity, experimental and limitation sections were read. Welleweerd PDF p4 equations were additionally checked in a rendered image because plain-text extraction obscures the low-energy branch. The initial literature phase was read-only on production code. Its extraction/report artifacts are in this papers directory. Subsequent authorized implementation is recorded in the addendum below. No web, hardware, or additional source-paper claims are involved. Page numbers below are PDF pages, starting at one.

## Finding

The current `CommandBudget` implements the same *kind* of external-port balance as Secchi/Benzi: actual completed logical epochs add `W_env^T V_final dt`. It already replenishes on positive port work. Persistent motion against resistance normally gives negative port work and drains a finite one-port budget. Neither passivity nor a tank promises indefinite autonomous positive net mechanical work from finite initial energy. A nonzero static force at exactly zero velocity does **not** consume mechanical port energy; drift, scan friction, indentation, force correction and torque/rotation can consume it. Electrical holding losses are a different port.

The papers that harvest damping have a different complete model. Dissipation is transferred from linked nominal/robot storage into the tank, and task references or a human may be explicit supply ports. Adding a positive `D*v²` credit to our standalone external-port account, while keeping no linked storage/source debit, would invent usable energy relative to its existing certificate.

A task-powered logical budget is coherent and feasible, but changes the certificate to a supplied/two-port model. It must explicitly record supplied task work and cannot be presented as external-port passivity, or as implementation of Lee's controller.

## Exact literature evidence

### Secchi and Ferraguti, 2019, Energy Optimization for a Robust and Flexible Interaction Control

- PDF p2 Sec.II explicitly assumes velocity tracking `v(t) ≈ vd(t)` and then equates their port behaviors. This is an idealization, not a bound on unknown vendor inner loops.
- PDF pp2–3 Eqs.(5)–(10): `xdot_t=u_t`, `y_t=x_t`, `T=x_t²/2`; modulator `u_t=A^T u`, `y=A y_t`, `A=gamma/x_t`, hence `Tdot=u^T y`. In the application `u=F_e`, `y=V_cmd`. Positive external work recharges; negative work drains.
- PDF p3 Eq.(11): `integral u^T y = T(t)-T(0) >= -T(0)`. Positive floor `epsilon` avoids division by zero; upper cap prevents excessive stored energy from permitting poor practical behavior.
- PDF p3 Proposition 2: a passive target with storage S can be reproduced without depletion if `T(0)>=S(0)+epsilon`. This is a bounded-extractable-energy claim for a passive target, not a promise for persistent active autonomous work.
- PDF p4 Eqs.(12)–(13): optimize distance to desired output under hard tank floor. No low-tank recharge timer exists; the optimizer returns an affordable passive approximation. Eq.(13b) in the supplied PDF extraction appears to omit a minus before T(0); its correct sign follows unequivocally from Eq.(11), Eq.(24), and Benzi Eq.(14). Do not reproduce that apparent typesetting/extraction error as an implementation rule.
- PDF pp4–5 Eqs.(17)–(24): Euler integration of **tank state** creates `Delta(k)=h²(u^T y)²/(2 x_t²) >=0`. Accumulated Delta must be withheld from spendable energy. Updating E directly with held port work, as our code does, avoids this specific state-squaring artifact.
- PDF p5 discussion permits expanding the tank's upper storage capacity as artificial discretization reserve grows. Increasing capacity does not authorize an unexplained increase of the actual initial/current energy balance.
- PDF p6 experiment is human guidance with changing virtual inertia; human input can replenish the tank. It is not autonomous frictional scanning forever.

### Benzi and Secchi, 2021

- PDF pp2–3 Eqs.(7)–(12): same modulated scalar tank, `T=x_t²/2`, `A=gamma/x_t`, `Tdot=gamma^T F_e`.
- PDF p3 Eqs.(13)–(14): `T>=epsilon`, minimize `||V_des-V_adm||²`, subject to `integral F_e^T V_des >= -T(0)+epsilon`.
- PDF p4 Eqs.(17)–(18): task constraints have slack; the energy constraint has no dedicated relaxing slack. Output `gamma=J(q) qdot` spends one full task-space port balance.
- PDF p4 Proposition 2, Eqs.(19)–(20): in free motion F_e=0, the tank constraint is automatically satisfied. This means no further **external mechanical port** expenditure; it does not mean robot actuation energy is zero or the tank recharges in air.
- PDF pp5–6 Eq.(23), experiments: goal reaching resumes after human interaction, and a human interacts against a changing repulsive potential. Tank near epsilon causes passive approximation. No guaranteed force tracking or indefinite active work is proved.
- Model is a velocity-controlled kinematic robot; physical equivalence retains the tracking assumption.

### Lee et al., 2024, Bidirectional Energy Flow Modulation for Passive Admittance Control

This is an inner torque/nominal-robot interconnection architecture, substantially different from an outer command filter.

- PDF p5 Eq.(8) defines **two power supplies**: `Sdot <= pdot_des^T F_o + qdot^T tau_ext`. A moving desired trajectory can supply energy explicitly. Removing `pdot_des^T F_o` changes the theorem.
- PDF pp5–6 Eqs.(9)–(13): unmeasured disturbances assumed dissipative; nominal/environment storage cancellation leaves the indefinite mismatch term `(V_nominal-V_real)^T F_ext`. Finite inner-loop tracking bandwidth is central, not ignored.
- PDF p6 Eqs.(14)–(17): auxiliary torque reshapes robot dynamics, PID compensation torque is `tau_c`, motor command `tau_m=tau_a+(1-alpha)tau_c`, nominal feedback torque `tau_bd=lambda alpha tau_c`. `alpha=1` removes PID influence on the real robot, leaving their designed impedance controller.
- PDF p7 Eqs.(18)–(19): alpha is chosen from the ratio of mismatch external power to compensator power when positive, capped at 1; if tank is at/below its lower threshold, force `alpha=1`. It is a structural admittance/impedance blend parameter, not scan speed alpha.
- PDF p7 Eqs.(20)–(22): `Sr=(lambda/2)qdot^T M qdot + edr^T K edr/2`, `Sn=qdot_n^T M_n qdot_n/2 + edn^T K edn/2`, `S=(1-alpha)Sn+alpha Sr`.
- PDF p8 Eqs.(28)–(32): `T=z²/2`; `zdot=psi/z` when `T<Tmax or psi<0`, otherwise zero. With mu the residual mismatch power admitted by their alpha rule,
  `psi=(1-alpha)V_n^T D V_n + alpha V^T D V - alpha_dot(Sr_hat-Sn) -(1-alpha)mu`.
  The first two terms harvest dissipation **already present with opposite sign in Sdot**. The remaining terms pay storage change from structure modulation and uncompensated generation. They are not optional.
- PDF p8 Eq.(33): total `S'=S+T` satisfies the same two-port supply inequality. When depleted they use their passive impedance fallback, with reduced disturbance rejection/tracking performance; they do not replenish merely by changing alpha.
- PDF p14 Sec.VII-C (not VII-B): omitted inertia-model term is `-(lambda/2)alpha_dot qdot^T(M-Mhat)qdot`. The authors explicitly say this unmonitored indefinite term means passivity cannot be guaranteed with that modeling error. Additional noise/discretization/flexibility sources are also outside their proof.

### Welleweerd et al., 2020, automatic robotic ultrasound

- PDF p3 Sec.II-D Eqs.(1)–(3): image mean confidence drives normal translation, confidence barycentre drives in-plane rotation; remaining DOFs follow a predefined trajectory. This is not an external-force setpoint controller.
- PDF pp3–4 Eqs.(4)–(11): underlying SAIP impedance controller uses robot torque dynamics, compensation, kinetic plus virtual-spring potential energy. `Etotal=Kinetic+Potential`; stiffness scales when it exceeds Emax; damping scales to restrict motion power. These require quantities/control authority absent from a bare velocity command budget.
- PDF p4 Eqs.(12)–(14): one tank per joint, spring energy `H_n=k s_n²/2`; the printed normalized interconnection is `sdot_n=u_n qdot_n`, `tau_on=-u_n s_n`. High-energy `u_n=-tau_cn/s_n`. Low-energy `u_n=-(tau_cn/gamma²)s_n`, `gamma=sqrt(2 epsilon)`, giving output torque `tau_cn s_n²/gamma²`, smoothly attenuated toward zero. Consequently, in the printed normalized k=1 form, `Hdot_n=-tau_on qdot_n`; negative controller work can replenish. For general k the energy-gradient normalization must be carried consistently; do not silently mix `k s²/2` with the normalized port equations.
- The low-energy branch retains control action; the text says this prevents recharging due to inertial motion of other links. It is not a statement that every low-energy tank automatically receives damping energy.
- PDF p6 Discussion explicitly says no force sensor is integrated and that future force data are needed to support their lower-force claim. Application-specific tank budgets and power/energy limits are also proposed for further analysis. This paper cannot supply a validated 6N force limit or replenishment parameter for ours.

### Heck et al., 2018, Direct Force-Reflecting Two-Layer Approach for Passive Bilateral Teleoperation With Time Delays

- PDF p3 Eqs.(1a,b): controller energy mismatches initially satisfy `Edot_m,diff=P_e-P_mc`, `Edot_s,diff=P_h-P_sc`. Inputs are real operator and environment powers, with their sign convention `P_h=F_h^T xdot_m`, `P_e=-F_e^T xdot_s`.
- PDF pp3–4: negative mismatch requests extra harvested power; physical damping is applied to moving master/slave. If velocity is zero/low, insufficient energy is harvested, so performance force is reduced. Lambda decreases from one to zero between two negative lower bounds; recovery damping is added. Empty-energy behavior does not guarantee task tracking.
- PDF p5 Eqs.(4)–(6): `F_ic*=lambda_i F_ic-lambda_i F_i,harv-(1-lambda_i)F_i,rec`; `F_j,harv=-beta_j E_j,harv xdot_j`, `F_i,rec=-gamma_i Ebar_i,bal xdot_i`. Harvester energy is negative debt, hence the applied negative harvesting force is dissipative.
- PDF p5 Eqs.(9)–(11): `P_diss=alpha_i max(E_i,diff,0)`, `P_gen=alpha_i max(-E_i,diff,0)`, `Edot_j,harv=lambda_j F_j,harv^T xdot_j-P_i,gen_delayed`. P_gen is simultaneously recorded as remote harvesting debt; copying P_gen alone as recharge would omit its counter-account.
- PDF pp5–6 Eqs.(12)–(16) include actual harvesting work, controller expenditure, recovery, discarded excess and communication energy. Eq.(14) constructs an available lower bound using delayed records and outstanding requests.
- PDF p6 Eqs.(19)–(20): total storage lower bounded and `Vdot<=2 u^T y`, factor two from their carefully analyzed duplication, with `u=[F_h;-F_e]`, `y=[xdot_m;xdot_s]`. Their assumptions bound forces and prohibit input/controller energy injection above a velocity threshold. It is not authorization to duplicate work in an arbitrary tank.

### De Stefano et al., 2020, Passivity-Based Approach for Simulating Satellite Dynamics With Robots

- PDF p3 Sec.II models actual velocity as **perfectly tracked delayed command**, `V_real(k)=V_s(k-mu)`, assuming integer sample delay. This omits arbitrary tracking error.
- PDF p6 Eq.(9): sampled port passivity `E(m)=E(0)+sum u(k)^T y(k) h>=0`.
- PDF pp6–7 Eqs.(12)–(18): time-delay observer debits measured port activity and credits **implemented correction** work. `F_pc=alpha V_s`, `F_c=F_e-F_pc`; each component uses `alpha_i=-Eobs_i/(V_s,i² h)` if observer negative, else zero. This changes the force input to nominal dynamics; it is not a positive credit added while leaving that dynamics unchanged. Zero-velocity division and finite execution need practical treatment.
- PDF p7 Fig.13 explicitly shows `(F_e,V_s)` observer energy differing from physical interaction energy using `(F_e,V_s delayed)`. The authors inspect both ports; one cannot substitute them freely.
- PDF p8 Eqs.(19)–(27): translational Euler error `DeltaH=h² F_e^T M^-1 F_e/2`; rotational error includes coupled free-body and torque terms. A second observer pays this error, and velocity correction `V_c=V_s-beta F_e` dissipates it. Delay and discretization are distinct corrections.
- The article is a two-PC/observer construction, not an inexhaustible tank-replenishment recipe.

### Samuel et al., A Perturbation-Robust Framework for Admittance Control of Robotic Systems With High-Stiffness Contacts and Heavy Payload

- No energy-tank refill or switch law is supplied. This is transfer-function/observer design for inner-loop and payload/contact perturbations.
- PDF p3 Eqs.(1)–(3): PI velocity inner loop C, mass-damper robot R, delay Gamma, payload P, environment E and desired admittance A explicitly modeled.
- PDF pp3–4 Eqs.(7)–(9): even infinite proportional inner gain leaves `Y=Gamma A/(1+Gamma A P^-1)` and contact force dynamics with delay/payload; disturbance suppression alone does not recover desired admittance.
- PDF p4 Eq.(10): `Pert_hat=Q[Tn^-1 V_m-V_i]+N1 F_m-N2 V_m`; actual task-space velocity and nominal inner-loop model are required.
- PDF pp5–6 Eqs.(20)–(21), (27)–(32): CDYOB constructs specific filters and nominalization conditions; ideal cancellation cannot be imported by naming our budget a tank. PDF p6 states bandwidth removal in Eq.(21) is under ideal conditions.

### Keemink et al., 2018, Admittance Control for Physical Human–Robot Interaction

- No energy-tank recharge or switch law. It reviews apparent admittance and inner-loop effects.
- PDF p5 Sec.4.2.1 Eq.(1): mechanical port passivity concerns the integral of conjugate external force and actual velocity, with storage history implicit in lower limit minus infinity.
- PDF p7 Sec.5.2 and Fig.4: force sampling/filtering, virtual dynamics, velocity/current control, zero-order hold, post-sensor inertia and actual mechanical motion are distinct blocks. Command speed is not automatically actual speed.
- PDF p9 Eqs.(7)–(8): for its simple pure-virtual-inertia/PI baseline, passivity requires `m_v>=K_p m_r/(K_p+b_r)` and `-b_r K_i>=0`. These are architecture-specific conditions; they illustrate why good virtual dynamics alone do not prove apparent-port passivity.

## Current code comparison

`command_budget.py` has one spendable balance, frozen successful final W/V epochs, six-axis net dot product, upper capacity clipping, reserve floor, and explicit logical expiry/unknown physical tail. `_settle` already credits positive `dt*power_w`; its constructor requires zero damping/errors/rates as model definition. There is no tank-state Euler squaring error. The 20260911 review's final Sec.7–8 contract matches this code; earlier Sec.4 defects describe its older predecessor and must not be reasserted as current failures.

`available=balance-stopping_reserve-active_remaining_liability-pending_liability` can be far below balance. A 50ms model lease reserves future work even when cycles are shorter. This reservation is not all consumed work: successful replacement releases unused old logical time after settling its used prefix. Low available energy may reflect both consumption and conservative overlapping publication liability.

The model freezes wrench with final speed until replacement. It therefore cannot discover actual elastic return, changing contact wrench, or slip within an epoch. Real aligned work is deliberately nonspendable; it is useful for diagnosing model mismatch, not silently refinancing this balance. This is an explicit model limitation.

## Feasible choices

1. Retain strict one-port budget: tune an explicit finite experiment energy, reduce positive task output when low, allow genuine positive completed model work to refill, and accept that persistent resistive autonomous work eventually exhausts it. A task may run indefinitely if its total net extracted work is finite; fixed force with zero normal motion is an example. No universal finite lifetime or universal infinite lifetime follows.
2. Implement linked nominal storage/damping: possible only after deriving storage and the actual baseline dynamics/reference supply and matching final speed/inner-loop changes. For this complex TFF/torque baseline it is not a narrow patch. No casual `D*v²` credit.
3. Explicit task-power source plus final-port budget: narrow, mathematically coherent for an experiment intended to keep baseline task motion available. This deliberately weakens the claim to supplied logical-port accounting.

### Preferred narrow supplied-model variant, if continuous baseline is the intended contract

At review, independently form baseline final-mapped velocity `V_b` without image objectives, and freeze it in the same frame and epoch as W and actual final `V_f`. Define

`P=W^T V_f`, `P_b=W^T V_b`, `A=max(0,-P_b)`.

A is authorized task-source power, not harvested mechanical power. A separately configured bound can limit it. If source is credited at full A, `Edot=P+A` is a coherent two-input model, but cancelled final motion `V_f=0` fills the tank at rate A. That permits waiting/cancellation to bank hypothetical baseline power for later visual work. It is mathematically allowed only because a continuous external source was declared; it is often an undesirable policy.

Prefer drawing only the source actually needed by positive final output:

`S_used=min(A,max(0,-P))`,

`Edot=P+S_used=max(P,0)-max(-P-A,0)`.

Thus baseline output can run without consuming image reserve, environmental positive work recharges, more output than baseline authorization spends the tank, and unused authorization is discarded. This policy does not bank task-source power at rest. Upper-cap clipping discards further recovery.

For held W/V/A and duration h, the hard admission condition remains affine:

`h*(W^T V_f+A)+E_available>=0`,

with actual liability `h*max(0,-W^T V_f-A)`. Preserve old/new publication liability rules and use the identical source power in QP, final review, reservation and settlement. No positive future source or recovery is pre-credited to the balance. Source authorization for an uncommitted candidate is not earned input.

Track cumulative absolute port work `I_P`, used source `I_S` and capacity discard `I_C`. Exact account:

`E(t)-E(0)=I_P+I_S-I_C`, with `I_C>=0`.

This gives `I_P+I_S>=Emin-E(0)`. It does **not** give `I_P>=Emin-E(0)` if source continues. Claim: "final command work constrained against explicitly supplied nominal-task power plus finite stored energy; physical port uncertified."

### Critical policy and implementation red flags

- An energy scalar cannot uniquely isolate image correction: a visual motion with the same total W-dot-V as baseline is freely authorized; a power-neutral turn can be arbitrarily large subject to separate velocity/geometry constraints. If low tank must suppress **all** visual motion, implement an explicit image-authority blend/constraint, not a false energy claim.
- Counterfactual baseline must be independent of image decisions and use the same real final command mapping, actuator limits, coordinate point and rail contribution. An inflated raw nominal speed, or one the final publisher cannot realize, artificially expands source authorization. Freeze baseline source before image optimization and log V_b explicitly.
- If image corrections feed into baseline state/reference next cycle, they can become future task-supply authority. Define which state/reference changes belong to the baseline source contract; a mere label does not separate them.
- Source allowance measures **net full-port mechanical output**, not each-axis cost or `W^T(V_f-V_b)` alone. Net power can mix translation and rotation. Keep full W/V ledger visible.
- With the nonbanking source rule, when baseline already absorbs external power, reducing that absorption does not spend tank unless final net work becomes active. This is a policy consequence; do not claim exact incremental image-energy charging.
- Continuous nominal force/path can now deliver unbounded external cumulative energy. Independent force, velocity, confidence authority and stop protections remain material; this account supplies no 6N force bound.
- The source is a declared mathematical task authorization unless a physical task-reference port/storage model is actually proved. Motor electrical supply alone does not certify the mechanical sensor/command accounting.
- Changing certificate/mode must be explicit. Existing one-port logs should not silently acquire supplied work; include `task_source_enabled`, `source_model`, `source_available_w`, `source_used_w`, absolute P, source/port accumulated J, tank work, rejected/unused supply and physical uncertified status.
- No budget reset on low-energy/retry/expiry. Source usage belongs only to successfully active model epochs. Physical stopping tail and tracking remain unknown under either model.

Minimum independent verification: baseline-only does not drain while absolute negative work/source used accumulate equally; extra output spends exact difference; cancelled/zero output does not recharge; positive external work credits once; capacity discard accounted; baseline source cannot depend on visual candidate; delayed/partial publication and expiry do not earn uncommitted source; random six-axis epochs satisfy the independent supplied-work identity; zero/source-disabled mode recovers the old strict model exactly.


## Implementation addendum

Following explicit root authorization, HIGH implementation added optional nominal-command supply in `command_budget.py` and `port_constraint.py`, tests, runtime configuration, daemon capability negotiation and trial profile settings. Strict one-port behavior remains the default. Supplied mode records absolute port work, actual task-source draw, tank change and capacity discard separately; unused allowance cannot recharge the tank. Profile initial/capacity/reserve are .10/.15/.05 J, confidence deadband .03, and `feature.dropout_policy=pause_visual`. Missing/stale images after established feedback pause visual correction while fresh force baseline can continue.

The integrated runtime currently derives allowance from the existing pre-QP nominal tool command, whose internal state can contain earlier applied visual velocity. It is consequently a declared current-command task authorization, **not** an independent no-vision counterfactual or a causal measurement of pure visual energy. The stronger independently final-mapped baseline suggested above is not claimed as implemented. Physical command tracking and actual stop tails remain uncertified.

An additive nominal-increment visual formulation was investigated by the visual worker and rejected after repeated actual TorqueTilt/QP state-update tests showed accumulation toward the angular speed cap. Existing total-velocity fusion stays in place; no incremental field/capability is shipped.

Verification: first ledger/constraint run 90 passed (new task-power tests, legacy command-budget tests and existing port-constraint tests); final config/capability/task/legacy/continuous-visual integration subset 98 passed. Final subset used rm75 Python with its cmeel Python/library paths and disabled unrelated pytest plugin autoload. This implementation phase did not run hardware.
