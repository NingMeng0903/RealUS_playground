# Contact adaptation and ablation (paste-ready LaTeX)

Figures (both `figure*` at `\textwidth`, drawn at 7.16 in so no rescaling):

- `figures/coupling_progression.{png,pdf}` — mechanism 1, two panels: (a) establishment episode at cycle resolution, (b) near-field frames.
- `figures/contact_correction.{png,pdf}` — mechanism 2, three panels: (a) angular channels, (b) CoP pairing, (c) force error.
- Ablation is `table*` only: Coverage, $D_{\mathrm{under}}$, $T$, $T_{\mathrm{rec}}$, $F_z$ rms. No ablation figure. $D_{\mathrm{rec}}$ dropped. $T$ is `wall_s` (whole run, including press-in), not `track_wall_s`.

Scripts: `scripts/contact_adaptation_data.py` (all numbers below are printed by it), `scripts/plot_coupling_progression.py`, `scripts/plot_contact_correction.py`, shared style in `scripts/contact_figure_style.py`.

Mechanism numbers are from `Comparison Study/ultrapoc/004noise` (6361 tracking cycles, c_{\mathrm{th}}=0.8, N=10 regions). Ablation numbers are from `Ablation Study/analysis/README.md` and must not be mixed with `004noise`.

No-redundancy rules kept while writing: the ultrasound evidence appears once (Fig. 1b, also serving the one-sided window of mechanism 2); Figs. 1a and 2a share the establishment window but plot disjoint signals (coupling and speed vs. angular channels); both figures use the scan clock ($t=0$ is the first tracking cycle); whole-scan statistics stay in the text instead of a long, unreadable time axis; each panel carries one claim.

```
FORBIDDEN CLAIMS (do not write these back in):
- CoP shortened recovery time or recovery distance.
- CoP reduced force chatter / Fz RMS (the data show force error is unchanged, which is the point).
- CoP pairing as a third mechanism (it is how mechanism 2 executes its rocking).
- moment-free means image-only (normal force, image task, CoP, and alpha remain).
- fixed alpha is faster and therefore better.
- mid-scan fusion window on 004noise (the unmet-request cycles are all at establishment).
- CoP residual is "small" (it is numerically zero; report |c_p Delta omega_n| instead).
- a repeated zoom of a signal already plotted elsewhere (cycle-level numbers go in the text).
```

```latex
\subsection{Analysis of Contact Adaptation}
\label{sec:contact_adaptation}
Section~\ref{sec:comparison} established that \method{} preserves acoustic coupling under an orientation-biased plan.
This section uses one representative scan of that study to show \emph{how} the coupling is preserved, through the two mechanisms the formulation adds on top of force regulation: coupling-aware progression and image-driven contact correction.
Time is counted from the first path-tracking cycle, so $t=0$ is the start of the scan and the press-in occupies $t<0$.
A near-field region has a lost acoustic window when its confidence falls below the threshold, $d_i=\max(c_{\mathrm{th}}-c_i,0)>0$; that happens in $2.6\%$ of the $6361$ tracking cycles.

\subsubsection{Coupling-aware progression}
Progress along the path is slowed rarely, and only where the window is lost.
The executed scan-speed factor leaves its nominal value ($\alpha^{*}<0.99$) in $3.3\%$ of tracking cycles, $91\%$ of which lie within $0.3\,\mathrm{s}$ of a cycle with $\max_i d_i>0$, and the three grouped slow-downs of the scan scale with the severity of the window rather than with elapsed time: the two mid-scan dips ($\min_i c_i=0.76$ and $0.72$) reduce the factor only to $0.88$ and $0.85$, whereas the establishment episode of Fig.~\ref{fig:coupling_progression}a, whose worst region falls to $\min_i c_i=0.28$, drives it to $0.43$.
The reduction is an image decision that the QP passes through, not a by-product of the motion limits: the executed factor equals the image-derived reference $\alpha_{\mathrm{us}}=\mathrm{clip}(1-\gamma_\alpha\max_i d_i/c_{\mathrm{th}},\underline{\alpha},1)$ to within $0.01$ in $99.9\%$ of cycles.
The near-field frames marked in that episode (Fig.~\ref{fig:coupling_progression}b) show what the factor is responding to: all ten regions deficient at $t=0$ (frame~1), a one-sided window at $t=0.89\,\mathrm{s}$ (frame~2), and the restored window at $t=1.10\,\mathrm{s}$ (frame~3), when every region is already $\approx0.98$ and $\alpha^{*}$ has returned to one.
The gate is therefore selective rather than conservative: over the whole scan the mean factor is $0.995$, the total deferred time is $1.26\,\mathrm{s}$, and only $10\,\mathrm{mm}$ of the $462\,\mathrm{mm}$ path is acquired while a region is deficient.

\subsubsection{Image-driven contact correction}
The second mechanism acts when the moment channel cannot see the coupling problem.
Over the interval in which the moment reference does not meet the image request, $0$--$1.03\,\mathrm{s}$, the moment-derived rocking rate stays at $\lvert\omega_m\rvert\le0.006\,\mathrm{rad/s}$ while the request reaches $\lvert\omega_{\mathrm{us}}\rvert=0.160\,\mathrm{rad/s}$, and the QP still commands increments of up to $\lvert\Delta\omega_n\rvert=0.042\,\mathrm{rad/s}$, in the requested direction in $90\%$ of those cycles (Fig.~\ref{fig:contact_correction}a).
The image has the information the moment lacks because the deficit is one-sided: six of the ten regions read below $c_{\mathrm{th}}$ on one half of the aperture while the other half stays above $0.9$ (Fig.~\ref{fig:coupling_progression}b, frame~2).
The moment channel is silent because the load is still being built at that time: $6.3\,\mathrm{mm}$ of commanded normal travel is needed before the measured load reaches the $4\,\mathrm{N}$ reference, so no contact moment is available to rock on.
Once the load is established the roles swap without any switching logic: from $1.1\,\mathrm{s}$ the request is zero and the moment reference alone drives rocking up to $0.093\,\mathrm{rad/s}$, and in the mid-scan dip at $33.1$--$33.5\,\mathrm{s}$ the moment reference ($\omega_m\in[0.071,0.109]\,\mathrm{rad/s}$) already exceeds the request ($\lvert\omega_{\mathrm{us}}\rvert\le0.036\,\mathrm{rad/s}$), which is why no unmet cycle occurs there.
The two angular channels are thus complementary in time rather than redundant, and the controller never rocks in a direction the image does not support.

How that rocking is executed decides where it puts the load.
Whenever the QP changes the rocking rate it changes the normal speed by the paired amount, so that the increment is an instantaneous rotation about the measured centre of pressure,
\begin{equation}
\Delta v_n=c_p\Delta\omega_n .
\label{eq:cop_pair}
\end{equation}
This is not a corner case: a rocking increment is present in $64\%$ of tracking cycles, and the pressure centre is measurably eccentric, with $\lvert c_p\rvert$ at a median of $5.9\,\mathrm{mm}$, exceeding $5\,\mathrm{mm}$ in $58\%$ of cycles and reaching $24.9\,\mathrm{mm}$ on a $25\,\mathrm{mm}$ half-aperture (Fig.~\ref{fig:contact_correction}b, inset), so rotating about the probe centre and rotating about the loaded point are materially different motions.
The residual of~\eqref{eq:cop_pair} is numerically zero ($\le3\times10^{-14}\,\mathrm{m/s}$), and the informative quantity is the paired translation it produces: $\lvert c_p\Delta\omega_n\rvert$ has an rms of $0.44\,\mathrm{mm/s}$ and a peak of $3.06\,\mathrm{mm/s}$, against an admittance normal speed of $2.08\,\mathrm{mm/s}$ rms.
The same increments without pairing would appear instead as an indentation rate at the loaded point, accumulating up to $1.19\,\mathrm{mm}$ within a single rocking episode, which is about a fifth of the $6.3\,\mathrm{mm}$ indentation that carries the nominal load: the pairing moves the probe centre by that amount so that the loaded region is not pressed further.

Neither mechanism is paid for with force accuracy.
The force reference is fixed at $F_d=4\,\mathrm{N}$ and is never modified by the image channel, and once the load is established the measured force holds $3.94\,\mathrm{N}$ with a $0.176\,\mathrm{N}$ rms deviation and a $0.73\,\mathrm{N}$ worst case.
Grouping those cycles by how much paired normal motion the QP injects leaves the force error essentially unchanged ($0.170$, $0.174$ and $0.202\,\mathrm{N}$ rms for $\lvert c_p\Delta\omega_n\rvert\le0.001$, $0.001$--$0.35$ and $>0.35\,\mathrm{mm/s}$; Fig.~\ref{fig:contact_correction}c), and cycles with an active image request are no worse than idle ones ($0.183$ versus $0.176\,\mathrm{N}$).
The coupling recovered above is therefore obtained by redistributing contact, not by pressing harder.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{coupling_progression}
\caption{Coupling-aware progression on the representative \method{} scan.
(a)~Confidence and scan speed over the establishment episode ($t=0$ is the start of path tracking). Left axis: worst-region confidence $c_i$; right axis: executed scan-speed factor $\alpha^{*}$. Yellow shading marks contact-degraded intervals, in which at least one near-field region is below $c_{\mathrm{th}}$. Markers 1--3 on the confidence trace are the instants shown in (b).
(b)~Near-field B-mode at those instants; the ribbon marks regions below $c_{\mathrm{th}}$, showing a full-aperture deficit, a one-sided deficit, and the restored window.}
\label{fig:coupling_progression}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{contact_correction}
\caption{Image-driven contact correction and its execution.
(a)~Angular channels over the establishment window, on the same scan clock as Fig.~\ref{fig:coupling_progression}: the image request $\omega_{\mathrm{us}}$, the moment-derived reference $\omega_m$, and the command $\omega_n^{*}$; shading marks cycles in which $\omega_m$ does not meet the request, after which the moment channel takes over unaided.
(b)~Normal increment against the paired centre-of-pressure term for all rocking cycles, on the dashed identity line, with the signed $c_p$ distribution on the $50\,\mathrm{mm}$ aperture inset.
(c)~Absolute force error after establishment, grouped by the paired normal increment $\lvert c_p\Delta\omega_n\rvert$ into $\leq0.001$ ($\approx0$), $0.001$--$0.35$ and $>0.35\,\mathrm{mm/s}$; medians in black, overall rms dashed, per-group rms above each violin.}
\label{fig:contact_correction}
\end{figure*}

\subsection{Ablation Study}
\label{sec:ablation}
The ablations isolate the three ingredients exercised above: CoP pairing, coupling-aware progression, and the two complementary angular channels.
All four variants share one $20^{\circ}$-offset Lissajous plan; each deletes one ingredient while the force loop and the remaining components stay active, so each row of Table~\ref{tab:ablation} is one missing piece rather than a different controller.
$D_{\mathrm{under}}$ is the path length acquired while $\max_i d_i>0$, and $T$ is the wall-clock duration of the scan.
Three takeaways follow, and none is confounded by force tracking: the force-error rms stays within $0.265$--$0.297\,\mathrm{N}$ across all five configurations.

Removing CoP coupling primarily reduces acoustic-coupling coverage ($98.0\%$ to $88.5\%$; $D_{\mathrm{under}}$ from $8.1\,\mathrm{mm}$ to $51.9\,\mathrm{mm}$), while the force error is essentially unchanged ($0.273$ versus $0.276\,\mathrm{N}$).
Fixing $\alpha=1$ disables the coupling-aware gate: $D_{\mathrm{under}}$ rises from $8.1\,\mathrm{mm}$ to $59.9\,\mathrm{mm}$ and coverage falls to $86.8\%$, while $T$ only shortens from $52.2\,\mathrm{s}$ to $50.9\,\mathrm{s}$.
The extra $1.3\,\mathrm{s}$ on the full controller is therefore spent on the brief, poorly coupled intervals of Fig.~\ref{fig:coupling_progression}, not on a uniformly slower scan; once the window is restored the factor returns to one, so coverage is not bought by lingering on already acceptable contact.
Removing either ultrasound- or moment-driven orientation feedback leaves coverage at the same $89.0\%$, but prolongs contact recovery from $0.65\,\mathrm{s}$ to $1.36\,\mathrm{s}$ and $1.60\,\mathrm{s}$ respectively, the population counterpart of Fig.~\ref{fig:contact_correction}a.

\begin{table*}[t]
\caption{Ablation on the shared offset Lissajous plan.
Coverage is the fraction of the path grid with $\max_i d_i=0$.
$D_{\mathrm{under}}$ is the path length acquired while any near-field region is below $c_{\mathrm{th}}$.
$T$ is the wall-clock duration of the scan.
$T_{\mathrm{rec}}$ is the time from the first valid image to the first $0.3\,\mathrm{s}$ of full window.
$F_z$ rms is the force-error rms.
Full is the best of five repeats on this plan (coverage $93.8$--$98.0\%$).}
\label{tab:ablation}
\centering
\begin{tabular}{lccccc}
\hline
Variant & Coverage & $D_{\mathrm{under}}$ & $T$ & $T_{\mathrm{rec}}$ & $F_z$ rms \\
\hline
Full & $98.0\%$ & $8.1\,\mathrm{mm}$ & $52.2\,\mathrm{s}$ & $0.65\,\mathrm{s}$ & $0.276\,\mathrm{N}$ \\
Fixed $\alpha$ & $86.8\%$ & $59.9\,\mathrm{mm}$ & $50.9\,\mathrm{s}$ & $0.77\,\mathrm{s}$ & $0.297\,\mathrm{N}$ \\
w/o US orientation & $89.0\%$ & $49.6\,\mathrm{mm}$ & $54.5\,\mathrm{s}$ & $1.36\,\mathrm{s}$ & $0.281\,\mathrm{N}$ \\
w/o moment reference & $89.0\%$ & $49.5\,\mathrm{mm}$ & $55.3\,\mathrm{s}$ & $1.60\,\mathrm{s}$ & $0.265\,\mathrm{N}$ \\
w/o CoP coupling & $88.5\%$ & $51.9\,\mathrm{mm}$ & $54.8\,\mathrm{s}$ & $0.71\,\mathrm{s}$ & $0.273\,\mathrm{N}$ \\
\hline
\end{tabular}
\end{table*}
```



Notes:

- Do not mix the 004noise under-coupled length ($10\,\mathrm{mm}$ of $462\,\mathrm{mm}$) with ablation Full $D_{\mathrm{under}}=8.1\,\mathrm{mm}$: different runs, different plans.
- `\underline{\alpha}` is the floor on the speed factor (0.25 in the deployed configuration); use whatever symbol the method section already defines.
- `\ref{sec:comparison}` is the comparison-study subsection label; rename to match the main file.
- Regenerate with `python scripts/contact_adaptation_data.py`, then the two plotting scripts; the extractor prints every number quoted above, including the CoP eccentricity, the unpaired-excursion bound, and the force-error groups.

