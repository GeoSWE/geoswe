# Governing equations

GeoSWE solves the **2D nonlinear shallow-water equations** (SWE) in conservative
form. With water depth $h$, depth-averaged velocity $\mathbf{u}=(u,v)$, and
momenta $hu, hv$, the state vector is

$$
\mathbf{q} = \begin{pmatrix} h \\ hu \\ hv \end{pmatrix},
$$

and the system reads

$$
\partial_t \mathbf{q}
+ \partial_x \mathbf{F}(\mathbf{q})
+ \partial_y \mathbf{G}(\mathbf{q})
= \mathbf{S}_b + \mathbf{S}_f + \mathbf{S}_r .
$$

The fluxes are

$$
\mathbf{F} = \begin{pmatrix} hu \\ hu^2 + \tfrac12 g h^2 \\ huv \end{pmatrix},
\qquad
\mathbf{G} = \begin{pmatrix} hv \\ huv \\ hv^2 + \tfrac12 g h^2 \end{pmatrix},
$$

where $g$ is gravity (default $9.81\,\mathrm{m\,s^{-2}}$, `Config.g`). In the
arrays, `q[0]=h`, `q[1]=hu`, `q[2]=hv`.

## Source terms

The right-hand side carries three sources. The bed slope enters the residual together with the fluxes; friction and the depth sinks are applied after it (see [numerical methods](numerical_methods.md)).

**Bed slope** $\mathbf{S}_b$ — the topographic forcing,

$$
\mathbf{S}_b = \begin{pmatrix} 0 \\ -g h\, \partial_x b \\ -g h\, \partial_y b \end{pmatrix},
$$

with bed elevation $b(x,y)$. Discretizing $\mathbf{S}_b$ so that it exactly
balances the pressure flux for still water is the **well-balanced** property
(see [well-balanced schemes](well_balanced.md)).

**Friction** $\mathbf{S}_f$ — Manning bed friction, treated point-implicitly for
stability (see [friction](friction.md)):

$$
\mathbf{S}_f = \begin{pmatrix} 0 \\ -C_f\,u\,|\mathbf{u}| \\ -C_f\,v\,|\mathbf{u}| \end{pmatrix},
\qquad C_f = \frac{g\,n^2}{h^{1/3}},
$$

with Manning roughness $n$ and $|\mathbf{u}| = \sqrt{u^2+v^2}$. The term is stiff in thin films, which is why it is integrated point-implicitly.

**Mass source** $\mathbf{S}_r$ — the gridded rainfall rate $R$ (m/s), added to the depth equation only,

$$
\mathbf{S}_r = \begin{pmatrix} R \\ 0 \\ 0 \end{pmatrix}.
$$

Rain adds depth but no horizontal momentum, so discharge is unchanged and velocity falls as depth rises. Infiltration (Green-Ampt) and the uniform recession sink are **not** entries in $\mathbf{S}_r$: they are operator-split depth sinks applied after the residual update, and they remove water at the local velocity by scaling $(hu, hv)$ by the remaining-depth ratio (see [forcings](forcings.md)).

## Wetting and drying

A cell with $h < h_\min$ is treated as dry: its momentum is zeroed and it contributes no outgoing flux, but any positive sub-threshold depth is retained, so the cell can still be wetted by a neighbouring flux. The threshold is `Config.h_min` (default $10^{-10}$ in float64; automatically raised to $10^{-6}$ in float32). The floor used in the velocity estimate of the CFL condition, `Config.h_min_cfl`, may be set to a different value; every run reported in the paper uses the same value for both. See [configuration](../configuration.md).
