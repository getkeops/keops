"""
1) Blur parameter, scaling strategy
=====================================================

Dating back to the work of `Schrödinger <http://www.numdam.org/article/AIHP_1932__2_4_269_0.pdf>`_ 
- see e.g. `(Léonard, 2013) <https://arxiv.org/abs/1308.0215>`_ for a modern review -
entropy-regularized Optimal Transport is all about
solving the convex primal/dual problem:
"""

##################################################
# .. math::
#   \text{OT}_\varepsilon(\alpha,\beta)~&=~
#       \min_{0 \leqslant \pi \ll \alpha\otimes\beta} ~\langle\text{C},\pi\rangle
#           ~+~\varepsilon\,\text{KL}(\pi,\alpha\otimes\beta) \quad\text{s.t.}~~
#        \pi\,\mathbf{1} = \alpha ~~\text{and}~~ \pi^\intercal \mathbf{1} = \beta\\
#    &=~ \max_{f,g} ~~\langle \alpha,f\rangle + \langle \beta,g\rangle
#         - \varepsilon\langle \alpha\otimes\beta,
#           \exp \tfrac{1}{\varepsilon}[ f\oplus g - \text{C} ] - 1 \rangle,
#
# where the linear `Kantorovitch program <https://en.wikipedia.org/wiki/Transportation_theory_(mathematics)>`_
# is convexified by the addition of an entropic penalty
# - here, the generalized Kullback-Leibler divergence
#
# .. math::
#   \text{KL}(\alpha,\beta) ~=~
#   \langle \alpha, \log \tfrac{\text{d}\alpha}{\text{d}\beta}\rangle
#   - \langle \alpha, 1\rangle + \langle \beta, 1\rangle.
#
# The celebrated `IPFP <https://en.wikipedia.org/wiki/Iterative_proportional_fitting>`_,
# `SoftAssign <https://en.wikipedia.org/wiki/Point_set_registration#Robust_point_matching>`_
# and `Sinkhorn <https://arxiv.org/abs/1803.00567>`_ algorithms are all equivalent
# to a **block-coordinate ascent** on the **dual problem** above
# and can be understood as smooth generalizations of the
# `Auction algorithm <https://en.wikipedia.org/wiki/Auction_algorithm>`_,
# where a **SoftMin operator**
#
# .. math::
#   \text{min}_{\varepsilon, x\sim\alpha} [ \text{C}(x,y) - f(x) ]
#   ~=~ - \varepsilon \log \int_x \exp \tfrac{1}{\varepsilon}[ f(x) - \text{C}(x,y)  ]
#   \text{d}\alpha(x)
#
# is used to update prices in the bidding rounds.
# This algorithm can be shown to converge as a `Picard fixed-point iterator <https://en.wikipedia.org/wiki/Fixed-point_iteration>`_,
# with a worst-case complexity that scales in
# :math:`O( \max_{\alpha\otimes\beta} \text{C} \,/\,\varepsilon )` iterations
# to reach a target numerical accuracy, as :math:`\varepsilon` tends to zero.
#
# **Limitations of the (baseline) Sinkhorn algorithm.**
# In most applications, the cost function is the **squared Euclidean distance**
# :math:`\text{C}(x,y)=\tfrac{1}{2}\|x-y\|^2`
# studied by `Brenier and subsequent authors <http://www.math.toronto.edu/mccann/papers/FiveLectures.pdf>`_,
# with a temperature :math:`\varepsilon` that is
# homogeneous to the **square** of a **blurring scale** :math:`\sigma = \sqrt{\varepsilon}`.
#
# With a complexity that scales in :math:`O( (\text{diameter}(\alpha, \beta) / \sigma)^2)` iterations
# for typical configurations,
# the Sinkhorn algorithm thus seems to be **restricted to high-temperature problems**
# where the point-spread radius :math:`\sigma` of the **fuzzy transport plan** :math:`\pi`
# does not go below ~1/20th of the configuration's diameter.
#
# **Scaling heuristic.**
# Fortunately though, as often in operational research,
# `simulated annealing <https://en.wikipedia.org/wiki/Simulated_annealing>`_
# can be used to break this computational bottleneck.
# First introduced for the :math:`\text{OT}_\varepsilon` problem
# in `(Kosowsky and Yuille, 1994) <https://www.ics.uci.edu/~welling/teaching/271fall09/InvidibleHandAlg.pdf>`_,
# this heuristic is all about **decreasing the temperature** :math:`\varepsilon`
# across the Sinkhorn iterations, letting prices adjust in a coarse-to-fine fashion.
#
# The default behavior of the :mod:`SamplesLoss("sinkhorn") <geomloss.SamplesLoss>` layer
# is to let :math:`\varepsilon` decay according to an **exponential schedule**.
# Starting from a large value of :math:`\sigma = \sqrt{\varepsilon}`,
# estimated from the data or given through the **diameter** parameter,
# we multiply this blurring scale by a fixed **scaling**
# coefficient in the :math:`(0,1)` range and loop until :math:`\sigma`
# reaches the target **blur** value.
# We thus work with decreasing values of the temperature :math:`\varepsilon` in
#
# .. math::
#   [ \text{diameter}^2,~(\text{diameter}\cdot \text{scaling})^2,
#       ~(\text{diameter}\cdot \text{scaling}^2)^2,~ \cdots~ , ~\text{blur}^2~],
#
# with an effective number of iterations that is equal to:
#
# .. math::
#   N_\text{its}~=~ \bigg\lceil \frac{ \log ( \text{diameter}/\text{blur} )}{ \log (1 / \text{scaling})} \bigg\rceil.
#
# Let us now illustrate the behavior of the Sinkhorn loop across
# these iterations, on a simple 2d problem.


##############################################
# Setup
# ---------------------
#
# Standard imports:

import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from torch.autograd import grad

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

###############################################
# Display routines:

from imageio import imread


def load_image(fname):
    img = np.mean(imread(fname), axis=2)  # Grayscale
    img = (img[::-1, :]) / 255.0
    return 1 - img


def draw_samples(fname, sampling, dtype=torch.FloatTensor):
    A = load_image(fname)
    A = A[::sampling, ::sampling]
    A[A <= 0] = 1e-8

    a_i = A.ravel() / A.sum()

    x, y = np.meshgrid(np.linspace(0, 1, A.shape[0]), np.linspace(0, 1, A.shape[1]))
    x += 0.5 / A.shape[0]
    y += 0.5 / A.shape[1]

    x_i = np.vstack((x.ravel(), y.ravel())).T

    return torch.from_numpy(a_i).type(dtype), torch.from_numpy(x_i).contiguous().type(
        dtype
    )


def display_potential(ax, F, color, nlines=21):
    # Assume that the image is square...
    N = int(np.sqrt(len(F)))
    F = F.view(N, N).detach().cpu().numpy()
    F = np.nan_to_num(F)

    # And display it with contour lines:
    levels = np.linspace(-1, 1, nlines)
    ax.contour(
        F,
        origin="lower",
        linewidths=2.0,
        colors=color,
        levels=levels,
        extent=[0, 1, 0, 1],
    )


def display_samples(ax, x, weights, color, v=None):
    x_ = x.detach().cpu().numpy()
    weights_ = weights.detach().cpu().numpy()

    weights_[weights_ < 1e-5] = 0
    ax.scatter(x_[:, 0], x_[:, 1], 10 * 500 * weights_, color, edgecolors="none")

    if v is not None:
        v_ = v.detach().cpu().numpy()
        ax.quiver(
            x_[:, 0],
            x_[:, 1],
            v_[:, 0],
            v_[:, 1],
            scale=1,
            scale_units="xy",
            color="#5CBF3A",
            zorder=3,
            width=2.0 / len(x_),
        )


###############################################
# Dataset
# --------------
#
# Our source and target samples are drawn from measures whose densities
# are stored in simple PNG files. They allow us to define a pair of discrete
# probability measures:
#
# .. math::
#   \alpha ~=~ \sum_{i=1}^N \alpha_i\,\delta_{x_i}, ~~~
#   \beta  ~=~ \sum_{j=1}^M \beta_j\,\delta_{y_j}.

sampling = 10 if not use_cuda else 2

A_i, X_i = draw_samples("data/ell_a.png", sampling)
B_j, Y_j = draw_samples("data/ell_b.png", sampling)

###############################################
# Scaling strategy
# -------------------
#
# We now display the behavior of the Sinkhorn loss across
# our iterations.

from geomloss import SamplesLoss


def display_scaling(scaling=0.5, Nits=9, debias=True):

    plt.figure(figsize=((12, ((Nits - 1) // 3 + 1) * 4)))

    for i in range(Nits):
        blur = scaling ** i
        Loss = SamplesLoss(
            "sinkhorn", p=2, blur=blur, diameter=1.0, scaling=scaling, debias=debias
        )

        # Create a copy of the data...
        a_i, x_i = A_i.clone(), X_i.clone()
        b_j, y_j = B_j.clone(), Y_j.clone()

        # And require grad:
        a_i.requires_grad = True
        x_i.requires_grad = True
        b_j.requires_grad = True

        # Compute the loss + gradients:
        Loss_xy = Loss(a_i, x_i, b_j, y_j)
        [F_i, G_j, dx_i] = grad(Loss_xy, [a_i, b_j, x_i])

        #  The generalized "Brenier map" is (minus) the gradient of the Sinkhorn loss
        # with respect to the Wasserstein metric:
        BrenierMap = -dx_i / (a_i.view(-1, 1) + 1e-7)

        # Fancy display: -----------------------------------------------------------
        ax = plt.subplot(((Nits - 1) // 3 + 1), 3, i + 1)
        ax.scatter([10], [10])  # shameless hack to prevent a slight change of axis...

        display_potential(ax, G_j, "#E2C5C5")
        display_potential(ax, F_i, "#C8DFF9")

        display_samples(ax, y_j, b_j, [(0.55, 0.55, 0.95)])
        display_samples(ax, x_i, a_i, [(0.95, 0.55, 0.55)], v=BrenierMap)

        ax.set_title("iteration {}, blur = {:.3f}".format(i + 1, blur))

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.axis([0, 1, 0, 1])
        ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()


#################################################
# **The entropic bias.** In the first round of figures, we focus on the
# classic, **biased** loss
#
# .. math::
#   \text{OT}_\varepsilon(\alpha,\beta)~=~ \langle \alpha, f\rangle + \langle\beta,g\rangle,
#
# where :math:`f` and :math:`g` are solutions of the dual problem above.
# Displayed in the background, the dual potentials
#
# .. math::
#   f ~=~ \partial_{\alpha} \text{OT}_\varepsilon(\alpha,\beta) \qquad\text{and}\qquad
#   g ~=~ \partial_{\beta} \text{OT}_\varepsilon(\alpha,\beta)
#
# evolve from simple convolutions of the form :math:`\text{C}\star\alpha`, :math:`\text{C}\star\beta`
# (when :math:`\varepsilon` is large) to genuine Kantorovitch potentials (when :math:`\varepsilon` tends to zero).
#
# Unfortunately though, as was first illustrated in
# `(Chui and Rangarajan, 2000) <http://www.cvl.iis.u-tokyo.ac.jp/class2013/2013w/paper/correspondingAndRegistration/05_RPM-TPS.pdf>`_,
# the :math:`\text{OT}_\varepsilon` loss suffers from an **entropic bias**:
# its Lagrangian gradient :math:`-\tfrac{1}{\alpha_i}\partial_{x_i} \text{OT}(\alpha,\beta)`
# (i.e. its gradient for the `Wasserstein metric <https://arxiv.org/abs/1609.03890>`_)
# points towards the **inside** of the target measure :math:`\beta`,
# as points get attracted to the **Fréchet mean** of their :math:`\varepsilon`-targets
# specified by the **fuzzy** transport plan :math:`\pi`.

display_scaling(scaling=0.5, Nits=9, debias=False)


#################################################
# **Unbiased Sinkhorn divergences.** To alleviate this **mode collapse** phenomenon,
# an idea that recently emerged in the Machine Learning community
# is to use the **unbiased** Sinkhorn loss `(Ramdas et al., 2015) <https://arxiv.org/abs/1509.02237>`_:
#
# .. math::
#   \text{S}_\varepsilon(\alpha,\beta)~=~ \text{OT}_\varepsilon(\alpha,\beta)
#   - \tfrac{1}{2}\text{OT}_\varepsilon(\alpha,\alpha)
#   - \tfrac{1}{2}\text{OT}_\varepsilon(\beta,\beta),
#
# which interpolates between a Wasserstein distance (when :math:`\varepsilon \rightarrow 0`)
# and a kernel norm (when :math:`\varepsilon \rightarrow +\infty`).
# In `(Feydy et al., 2018) <https://arxiv.org/abs/1810.08278>`_,
# this formula was shown to define a **positive**, definite, convex loss function
# that **metrizes the convergence in law**.
# Crucially, as detailed in `(Feydy and Trouvé, 2018) <https://hal.archives-ouvertes.fr/hal-01827184/>`_,
# it can also be written as
#
# .. math::
#   \text{S}_\varepsilon(\alpha,\beta)~=~
#   \langle ~\alpha~, ~\underbrace{b^{\beta\rightarrow\alpha} - a^{\alpha\leftrightarrow\alpha}}_F~\rangle
#   + \langle ~\beta~, ~\underbrace{a^{\alpha\rightarrow\beta} - b^{\beta\leftrightarrow\beta}}_G~\rangle
#
# where :math:`(f,g) = (b^{\beta\rightarrow\alpha},a^{\alpha\rightarrow\beta})`
# is a solution of :math:`\text{OT}_\varepsilon(\alpha,\beta)`
# and :math:`a^{\alpha\leftrightarrow\alpha}`, :math:`b^{\beta\leftrightarrow\beta}`
# are the unique solutions of :math:`\text{OT}_\varepsilon(\alpha,\alpha)`
# and :math:`\text{OT}_\varepsilon(\beta,\beta)` on the diagonal of the space of potential pairs.
#
# As evidenced by the figures below, the **unbiased** dual potentials
#
# .. math::
#   F ~=~ \partial_{\alpha} \text{S}_\varepsilon(\alpha,\beta) \qquad\text{and}\qquad
#   G ~=~ \partial_{\beta} \text{S}_\varepsilon(\alpha,\beta)
#
# interpolate between simple linear forms (when :math:`\varepsilon` is large)
# and genuine Kantorovitch potentials (when :math:`\varepsilon` tends to zero).
#
# **A generalized Brenier map.**
# Instead of suffering from shrinking artifacts, the Lagrangian gradient of the Sinkhorn divergence
# interpolates between an **optimal translation** and an **optimal transport plan**.
# Understood as a **smooth generalization of the Brenier mapping**, the displacement field
#
# .. math::
#   v(x_i) ~=~ -\tfrac{1}{\alpha_i}\partial_{x_i} \text{S}(\alpha,\beta) ~=~ -\nabla F(x_i)
#
# can thus be used as a **blurred transport map** that registers measures
# up to a **detail scale** specified through the **blur** parameter.

# sphinx_gallery_thumbnail_number = 2
display_scaling(scaling=0.5, Nits=9)


#################################################
# As a final note: please remember that the trade-off between
# **speed** and **accuracy** can be simply set by changing
# the value of the **scaling** parameter.
# By choosing a decay of :math:`.7 \simeq \sqrt{.5}` between successive
# values of the blurring radius :math:`\sigma = \sqrt{\varepsilon}`,
# we effectively double the number of iterations spent to solve
# our dual optimization problem and thus improve the quality of our matching:

display_scaling(scaling=0.7, Nits=18)
plt.show()
