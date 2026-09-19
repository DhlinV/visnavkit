"""One (H, W, 3) uint8 plan panel per window, after the reference's ``autopilot/utils/plan_viz.plan_figure``:
[current frame with the GT and the top-k predicted plans projected on the ground (pinhole or Kannala-Brandt fisheye,
cam_height_m above z = 0) | BEV: route patch, past (grey), GT (black, red heading arrows every 0.5 s), the top-k modes
(rank colours, heading arrows, probability in the legend; the top mode thick), goal (magenta star) | v-t, w-t, yaw-t
stacked; the yaw panel also draws the GT travel direction atan2(dy, dx) (red dashed, where v > 0.3 m/s)].
Needs matplotlib (the ``plot`` extra).
"""

import numpy as np

MODE_COLORS = ("tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan")
ROUTE_RGB = np.array([[0, 0, 0], [0, 0, 255], [0, 255, 0], [255, 0, 0], [128, 128, 128], [255, 255, 0]]) / 255.0


def project(xy, camera, wh):
    """Ego ground points (N, 2) [forward, left] -> pixels (N, 2), nan behind the camera or off the image."""
    fx, fy, cx, cy, k1, k2, k3, k4, h, cam_type = (float(v) for v in camera)
    X, Y, Z = -xy[:, 1], np.full(len(xy), h), xy[:, 0]
    ok = Z > 0.05
    xn, yn = X / np.where(ok, Z, 1), Y / np.where(ok, Z, 1)
    if int(cam_type) == 1:  # fisheye
        r = np.hypot(xn, yn)
        th = np.arctan(r)
        s = th * (1 + k1 * th**2 + k2 * th**4 + k3 * th**6 + k4 * th**8) / np.maximum(r, 1e-9)
        xn, yn = xn * s, yn * s
    u, v = fx * xn + cx, fy * yn + cy
    ok &= (u >= 0) & (u < wh[0]) & (v >= 0) & (v < wh[1])
    return np.where(ok[:, None], np.stack([u, v], 1), np.nan)


def plan_figure(frame, gt, modes, probs, ego_vw, past=None, camera=None, goal=None, route=None, hz=20, title=""):
    """``frame`` (H, W, 3) uint8, ``gt`` (T, 5) and ``modes`` (k, T, 5) metric [x, y, yaw, v, w] in the current ego
    frame, ``probs`` (k,), ``ego_vw`` (2,), ``past`` (S, 3) [x, y, yaw], ``camera`` (10,), ``goal`` (3,) [distance m,
    cos, sin], ``route`` (h, w) class ids -> the panel (H', W', 3) uint8."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H, W = frame.shape[:2]
    t = (np.arange(len(gt)) + 1) / hz
    colors = [MODE_COLORS[j % len(MODE_COLORS)] for j in range(len(modes))]
    widths = [2.2] + [1.0] * (len(modes) - 1)
    fig = plt.figure(figsize=(14, 4.8))
    gs = fig.add_gridspec(3, 3, width_ratios=[1.9, 1.25, 1.0])
    ax0, b, curves = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[:, 1]), [fig.add_subplot(gs[j, 2]) for j in range(3)]
    fig.suptitle(title, fontsize=9)
    gx = gy = None
    if goal is not None:
        gx, gy = goal[0] * goal[1], goal[0] * goal[2]
    # image: modes (lowest rank on top) under the GT
    ax0.imshow(frame)
    if camera is not None:
        for a, c, lw in [*zip(modes[::-1], colors[::-1], widths[::-1]), (gt, "lime", 2.0)]:
            p = project(a[:, :2], camera, (W, H))
            ax0.plot(p[:, 0], p[:, 1], "-", color=c, lw=lw, alpha=0.85)
        if goal is not None:
            p = project(np.array([[gx, gy]]), camera, (W, H))
            ax0.plot(p[:, 0], p[:, 1], "*", color="magenta", ms=12)
    ax0.set_xlim(0, W), ax0.set_ylim(H, 0), ax0.axis("off")
    # BEV: forward up, +y (left) to the left (x axis inverted); the route patch is drawn image-left = ego-left
    if route is not None:
        b.imshow(ROUTE_RGB[route.astype(int).clip(0, len(ROUTE_RGB) - 1)], extent=[20, -20, -20, 20], alpha=0.35)
    if past is not None:
        b.plot(past[:, 1], past[:, 0], "-", color="grey", lw=1)
    k = slice(hz // 2 - 1, None, hz // 2)
    lines = [(gt, "k", "tab:red", 1.8, "gt")] + [
        (m, c, c, lw, f"#{j + 1} p={p:.2f}") for j, (m, c, lw, p) in enumerate(zip(modes, colors, widths, probs))
    ]
    for a, c, ac, lw, label in lines[::-1]:
        b.plot(a[:, 1], a[:, 0], "-", color=c, lw=lw, label=label)
        # heading [cos, sin] in [forward, left] = [sin, cos] on the [left, forward] axes
        b.quiver(a[k, 1], a[k, 0], np.sin(a[k, 2]), np.cos(a[k, 2]), color=ac, angles="xy", width=0.006 * lw)
    if goal is not None:
        b.plot(gy, gx, "*", color="magenta", ms=12)
    b.plot(0, 0, "s", color="k", ms=5)
    xy = np.concatenate([gt[:, :2], *[m[:, :2] for m in modes], *([past[:, :2]] if past is not None else [])])
    fr = max(2.0, 1.15 * xy[:, 0].max())  # forward range; the left range at least half of it (the box stays ~square)
    lr = max(0.5 * fr, 1.15 * np.abs(xy[:, 1]).max())
    b.set_xlim(lr, -lr), b.set_ylim(min(0, xy[:, 0].min()) - 0.1 * fr, fr), b.set_aspect("equal")
    b.set_xlabel("left [m]"), b.set_ylabel("forward [m]"), b.grid(alpha=0.3)
    handles, labels = b.get_legend_handles_labels()
    b.legend(handles[::-1], labels[::-1], fontsize=6, loc="lower right")
    # v / w / yaw curves (t = 0: the ego status, yaw 0); yaw: + the GT travel direction (must sit on the GT yaw)
    d = np.gradient(gt[:, :2], axis=0) * hz
    travel = np.where(np.hypot(d[:, 0], d[:, 1]) > 0.3, np.arctan2(d[:, 1], d[:, 0]), np.nan)
    for a, (c, name, e0) in zip(
        curves, [(3, "v [m/s]", ego_vw[0]), (4, "w [rad/s]", ego_vw[1]), (2, "yaw [rad]", 0.0)]
    ):
        for m, color, lw in list(zip(modes, colors, widths))[::-1]:
            a.plot(np.r_[0, t], np.r_[e0, m[:, c]], "-", color=color, lw=lw * 0.7)
        a.plot(np.r_[0, t], np.r_[e0, gt[:, c]], "k-", lw=1.5, label="gt")
        if c == 2:
            a.plot(t, travel, "--", color="tab:red", lw=1, label="gt travel dir")
        a.set_ylabel(name), a.grid(alpha=0.3), a.set_xlim(0, t[-1])
    curves[2].legend(fontsize=7, loc="best"), curves[2].set_xlabel("t [s]")
    for a in curves[:2]:
        a.tick_params(labelbottom=False)
    fig.tight_layout()
    fig.canvas.draw()
    out = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return out
