"""Plot route vs executed path from path_tracking_diagnostics.csv.
Usage: python plot_tracking.py --csv <diag.csv> --route cmdp_path_world.csv --out tracking.png"""
import argparse, csv, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True); ap.add_argument("--route", required=True)
ap.add_argument("--out", default="tracking.png"); a = ap.parse_args()
route = [(float(r["x"]), float(r["y"])) for r in csv.DictReader(open(a.route))]
rows = list(csv.DictReader(open(a.csv)))
f = lambda k: [float(r[k]) for r in rows]
t, x, y = f("time_s"), f("robot_x"), f("robot_y")
fig, ax = plt.subplots(1, 2, figsize=(16, 7))
ax[0].plot(*zip(*route), "k-", lw=1, label="planned route")
ax[0].plot(x, y, "r-", lw=1.5, label="actual")
ax[0].plot(*route[0], "go", label="start"); ax[0].plot(*route[-1], "b*", ms=14, label="goal")
for i in range(0, len(rows), max(1, len(rows)//40)):
    by = math.radians(float(rows[i].get("body_yaw_deg", 0))) - math.pi  # travel heading
    ax[0].arrow(x[i], y[i], 0.8*math.cos(by), 0.8*math.sin(by), head_width=.25, color="orange")
ax[0].plot(f("look_x"), f("look_y"), "c.", ms=3, label="target")
for r in rows:
    if float(r["local_route_error_m"]) > 0.85:
        ax[0].plot(float(r["robot_x"]), float(r["robot_y"]), "mx", ms=12, label="first local err>0.85 m"); break
ax[0].set_aspect("equal"); ax[0].legend(); ax[0].grid(alpha=.3); ax[0].set_title("route vs actual (equal axes)")
ax[1].plot(t, f("local_route_error_m"), label="local err m")
ax[1].plot(t, [v/10 for v in f("heading_error_deg")], label="heading err deg/10")
ax[1].plot(t, f("steering"), label="steering")
ax[1].plot(t, [v/10 for v in f("yaw_rate_deg_s")], label="yaw rate deg/s /10")
ax[1].legend(); ax[1].grid(alpha=.3); ax[1].set_xlabel("t (s)")
fig.savefig(a.out, dpi=130); print("saved", a.out)
