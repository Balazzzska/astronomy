import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplcursors

from datetime import datetime
from zoneinfo import ZoneInfo

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, get_body
from astroplan import Observer, FixedTarget

# your Messier list
FIXED_TARGETS = [f"M{m:03d}" for m in range(1, 111)]

# the extra solar‐system bodies you care about
# note: use the same name keys you want in your plot legend/DataFrame
EPHEMERIS_TARGETS = {
    "Venus": lambda t, obs: get_body("venus", t, obs.location),
    "Mars": lambda t, obs: get_body("mars", t, obs.location),
    "Jupiter": lambda t, obs: get_body("jupiter", t, obs.location),
    "Saturn": lambda t, obs: get_body("saturn", t, obs.location),
    # "Moon": lambda t, obs: get_body("moon", t, obs.location),
}

zi = ZoneInfo("Europe/Budapest")
now = datetime.now(zi)

observer = Observer(
    latitude=46.189562 * u.deg,
    longitude=20.041981 * u.deg,
    elevation=76 * u.m,
)

# load or build DataFrame
try:
    stats = pd.read_pickle("when-to-see-objects.pkl")
except FileNotFoundError:
    stats = pd.DataFrame()

    # loop Messier *and* solar‐system bodies
    for name in FIXED_TARGETS + list(EPHEMERIS_TARGETS):
        print(f"Processing {name}...")
        # build a function that, given Time, returns a SkyCoord
        if name in FIXED_TARGETS:
            coord_fn = lambda t, obs, name=name: SkyCoord.from_name(name)
        else:
            coord_fn = EPHEMERIS_TARGETS[name]

        local_dt = datetime(now.year, now.month, now.day, 21, 0, tzinfo=zi)
        for i in range(366):
            t = Time(local_dt)
            coord = coord_fn(t, observer)
            # wrap it into an astroplan target so altaz() will work uniformly
            tgt = FixedTarget(coord=coord, name=name)

            altaz = observer.altaz(t, tgt)
            stats = pd.concat(
                [
                    stats,
                    pd.DataFrame(
                        {
                            "target": [name],
                            "date": [local_dt],
                            "alt_deg": [altaz.alt.to(u.deg).value],
                            "az_deg": [altaz.az.to(u.deg).value],
                        }
                    ),
                ],
                ignore_index=True,
            )

            print(f"{name} {local_dt:%Y-%m-%d} {altaz.alt:.1f}° {altaz.az:.1f}°")

            local_dt += pd.DateOffset(days=1)

    stats.to_pickle("when-to-see-objects.pkl")


stats["alt_deg"] = stats["alt_deg"].apply(lambda x: np.nan if x < 0 else x)

fig, ax = plt.subplots()
ax = [ax]
bestday = {}

for cnt, name in enumerate(stats["target"].unique()):
    if "oo" in name:
        continue

    df = stats[stats["target"] == name].sort_values("date")

    # mark max-altitude day
    idx = df["alt_deg"].idxmax()
    md = df.loc[idx, "date"]
    ma = df.loc[idx, "alt_deg"]

    bestday[name] = md.isoformat()

    kwargs = {
        "color": f"C{cnt}",
        "lw": 1 if name in FIXED_TARGETS else 2,
    }

    if "oo" in name:
        kwargs["lw"] = 3
        kwargs["alpha"] = 0.5

    ax[0].plot(
        df["date"],
        df["alt_deg"],
        label=f"{name} max alt {ma:.1f}° on {md:%Y-%m-%d}",
        **kwargs,
    )
    ax[0].scatter([md], [ma], color=f"C{cnt}", marker="o", s=100)

for a in ax:
    a.xaxis.set_major_locator(mdates.MonthLocator())
    a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%b-%d"))
    a.grid(True, which="major", ls="--", alpha=0.4)
    # a.legend()

json.dump(
    bestday,
    open("bestday.json", "w"),
    indent=4,
)

ax[0].set_ylabel("Altitude (degrees)")

mplcursors.cursor()
plt.show()
