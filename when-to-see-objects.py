import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from datetime import datetime
from zoneinfo import ZoneInfo

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, get_body
from astroplan import Observer, FixedTarget

# your Messier list
FIXED_TARGETS = [f"M{m:03d}" for m in range(1, 110)]

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

fig = go.Figure()
bestday = {}

palette = px.colors.qualitative.Plotly

for cnt, name in enumerate(stats["target"].unique()):
    if "oo" in name:
        continue

    df = stats[stats["target"] == name].sort_values("date")

    # mark max-altitude day
    idx = df["alt_deg"].idxmax()
    md = df.loc[idx, "date"]
    ma = df.loc[idx, "alt_deg"]

    bestday[name] = md.isoformat()

    color = palette[cnt % len(palette)]
    width = 1 if name in FIXED_TARGETS else 2

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["alt_deg"],
            mode="lines",
            name=f"{name} max alt {ma:.1f}° on {md:%Y-%m-%d}",
            line=dict(color=color, width=width),
            opacity=0.5 if "oo" in name else 1,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[md],
            y=[ma],
            mode="markers",
            marker=dict(color=color, size=10),
            showlegend=False,
        )
    )

fig.update_layout(
    yaxis_title="Altitude (degrees)",
    xaxis=dict(dtick="M1", tickformat="%Y-%b-%d", showgrid=True),
)

json.dump(bestday, open("bestday.json", "w"), indent=4)

fig.write_html("when-to-see-objects.html", include_plotlyjs="cdn")
fig.show()
