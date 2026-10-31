# make_ridehail_data.py - synthetic ride-hailing data for Lab 2
# Writes raw/trips.csv (structured), raw/gps_pings.jsonl (semi-structured) and raw/reviews.jsonl (text)
import json, os
import numpy as np
import pandas as pd

rng = np.random.default_rng(2)
os.makedirs("raw", exist_ok=True)

# Addis Ababa sub-cities with approximate centres (synthetic data; positions are rough)
zones = pd.DataFrame({
    "zone": ["Addis Ketema", "Akaky Kaliti", "Arada", "Bole", "Gulele", "Kirkos", "Kolfe Keranio",
             "Lemi Kura", "Lideta", "Nifas Silk-Lafto", "Yeka"],
    "lat": [9.030, 8.890, 9.035, 8.985, 9.065, 9.005, 9.010, 9.010, 9.010, 8.960, 9.045],
    "lon": [38.725, 38.780, 38.752, 38.800, 38.740, 38.765, 38.690, 38.870, 38.735, 38.745, 38.815],
    "demand": [0.08, 0.05, 0.11, 0.20, 0.06, 0.14, 0.08, 0.04, 0.07, 0.08, 0.09],
    "drivers": [0.08, 0.07, 0.10, 0.15, 0.07, 0.12, 0.10, 0.06, 0.08, 0.09, 0.08],
})
hour_w = np.array([1, 0.5, 0.3, 0.3, 0.5, 1.5, 3, 6, 7, 5, 4, 4, 4.5, 4.5, 4, 4.5, 5.5, 7.5, 8, 6, 4.5, 3.5, 2.5, 1.5])
hour_w /= hour_w.sum()

# ---------------- trips: 60 days, about 25,000 requests a day ----------------
days = pd.date_range("2026-09-01", "2026-10-30")
n_per_day = rng.poisson(np.where(days.dayofweek >= 5, 21_000, 26_000))
N = int(n_per_day.sum())
day = np.repeat(days.values, n_per_day.astype(np.intp))   # intp: repeat counts must match platform int (32-bit Python)
hour = rng.choice(24, N, p=hour_w)
ts = pd.to_datetime(day) + pd.to_timedelta(hour * 3600 + rng.integers(0, 3600, N), "s")
zi = rng.choice(len(zones), N, p=zones.demand.values)
dzi = rng.choice(len(zones), N, p=zones.demand.values)
dist = np.round(np.clip(rng.gamma(2.2, 2.6, N) + 0.8 * (zi != dzi), 0.5, 40), 2)
peak = np.isin(hour, [7, 8, 17, 18])
dur = np.round(dist * rng.uniform(2.2, 3.2, N) * np.where(peak, 1.6, 1.0) + 2, 1)
# supply gap: Bole, Kirkos and Lemi Kura run short of drivers at peak hours
short = peak & np.isin(zones.zone.values[zi], ["Bole", "Kirkos", "Lemi Kura"])
u = rng.random(N)
status = np.where(u < np.where(short, 0.22, 0.04), "no_driver", np.where(u < np.where(short, 0.27, 0.09), "cancelled", "completed"))
surge = np.where(peak, np.where(short, 1.5, 1.2), 1.0)
fare = np.where(status == "completed", np.round((60 + 18 * dist + 2 * dur) * surge, 0), 0)
trips = pd.DataFrame({
    "trip_id": np.arange(1, N + 1), "request_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
    "pickup_zone": zones.zone.values[zi], "dropoff_zone": zones.zone.values[dzi],
    "distance_km": dist, "duration_min": np.where(status == "completed", dur, 0),
    "fare_birr": fare, "surge": surge, "status": status,
    "vehicle": rng.choice(["economy", "comfort", "minivan"], N, p=[0.72, 0.2, 0.08]),
    "payment": rng.choice(["cash", "mobile money", "card"], N, p=[0.55, 0.38, 0.07]),
    "driver_id": np.where(status == "no_driver", 0, rng.integers(1, 3001, N)),
    "rider_id": rng.integers(1, 250_001, N),
})
trips = trips.sort_values("request_time", kind="stable").reset_index(drop=True)
trips["trip_id"] = np.arange(1, N + 1)                                   # ids in time order
trips.to_csv("raw/trips.csv", index=False)

# ---------------- GPS pings: 1,000 drivers on 15 October, one ping a minute ----------------
D = 1000
start_h = np.clip(rng.normal(9, 3, D), 0, 20)
shift_h = np.clip(rng.normal(8, 2, D), 3, 12)
home = rng.choice(len(zones), D, p=zones.drivers.values)
upgraded_at = np.where(rng.random(D) < 0.5, rng.uniform(6, 18, D), 99)   # app 5.0 rolls out during the day
parts = []
for d in range(D):
    mins = np.arange(int(start_h[d] * 60), int(min(24, start_h[d] + shift_h[d]) * 60))
    k = len(mins)
    parts.append(pd.DataFrame({
        "driver_id": d + 1, "sec": mins * 60 + rng.integers(0, 5, k),      # every minute, give or take a few seconds
        "lat": zones.lat.values[home[d]] + np.cumsum(rng.normal(0, 0.0012, k)),
        "lon": zones.lon.values[home[d]] + np.cumsum(rng.normal(0, 0.0012, k)),
        "on_trip": rng.random(k) < 0.55, "ios": d % 5 == 0}))
pg = pd.concat(parts, ignore_index=True)
n = len(pg)
pg["new_app"] = pg.sec / 3600 >= upgraded_at[pg.driver_id - 1]
pg["battery"] = rng.integers(5, 101, n)
pick = lambda frac, seed: np.random.default_rng(seed).random(n) < frac
pg.loc[pick(0.005, 1), ["lat", "lon"]] = np.nan                         # no GPS fix
pg.loc[pick(0.001, 2), ["lat", "lon"]] = 0.0                            # cold-start default
pg.loc[pick(0.003, 3) & (pg.lat > 1), "lat"] += 0.08                    # a GPS jump of about 9 km
pg["recv"] = pg.sec + np.where(pick(0.02, 4), rng.integers(31, 181, n), rng.integers(0, 6, n))  # 2% arrive late
pg = pd.concat([pg, pg[pick(0.01, 5)]]).sort_values("recv", kind="stable")   # 1% sent twice
base = pd.Timestamp("2026-10-15")
fmt = lambda s: (base + pd.to_timedelta(s, "s")).strftime("%Y-%m-%dT%H:%M:%S")
pg["ts"], pg["received_at"] = fmt(pg.sec.values), fmt(pg.recv.values)
with open("raw/gps_pings.jsonl", "w") as f:
    for r in pg.itertuples(index=False):
        ev = {"driver_id": int(r.driver_id), "ts": r.ts,
              "lat": None if np.isnan(r.lat) else round(float(r.lat), 5),
              "lon": None if np.isnan(r.lon) else round(float(r.lon), 5),
              "status": "on_trip" if r.on_trip else "available",
              "app": {"version": "5.0" if r.new_app else "4.7", "os": "ios" if r.ios else "android"}}
        if r.new_app:
            ev["battery"] = int(r.battery)                                 # field added in app 5.0
        ev["received_at"] = r.received_at
        f.write(json.dumps(ev) + "\n")

# ---------------- reviews: free text for 20,000 completed trips ----------------
done = trips[trips.status == "completed"].sample(20_000, random_state=7)
issues = {
    "late":   (["The driver was late", "Waited too long for the car", "Driver took 20 minutes to arrive"], [1, 2, 3], 0.18),
    "price":  (["Too expensive", "The price was higher than shown", "Surge price is too high"], [2, 3], 0.15),
    "app":    (["The app crashed twice", "App could not find my location", "Payment failed in the app"], [1, 2, 3], 0.08),
    "rude":   (["The driver was rude", "Driver talked on the phone all the time"], [1, 2], 0.07),
    "clean":  (["Clean car and friendly driver", "Very clean car"], [4, 5], 0.20),
    "safe":   (["Safe and careful driving", "Driver knew the road well"], [4, 5], 0.20),
    "amharic": (["በጣም ጥሩ አገልግሎት", "ጥሩ ሹፌር"], [4, 5], 0.12),   # "very good service", "good driver"
}
keys = list(issues)
w = np.array([issues[k][2] for k in keys])
w_surge = w.copy(); w_surge[keys.index("price")] = 0.40          # riders on surge prices complain about price more
choice = np.where(done.surge.values > 1, rng.choice(len(keys), len(done), p=w_surge / w_surge.sum()),
                  rng.choice(len(keys), len(done), p=w / w.sum()))
reviews = []
for rid, (tid, c) in enumerate(zip(done.trip_id.values, choice), start=1):
    texts, ratings, _ = issues[keys[c]]
    reviews.append({"review_id": rid, "trip_id": int(tid), "rating": int(rng.choice(ratings)),
                    "text": str(rng.choice(texts)) + ("" if rng.random() < 0.7 else ". Thanks")})
with open("raw/reviews.jsonl", "w", encoding="utf-8") as f:
    for r in reviews:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

size = lambda p: os.path.getsize(p) / 1e6
print(f"trips {len(trips):,} rows ({size('raw/trips.csv'):.1f} MB); pings {len(pg):,} "
      f"({size('raw/gps_pings.jsonl'):.1f} MB); reviews {len(reviews):,} ({size('raw/reviews.jsonl'):.1f} MB)")
