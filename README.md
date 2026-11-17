# GTA VI Launch-Day Twitch Forecaster

Entry for the make.fun forecasting bounty
[`gta6-twitch-ccu-forecast-01`](https://make.fun/bounty/gta6-twitch-ccu-forecast-01):
forecast the hourly Twitch viewer count of the "Grand Theft Auto VI"
category on its release day (19 Nov 2026, UTC), as five quantiles
(5%, 25%, 50%, 75%, 95%) for hours 2–24, given the first hour's count.

## How it works

Big launches have a strong diurnal shape: viewers dip to ~0.55× the
first hour in the UTC morning, then rise to ~1.3× (up to ~9× for the
biggest single-player launches) in the US evening — while the house
baseline assumes every hour equals the first hour. `main.py` measures
that shape directly from the `comparables` release-day curves shipped
in the case itself: for each hour, the empirical quantiles of the
log-ratio `viewers_h / viewers_1` across all comparable launches,
smoothed across hours, multiplied by the observed first-hour count.
Empirical quantiles are exactly what the pinball-loss score rewards,
so calibration comes from real data, not from a guessed spread.

## Usage

```sh
python3 main.py /input/<id>.in /output/<id>.out
```

Standard library only, deterministic, ~0.03 s per case.

## Local test results (practice cases p00–p03)

| Case | This program | House baseline |
|---|---|---|
| p00 Wolverine | 0.4054 | 0.7437 |
| p01 RDR2 | 0.2887 | 0.2868 |
| p02 God of War | 0.2881 | 0.2146 |
| p03 Spider-Man 2 | 0.3172 | 0.3033 |
| **Mean** | **0.3249** | **0.3871** |

Mean pinball loss (log scale) ~16% better than the house baseline.
See `NOTES.md` for the full test report, assumptions, and known gaps.

## License

MIT — see `LICENSE`.
