# Flight Delay Intelligence: results portal

Static results portal for a study of 2015 US flight delays: how much weather changes delays and cancellations, and how well a flight's delay (15+ minutes late) can be predicted one hour before departure (T-60), the day before (day-ahead) and, for comparison with a published base paper, after take-off.

Pages: Overview, T-60, Day-ahead, Post-departure, Metric evaluation (confidence intervals, precision-recall, ROC and lift curves, threshold sweep, base-paper scorecard, tuning), Explore data, About and limits.

This hosted copy shows the saved results only. The live flight predictor needs the Python model server and runs in the full (private) project with `bash run_dashboard.sh`.

Built with plain HTML, CSS and JavaScript and d3 v7. Data: 2015 US DOT on-time performance and NOAA GHCN-Daily weather; all numbers in `data/` are aggregated results.
