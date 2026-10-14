"""Domain: forecasting and time (预测与时间).

Questions about events that resolved after the question's as-of date. The state holds the
question, its resolution criteria and the as-of date; gold is the recorded resolution. Market
prices at the as-of date, where available, are kept in meta as a calibration reference and are
never shown to the model. Static items are contaminated for models whose training data
postdates the resolution.
"""
