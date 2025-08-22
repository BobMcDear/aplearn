]Import # ../../APLSource

(data header)←⎕CSV 'housing.csv' ⍬ 4 1
(X_t y_t) (X_v y_v)←(¯1+≢header) MISC.SPLIT.xy⍨¨data MISC.SPLIT.train_val 0.2
X_t X_v←(X_t PREPROC.NORM.fit ⍬)∘(PREPROC.NORM.trans⍨)¨X_t X_v
⎕←y_v MISC.METRICS.rmse X_v SUP.RIDGE.pred X_t y_t SUP.RIDGE.fit 1
