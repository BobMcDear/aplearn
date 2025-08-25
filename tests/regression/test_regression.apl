]Import # ../../APLSource

eval←{⎕←⍺ (y MISC.METRICS.rmse X ⍺.pred X y ⍺.fit ⍵)}

X y←0 MISC.SPLIT.xy⍨⊃⎕CSV 'regression.csv' ⍬ 4 1
SUP.KNN eval 0 5
SUP.RIDGE eval 0.01
SUP.LASSO eval 0
SUP.RF eval 0 10 2 2
