]Import # ../../APLSource

eval_hard←{⎕←⍺ (y MISC.METRICS.acc X ⍺.pred X y ⍺.fit ⍵)}
eval_soft←{⎕←⍺ (y MISC.METRICS.acc 0⌷⍉⍒⍤1⊢X ⍺.pred X y ⍺.fit ⍵)}

X y←0 MISC.SPLIT.xy⍨⊃⎕CSV 'classification.csv' ⍬ 4 1
SUP.LIN_SVC eval_hard 0.1 0
SUP.LDA eval_hard ⍬
SUP.KNN eval_soft 1 5
SUP.LOG_REG eval_soft 1
SUP.NB eval_soft ⍬
SUP.RF eval_soft 1 10 2 2
