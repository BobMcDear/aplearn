]Import # ../../APLSource

(data header)←⎕CSV 'iris.csv' ⍬ 4 1
X y←data MISC.SPLIT.xy 0
X←X PREPROC.NORM.trans X PREPROC.NORM.fit ⍬
cs←X UNSUP.KMEANS.pred X UNSUP.KMEANS.fit 3
{⎕←'Cluster:' ⍵ (y⌷⍨⊂⍸cs=⍵)}¨⍳3
