# Assignment4

Actually to my limited knowledge of both English and NLP models, I can not analyze deeper why nmt gives specific errors. 

Thus following answers in `(a)` and `(b)` are just some of my observations, without strict analysis.

## (a)

### (i)

`favorite` should be replaced by `one` since it appears twice.

Maybe attention forces `favorite` to focus more on  `facoritos`, or focus less.

A positive weight vector can be set to adjust attention vector, so that a target word initially would focus on source word at the same position.

### (ii)

`more` is actually `most`, and it should be an adjective `most reading` before `author`, rather than weirdly at the end.

It seems that following the order in source sentence leads to the result.

This case is the opposite from the one above, maybe the reason is that this sentence is longer, where attention can be modified to learn longer.

### (iii)

`Bolingbroke` is not in corpus and is translated as \<unk\>.

Solutions include:

+ use char-level language model
+ change to char-level only when coming with \<unk\> (Mix word/char model)

+ use word-piece language model, like `Jessica` will be split into `Je`,`ssi`,`ca`

### (iv)

`block` and `go around` are mistranslated as `apple` and `go back`.

Maybe `go back` is a more common phrase in corpus?

Maybe lower the weight of last predicted word.

### (v) 

`teachers' lounge` is mistranslated as `women's room`.

It seems `women's room` is related to the context `she` and `bathroom`, but `teachers' lounge` is not shown in the context. 

Attention lets model to lay attention to context. It's beneficial in most cases, but sometimes there is no need. There should be a trade-off. Maybe there can be a binary classifier to determine before attention to determine whether attention is needed at current position.

### (vi)

There's a miscalculation of units.

Language model can not know 1 acre equals 0.4 hectare.

When coming with a number, pay attention to the next word because it is most likely to be units. Usually numbers are directly mapped without translation. We need a "unit convert" table to handle it.



## (b)

### (i)

es: `Lo haca en secreto`

en(reference): `She did it in secret.`

en(nmt): `I was doing it in secret.`

### (ii)

es: `Estoy desilusionada que de adultos nunca llegamos a conocernos.`

en(reference): `I'm disappointed  that we never got to know each other as adults.`

en(nmt): `I'm sure, as adults never get to know us.`



## (c)

###  (i)

**c1**
$$
p1=0.6,p2=0.5,BP=1,BLEU=e^{(0.5\log{0.6}+0.5log0.5)}
$$
**c2**
$$
p1=0.8,p2=0.5,BP=1,BLEU=e^{(0.5\log{0.8}+0.5log0.5)}
$$
c2 is better regarding  BLEU, I agree on it.

### (ii)

**c1**
$$
p1=0.6,p2=0.5,BP=e^{-0.2},BLEU==e^{-0.2}*e^{(0.5\log{0.6}+0.5log0.5)}
$$


**c2**
$$
p1=0.4,p2=0.25,BP=e^{-0.2},BLEU==e^{-0.2}*e^{(0.5\log{0.4}+0.5log0.25)}
$$
c1 is better regarding BLEU, I think c2 is better.

### (iii)

Single reference will lead to bias, `(ii)` is an example. 

### (iv)

**PROs**

+ easy to compute automatically
+ a unified standard

**CONs**

+ high score does not indicates high performance
+ reference quality is important



