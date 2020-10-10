# Assignment5

## (a)

`char` has less meaning than `word`, and `word` will be computed by multiple `char`.  And vocab size of `char` is much less than `word`, which means representing a `char` is easier. Thus lower dimensions for `char` is enough.



## (b)

### character based

$$
V_{char}*e_{char}+e_{char}^2*(k+2)
$$





### word based

$$
V_{word}*e_{word}
$$





## (c)

`RNN` focuses too much on final steps (or later steps), which leads to biased char-level embedding depending on position. `CNN` uses sliding window to extract features, which will exert same weight on each position when window slides.



## (d)

`max-pooling` detects whether something has happened by keeping the max value on each dimension. For example, `max-pooling` over time step means catching max signal on each feature dimension along the whole sentence, which serves well for classification.

`average-pooling` keeps average information, which may mediate some features along the sentence. But it also contains more information than `max-pooling` by averaging all words.



Analysis of model output omitted.



