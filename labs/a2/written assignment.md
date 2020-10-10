# Assignment2

将$u,v,y$都视作列向量,$U,V$的列向量视作$u,v$。

## (a)

$$
-\sum_{w \in V} y_w(\log \hat{y}_w) =-1·\log \hat{y}_o -\sum_{w!=o}0·\log \hat{y}_w=-\log \hat{y}_o
$$



## (b)

$$
\frac{\partial{J(v_c, o, U)}}{\partial{v_c}}=u_o-\sum_{w \in V}u_wP(w|c)=u_o-\sum_{w \in V}u_w \hat{y}_w=Uy-U\hat{y}=U(y-\hat{y})
$$

注意最后要变成矩阵形式

## (c)

#### $w=o$

$$
\frac{\partial{J(v_c, o, U)}}{\partial{u_o}}=v_c-v_c\frac{e^{u_o^Tv_c}}{\sum e^{u_wTv_c}}=v_c(1-\hat{y}_o)
$$

#### $w \neq o$

$$
\frac{\partial{J(v_c, o, U)}}{\partial{u_w}}=0-v_c =-v_c\frac{e^{u_w^Tv_c}}{\sum e^{u_wTv_c}}=-v_c\hat{y}_w
$$

#### $U$

$$
\frac{\partial{J(v_c, o, U)}}{\partial{U}}=v_c(y-\hat{y})^T
$$

注意最后要变成矩阵形式

## (d)

$$
\frac{\partial{\sigma{(x)}}}{\partial x}=\sigma{(x)}·(1-\sigma(x))
$$

## (e)

#### $v_c$

$$
\frac{\partial{J(v_c, o, U)}}{\partial{v_c}}=u_o(1-\sigma{(u_o^Tv_c)})-\sum u_k(1-\sigma{(-u_k^Tv_c)})
$$

#### $u_o$

$$
\frac{\partial{J(v_c, o, U)}}{\partial{u_o}}=v_c(1-\sigma({u_o^Tv_c}))
$$

#### $u_k$

$$
\frac{\partial{J(v_c, o, U)}}{\partial{u_k}}=-v_c(1-\sigma(-u_k^Tv_c))
$$



negative-sampling比naive softmax更高效因为

+ 不用计算e的幂次
+ 每次只要计算K组点积，而不是所有$w \in V$

## (e)

用naive softmax:

#### (i)

$$
\frac{\partial{J(v_c, w_{t-m},...,w_{t+m}, U)}}{\partial{U}}=\sum_{j}\frac{\partial{J(v_c, w_{t+j}, U)}}{\partial{U}}
$$

#### (ii)

$$
\frac{\partial{J(v_c, w_{t-m},...,w_{t+m}, U)}}{\partial{v_c}}=\sum_{j}\frac{\partial{J(v_c, w_{t+j}, U)}}{\partial{v_c}}
$$



#### (iii)

$$
\frac{\partial{J(v_c, w_{t-m},...,w_{t+m}, U)}}{\partial{v_w}}=0
$$