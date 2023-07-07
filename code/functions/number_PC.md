#### Rule of Thumb
`rule_thumb()`

Stop at a $k$ such that the $(k+1)$-th PC does not add much to the already explained variance (say $<3 \%$ ).


#### Informal Way
`informal_way()`
Choose the number of components such that a large portion (say 90\%) of the variance is explained.


#### Biggest Drop
`biggest_drop()` 
Onatski (2010) suggests looking for the biggest drop computing

$$
r:=\underset{1 \leq j < n}{\arg \max} \frac{\lambda_j}{\lambda_{j+1}} 
$$