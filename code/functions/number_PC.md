#### Rule of Thumb

Stop at a $k$ such that the $(k+1)$-th PC does not add much to the already explained variance (say $<3 \%$ ).


#### Informal Way
Choose the number of components such that a large portion (say 90\%) of the variance is explained.


#### Biggest Drop 
Onatski (2010) suggests looking for the biggest drop computing
```math
r:=\arg \max _{1 \leq j<n} \frac{\lambda_j}{\lambda_{j+1}} .
```