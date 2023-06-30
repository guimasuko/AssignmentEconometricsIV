#### Residual Sum of Squares 
`rss()`

$$
RSS = \sum_{t=1}^T (y_t - \hat{y}_t)^2
$$


#### Explained Sum of Squares`
`ess()`

$$
ESS = \sum_{t=1}^T (\hat{y}_t - \bar{y})^2
$$


#### Total Sum of Squares`
`tss()`

$$
TSS = \sum_{t=1}^T (y_t - \bar{y})^2
$$


#### R-Squared 
`r_squared()`

$$
R^2 = 1 - \frac{RSS}{TSS}
$$


#### Root Mean Squared Error
`rmse()`

$$
RMSE = \sqrt{\left(\frac{1}{T}\sum_{t=1}^T (y_t - \hat{y}_t)^2\right)}
$$