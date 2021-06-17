# Automatic_Regression_Line_Plot

Purpose: Automate the construction of a "nice-looking" linear regression plot with a table of confidence intervals for the slope and y-intercept of the regression line alongside. Also, I got sick of undergraduate CS students telling me I'll never understand ML because I'm too dumb to even do regression properly.

## About:
Regression analysis is a method designed to ascertain a functional relationship among variables [[1]](#1). A particular type of regression is linear regression when there are two variables: a predictor variable X and a response variable Y, and we assume that the regression of Y on X is linear, meaning that E(Y|X=x) = mx + b [[1]](#1). In the standard notation of statistics, the y-intercept is denoted \beta_{0} and the slope is denoted \beta{1} [[1]](#1). To obtain a functional relationship among X and Y is to analyze the raw data and find a "line of best fit" that seems to closely match the trend in the data. Because we are assuming the regression of Y on X is linear but do not know the population values characterizing a the linear regression, we construct a sample regression line in the form y = b_{0} + b_{1}x [[1]](#1). A very popular method of choosing b_{0} and b_{1} is called the method of least squares. The method of least squares seeks to minimize the sum of the squares of the residuals, the residuals being the difference between the actual value of y (the data point) and the predicted value \hat{y} [[1]](#1). Performing this optimization problem leads to "normal equations" that can be solved to obtain formulas for both the sample slope and y-intercept of the regression line [[1]](#1).

This program employs the method of least squares to obtain a functional form of the regression line. Further, it test the hypotheses that the slope and y-intercept of the regression line are 0 using 80%, 90%, 95%, 98%, 99%, and 99.9% confidence intervals. Both the plot of the raw data with the overlayed regression line (and equation) and a table displaying upper and lower bounds for the regression line slope and y-intercept are present in the figure. At the present moment, the figure looks what one might call "pretty good," but it definitely can be optimized. It's somewhat of a pain wrestling with Matplotlib. Any and all suggestions are welcome.

## References;
<li>
<a id = "1">[1]</a>
 Sheather, Simon J. <i>A Modern Approach to Regression with R.</i> Springer, 2009.
</li>

## Parenthetical:
https://stackoverflow.com/a/327011 -> for math.sqrt(x) being faster than x^0.5 
https://stackoverflow.com/a/42827330 -> for finding the "longest" list in another list
https://stackoverflow.com/a/4690655 -> for how to access the error as "e"
https://stackoverflow.com/a/3411435 -> for rounding a number to n
https://stackoverflow.com/q/30281485 -> for typesetting LaTeX in Matplotlib 
https://stackoverflow.com/a/2476868 -> for dealing with annoying LaTeX string parsing with Matplotlib
https://stackoverflow.com/a/46664216 -> for future reference, consider using Pandas DF for simple Matplotlib tables
https://stackoverflow.com/a/15514091 -> for why Matplotlib tables' fontsize is broken & need a workaround
https://stackoverflow.com/a/46664216 -> for why and how on earth I need to color every table cell and column
https://stackoverflow.com/a/38173860 -> for horizonally aligning text within Matplotlib table
https://stackoverflow.com/a/17793421 -> for getting things in a dictionary in a for loop
https://stackoverflow.com/a/30280874 -> for "generating" a dictionary in a for loop
https://stackoverflow.com/a/46664216 -> for future reference, consider using Pandas DF for simple Matplotlib tables
https://stackoverflow.com/a/15514091 -> for why Matplotlib tables' fontsize is broken & need a workaround
https://stackoverflow.com/a/46664216 -> for why and how on earth I need to color every table cell and column
