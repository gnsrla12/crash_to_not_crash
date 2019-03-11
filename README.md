# GTACrash

# 1. Symbolic Regression
The goal of this program is to find a expression that fits the given dataset with as small error as possible. For example, suppose your training data set includes tuples (x, y), from which you want to learn the best model f such that y = f(x) explains the given data set, as well as unseen test data set, as precisely as possible. If the training data consist of X = {1.2, 2, 3} and Y = {3.1, 4.6, 6.8}, and the test data set consist of X0 = {6, 5} and Y 0 = {13, 10.5}, one possible symbolic regression model would be y = 2x + 1. Figure 1 shows the result of symbolic regression.

![Alt text](Figures/Figure1.png?raw=true "Title")

With Genetic Programming, you can build candidate expressions for f , and evaluate them using Mean Square Error, which is calculated as follows:

![Alt text](Figures/Figure2.png?raw=true "Title")

# 2. Dataset
The symbolic regression dataset we are using contains 57 input variables (x1,...,x57), 1 output variables, y and contains 747 rows. Using this dataset, this program evolves a symbolic regression model.

# 3. How to get started

The first program, "train.py" is an implementation of GP that takes a .csv file containing the training data as input, and prints out the evolved expression using Reverse Polish Notation. 

The second program, "test.py" takes two inputs: the evolved RPN expression in one string, and a .csv file containing the test data. Subsequently, it evaluatse the given RPN expression on the test data and print out the MSE.

Following ternimal and non-terminal nodes for GP, and corresponding symbols when printing out the evolved expression in RPN are used:
-  Terminals: x1,. . ., x57, as well as any floating point constant numbers
-  Unary Operators: ⇠ (unary minus), abs, sin, cos, tan, asin, acos, atan, sinh, cosh,
tanh, exp, sqrt, log
-  Binary Operators: +, -, *, /,ˆ(power)


```sh
# For example, x24   sin( 2x3) would be represented by x24 2 ^ 2 ~ x3 * sin -.
# in directory "Simple_Genetic_Programming/"
$ python training.py test.csv
...
x24 2 ^ 2 ~ x3 * sin - # this RPN equation can be different everytime the program is run
$ python test.py "x24 2 ^ 2 ~ x3 * sin -" test.csv

```
