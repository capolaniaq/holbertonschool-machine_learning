# 0x02. Calculus

By Alexa Orrico, Software Engineer at Holberton School

## Learning Objectives

-   Summation and Product notation
-   What is a series?
-   Common series
-   What is a derivative?
-   What is the product rule?
-   What is the chain rule?
-   Common derivative rules
-   What is a partial derivative?
-   What is an indefinite integral?
-   What is a definite integral?
-   What is a double integral?
## Tasks
### 0. Sigma is for Sum

![](https://latex.codecogs.com/gif.latex?\sum_{i=2}^{5}&space;i "\sum_{i=2}^{5} i")

### 1. The Greeks pronounce it sEEgma

![](https://latex.codecogs.com/gif.latex?\sum_{k=1}^{4}&space;9i&space;-&space;2k "\sum_{k=1}^{4} 9i - 2k")

### 2. Pi is for Product

![](https://latex.codecogs.com/gif.latex?\prod_{i&space;=&space;1}^{m}&space;i "\prod_{i = 1}^{m} i")

### 3. The Greeks pronounce it pEE
![](https://latex.codecogs.com/gif.latex?\prod_{i&space;=&space;0}^{10}&space;i "\prod_{i = 0}^{10} i")

### 4. Hello, derivatives!

![](https://latex.codecogs.com/gif.latex?\frac{dy}{dx} "\frac{dy}{dx}") where ![](https://latex.codecogs.com/gif.latex?y&space;=&space;x^4&space;+&space;3x^3&space;-&space;5x&space;+&space;1 "y = x^4 + 3x^3 - 5x + 1")

### 5. A log on the fire

![](https://latex.codecogs.com/gif.latex?\frac{d&space;(xln(x))}{dx} "\frac{d (xln(x))}{dx}")

### 6. It is difficult to free fools from the chains they revere

![](https://latex.codecogs.com/gif.latex?\frac{d&space;(ln(x^2))}{dx} "\frac{d (ln(x^2))}{dx}")

### 7. Partial truths are often more insidious than total falsehoods

![](https://latex.codecogs.com/gif.latex?\frac{\partial}{\partial&space;y}&space;f(x,&space;y) "\frac{\partial f(x, y)}{\partial y}") where ![](https://latex.codecogs.com/gif.latex?f(x,&space;y)&space;=&space;e^{xy} "f(x, y) = e^{xy}") and ![](https://latex.codecogs.com/gif.latex?\frac{\partial&space;x}{\partial&space;y}=\frac{\partial&space;y}{\partial&space;x}=0 "\frac{\partial&space;x}{\partial&space;y}=\frac{\partial&space;y}{\partial&space;x}=0")

### 8. Put it all together and what do you get?

![](https://latex.codecogs.com/gif.latex?\frac{\partial^2}{\partial&space;y\partial&space;x}(e^{x^2y}) "\frac{\partial^2}{\partial&space;y\partial&space;x}(e^{x^2y})") where ![](https://latex.codecogs.com/gif.latex?\frac{\partial&space;x}{\partial&space;y}=\frac{\partial&space;y}{\partial&space;x}=0 "\frac{\partial&space;x}{\partial&space;y}=\frac{\partial&space;y}{\partial&space;x}=0")

### 9. Our life is the sum total of all the decisions we make every day, and those decisions are determined by our priorities

Write a function `def summation_i_squared(n):` that calculates ![](https://latex.codecogs.com/gif.latex?\sum_{i=1}^{n}&space;i^2 "\sum_{i=1}^{n} i^2"):
### 10. Derive happiness in oneself from a good day's work
Write a function `def poly_derivative(poly):` that calculates the derivative of a polynomial:

### 11. Good grooming is integral and impeccable style is a must

### 12. We are all an integral part of the web of life

### 13. Create a definite plan for carrying out your desire and begin at once

### 14. My talents fall within definite limitations

### 15. Winners are people with definite purpose in life

### 16. Double whammy

### 17. Integrate

Write a function `def poly_integral(poly, C=0):` that calculates the integral of a polynomial:

