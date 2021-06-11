# Approximation of Functions

For ML, one needs to implement several common complex functions in hardware using digital logic. Many of the functions can have different implementations based on whether combinational logic or sequential logic can be used. Different implementations depend on what kind of approximation is applied to create a simpler approximated version of the original function. This idea ties directly to concepts in the mathematics field of Approximation Theory. For this work, it is assumed that a signed fixed-point representation is being used.


## Polynomial Approximations using Combinational Logic

If we use digital logic to implement complex functions, we first need to understand what kinds of functions we can build easily with digital logic. Two functions common in digital design and have been well studied and implemented are addition and multiplication. Most, if not all, FPGA tooling supports implementing and synthesizing these operations "natively," and much research has been done on various implementations and optimations around these functions.

Therefore it makes sense to constrain ourselves to use only addition and multiplication to approximate more complex functions. One such class of these approximation functions that follow these constraints is polynomials.

First, we need to know if its even posbble to apprximate any function using polynomials. We are in luck. One of the core thereoms in appromziation theory, Weierstrass approximation theorem, proves that is posbile to appximate any continus fuction on a closed interval with any desired error on that interval using polynomials. In other temrs, if I have a function I want to approximate and i know i need an error less than some vale in specifc range, I can be certian that there is anpolynial somehere i can use that can meet those contions nad have an error smaller that the one I want for the range im intresedf in. Now finding out what that polynomial actually is for a given function, error, and range is the real fun in this process. And as we will see, many clever people have come up with different ways to do so.

Putting it all together:
1. I have some complex function I want to implement on an FPGA.
2. I can approximate the function using a polynomial (because I know it is always possible to do so) using some approximation method.
3. I can  implement the approximated polynomial version of my original complex function using addition and multiplication operation on the FPGA, which is relatively easy to do.

Like all good things in engineering, there is a tradeoff here that we need to address. We get a tradeoff between approximation error and the complexity of our approximation method, which is directly proportional to the number of resources we need to implement our approximation. We can think about this as how "close enough" we are to being 100% accurate. The more effort we put in, the "closer enough" we are to what we are approximating. The primary tradeoff is between the number of terms in our polynomials, also known as the degree of the polynomial, and the approximation error. The higher the degree (number of terms), the more error we will have in our approximation. The inverse is also true, where the lower the degree is, the higher the approximation error will be. There is also a relationship between the degree of the polynomial and the number of resources needed to implement the polynomial function on an FPGA. The higher the degree of the polynomial, the more terms there are in the polynomial requiring more addition and multiplication operations needed to implement the polynomial operation of the FPGA. Addially, the method used to construe polynomial approximation can determine how much approximation error will be present in different parts of the function depending on the function you are trying to approximate. Therefore, we must put on our engineering hats and explore how to develop the best function approximation that balances error and complexity.

### Using Taylor Seires

The first idea that came to mind was uing a taylor series to appromximate a function. This concetp is commonly introduced in hgh school and college calcualus and is probbly the most expsoure that most students get to funciton approxmiation. Lets dive in.

This idea stems from a more straightforward approximation idea called linearization. In linearization, we can approximate a function around a certain point by using a straight line. The line we use is tangent to the function at the point we are uninterested in approximating around. Using a tangent achieves two goals. First, it matches the slope of our line as closely as possible to the slope of the function we are trying to approximate around the point we are interested in. Second, it makes sure that our line is near the function we want to approximate. This is done by shifting the line up or down to ensure it is close to our point of interest).What you end up with is a line that is a close approximation of a function but only near the point with which you chose to make your approximation. When I use the term near, I mean near as in the values immediately to the left and right of the point.

In theory, a linear approximation is also a polynomial approximation with degree 1. It turns out that we can extend this idea of using a derivative to approximate around a point to higher orders by taking nth order derivatives from solving for the nth degree term of a polynomial. We can choose the degree of our final polynomial and solve for the taylor expansion of our function to that degree to create an approximated version of our function.

### Using Chebyshev Polynomials


## Lookup Table Approximations using Combinational Logic

## Linear Interpolation Approximations using Combinational Logic