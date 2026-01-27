# Epicyclificator 

Date: 2019-06-15 <br>
Author: Diego Plascencia-Vega

## Background

Many articles, blog posts and videos have described the process by which arbitrary closed curve shapes can be approximated by epicycle trajectories. <a href="https://www.youtube.com/watch?v=qS4H6PEcCCA">Mathologer [1]</a> walks through an example of how to extract points from a rasterized image and approximate the shape in the image using epicycle trajectories. <a href="https://www.youtube.com/watch?v=2hfoX51f6sg"> GoldPlatedGoof [2] </a> walks through a similar process in his video, but starts with a vectorized image in .svg format. Both create these epicycle animations in Mathematica.

In this article I will explain the theory behind how this is done, and show some animations produced by the epicyclificator.


## Theory

We can describe motion around a circle using the following equation.

$$E(t)=(a+bi)e^{nit}$$

Where $(a+bi)$ defines the diameter of the circular motion and where on the circle the cycle starts. $(n)$ defines how long it takes the cycle to complete, and which direction the cycle turns.


An epicycle trajecotry can be defined using a series of nested cycles with $n=-N,\ldots,-1,0,1,\ldots,N$ . We will see later why this is a convenient series of $n$ to choose. The trajectory of the epicycle is defined by the following equation:

$$P(t)=\sum_{n=-N}^{N}(a_n+b_ni)e^{nit}=\sum_{n=-N}^{N}C_ne^{nit}$$


To find $C_n$ we multiply both sides times $e^{-int}$ resulting in:

$$P(t)e^{-nit}=\ldots+C_{n-1}e^{(n-1)it}e^{-nit}+C_ne^{nit}e^{-nit}+C_{n+1}e^{(n+1)it}e^{-nit}+\ldots$$
$$P(t)e^{-nit}=\ldots+C_{n-1}e^{-it}+C_n+C_{n+1}e^{it}+\ldots$$


We take the integral of this expression from $0$ to $2\pi$.

$$\int\limits_0^{2\pi}P(t)e^{-nit}=\ldots+\int\limits_0^{2\pi}C_{n-1}e^{-it}dt+\int\limits_0^{2\pi}C_ndt+\int\limits_0^{2\pi}C_{n+1}e^{it}dt+\ldots$$

For each epicycle, we can use the following property of epicycles:

$$\int\limits_0^{2\pi}e^{nit}dt=0$$


Which makes all of the terms on the right zero except for $C_n$. This simplifies our expression:

$$\int\limits_0^{2\pi}P(t)e^{-nit}=C_n2\pi$$
$$\frac{1}{2\pi}\int\limits_0^{2\pi}P(t)e^{-nit}=C_n$$

We can use the above expression to find all $C_n$ terms that make up the epicycle approximating the closed curve of interest.


## Epicyclificator

The goal of this python script will be to input an `.svg` file and produce a `.gif` file with an epicycle animation approximating each closed curve in the `.svg` file. For fun, we will preserve the color of the original curve in the `.svg` file in the animation.

The script can be found here:
<a href="https://github.com/diego4rmando/epicycles">make_epicycles.py</a>


## Use

<code>python epicyclificator.py path/to/file.svg --n number_of_cycles_per_epicycle</code>

<table>
    <tr>
        <th>Parameter</th>
        <th>Type</th>
        <th>Description</th>
    </tr>
    <tr>
        <td><code>filename</code></td>
        <td>str</td>
        <td>Path to <code>.svg</code> file containing paths to approximate via epicycles.</td>
    </tr>
    <tr>
        <td><code>--n</code></td>
        <td>int</td>
        <td>Number of epicycles to use in shape approximation.</td>
    </tr>
    <tr>
        <td><code>--gif_filename</code></td>
        <td>str</td>
        <td>If this parameter is set, the animation will be output in <code>.gif</code> file of this name.</td>
    </tr>
    <tr>
        <td><code>--csv_filename</code></td>
        <td>str</td>
        <td>If this parameter is set, the list of Cn values will be output in <code>.csv</code> file of this name.</td>
    </tr>
    <tr>
        <td><code>--color_theme</code></td>
        <td>str</td>
        <td>If this parameter is set to 'electric_blue' the animation colors will be overriden to shades of blue.</td>
    </tr>
</table>

## Results

<img src="mano.gif" alt="Mano" width="600">

Figure 1 - Electric Blue Mano

<img src="car_15.gif" alt="Car" width="600">

Figure 2 - Car

<img src="cactus.gif" alt="Cactus" width="600">

Figure 3 - Cactus Flowers

## References
<ul style="list-style-type:none;">
    <li><a href="https://www.youtube.com/watch?v=qS4H6PEcCCA">[1] Epicycles, complex Fourier series and Homer Simpson's orbit - Mathologer</a></li>
    <li><a href="https://www.youtube.com/watch?v=2hfoX51f6sg">[2] Fourier Analysis For The Rest Of Us - GoldPlatedGoof</a></li>
    <li><a href="https://pypi.org/project/svgpathtools/">[3] Svgpathtools - Mathandy</a></li>
    <li><a href="https://developer.mozilla.org/en-US/docs/Web/SVG/Tutorial/Paths">[4] SVG Tutorials</a></li>
</ul>

