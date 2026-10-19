# LaplaceBlur
Presents, Gaussian like, low level algo - Laplace Blur.
Just a little 3% error and much more performance &amp; portablity (shift &amp; mask arithmetic &amp; bit pack magic)

Drag youre image file & click n-times for check accuracy yourself!
Console log time statistic. 

## Python
`pip install numpy pillow`, then:

    python blur.py input.png -o blurred.png -n 3

`-n` = number of blur passes (same as clicking n times).
