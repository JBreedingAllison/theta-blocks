# theta-blocks

This repo provides Python scripts for computing theta blocks.

To install all dependencies, run 

    $ pip install -r requirements.txt

in your shell.

A theta block is a sequence of integers with finite support that defines a quotient of theta functions and eta functions. This quotient turns out to be a weakly meromorphic Jacobi form with character. 

Theta blocks were invented by V. Gritsenko, N.P. Skoruppa, and D. Zagier and can be used to construct bases of spaces of Jacobi cusp forms.

See the included Jupyter notebook for details on the functions available in this package.

Computing theta blocks relies on quickly generating sequences of integers such that the sum of their squares is equal to the index of the theta blocks of interest. The author wrote a Python version of a Perl script that generates ways of writing an integer n as the sum of k pth powers. The original code, written by another developer, can be accessed through the following link:

[representing-a-number-as-a-sum-of-k-pth-powers](https://math.stackexchange.com/questions/485159/representing-a-number-as-a-sum-of-k-pth-powers)