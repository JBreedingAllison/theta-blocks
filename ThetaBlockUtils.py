#!/usr/bin/env python
"""
This file provides functions for theta block computations.

Author: Jeff Breeding-Allison
  Date: March 2019
"""

import numpy as np
from sympy import Matrix
import matplotlib.pyplot as plt


def get_divisors(n):
    """
    This function returns a list of all divisors of an integer n.

    Args:
        n          (int): An integer
    Returns:
        n_divs    (list): The list of divisors of n.
    """
    # Replace n by its absolute value
    n = abs(n)

    # If n is 0, return only the list with 0 as an item
    if n == 0:
        return [0]

    # Compute the upperbound of divisors less than n
    div_upperbound = int(np.floor(n / 2))

    # Create container for the divisors
    n_divs = []

    # Find divisors
    for i in range(1, div_upperbound + 1):
        if n % i == 0:
            n_divs.append(i)

    # Append n to the list of divisors
    n_divs.append(n)

    return n_divs


def is_square(m):
    """
    This function checks if an integer is a square.

    Args:
        m    (int): An integer
    Returns:
        is_square_bool    (bool): True or False, indicating if m is a square
    """
    is_square_bool = (np.sqrt(m) == int(np.sqrt(m)))

    return is_square_bool


def dimension_cusp_forms(k):
    """
    This function computes the dimension of the space of cusp forms
    of weight k and full level.

    Args:
        k         (int): A positive integer indicating the weight of the
                         space of modular forms

    Returns:
        dim_sk    (int): The dimension of the space of cusp
                                 forms of weight k and full level
    """
    if k % 2 == 1:
        dim_sk = 0
        return dim_sk

    # Compute integers j and r such that k = 12*j + r
    # with r < 12
    j = int(k / 12)
    r = k - 12 * j

    if r in [0, 4, 6, 8, 10]:
        dim_mk = j + 1
    else:
        dim_mk = j

    dim_sk = dim_mk
    if k >= 4:
        dim_sk -= 1

    return dim_sk


def dim_jkm_cusp(k,m):
    """
    This function computes the dimension of the space of Jacobi cusp forms
    of weight k and index m

    Args:
        k   (int): A positive integer indicating the weight
        m   (int): A positive integer indicating the index

    Returns:
        dim_jkm_cusp  (int): The dimension of the space of Jacobi cusp forms
                           of weight k, index m
    """
    # Get the divisors of m
    m_divs = get_divisors(m)
    num_m_divs = len(m_divs)

    # Set delta_km value
    delta_km = 0

    # Special case for delta_km value
    if k == 2:
        if is_square(m):
            delta_km = (1 / 2) * (num_m_divs - 1)
        else:
            delta_km = (1 / 2) * num_m_divs - 1

    # Even weight case
    if k % 2 == 0:
        dim_jkm_cusp = \
            delta_km + sum(dimension_cusp_forms(k + 2*j) \
                - np.floor((j**2)/(4*m)) for j in range(m+1))

    else:
        if k >= 3:
            dim_jkm_cusp = \
                sum(dimension_cusp_forms(k + 2*j - 1)
                    - np.floor((j**2)/(4*m)) for j in range(1, m))
        else:
            dim_jkm_cusp = 0

    # Convert to integer
    dim_jkm_cusp = int(dim_jkm_cusp)

    return dim_jkm_cusp


def power_representations(n, k, p):
    """
    This function returns all of the ways of writing a positive integer
    n as the sum of k pth powers of non-negative integers (up to permutation).

    Args:
        n                   (int): A positive integer
        k                   (int): A positive integer indicating
                                   the number of pth power terms in the sum
        p                   (int): A positive integer, the power of each term
                                   in the sum

    Returns:
        power_rep_solns    (list): A list of tuples of solutions
                                      (x1, x2, ... xk)
                                   in ZZ_{>=0} to the equation
                                       x1^p + x2^p + ... + xk^p = n
                                   with x1 <= x2 <= ... <= xk
    """
    # Create containers for the dictionaries used to build the solution set
    known_power_reps = {}
    power_menu = {}

    # Build the power_menu dictionary and initial values in
    # the known_power_reps dictionary
    for q in range(1, int(n**(1/p))+1):
        val = q**p
        cur_power_menu_key = str(q)
        cur_kpr_key = "{}-1".format(val) # “n-1” is the key to lookup the
                                         # pth root of n, or, equivalently,
                                         # how to write n as
                                         # the sum of 1 pth power

        power_menu.update({cur_power_menu_key: val})
        known_power_reps.update({cur_kpr_key: [q]})

    # Create a container for the solutions
    power_rep_solutions = []

    # Set maximum number of zero terms
    zero_terms = k-1

    # Generate solutions
    while zero_terms >= 0:
        non_zero_parts, power_menu, known_power_reps = \
            power_representations_builder(n,
                                          k-zero_terms,
                                          p,
                                          power_menu,
                                          known_power_reps)

        if non_zero_parts != []:
            for non_zero_part in non_zero_parts:
                cur_solution = [0]*zero_terms
                cur_solution.extend(non_zero_part)
                power_rep_solutions.append(cur_solution)

        zero_terms -= 1

    power_rep_solutions = list(set(tuple(i) for i in power_rep_solutions))
    return power_rep_solutions


def power_representations_builder(n, k, p, power_menu, known_power_reps):
    """
    This function helps build representations of n by a sum of k pth powers.

    Args:
        n                   (int): A positive integer
        k                   (int): A positive integer indicating
                                   the number of pth power terms in the sum
        p                   (int): A positive integer, the power of each term
                                   in the sum
        power_menu         (dict): A dictionary used to quickly locate of pth
                                   powers of integers with entries in the form
                                   {m^p: m}
        known_power_reps   (dict): A dictionary of known power representations

    Returns:
        reps               (list): A list of power representations of n by
                                   k pth powers
        power_menu         (dict): A dictionary used to quickly locate of pth
                                   powers of integers with entries in the form
                                   {m^p: m}
        known_power_reps   (dict): The updated dictionary of known power
                                   representations that includes representing
                                   n as the sum of k pth powers
    """
    cur_kpr_key = "{}-{}".format(n, k) # “n-k” is the key to look up
                                       # the ways that n can be written
                                       # as the sum of k pth powers
    if cur_kpr_key in known_power_reps.keys():
        return known_power_reps[cur_kpr_key], power_menu, known_power_reps

    if k == 1:
        return [], power_menu, known_power_reps

    reps = []
    for b in power_menu.keys():
        bp = power_menu[b]
        if bp < n:
            cur_kpr, power_menu, known_power_reps = \
                power_representations_builder(n-bp, k-1, p,
                                              power_menu, known_power_reps)
            if cur_kpr != []:
                for rep_soln in cur_kpr:
                    cur_soln = [int(b)]
                    if type(rep_soln) is int:
                        rep_soln = [rep_soln]
                    cur_soln.extend(rep_soln)
                    cur_soln.sort()
                    reps.append(cur_soln)
    known_power_reps.update({cur_kpr_key: reps})

    return reps, power_menu, known_power_reps


def non_zero_power_representations(n, k, p):
    """
    This function returns all of the ways of writing a positive integer n
    as the sum of k pth powers of positive integers (up to permutation).

    Args:
        n                       (int): A positive integer
        k                       (int): A positive integer indicating
                                       the number of pth power terms
                                       in the sum
        p                       (int): A positive integer, the power of
                                       each term in the sum

    Returns:
        non_zero_power_reps    (list): A list of tuples of solutions
                                                (x1, x2, ... xk)
                                       in ZZ_{>0} to the equation
                                            x1^p + x2^p + ... + xk^p = n
                                       with x1 <= x2 <= ... <= xk
    """
    # Get the power representations with zeros allowed
    all_power_reps = power_representations(n, k, p)

    # Create a container for the non-zero power representations
    non_zero_power_reps = []
    for cur_power_rep in all_power_reps:
        if cur_power_rep[0] > 0:
            non_zero_power_reps.append(cur_power_rep)

    return non_zero_power_reps


def integer_counter(int_tuple):
    """
    This function counts the occurences of integers in a tuple.

    Args:
        int_tuple    (tuple): A tuple of positive integers

    Returns:
        int_counts    (list): A list of counts of the integers in int_tuples
                                in the form [n1, n2, ..., nk]
                                where ni is the count of the occurrences of the
                                integer i in int_tuples and k is the largest
                                integer in int_tuple
    """
    # Create a container for the integer counts
    int_counts = []

    tuple_max = max(int_tuple)
    for i in range(1, tuple_max+1):
        cur_count = int_tuple.count(i)
        int_counts.append(cur_count)

    return int_counts


def convert_to_theta_block_notation(prelim_tb, weight):
    """
    This function converts a tuple of positive integers and a weight
    to a theta block

    Args:
        prelim_tb    (tuple): A tuple of positive integers
        weight         (int): A positive integer

    Returns:
        theta_block   (list): A list of integers obtained in the form
                              [2*weight, f(1), f(2), ..., f(k)]
                              where f(n) is the number of occurences of n in
                              prelim_tb. This new list defines a theta block.
    """
    first_part = [2*weight]
    second_part = integer_counter(prelim_tb)

    theta_block = first_part + second_part

    return theta_block


def convert_from_theta_block_notation(theta_block):
    """
    This function converts a tuple of positive integers and the weight
    of a theta block

    Args:
        theta_block    (tuple): A list of integers giving a theta block
                                in the form [2*weight, f(1), f(2), ..., f(n)]

    Returns:
        int_reps        (list): A tuple of integers in ascending order,
                                each positive integer k for 1<=k<=n occurs
                                precisely f(k) times.
    """
    int_reps = []
    for i in range(1, len(theta_block)):
        i_mult = theta_block[i]
        for _ in range(i_mult):
            int_reps.append(i)

    int_reps = tuple(int_reps)

    return int_reps


def theta_block_L(theta_block):
    """
    This function computes the number L that is the power of the character
    \nu_H for a theta block.

    Args:
        theta_block    (list): A list of integers giving a theta block

    Returns:
        L               (int): The power of the character
                               nu_H for the theta block
    """
    L = \
        theta_block[0] + \
            sum(theta_block[i] for i in range(1, len(theta_block)))

    return L


def theta_block_K(theta_block):
    """
    This function computes the number K that is the power of the character
    epsilon for a theta block.

    Args:
        theta_block    (list): A list of integers giving a theta block

    Returns:
        K               (int): The power of the character epsilon
                               for the theta block
    """
    K = \
        theta_block[0] + \
            2*sum(theta_block[i] for i in range(1, len(theta_block)))

    return K


def get_theta_block_shapes(weight, num_shapes=1):
    """
    This function gets the shapes of the theta blocks
    that have trivial character.

    Args:
        weight        (int): The weight of the theta blocks

        num_shapes    (int): The number of shapes

    Returns:
        shapes       (list): A list of the allowable lengths of sums of squares

    """
    # Create a container to hold the shapes
    shapes = []

    for j in range(1, num_shapes+1):
        cur_K = 24*j
        cur_num_squares = int((cur_K - 2*weight)/2)
        shapes.append(cur_num_squares)

    return shapes


def get_weight(theta_block):
    """
    This function gets the weight of a theta block.

    Args:
        theta_block          (list): A list that defines a theta block

    Returns:
        theta_block_weight    (int): The weight of the theta block
    """
    theta_block_weight = int(theta_block[0]/2)

    return theta_block_weight


def theta_block_index(theta_block):
    """
    This function computes the index of a theta block.

    Args:
        theta_block       (list): A list that defines a theta block

    Returns:
        theta_block_idx    (int): The index of the theta block
    """
    theta_block_size = len(theta_block)
    theta_block_idx = \
        sum((1/2)*(i**2)*theta_block[i] for i in range(1, theta_block_size))

    theta_block_idx = int(theta_block_idx)

    return theta_block_idx


def theta_block_order_function(theta_block, x):
    """
    This function evaluates a theta block's order at x

    Args:
        theta_block    (list): A list of integers defining a theta block
        x             (float): A real number

    Returns:
        order_at_x    (float): The theta block's order at x
    """
    theta_block_size = len(theta_block)
    theta_block_weight = get_weight(theta_block)
    order_at_x = (1/12)*theta_block_weight
    for i in range(1, theta_block_size):
        b_factor = (i*x - np.floor(i*x))**2 - (i*x - np.floor(i*x)) + 1/6
        cur_term = (1/2)*theta_block[i]*b_factor
        order_at_x += cur_term

    return order_at_x


def theta_block_minimum_order(theta_block):
    """
    This function computes the minimum order of a theta block

    Args:
        theta_block    (list): A list of integers defining a theta block

    Returns:
        min_order     (float): The minimum order of the theta block
    """
    theta_block_idx = theta_block_index(theta_block)
    min_order = theta_block_order_function(theta_block, 0)

    for i in range(2*theta_block_idx +1):
        cur_order = \
            theta_block_order_function(theta_block, i/(2*theta_block_idx))
        if cur_order < min_order:
            min_order = cur_order

    return min_order


def show_theta_block_order_graph(theta_block):
    """
    This function graphs the order of a theta block on the interval [0,1]

    Args:
        theta_block         (list): A list of integers defining a theta block

    Returns:
        plt    (matplotlib.pyplot): A plot of the order of the theta block
                                    at x for x in the interval [0, 1]
    """
    t = np.arange(0, 1, 0.001)

    plt.plot(t, theta_block_order_function(theta_block, t),)
    plt.title("Order of {}".format(str(theta_block)))
    plt.grid()
    plt.show()

    return plt


def is_holomorphic(theta_block):
    """
    This function tests if a theta block is holomorphic

    Args:
        theta_block          (list): A list of integers defining a theta block

    Returns:
        is_holomorphic_val   (bool): Is the theta block holomorphic?
    """
    tb_min_order = theta_block_minimum_order(theta_block)

    if tb_min_order > 0:
        is_holomorphic_val = True
    else:
        is_holomorphic_val = False

    return is_holomorphic_val


def is_cuspidal(theta_block):
    """
    This function tests if a theta block is a holomorphic Jacobi cusp form

    Args:
        theta_block       (list): A list of integers defining a theta block

    Returns:
        is_cuspidal_val   (bool): Is the theta block cuspidal?
    """
    tb_min_order = theta_block_minimum_order(theta_block)

    if tb_min_order > 0:
        is_cuspidal_val = True
    else:
        is_cuspidal_val = False

    return is_cuspidal_val


def get_theta_blocks(idx, weight, num_shapes=1):
    """
    This function gets all theta blocks of weight 2 with the shape
    of 10 thetas over 6 etas.

    Args:
        idx              (int): The index of the theta blocks

        weight           (int): The weight of the theta blocks

        num_shapes      (list): The number of theta block shapes to consider

    Returns:
        theta_blocks    (list): A list of theta blocks given as tuples
    """
    tb_shapes = get_theta_block_shapes(weight, num_shapes)

    # Create a container to hold the theta blocks
    theta_blocks = []
    for tb_shape in tb_shapes:
        tbs = non_zero_power_representations(2*idx, tb_shape, 2)

        for tb in tbs:
            converted_tb = convert_to_theta_block_notation(tb, weight)
            theta_blocks.append(converted_tb)

    return theta_blocks


def get_cuspidal_theta_blocks(idx, weight, num_shapes=1):
    """
    This function gets all cuspidal theta blocks of weight 2 with the
    shape of 10 thetas over 6 etas

    Args:
        idx              (int): The index of Jacobi forms

        weight           (int): The weight of the theta blocks

        num_shapes      (list): The number of theta block shapes to consider


    Returns:
        cuspidal_tbs    (list): A list of theta blocks given as tuples
    """
    # Get all theta blocks:
    theta_blocks = get_theta_blocks(idx, weight, num_shapes)

    # Create a container for the cuspidal theta blocks
    cuspidal_tbs = []

    # Find which theta blocks are cuspidal
    for tb in theta_blocks:
        if is_cuspidal(tb):
            cuspidal_tbs.append(tb)

    return cuspidal_tbs


def dedekind_eta(max_power):
    """
    This function is the Dedekind eta function eta(q)/q^(1/24) expanded up to
    a given maximum power

    Args:
        max_power              (int): The maximum power of the series expansion

    Returns:
        summation_dict        (dict): A dictionary of terms in the q-expansion
                                      with entries in the form
                                      {q_power: coefficient}
    """
    # Define the initial dictionary that will hold the series terms
    summation_dict = {0: 1}

    # Compute an upper bound for the n's to consider
    n_ub = int((1 + np.sqrt(1 + 24*max_power))/6)

    for n in range(1, n_ub + 1):
        cur_coeff = (-1)**n
        cur_first_q_term_power = n*(3*n - 1)/2

        if cur_first_q_term_power == int(cur_first_q_term_power):
            cur_first_q_term_power = int(cur_first_q_term_power)
        if cur_first_q_term_power in summation_dict.keys():
            summation_dict[cur_first_q_term_power] += cur_coeff
        else:
            if cur_first_q_term_power <= max_power:
                summation_dict.update({cur_first_q_term_power: cur_coeff})

        cur_second_q_term_power = n*(3*n + 1)/2

        if cur_second_q_term_power == int(cur_second_q_term_power):
            cur_second_q_term_power = int(cur_second_q_term_power)
        if cur_second_q_term_power in summation_dict.keys():
            summation_dict[cur_second_q_term_power] += cur_coeff
        else:
            if cur_second_q_term_power <= max_power:
                summation_dict.update({cur_second_q_term_power: cur_coeff})

    # Complete the dictionary
    dict_keys = sorted(summation_dict.keys())
    max_key = dict_keys[-1]

    if max_key == max_power:
        return summation_dict

    else:
        summation_dict.update({max_power:0})

    return summation_dict


def print_one_var_expansion(summation_dict, var_name):
    """
    This function prints a pretty version of the series expansion indicated
    by a dicionary.

    Args:
        summation_dict        (dict): A dictionary of terms in the series
                                      expansion with entries in the form
                                      {variable_power: coefficient}

        var_name              (str): The variable name

    Returns:
        printed_expansion    (str): A string showing the series expansion
                                    in the input var_name
    """
    sorted_powers = sorted(summation_dict.keys())
    max_power = sorted_powers[-1]

    printed_expansion = ""
    # Make a pretty expansion
    for var_power in sorted_powers:
        if var_power == 0:
            printed_expansion += "1"
        else:
            if summation_dict[var_power] != 0:
                printed_expansion += \
                    " + ({}){}^({})".format(summation_dict[var_power],
                                            var_name,
                                            var_power)

    printed_expansion += " + O({}^({}))".format(var_name, max_power+1)

    printed_expansion = \
        printed_expansion.replace("{}^(1)".format(var_name),
                                  "{}".format(var_name))
    # printed_expansion = printed_expansion.replace("+ (1)", "+ ")
    # printed_expansion = printed_expansion.replace("+ (-1)", "- ")

    return printed_expansion


def jacobi_odd_theta_function(max_q_order, ell=1):
    """
    This functions computes the series expansion of Jacobi's
    odd theta function theta(tau, ell*z) up to a maximum q order,
    where Q^8 = q, and Z^2 = zeta.

    Args:
        max_q_order            (int): The maximum q power in the
                                      returned series expansion

        ell                    (int): A positive integer

    Returns:
        summation_dict        (dict): A dictionary of terms in the q-expansion
                                      with entries in the form
                                      {"Q_power|Z_power": coefficient}
    """
    # Define the initial dictionary that will hold the series terms
    summation_dict = {"1|{}".format(ell): 1}

    # Compute how many terms to calculate
    max_n = int((np.sqrt(8*max_q_order)+1)/2)

    # Compute the terms in the series expansion
    for n in range(1, max_n):
        cur_coeff = (-1)**n

        first_term_Q_power = (1 + 2*n)**2
        first_term_Z_power = ell*(1 + 2*n)
        second_term_Q_power = (1 - 2*n)**2
        second_term_Z_power = ell*(1 - 2*n)
        first_term_key = \
            "{}|{}".format(first_term_Q_power, first_term_Z_power)
        second_term_key = \
            "{}|{}".format(second_term_Q_power, second_term_Z_power)

        if first_term_key not in summation_dict:
            summation_dict.update({first_term_key: cur_coeff})
        else:
            summation_dict[first_term_key] += cur_coeff

        if second_term_key not in summation_dict:
            summation_dict.update({second_term_key: cur_coeff})
        else:
            summation_dict[second_term_key] += cur_coeff

    return summation_dict


def print_QZ_function(summation_dict, max_q_order):
    """
    This function prints a function expanded in terms of Q and Z
    """
    summation_keys = summation_dict.keys()
    printed_expansion = ""
    for i in range(1, 8*max_q_order +1):
        cur_keys = []
        for skey in summation_keys:
            skey_split = skey.rsplit('|')
            skey_Q_power = int(skey_split[0])
            if skey_Q_power == i:
                cur_keys.append(skey)

        if cur_keys != []:
            cur_print_term = "("
            for good_key in cur_keys:
                good_key_split = good_key.rsplit('|')
                good_key_Q_power = int(good_key_split[0])
                good_key_Z_power = int(good_key_split[1])

                if good_key_Z_power != 0:
                    cur_print_term += \
                        "({})Z^({}) + ".format(summation_dict[good_key],
                                               good_key_Z_power)
                else:
                    cur_print_term += \
                        "({}) + ".format(summation_dict[good_key])

            cur_print_term += ")*Q^({}) + ".format(good_key_Q_power)
            printed_expansion += cur_print_term

    # Make pretty
    printed_expansion = printed_expansion.replace(" + )", ")")
    printed_expansion = printed_expansion.replace("((1)", "(")
    # printed_expansion = printed_expansion.replace("+ (1)", "+ ")
    # printed_expansion = printed_expansion.replace("+ (-1)", "- ")

    printed_expansion += "O(Q^({}))".format(8*(max_q_order+1))

    return printed_expansion


def q_to_Q_expansion(q_summation_dict):
    """
    This function produces the series expansion of a q-expansion
    if q was replaced with Q^8

    Args:
        q_summation_dict    (dict): A dictionary indicating a
                                    q-expansion

    Returns:
        Q_summation_dict    (dict): A dictionary indicating the
                                    Q-expansion of q_summation_dict
    """
    q_keys = sorted(q_summation_dict.keys())

    Q_summation_dict = {}
    for cur_q_key in q_keys:
        cur_Q_key = 8*cur_q_key
        cur_Q_val = q_summation_dict[cur_q_key]

        Q_summation_dict.update({cur_Q_key: cur_Q_val})

    return Q_summation_dict


def Q_to_QZ_expansion(Q_summation_dict):
    """
    This function makes a Q expansion dictionary into a QZ
    expansion dictionary

    Args:
        Q_summation_dict     (dict): A dictionary indicating a
                                     Q-expansion

    Returns:
        QZ_summation_dict    (dict): A dictionary indicating the
                                     QZ-expansion of the Q-expansion
    """
    Q_keys = Q_summation_dict.keys()

    QZ_summation_dict = {}
    for cur_Q_key in Q_keys:
        cur_QZ_key = "{}|{}".format(cur_Q_key, 0)
        cur_QZ_val = Q_summation_dict[cur_Q_key]

        QZ_summation_dict.update({cur_QZ_key: cur_QZ_val})

    return QZ_summation_dict


def multiply_QZ_expansions(summation_dict1, summation_dict2):
    """
    This function multiplies two series expansions.

    Args:
        summation_dict1        (dict): A dictionary indicating
                                       terms in a QZ-expansion
        summation_dict2        (dict): A dictionary indicating
                                       terms in a QZ-expansion

    Returns:
        mult_summation_dict    (dict): A dictionary indicating
                                       terms in the QZ-expansion
                                       determined by the product
                                       of summation_dict1 and
                                       summation_dict2
    """
    # Create a container for the QZ-expansion dictionary
    mult_summation_dict = {}

    keys1 = summation_dict1.keys()
    keys2 = summation_dict2.keys()

    for cur_key1 in keys1:
        cur_key1_coeff = summation_dict1[cur_key1]
        cur_key1_split = cur_key1.split("|")
        cur_key1_Q_pow = int(cur_key1_split[0])
        cur_key1_Z_pow = int(cur_key1_split[1])
        for cur_key2 in keys2:
            cur_key2_coeff = summation_dict2[cur_key2]
            cur_key2_split = cur_key2.split("|")
            cur_key2_Q_pow = int(cur_key2_split[0])
            cur_key2_Z_pow = int(cur_key2_split[1])

            cur_mult_coeff = cur_key1_coeff * cur_key2_coeff
            cur_mult_Q_pow = cur_key1_Q_pow + cur_key2_Q_pow
            cur_mult_Z_pow = cur_key1_Z_pow + cur_key2_Z_pow

            cur_mult_key = \
                "{}|{}".format(cur_mult_Q_pow, cur_mult_Z_pow)

            if cur_mult_key in mult_summation_dict.keys():
                mult_summation_dict[cur_mult_key] += cur_mult_coeff
            else:
                mult_summation_dict.update({cur_mult_key: cur_mult_coeff})

    return mult_summation_dict


def get_theta_block_theta_factor(theta_block, max_q_power):
    """
    This function computes the factor composed of Jacobi's odd
    theta function for a theta block

    Args:
        theta_block      (list): A theta block
        max_q_power       (int): The largest q-power in the expansion

    Returns:
        theta_dict       (dict): A dictionary giving the terms in the
                                 theta factor's q-expansion
    """
    tb2 = convert_from_theta_block_notation(theta_block)

    # Get the summation dictionaries of the factors
    theta_factors = []
    theta_factor_count = 0
    for i in range(len(tb2)):
        cur_theta_factor_dict = \
            jacobi_odd_theta_function(max_q_power, tb2[i])
        theta_factors.append(cur_theta_factor_dict)
        if theta_factor_count == 0:
            theta_dict = cur_theta_factor_dict
            theta_factor_count += 1
        else:
            theta_dict = \
                multiply_QZ_expansions(theta_dict, cur_theta_factor_dict)

            # Truncate the dictionary
            cur_mult_keys = theta_dict.keys()
            good_keys = []
            for new_mult_key in cur_mult_keys:
                new_mult_key_split = new_mult_key.split("|")
                new_mult_key_Q_pow = int(new_mult_key_split[0])
                if new_mult_key_Q_pow <= 8*max_q_power:
                    good_keys.append(new_mult_key)

            theta_dict = \
                {k:theta_dict[k] for k in good_keys}

    return theta_dict


def get_expansion_reciprocal(summation_dict, max_power):
    """
    This function computes a dictionary that gives an expansion
    of the reciprocal of a series expansion in x.

    Args:
        summation_dict          (dict): A dictionary giving terms in a
                                        function's series expansion

        max_power                (int): The largest x-power in the returned
                                        expansion

    Returns:
        recip_summation_dict    (dict): A dictionary giving terms in the
                                        reciprocal function's expansion up to
                                        and including the max_x_power
    """
    # Get the keys
    x_keys = summation_dict.keys()

    # Initialize the recip_summation_dict
    recip_summation_dict = {}

    # Check if the reciprocal exists
    try:
        b0 = summation_dict[0]
        if b0 != 0:
            c0 = 1./b0
            recip_summation_dict.update({0: c0})
        else:
            print("The reciprocal does not exist")
            return None
    except:
        print("The reciprocal does not exist")
        return None

    # Build the dictionary
    for i in range(1, max_power+1):
        cur_recip_keys = recip_summation_dict.keys()

        cur_c = 0
        for k in range(0, i):
            if (k in cur_recip_keys) and ((i - k) in x_keys):
                cur_ck = recip_summation_dict[k]
                cur_b_n_minus_k = summation_dict[i - k]
                cur_c -= c0 * cur_ck * cur_b_n_minus_k

        recip_summation_dict.update({i: cur_c})

    return recip_summation_dict


def multiply_one_var_expansions(summation_dict1, summation_dict2, max_x_power):
    """
    This function multiplies two series expansions.

    Args:
        summation_dict1         (dict): A dictionary indicating
                                        terms in a series expansion
        summation_dict2         (dict): A dictionary indicating
                                        terms in a series expansion
        max_x_power              (int): The largest x-power in the returned
                                        expansion

    Returns:
        mult_summation_dict     (dict): A dictionary indicating
                                        terms in the series expansion
                                        determined by the product
                                        of summation_dict1 and
                                        summation_dict2
    """
    # Get the powers
    keys1 = summation_dict1.keys()
    keys2 = summation_dict2.keys()

    # Create a container for the expansion dictionary
    mult_summation_dict = {}

    for cur_key1 in keys1:
        cur_key1_coeff = summation_dict1[cur_key1]
        for cur_key2 in keys2:
            cur_key2_coeff = summation_dict2[cur_key2]

            cur_mult_coeff = cur_key1_coeff * cur_key2_coeff
            cur_mult_key = cur_key1 + cur_key2

            if cur_mult_key in mult_summation_dict.keys():
                mult_summation_dict[cur_mult_key] += cur_mult_coeff
            else:
                mult_summation_dict.update({cur_mult_key: cur_mult_coeff})

    # Restrict dictionary to only include keys up to the max_x_power
    # and remove zero terms
    mult_summation_dict = \
        dict((k, mult_summation_dict[k])
             for k in range(max_x_power+1)
             if (k in mult_summation_dict)
             and (mult_summation_dict[k]!= 0))

    return mult_summation_dict


def get_eta_power(theta_block):
    """
    This function computes the power of the Dedekind eta function in the
    factorization of a theta block.

    Args:
        theta_block      (list): A theta block

    Returns:
        eta_power         (int): The power of the eta function for the
                                 theta block
    """
    eta_power = theta_block[0]

    for k in range(1, len(theta_block)):
        eta_power -= theta_block[k]

    return eta_power


def get_theta_block_eta_factor(theta_block, max_q_power):
    """
    This function computes the factor composed of a power of the
    Dedekind eta function for a theta block.

    Args:
        theta_block      (list): A theta block
        max_q_power       (int): The largest q-power in the expansion

    Returns:
        eta_Q_dict         (dict): A dictionary giving the terms in the
                                   eta factor's Q-expansion, where Q^8=q
    """
    eta_power = get_eta_power(theta_block)

    if eta_power == 0:
        eta_Q_dict = {0: 1}

    else:
        abs_eta_power = np.abs(eta_power)

        # Get an expansion of the Dedekind eta function
        dedekind_eta_dict = dedekind_eta(max_q_power)

        # Build the dictionary for the expansion of the eta function
        # raised to abs(eta_power)
        eta_pow_dict = dedekind_eta(max_q_power)
        for _ in range(abs_eta_power-1):
            eta_pow_dict = \
                multiply_one_var_expansions(eta_pow_dict,
                                            dedekind_eta_dict,
                                            max_q_power)

    if eta_power < 0:
        eta_pow_dict = \
            get_expansion_reciprocal(eta_pow_dict, max_q_power)

    eta_pow_Q_dict = q_to_Q_expansion(eta_pow_dict)

    # Take the eta_power of q^(1/24)
    Q_factor_pow = int((8*eta_power)/24)

    # Make the final dictionary
    eta_Q_dict = {}
    for cur_pow in eta_pow_Q_dict.keys():
        cur_new_pow = cur_pow + Q_factor_pow

        eta_Q_dict.update({cur_new_pow: eta_pow_Q_dict[cur_pow]})

    return eta_Q_dict


def get_theta_block_QZ_expansion(theta_block, max_q_power):
    """
    This function computes the QZ expansion of a theta block
    up to a maximum q power.

    Args:
        theta_block      (list): A theta block
        max_q_power       (int): The largest q-power in the expansion

    Returns:
        tb_QZ_dict       (dict): A dictionary giving the terms in the
                                 theta block's QZ-expansion, where Q^8=q
    """
    # Get the eta factor
    tb_eta_factor = \
        get_theta_block_eta_factor(theta_block, max_q_power)

    # Convert the dictionary giving the Q-expansion of the eta factor
    # to a QZ expansion dictionary
    tb_eta_factor = Q_to_QZ_expansion(tb_eta_factor)

    # Get the theta factor
    tb_theta_factor = \
        get_theta_block_theta_factor(theta_block, max_q_power)

    # Multiply the two factors
    tb_QZ_dict = multiply_QZ_expansions(tb_eta_factor, tb_theta_factor)

    # Truncate the dictionary
    tb_QZ_keys = tb_QZ_dict.keys()
    good_keys = []
    for cur_key in tb_QZ_keys:
        cur_key_split = cur_key.split("|")
        cur_key_Q_pow = int(cur_key_split[0])
        if cur_key_Q_pow <= 8*max_q_power:
            good_keys.append(cur_key)

    tb_QZ_dict = \
        {k:tb_QZ_dict[k] for k in good_keys}

    return tb_QZ_dict


def QZ_to_qz_expansion(QZ_summation_dict):
    """
    This function makes a QZ-expansion dictionary into a qz
    expansion dictionary

    Args:
        QZ_summation_dict     (dict): A dictionary indicating a
                                      QZ-expansion

    Returns:
        qz_summation_dict    (dict): A dictionary indicating the
                                     qz-expansion of the QZ-expansion
    """
    QZ_keys = QZ_summation_dict.keys()

    qz_summation_dict = {}
    for cur_QZ_key in QZ_keys:
        cur_QZ_key_split = cur_QZ_key.split('|')
        cur_QZ_key_Q_pow = int(cur_QZ_key_split[0])
        cur_QZ_key_Z_pow = int(cur_QZ_key_split[1])

        cur_QZ_key_q_pow = cur_QZ_key_Q_pow/8
        cur_QZ_key_z_pow = cur_QZ_key_Z_pow/2

        cur_qz_key = "{}|{}".format(cur_QZ_key_q_pow, cur_QZ_key_z_pow)
        cur_qz_val = QZ_summation_dict[cur_QZ_key]

        qz_summation_dict.update({cur_qz_key: cur_qz_val})

    return qz_summation_dict


def print_qz_function(summation_dict, max_q_order):
    """
    This function prints a function expanded in terms of q and z
    """
    summation_keys = summation_dict.keys()
    printed_expansion = ""
    for i in range(1, max_q_order +1):
        cur_keys = []
        for skey in summation_keys:
            skey_split = skey.rsplit('|')
            skey_q_power = skey_split[0]
            if (float(skey_q_power) <= i) and (float(skey_q_power) > i-1):
                cur_keys.append(skey)

        if cur_keys != []:
            cur_print_term = "("
            for good_key in cur_keys:
                good_key_split = good_key.rsplit('|')
                good_key_q_power = good_key_split[0]
                good_key_z_power = good_key_split[1]

                if good_key_z_power != 0:
                    cur_print_term += \
                        "({})z^({}) + ".format(summation_dict[good_key],
                                               good_key_z_power)
                else:
                    cur_print_term += \
                        "({}) + ".format(summation_dict[good_key])

            cur_print_term += ")*q^({}) + ".format(good_key_q_power)
            printed_expansion += cur_print_term

    # Make pretty
    printed_expansion = printed_expansion.replace(" + )", ")")
    printed_expansion = printed_expansion.replace("((1)", "(")

    printed_expansion += "O(q^({}))".format(max_q_order+1)

    return printed_expansion


def theta_block_order_sorter(idx, weight):
    """
    This function finds all theta blocks of a given index and weight
    and sorts them into lists of cuspidal theta blocks, non-cuspidal
    holomorphic theta blocks, and lists of theta blocks with the n
    largest negative minimum orders.

    Args:
        idx              (int): The index of Jacobi forms

        weight           (int): The weight of the theta blocks

    Returns:
        sorted_tbs      (list): A list of the lists of sorted theta blocks
    """
    print("Finding theta blocks...")
    all_tbs = get_theta_blocks(idx, weight)
    print("Done.")

    # Create containers for the theta blocks
    cuspidal_tbs = []
    holomorphic_tbs = []
    neg_order_tbs = {}

    print("Sorting theta blocks...")
    for cur_tb in all_tbs:
        cur_tb_min_order = \
            theta_block_minimum_order(cur_tb)

        cur_tb_min_order = np.round(cur_tb_min_order, 6)

        if cur_tb_min_order > 0:
            cuspidal_tbs.append(cur_tb)

        elif cur_tb_min_order == 0:
            holomorphic_tbs.append(cur_tb)

        else:
            if cur_tb_min_order in neg_order_tbs.keys():
                cur_list = neg_order_tbs[cur_tb_min_order]
                cur_list.append(cur_tb)
                neg_order_tbs.update({cur_tb_min_order: cur_list})
            else:
                neg_order_tbs.update({cur_tb_min_order: [cur_tb]})

    print("Done.")

    # Find the num_neg_mins largest negative minimum orders
    neg_orders = sorted(neg_order_tbs.keys())

    # Get the theta blocks with the largest negative minimum orders
    largest_neg_order_tbs = []

    i = 0
    while i < len(neg_orders):
        cur_neg_order = neg_orders[-(1+i)]
        cur_neg_tbs = neg_order_tbs[cur_neg_order]
        largest_neg_order_tbs.append([cur_neg_order, cur_neg_tbs])
        i += 1

    return cuspidal_tbs, holomorphic_tbs, largest_neg_order_tbs


def check_if_zero(summation_dict):
    """
    This function checks if a series expansion is 0

    Args:
        summation_dict    (dict): A dictionary giving first terms in a
                                  series expansion

    Return:
        is_zero           (bool): True/False depending on if the series is 0
    """
    coeffs = summation_dict.values()

    is_zero = True
    for coeff in coeffs:
        if np.abs(coeff) > 0.00000001:
            is_zero = False
            return is_zero

    return is_zero


def add_expansions(summation_dict_1, summation_dict_2):
    """
    This function adds two series expansions indicated with dictionaries.

    Args:
        summation_dict_1
        summation_dict_2

    Returns
        sum_of_summation_dicts    (dict):
    """
    # Get all keys
    sum_1_keys = list(summation_dict_1.keys())
    sum_2_keys = list(summation_dict_2.keys())

    all_keys = sum_1_keys + sum_2_keys
    all_keys = set(all_keys)
    all_keys = list(all_keys)

    # Create a container for the sum
    sum_of_summation_dicts = {}
    for cur_key in all_keys:
        cur_val = 0
        if cur_key in sum_1_keys:
            cur_val += summation_dict_1[cur_key]

        if cur_key in sum_2_keys:
            cur_val += summation_dict_2[cur_key]

        sum_of_summation_dicts.update({cur_key: cur_val})

    return sum_of_summation_dicts


def mult_expansion_by_scalar(summation_dict, scalar):
    """
    This function adds two series expansions indicated with dictionaries.

    Args:
        summation_dict

    Returns
        scaled_summation_dict    (dict):
    """
    # Get the keys
    sum_keys = list(summation_dict.keys())

    # Create a container for the sum
    scaled_summation_dict = {}
    for cur_key in sum_keys:
        cur_val = scalar*summation_dict[cur_key]

        scaled_summation_dict.update({cur_key: cur_val})

    return scaled_summation_dict


def get_common_keys(dicts):
    """
    This function gets the keys common to all dictionaries in a list.

    Args:
        dicts          (list): A list of dictionaries

    Returns:
        common_keys    (list): A list of keys common to all input dicts

    """
    num_dicts = len(dicts)
    common_keys = list(dicts[0].keys())

    for i in range(1, num_dicts):
        cur_dict_keys = set(dicts[i].keys())
        cur_common_keys = set(common_keys)
        common_keys = cur_common_keys.intersection(cur_dict_keys)

    return list(common_keys)


def get_common_keys_with_specified_value(dicts, good_val):
    """
    This function gets the keys common to all dictionaries in a list
    such that each dictionary takes a given value for those keys.

    Args:
        dicts           (list): A list of dictionaries
        good_val    (variable): The value a common key must take

    Returns:
        common_keys     (list): A list of keys common to all input dicts
                                such that each dict takes the shared value
                                key_val
    """
    num_dicts = len(dicts)
    common_keys = []

    dict_scan_ct = 0
    for i in range(num_dicts):
        cur_dict = dicts[i]

        # Find the keys in the current dictionary that take key_val
        cur_dict_good_keys = []
        for cur_key in cur_dict.keys():
            if cur_dict[cur_key] == good_val:
                cur_dict_good_keys.append(cur_key)

        if dict_scan_ct == 0:
            common_keys = cur_dict_good_keys

        else:
            common_keys = set(common_keys)
            cur_dict_good_keys = set(cur_dict_good_keys)
            common_keys = common_keys.intersection(cur_dict_good_keys)

        dict_scan_ct += 1

    return list(common_keys)


def get_common_keys_without_specified_value(dicts, bad_val):
    """
    This function gets the keys common to all dictionaries in a list
    such that each dictionary does not take a given value for those keys.

    Args:
        dicts          (list): A list of dictionaries
        bad_val    (variable): The value a common key cannot take

    Returns:
        common_keys    (list): A list of keys common to all input dicts
                               such that each dict takes the shared value
                               key_val
    """
    num_dicts = len(dicts)
    common_keys = []

    dict_scan_ct = 0
    for i in range(num_dicts):
        cur_dict = dicts[i]

        # Find the keys in the current dictionary that take key_val
        cur_dict_good_keys = []
        for cur_key in cur_dict.keys():
            if cur_dict[cur_key] != bad_val:
                cur_dict_good_keys.append(cur_key)

        if dict_scan_ct == 0:
            common_keys = cur_dict_good_keys

        else:
            common_keys = set(common_keys)
            cur_dict_good_keys = set(cur_dict_good_keys)
            common_keys = common_keys.intersection(cur_dict_good_keys)

        dict_scan_ct += 1

    return list(common_keys)


def check_if_all_dict_values_are_zero(input_dict):
    """
    This function checks if all values in a dictionary are zero.

    Args:
        input_dict    (dict): A dictionary

    Returns:
        is_zero       (bool): True/False if all values are zero
    """
    is_zero = True
    for cur_key in input_dict.keys():
        if input_dict[cur_key] != 0:
            is_zero = False

            return is_zero

    return is_zero


def make_vector(summation_dict, input_keys):
    """
    This function creates a vector from terms in a series.

    Args:
        summation_dict             (dict): A dictionary giving
                                           terms in a series
        input_keys                 (list): A list of keys to index components
                                           of the output vector

    Returns:
        summation_vector           (list): A vector
    """
    summation_vector = []
    for cur_key in input_keys:
        if cur_key in summation_dict.keys():
            summation_vector.append(summation_dict[cur_key])
        else:
            summation_vector.append(0)

    return summation_vector


def get_max_lin_indep_expansions(expansions_dict):
    """
    This functions find a maximal linearly independent set
    of cuspidal theta blocks.

    Args:
        expansions_dict         (dict): A dictionary giving series
                                        expansions. The keys are the
                                        function names, the values
                                        are the series expansions of
                                        the functions up to some order.

    Returns:
        lin_indep_expansions    (dict): A maximal linearly independent
                                        collection of expansions from
                                        expansions_dict
    """
    # Get all function names
    all_functions = list(expansions_dict.keys())
    
    # Create a container for the linearly independent functions
    lin_indep_expansions = {}
    
    # Loop through the functions and add only linearly independent
    # functions to the lin_indep_expansions dictionary
    for cur_function in all_functions:
        cur_expansion = expansions_dict[cur_function]
        
        # Check if the expansion is 0
        cur_zero_check = \
            check_if_all_dict_values_are_zero(cur_expansion)
            
        if not cur_zero_check:
            # Check if there are any functions in the lin_indep_expansions dict
            if lin_indep_expansions == {}:
                lin_indep_expansions.update({cur_function: cur_expansion})
                
            else:
                # Get all locations of terms for the current expansion
                # and the linearly independent expansions in order
                # to construct a vector of these terms
                cur_term_locs = list(cur_expansion.keys())
                for cur_lin_indep_exp in lin_indep_expansions.values():
                    cur_term_locs += list(cur_lin_indep_exp.keys())
                    
                # Remove duplicates and sort
                cur_term_locs = list(set(cur_term_locs))
                cur_term_locs = sorted(cur_term_locs)
                
                # Make vectors for the current dict of
                # linearly independent functions
                cur_lin_indep_vectors = []
                for cur_lin_indep_exp in lin_indep_expansions.values():
                    cur_lin_indep_vector = \
                        make_vector(cur_lin_indep_exp, cur_term_locs)
                    cur_lin_indep_vectors.append(cur_lin_indep_vector)
                    
                # Make a vector for the current expansion
                cur_test_vector = make_vector(cur_expansion, cur_term_locs)
                
                # Make a matrix of all vectors
                cur_all_vecs = cur_lin_indep_vectors + [cur_test_vector]
                cur_test_matrix = Matrix(cur_all_vecs)
                
                # Compute the rank of the matrix
                cur_test_matrix_rank = cur_test_matrix.rank()
                
                # Check if the current expansion is linearly independent
                # from the others. If so, add it to the collection.
                if len(cur_all_vecs) == cur_test_matrix_rank:
                    lin_indep_expansions.update({cur_function: cur_expansion})

    return lin_indep_expansions


def get_max_lin_indep_cusp_tbs(cusp_tbs):
    """
    This functions find a maximal linearly independent set
    of cuspidal theta blocks.

    Args:
        cusp_tbs                (list): A list of cuspidal theta blocks
                                        of the same weight and index

    Returns:
        lin_indep_expansions    (dict): A dictionary giving a maximal linearly
                                        independent collection of cuspidal
                                        theta blocks.
    """
    if cusp_tbs != []:
        index_count = 0
        lin_indep_expansions = {}
        for cur_tb in cusp_tbs:
            cur_tb_index = theta_block_index(cur_tb)
            cur_tb_weight = get_weight(cur_tb)

            if index_count == 0:
                tb_index = cur_tb_index
                tb_weight = cur_tb_weight
                jkm_cusp_dim = dim_jkm_cusp(tb_weight, tb_index)

                max_q_power = int(np.ceil(tb_index / 4)) + 5

                index_count += 1
            else:
                if cur_tb_index != tb_index:
                    print("The theta blocks have different indices.")
                    return None
                if cur_tb_weight != tb_weight:
                    print("The theta blocks have different weights.")
                    return None

            print("Computing an expansion for {}...".format(cur_tb))
            cur_tb_expansion = \
                get_theta_block_QZ_expansion(cur_tb, max_q_power)
            print("Done.")
            
            lin_indep_expansions.update({'{}'.format(cur_tb): cur_tb_expansion})

            # Get the maximal linearly independent set of expansions
            lin_indep_expansions = \
                get_max_lin_indep_expansions(lin_indep_expansions)
            
            # Check if a basis has been computed
            if len(lin_indep_expansions) == jkm_cusp_dim:
                print("Basis computed.")
                
                return lin_indep_expansions
            
        return lin_indep_expansions

    else:
        print("The list is empty.")
        return None
