"""
Code by Heyuan (Henry) Gao
"""

from fst import FST
import string, sys
from fsmutils import compose


def letters_to_numbers():
    """
    Returns an FST that converts letters to numbers as specified by
    the soundex algorithm
    """

    # Let's define our first FST
    f1 = FST('soundex-generate')

    # Indicate that 'start' is the initial state
    f1.add_state('start')
    f1.add_state('0')
    f1.add_state('1')
    f1.add_state('2')
    f1.add_state('3')
    f1.add_state('4')
    f1.add_state('5')
    f1.add_state('6')
    f1.initial_state = 'start'

    # Set all the final states
    f1.set_final('0')
    f1.set_final('1')
    f1.set_final('2')
    f1.set_final('3')
    f1.set_final('4')
    f1.set_final('5')
    f1.set_final('6')

    replace_letters = {'0': 'aehiouwy', '1': 'bfpv', '2': 'cgjkqsxz', '3': 'dt', '4': 'l', '5': 'mn', '6': 'r'}
    # retaining the first letter
    # removing letters and replacing letters with numbers.
    for state, in_strs in replace_letters.items():
        for in_str in in_strs:
            f1.add_arc('start', state, in_str, in_str)
            f1.add_arc('start', state, in_str.upper(), in_str.upper())
            f1.add_arc(state, state, in_str, '')

        for state_supplementary in replace_letters:
            if not state_supplementary.startswith(state):
                for str_out_state in replace_letters[state_supplementary]:
                    state_supplementary_out = '' if state_supplementary.startswith('0') else state_supplementary
                    f1.add_arc(state, state_supplementary, str_out_state, state_supplementary_out)
    return f1


def truncate_to_three_digits():
    """
    Create an FST that will truncate a soundex string to three digits
    """

    # Ok so now let's do the second FST, the one that will truncate
    # the number of digits to 3
    f2 = FST('soundex-truncate')

    # Indicate initial and final states
    f2.add_state('0')
    f2.add_state('1')
    f2.add_state('2')
    f2.add_state('3')
    f2.add_state('4')
    f2.initial_state = '0'
    f2.set_final('1')
    f2.set_final('2')
    f2.set_final('3')
    f2.set_final('4')

    # truncating extra digits.
    numbers = '0123456789'
    for letter in string.ascii_letters:
        f2.add_arc('0', '1', letter, letter)

    for number in numbers:
        f2.add_arc('0', '2', number, number)
        f2.add_arc('1', '2', number, number)
        f2.add_arc('2', '3', number, number)
        f2.add_arc('3', '4', number, number)
        f2.add_arc('4', '4', number, '')
    return f2


def add_zero_padding():
    # Now, the third fst - the zero-padding fst
    f3 = FST('soundex-padzero')

    # Indicate initial and final states
    f3.add_state('0')
    f3.add_state('1')
    f3.add_state('2')
    f3.add_state('3')
    f3.add_state('4')
    f3.add_state('11')
    f3.add_state('12')
    f3.add_state('13')
    f3.add_state('21')
    f3.add_state('22')
    f3.initial_state = '0'
    f3.set_final('4')
    f3.set_final('22')
    f3.set_final('13')

    # padding with zeros if required.
    numbers = '0123456789'
    for letter in string.ascii_letters:
        f3.add_arc('0', '1', letter, letter)
    for number in numbers:
        f3.add_arc('0', '2', number, number)
        f3.add_arc('1', '2', number, number)
        f3.add_arc('2', '3', number, number)
        f3.add_arc('3', '4', number, number)

    f3.add_arc('1', '11', '', '0')
    f3.add_arc('11', '12', '', '0')
    f3.add_arc('12', '13', '', '0')
    f3.add_arc('2', '21', '', '0')
    f3.add_arc('21', '22', '', '0')
    f3.add_arc('3', '4', '', '0')

    return f3


def soundex_convert(name_string):
    """Combine the three FSTs above and use it to convert a name into a Soundex"""
    f1 = letters_to_numbers()
    f2 = truncate_to_three_digits()
    f3 = add_zero_padding()
    output = compose(name_string, f1, f2, f3)[0]
    return ''.join(output)


if __name__ == '__main__':
    user_input = input().strip()

    if user_input:
        print("%s -> %s" % (user_input, soundex_convert(list(user_input))))
