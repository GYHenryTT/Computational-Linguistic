"""
Code by Heyuan (Henry) Gao
"""

from fst import FST
import string
from fsmutils import compose


class Parser():

    def __init__(self):
        pass

    def generate(self, analysis):
        """Generate the morphologically correct word 

        e.g.
        p = Parser()
        analysis = ['p','a','n','i','c','+past form']
        p.generate(analysis) 
        ---> 'panicked'
        """

        # Let's define our first FST
        f1 = FST('morphology-generate')

        # Indicate initial and final states
        f1.add_state('start')
        f1.add_state('vowel')
        f1.add_state('consonant')
        f1.add_state('c')
        f1.add_state('form_1')
        f1.add_state('form_2')
        f1.initial_state = 'start'
        f1.set_final('form_1')
        f1.set_final('form_2')

        # Generate
        vowels = 'aeiou'
        for vowel in vowels:
            f1.add_arc('start', 'vowel', vowel, vowel)
            f1.add_arc('vowel', 'vowel', vowel, vowel)
            f1.add_arc('consonant', 'vowel', vowel, vowel)
            f1.add_arc('c', 'vowel', vowel, vowel)

        for letter in string.ascii_lowercase:
            if letter in vowels:
                continue

            if not letter == 'c':
                f1.add_arc('vowel', 'consonant', letter, letter)
            f1.add_arc('start', 'consonant', letter, letter)
            f1.add_arc('consonant', 'consonant', letter, letter)
            f1.add_arc('c', 'consonant', letter, letter)

        f1.add_arc('vowel', 'c', 'c', 'c')
        f1.add_arc('c', 'form_1', '+past form', 'ked')
        f1.add_arc('c', 'form_1', '+present participle form', 'king')
        f1.add_arc('consonant', 'form_2', '+past form', 'ed')
        f1.add_arc('consonant', 'form_2', '+present participle form', 'ing')

        output = f1.transduce(analysis)[0]

        return "".join(output)

    def parse(self, word):
        """Parse a word morphologically 

        e.g.
        p = Parser()
        word = ['p','a','n','i','c','k','i','n','g']
        p.parse(word)
        ---> 'panic+present participle form'
        """

        # Ok so now let's do the second FST
        f2 = FST('morphology-parse')  # Detecting affixes ('ing' and 'ed')

        # Indicate initial and final states
        states = ['start', 'i', 'n', 'g', 'e', 'd']
        for state in states:
            f2.add_state(state)
        f2.initial_state = 'start'
        f2.set_final('d')
        f2.set_final('g')

        for letter in string.ascii_lowercase:
            if not letter in 'ie':
                f2.add_arc('start', 'start', letter, letter)
            if not letter == 'n':
                f2.add_arc('i', 'start', letter, ('i', letter))
            if not letter == 'd':
                f2.add_arc('e', 'start', letter, ['e', letter])
            if not letter == 'g':
                f2.add_arc('n', 'start', letter, ['i', 'n', letter])

        f2.add_arc('start', 'i', 'i', '')
        f2.add_arc('i', 'n', 'n', '')
        f2.add_arc('n', 'g', 'g', '+present participle form')
        f2.add_arc('start', 'e', 'e', '')
        f2.add_arc('e', 'd', 'd', '+past form')

        f3 = FST('morphology-parse')  # K-deletion
        # Indicate initial and final states
        f3.add_state('start')
        f3.add_state('vowel')
        f3.add_state('consonant')
        f3.add_state('c')
        f3.add_state('k')
        f3.add_state('lick_l')
        f3.add_state('lick_i')
        f3.add_state('lick_c')
        f3.add_state('lick_k')
        f3.add_state('parse')
        f3.initial_state = 'start'
        f3.set_final('parse')

        vowels = 'aeiou'
        for vowel in vowels:
            f3.add_arc('start', 'vowel', vowel, vowel)
            f3.add_arc('vowel', 'vowel', vowel, vowel)
            f3.add_arc('consonant', 'vowel', vowel, vowel)
            f3.add_arc('c', 'vowel', vowel, vowel)

        for letter in string.ascii_lowercase:
            f3.add_arc('parse', 'parse', letter, letter)

            if letter in vowels:
                continue

            if not letter == 'c':
                f3.add_arc('vowel', 'consonant', letter, letter)

            if not letter == 'l':
                f3.add_arc('start', 'consonant', letter, letter)

            if not letter == 'k':
                f3.add_arc('c', 'consonant', letter, letter)
            f3.add_arc('consonant', 'consonant', letter, letter)

        f3.add_arc('vowel', 'c', 'c', 'c')
        f3.add_arc('c', 'k', 'k', '')
        f3.add_arc('start', 'lick_l', 'l', 'l')
        f3.add_arc('lick_l', 'lick_i', 'i', 'i')
        f3.add_arc('lick_i', 'lick_c', 'c', 'c')
        f3.add_arc('lick_c', 'lick_k', 'k', 'k')
        f3.add_arc('lick_k', 'parse', '+', '+')
        f3.add_arc('k', 'parse', '+', '+')
        f3.add_arc('consonant', 'parse', '+', '+')
        f3.add_arc('parse', 'parse', ' ', ' ')

        output = compose(word, f2, f3)[0]

        return "".join(output)