# import the fst module
import fst

# import the string module
import string

# Define a list of all vowels for convenience
vowels = ['a', 'e', 'i', 'o', 'u']

# Instantiate an FST object with some name
f = fst.FST('devowelizer')

# All we need is a single state ...
f.add_state('1')

# and this same state is the initial and the final state
f.initial_state = '1'
f.set_final('1')

# Now, we need to add an arc for each letter; if the letter is
# a vowel then the transition outputs nothing but otherwise it
# outputs the same letter that it consumed.
for letter in string.ascii_lowercase:
    if letter in vowels:
        f.add_arc('1', '1', letter, '')
    else:
        f.add_arc('1', '1', letter, letter)

# Evaluate it on some example words
print(''.join(f.transduce(['v', 'o', 'w', 'e', 'l'])[0]))
print(''.join(f.transduce('exception')[0]))
print(''.join(f.transduce('consonant')[0]))
