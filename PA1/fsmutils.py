import string
from fst import FST

# This function returns fn o ... o f3 o f2 o f1 (input)
# where ALL transducers use characters as input symbols
def compose(input, *fsts):
	output_list = [input]
	for fst in fsts:
		next_output_list = []
		for o in output_list:
			new_output = ''.join(o)
			next_output_list.extend(fst.transduce(new_output))
		output_list = next_output_list
	return output_list

if __name__ == '__main__':
	f1 = FST('test-generate')

	# Indicate that '1' is the initial state
	f1.add_state('start')
	f1.add_state('next')
	f1.initial_state = 'start'

	# Set all the final states
	f1.set_final('next')

	# Add the rest of the arcs
	for letter in ['A','B','C','D']:
		f1.add_arc('start', 'next', letter, '1')
		f1.add_arc('next', 'next', letter, '0')

	f2 = FST('test-generate')
	f2.add_state('start')
	f2.add_state('next')
	f2.initial_state = 'start'
	f2.set_final('next')
	
	f2.add_arc('start', 'next', '1', 'a')
	f2.add_arc('start', 'next', '1', 'an')
	f2.add_arc('next', 'next', '0', 'b')

	output = compose(tuple('BAD'), f1, f2)
	print(output)
	for o in output:
		print(compose(tuple(o), f2.inverted(), f1.inverted()))


