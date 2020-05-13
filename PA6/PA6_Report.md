<div align=center>

# CS114 (Spring 2020) Programing Assignment 6

## Neural Transition-Based Dependency Parsing

### Heyuan (Henry) Gao

</div>

## Worked Example

### 1. Go through the sequence of transitions

Given the sentence:

    I parsed this sentence correctly

Stack | Buffer | New dependency | Transition
:-- | :-- | :-- | :--
[ROOT] | [I, parsed, this, sentence, correctly] |  | Initial Configuration
[ROOT, I] | [parsed, this, sentence, correctly] |  | SHIFT
[ROOT, I, parsed] | [this, sentence, correctly] |  | SHIFT
[ROOT, parsed] | [this, sentence, correctly] | parsed $\rightarrow$ I | LEFT-ARC
[ROOT, parsed, this] | [sentence, correctly] |  | SHIFT
[ROOT, parsed, this, sentence] | [correctly] |  | SHIFT
[ROOT, parsed, sentence] | [correctly] | sentence $\rightarrow$ this | LEFT-ARC
[ROOT, parsed] | [correctly] | parsed $\rightarrow$ sentence | RIGHT-ARC
[ROOT, parsed, correctly] | [] |  | SHIFT
[ROOT, parsed] | [] | parsed $\rightarrow$ correctly | RIGHT-ARC
[ROOT] | [] | ROOT $\rightarrow$ parsed | RIGHT-ARC

### 2. A sentence containing $n$ words will be parsed in how many steps

For each word of the sentence, it need first be shifted onto the stack and then reduced by right\left arc. Therefore, there would be $2*n$ parsing steps for a sentence containing $n$ words regardless of the initial configuration