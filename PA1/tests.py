import unittest
from soundex import letters_to_numbers, truncate_to_three_digits, add_zero_padding, soundex_convert
from morphology import Parser

class TestHW1(unittest.TestCase):

    def setUp(self):
        self.f1 = letters_to_numbers()
        self.f2 = truncate_to_three_digits()
        self.f3 = add_zero_padding()
        self.mparser = Parser()

    def test_letters(self):
        self.assertEqual("".join(self.f1.transduce("washington")[0]), "w25235")
        self.assertEqual("".join(self.f1.transduce("jefferson")[0]), "j1625")
        self.assertEqual("".join(self.f1.transduce("adams")[0]), "a352")
        self.assertEqual("".join(self.f1.transduce("bush")[0]), "b2")

    def test_truncation(self):
        self.assertEqual("".join(self.f2.transduce("a33333")[0]), "a333")
        self.assertEqual("".join(self.f2.transduce("123456")[0]), "123")
        self.assertEqual("".join(self.f2.transduce("11")[0]), "11")
        self.assertEqual("".join(self.f2.transduce("5")[0]), "5")

    def test_padding(self):
        self.assertEqual("".join(self.f3.transduce("3")[0]), "300")
        self.assertEqual("".join(self.f3.transduce("b56")[0]), "b560")
        self.assertEqual("".join(self.f3.transduce("c111")[0]), "c111")

    def test_soundex(self):
        self.assertEqual(soundex_convert("jurafsky"), "j612")

    def test_morphology(self):
        havocking = list('havocking')
        self.assertEqual(self.mparser.parse(havocking), "havoc+present participle form")
        lick = ['l','i','c','k','+past form']
        self.assertEqual(self.mparser.generate(lick), "licked")


if __name__ == '__main__':
    unittest.main()
