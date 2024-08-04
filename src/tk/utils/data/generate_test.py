import unittest
from .import generate


class Test(unittest.TestCase):
    def test_opdata_generateFromTuple(self):
        data = generate.OpData([("+", sum)])
        self.assertEqual(
            next(iter(data)),
            "0+0=0"
        )

    def test_opdata_generateFromString(self):
        data = generate.OpData(["+"])
        self.assertEqual(
            next(iter(data)),
            "0+0=0"
        )

    def test_opdata_generateNonZero(self):
        data = generate.OpData(["-"])
        self.assertEqual(
            data[1],
            "0-1=-1"
        )

    def test_opdata_multiOp_hasAll(self):
        data = generate.OpData(["-", "+"], n=2)
        self.assertIn("1+1=2", data)
        self.assertIn("1-1=0", data)
        self.assertEqual(len(data), 8)

    def test_opdata_withEof(self):
        data = generate.OpData(["+"], n=2, eof="eof")
        self.assertTrue(all(x.endswith("eof") for x in data))
    
    def test_generateNdigit(self):
        data = generate.ndigit_data(ndigits=4, n=50)
        self.assertTrue(all(len(str(x)) == 4 for x in data))
        self.assertEqual(len(set(data)), 50)

    def test_generateFormat(self):
        data = generate.ndigit_data(ndigits=2, n=90)
        data = generate.strfmt_ops(data, data, op="+", fmt="03d")
        self.assertIn("010+010=020", data)