import numpy as np
from unittest import TestCase
from unittest.mock import patch, MagicMock

from scipy.optimize import OptimizeWarning

from sigmoid import Sigmoid


class SigmoidTestCase(TestCase):
    # test __repr__
    def test_repr(self):
        sig = Sigmoid(0.36, 5, 1.5, -0.06)
        self.assertEqual("Sigmoid(L=0.36, x0=5, k=1.5, b=-0.06, derivative=0, symbol=x, e_symbol=exp)", repr(sig))
        self.assertEqual("Sigmoid(L=0.36, x0=5, k=1.5, b=-0.06, derivative=1, symbol=x, e_symbol=exp)",
                         repr(sig.deriv()))
        self.assertEqual("Sigmoid(L=0.36, x0=5, k=1.5, b=-0.06, derivative=2, symbol=x, e_symbol=exp)",
                         repr(sig.deriv(m=2)))

    # test __str__
    def test_str(self):
        sig = Sigmoid(0.36, 5, 1.5, -0.06)
        self.assertEqual('0.36 / ( 1 + exp(-1.5*(x-5)) ) + -0.06', str(sig))
        self.assertEqual('( 1.5*0.36*exp(-1.5*(x-5)) ) / ( ( 1+exp(-1.5*(x-5)) )**2 )', str(sig.deriv()))
        self.assertEqual(
            '( (1.5**2)*0.36*exp(1.5*(x+5)) * (exp(1.5*5)-exp(1.5*x)) ) / ( (exp(1.5*5) + exp(1.5*x))**3 )',
            str(sig.deriv(m=2)))

    # test equality
    def test_eq(self):
        self.assertTrue(Sigmoid(0.36, 5, 1.5, -0.06) == Sigmoid(0.360, 5.0, 1.5, -0.06))
        self.assertFalse(Sigmoid(0.36, 5, 1.5, -0.06) == Sigmoid(0.361, 5.0, 1.5, -0.06))

    # test inequality
    def test_ne(self):
        self.assertFalse(Sigmoid(0.36, 5, 1.5, -0.06) != Sigmoid(0.360, 5.0, 1.5, -0.06))
        self.assertTrue(Sigmoid(0.36, 5, 1.5, -0.06) != Sigmoid(0.361, 5.0, 1.5, -0.06))

    # test default sigmoid
    def test_default_sigmoid(self):
        sig = Sigmoid()
        self.assertEqual(0, sig.x0)
        self.assertEqual(0, sig.b)
        self.assertEqual(1, sig.L)
        self.assertEqual(1, sig.k)
        self.assertAlmostEqual(0.5, sig(0))
        self.assertAlmostEqual(0.7310585786300049, sig(1))
        self.assertAlmostEqual(0.2689414213699951, sig(-1))
        self.assertAlmostEqual(1.0, sig(20))
        self.assertAlmostEqual(0.0, sig(-20))

    # test __call__ evaluates sigmoid correctly (and works with numpy arrays in/out)
    def test_evaluating_parameterized_sigmoid_from_numpy_array(self):
        sig = Sigmoid(0.36, 5, 1.5, -0.06)
        x_vals = np.array(range(10))
        expected = np.array([
            -0.0598,
            -0.0591,
            -0.0560,
            -0.0429,
            0.0057,
            0.1200,
            0.2343,
            0.2829,
            0.2960,
            0.2991,
            0.2998,
        ])
        actual = sig(x_vals)
        for x in x_vals:
            self.assertAlmostEqual(expected[x], actual[x], places=4)

    # test fit
    def test_fit(self):
        x = np.linspace(0, 7, 28)
        y = np.array([
            0.22107034278607543, 0.22107034278607543, 0.22107034278607543, 0.22262523377160684, 0.23945219622120945,
            0.3139030902579709, 0.3499823387345087, 0.6101973532680808, 0.6139505901302286, 0.6620951092848685,
            0.6975204366373884, 0.7276852800528291, 0.7349146323440234, 0.7349146323440234, 0.8051763853261831,
            0.8051763853261831, 0.8051763853261831, 0.8149730342812037, 0.8163679060760755, 0.8163679060760755,
            0.8163679060760755, 0.818162472417683, 0.818162472417683, 0.8197171496245342, 0.8197171496245342,
            0.8197171496245342, 0.8197171496245342, 0.8197171496245342])
        sig = Sigmoid.fit(x, y)
        self.assertAlmostEqual(0.6232, sig.L, places=4)
        self.assertAlmostEqual(1.7996, sig.x0, places=4)
        self.assertAlmostEqual(2.3185, sig.k, places=4)
        self.assertAlmostEqual(0.1852, sig.b, places=4)

    # test fit when curve_fit fails least-squares fit
    @patch('scipy.optimize.curve_fit', MagicMock(side_effect=RuntimeError()))
    def test_fit_fails_least_squares(self):
        x = np.linspace(0, 1, 11)
        y = np.array([
            0.0003,
            0.0013,
            0.0059,
            0.0244,
            0.0805,
            0.1350,
            0.0805,
            0.0244,
            0.0059,
            0.0013,
            0.0003,
        ])
        self.assertRaises(RuntimeError, Sigmoid.fit, x, y)

    # test fit when curve_fit fails least-squares fit
    @patch('scipy.optimize.curve_fit', MagicMock(side_effect=OptimizeWarning()))
    def test_fit_fails_to_estimate_covariance(self):
        x = np.linspace(0, 1, 11)
        y = np.array([
            0.0003,
            0.0013,
            0.0059,
            0.0244,
            0.0805,
            0.1350,
            0.0805,
            0.0244,
            0.0059,
            0.0013,
            0.0003,
        ])
        self.assertRaises(OptimizeWarning, Sigmoid.fit, x, y)

    # test fit when curve_fit fails least-squares fit
    def test_fit_fails_due_to_nans_in_input(self):
        x = np.linspace(0, 1, 11)
        y = np.array([
            0.0003,
            np.nan,
            0.0059,
            0.0244,
            0.0805,
            0.1350,
            0.0805,
            0.0244,
            0.0059,
            0.0013,
            0.0003,
        ])
        self.assertRaises(ValueError, Sigmoid.fit, x, y)

    # test derivative 1
    # test derivative 1 __call__ evaluates correctly
    def test_evaluating_derivative_1_of_sigmoid_from_numpy_array(self):
        sig = Sigmoid(0.36, 5, 1.5, -0.06)
        derivative_of_sig = sig.deriv()
        x_vals = np.array(range(10))
        expected = np.array([
            0.0003,
            0.0013,
            0.0059,
            0.0244,
            0.0805,
            0.1350,
            0.0805,
            0.0244,
            0.0059,
            0.0013,
            0.0003,
        ])
        actual = derivative_of_sig(x_vals)
        for e, a in zip(expected, actual):
            self.assertAlmostEqual(e, a, places=4)

    # test derivative 2
    # test derivative 2 __call__ evaluates correctly
    def test_evaluating_derivative_2_of_sigmoid_from_numpy_array(self):
        sig = Sigmoid(0.36, 5, 1.5, -0.06)
        second_derivative_of_sig = sig.deriv(m=2)
        x_vals = np.array(range(10))
        expected = np.array([
            0.0004,
            0.0020,
            0.0086,
            0.0331,
            0.0767,
            0.0000,
            -0.0767,
            -0.0331,
            -0.0086,
            -0.0020,
            -0.0004,
        ])
        actual = second_derivative_of_sig(x_vals)
        for e, a in zip(expected, actual):
            self.assertAlmostEqual(e, a, places=4)

    # test derivative 1 of derivative 1
    def test_evaluating_derivative_of_derivative_of_sigmoid_from_numpy_array(self):
        sig = Sigmoid(0.36, 5, 1.5, -0.06)
        second_derivative_of_sig = sig.deriv().deriv()
        x_vals = np.array(range(10))
        expected = np.array([
            0.0004,
            0.0020,
            0.0086,
            0.0331,
            0.0767,
            0.0000,
            -0.0767,
            -0.0331,
            -0.0086,
            -0.0020,
            -0.0004,
        ])
        actual = second_derivative_of_sig(x_vals)
        for e, a in zip(expected, actual):
            self.assertAlmostEqual(e, a, places=4)

    # test derivative 3 raises
    # test derivative 2 of derivative 1 raises
    def test_unimplemented_derivatives_raise(self):
        sig = Sigmoid()
        try:
            sig.deriv(m=3)
            self.fail()
        except NotImplementedError:
            pass
        try:
            sig.deriv().deriv(m=2)
            self.fail()
        except NotImplementedError:
            pass
        try:
            sig.deriv().deriv().deriv(m=1)
            self.fail()
        except NotImplementedError:
            pass

    # test root of sigmoid is correct (when there should be one)
    def test_root_of_sigmoid(self):
        sig = Sigmoid(0.36, 5, 1.5, -0.06)
        roots = sig.roots()
        print(roots)
        self.assertEqual(1, len(roots))
        self.assertAlmostEqual(3.9270413917106, roots[0])

    # test root of sigmoid is correct (when there shouldn't be one)
    def test_root_of_sigmoid_doesnt_exist(self):
        k = 0
        sig = Sigmoid(0.36, 5, k, -0.06)
        roots = sig.roots()
        self.assertEqual(0, len(roots))

        b = 0
        sig = Sigmoid(0.36, 5, 1.5, b)
        roots = sig.roots()
        self.assertEqual(0, len(roots))

        L = 0
        sig = Sigmoid(L, 5, 1.5, -0.06)
        roots = sig.roots()
        self.assertEqual(0, len(roots))

        # (b+L) = 0
        b = -0.06
        L = 0.06
        sig = Sigmoid(L, 5, 1.5, b)
        roots = sig.roots()
        self.assertEqual(0, len(roots))

    # test root of derivative 1 doesn't exist
    def test_root_of_first_derivative_doesnt_exist(self):
        sig = Sigmoid(0.36, 5, 1.5, -0.06)
        derivative_of_sig = sig.deriv()
        extrema = derivative_of_sig.roots()
        self.assertEqual(0, len(extrema))

    # test root of derivative 2 is x0
    def test_root_of_second_derivative_is_x0(self):
        sig = Sigmoid(0.36, 5, 1.5, -0.06)
        x0 = sig.x0
        second_derivative_of_sig = sig.deriv(m=2)
        inflection_points = second_derivative_of_sig.roots()
        self.assertEqual(1, len(inflection_points))
        self.assertEqual(x0, inflection_points[0])

    # test copy
    def test_copy(self):
        sig = Sigmoid(0.36, 5, 1.5, -0.06)
        cp = sig.copy()
        self.assertIsNot(sig, cp)
        self.assertEqual(sig.L, cp.L)
        self.assertEqual(sig.x0, cp.x0)
        self.assertEqual(sig.k, cp.k)
        self.assertEqual(sig.b, cp.b)
        self.assertEqual(sig.derivative, cp.derivative)
        self.assertEqual(sig.symbol, cp.symbol)
        self.assertEqual(sig.e_symbol, cp.e_symbol)

    # test linspace
    def test_linspace(self):
        sig = Sigmoid(0.36, 5, 1.5, -0.06)
        expected = np.array([
            -0.0598,
            -0.0591,
            -0.0560,
            -0.0429,
            0.0057,
            0.1200,
            0.2343,
            0.2829,
            0.2960,
            0.2991,
            0.2998,
        ])
        actual = sig.linspace(0, 10, 1)
        for e, a in zip(expected, actual):
            self.assertAlmostEqual(e, a, places=4)
