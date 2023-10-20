import jax.numpy as jnp


class UNITS_class:
    def __init__(self):
        self.eV = 1e-9

        self.m = 5067225.6643 / self.eV
        self.kg = 5.6102071e35 * self.eV
        self.s = 1.519116e15 / self.eV
        self.C = 1.0
        self.K = 1 / 1.160451812e4 * self.eV

        self.cm = 1e-2 * self.m
        self.mm = 1e-3 * self.m
        self.um = 1e-6 * self.m
        self.nm = 1e-9 * self.m
        self.pm = 1e-12 * self.m
        self.fm = 1e-15 * self.m
        self.km = 1e3 * self.m
        self.angstrom = 1e-10 * self.m
        self.lightyear = 9460730472580800.0 * self.m
        self.astro_unit = 149597870700.0 * self.m
        self.pc = (648000.0 / jnp.pi) * self.astro_unit
        self.kpc = 1e3 * self.pc
        self.Mpc = 1e6 * self.pc
        self.Gpc = 1e9 * self.pc
        self.inch = 2.54 * self.cm
        self.foot = 12.0 * self.inch
        self.mile = 5280.0 * self.foot
        self.thou = 1e-3 * self.inch

        self.L = 1e-3 * self.m**3
        self.mL = 1e-3 * self.L
        self.uL = 1e-6 * self.L
        self.nL = 1e-9 * self.L
        self.pL = 1e-12 * self.L
        self.fL = 1e-15 * self.L
        self.aL = 1e-18 * self.L
        self.kL = 1e3 * self.L
        self.ML = 1e6 * self.L
        self.GL = 1e9 * self.L

        self.ms = 1e-3 * self.s
        self.us = 1e-6 * self.s
        self.ns = 1e-9 * self.s
        self.ps = 1e-12 * self.s
        self.fs = 1e-15 * self.s
        self.minute = 60.0 * self.s
        self.hour = 60.0 * self.minute
        self.day = 24.0 * self.hour
        self.week = 7.0 * self.day
        self.year = 365.256363004 * self.day

        self.Hz = 1.0 / self.s
        self.mHz = 1e-3 * self.Hz
        self.kHz = 1e3 * self.Hz
        self.MHz = 1e6 * self.Hz
        self.GHz = 1e9 * self.Hz
        self.THz = 1e12 * self.Hz
        self.PHz = 1e15 * self.Hz

        self.g = 1e-3 * self.kg
        self.mg = 1e-3 * self.g
        self.ug = 1e-6 * self.g
        self.ng = 1e-9 * self.g
        self.pg = 1e-12 * self.g
        self.fg = 1e-15 * self.g
        self.tonne = 1e3 * self.kg
        self.amu = 1.660538921e-27 * self.kg
        self.Da = self.amu
        self.kDa = 1e3 * self.Da
        self.lbm = 0.45359237 * self.kg

        self.J = (self.kg * self.m**2) / self.s**2
        self.mJ = 1e-3 * self.J
        self.uJ = 1e-6 * self.J
        self.nJ = 1e-9 * self.J
        self.pJ = 1e-12 * self.J
        self.fJ = 1e-15 * self.J
        self.kJ = 1e3 * self.J
        self.MJ = 1e6 * self.J
        self.GJ = 1e9 * self.J
        self.erg = 1e-7 * self.J

        self.meV = 1e-3 * self.eV
        self.keV = 1e3 * self.eV
        self.MeV = 1e6 * self.eV
        self.GeV = 1e9 * self.eV
        self.TeV = 1e12 * self.eV
        self.btu = 1055.056 * self.J
        self.smallcal = 4.184 * self.J
        self.kcal = 4184.0 * self.J
        self.Wh = 3600.0 * self.J
        self.kWh = 1e3 * self.Wh

        self.NA = 6.02214129e23
        self.mol = self.NA
        self.mmol = 1e-3 * self.mol
        self.umol = 1e-6 * self.mol
        self.nmol = 1e-9 * self.mol
        self.pmol = 1e-12 * self.mol
        self.fmol = 1e-15 * self.mol
        self.M = self.mol / self.L
        self.mM = 1e-3 * self.M
        self.uM = 1e-6 * self.M
        self.nM = 1e-9 * self.M
        self.pM = 1e-12 * self.M
        self.fM = 1e-15 * self.M

        self.N = (self.kg * self.m) / self.s**2
        self.dyn = 1e-5 * self.N
        self.lbf = 4.4482216152605 * self.N

        self.Pa = self.N / self.m**2
        self.hPa = 1e2 * self.Pa
        self.kPa = 1e3 * self.Pa
        self.MPa = 1e6 * self.Pa
        self.GPa = 1e9 * self.Pa
        self.bar = 1e5 * self.Pa
        self.mbar = 1e-3 * self.bar
        self.cbar = 1e-2 * self.bar
        self.dbar = 0.1 * self.bar
        self.kbar = 1e3 * self.bar
        self.Mbar = 1e6 * self.bar
        self.atm = 101325.0 * self.Pa
        self.torr = (1.0 / 760.0) * self.atm
        self.mtorr = 1e-3 * self.torr
        self.psi = self.lbf / self.inch**2

        self.W = self.J / self.s
        self.mW = 1e-3 * self.W
        self.uW = 1e-6 * self.W
        self.nW = 1e-9 * self.W
        self.pW = 1e-12 * self.W
        self.kW = 1e3 * self.W
        self.MW = 1e6 * self.W
        self.GW = 1e9 * self.W
        self.TW = 1e12 * self.W

        self.degFinterval = (5.0 / 9.0) * self.K
        self.degCinterval = self.K

        self.mC = 1e-3 * self.C
        self.uC = 1e-6 * self.C
        self.nC = 1e-9 * self.C
        self.Ah = 3600.0 * self.C
        self.mAh = 1e-3 * self.Ah

        self.A = self.C / self.s
        self.mA = 1e-3 * self.A
        self.uA = 1e-6 * self.A
        self.nA = 1e-9 * self.A
        self.pA = 1e-12 * self.A
        self.fA = 1e-15 * self.A

        self.V = self.J / self.C
        self.mV = 1e-3 * self.V
        self.uV = 1e-6 * self.V
        self.nV = 1e-9 * self.V
        self.kV = 1e3 * self.V
        self.MV = 1e6 * self.V
        self.GV = 1e9 * self.V
        self.TV = 1e12 * self.V

        self.ohm = self.V / self.A
        self.mohm = 1e-3 * self.ohm
        self.kohm = 1e3 * self.ohm
        self.Mohm = 1e6 * self.ohm
        self.Gohm = 1e9 * self.ohm
        self.S = 1.0 / self.ohm
        self.mS = 1e-3 * self.S
        self.uS = 1e-6 * self.S
        self.nS = 1e-9 * self.S

        self.T = (self.V * self.s) / self.m**2
        self.mT = 1e-3 * self.T
        self.uT = 1e-6 * self.T
        self.nT = 1e-9 * self.T
        self.G = 1e-4 * self.T
        self.mG = 1e-3 * self.G
        self.uG = 1e-6 * self.G
        self.kG = 1e3 * self.G
        self.Oe = (1000.0 / (4.0 * jnp.pi)) * self.A / self.m
        self.Wb = self.J / self.A

        self.F = self.C / self.V
        self.uF = 1e-6 * self.F
        self.nF = 1e-9 * self.F
        self.pF = 1e-12 * self.F
        self.fF = 1e-15 * self.F
        self.aF = 1e-18 * self.F
        self.H = self.m**2 * self.kg / self.C**2
        self.mH = 1e-3 * self.H
        self.uH = 1e-6 * self.H
        self.nH = 1e-9 * self.H

        self.c0 = 299792458.0 * self.m / self.s
        self.mu0 = 4.0 * jnp.pi * 1e-7 * self.N / self.A**2
        self.eps0 = 1.0 / (self.mu0 * self.c0**2)
        self.Z0 = self.mu0 * self.c0
        self.hPlanck = 6.62606957e-34 * self.J * self.s
        self.hbar = self.hPlanck / (2.0 * jnp.pi)
        self.kB = 1.3806488e-23 * self.J / self.K
        self.GNewton = 6.67384e-11 * self.m**3 / (self.kg * self.s**2)
        self.sigmaSB = 5.670373e-8 * self.W / (self.m**2 * self.K**4)
        self.alphaFS = 7.2973525698e-3

        self.Rgas = self.kB
        self.e = 1.602176565e-19 * self.C
        self.uBohr = 9.27400968e-24 * self.J / self.T
        self.uNuc = 5.05078353e-27 * self.J / self.T
        self.aBohr = 0.52917721092e-10 * self.m
        self.me = 9.10938291e-31 * self.kg
        self.mp = 1.672621777e-27 * self.kg
        self.mn = 1.674927351e-27 * self.kg
        self.Rinf = 10973731.568539 / self.m
        self.Ry = 2.179872171e-18 * self.J
        self.ARichardson = (
            4.0 * jnp.pi * self.e * self.me * self.kB**2
        ) / self.hPlanck**3
        self.Phi0 = 2.067833758e-15 * self.Wb
        self.KJos = 4.83597870e14 * self.Hz / self.V
        self.RKlitz = 2.58128074434e4 * self.ohm

        self.REarth = 6371.0 * self.km
        self.g0 = 9.80665 * self.m / self.s**2
        self.MSun = 1.98892e30 * self.kg
        self.MEarth = 5.9736e24 * self.kg


UNITS = UNITS_class()
