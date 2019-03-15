from __future__ import print_function

from datetime import datetime

import numpy as np

from neuron import h, gui

class BaseModel(object):
    def __init__(self, *args, **kwargs):
        h.celsius = kwargs.pop('celsius', 33)
        self.log = kwargs.pop('log', print)

        # Model params
        for (var, val) in zip(self.PARAM_NAMES, args):
            setattr(self, var, val)

    @property
    def stim_variable_str(self):
        return "clamp.amp"

    def init_hoc(self, dt, tstop):
        h.tstop = tstop
        h.steps_per_ms = 1./dt
        h.stdinit()

    def attach_clamp(self):
        h('objref clamp')
        clamp = h.IClamp(h.cell(0.5))
        clamp.delay = 0
        clamp.dur = h.tstop
        h.clamp = clamp

    def attach_stim(self, stim):
        # assign to self to persist it
        self.stimvals = h.Vector().from_python(stim)
        self.stimvals.play("{} = $1".format(self.stim_variable_str), h.dt)

    def attach_recordings(self, ntimepts):
        hoc_vectors = {
            'v': h.Vector(ntimepts),
            'ina': h.Vector(ntimepts),
            'ik': h.Vector(ntimepts),
            'ica': h.Vector(ntimepts),
            'i_leak': h.Vector(ntimepts),
            'i_cap': h.Vector(ntimepts),
        }
        
        hoc_vectors['v'].record(h.cell(0.5)._ref_v)
        hoc_vectors['ina'].record(h.cell(0.5)._ref_ina)
        hoc_vectors['ica'].record(h.cell(0.5)._ref_ica)
        hoc_vectors['ik'].record(h.cell(0.5)._ref_ik)
        hoc_vectors['i_leak'].record(h.cell(0.5).pas._ref_i)
        hoc_vectors['i_cap'].record(h.cell(0.5)._ref_i_cap)

        return hoc_vectors

    def simulate(self, stim, dt=0.02):
        _start = datetime.now()
        
        ntimepts = len(stim)
        tstop = ntimepts * dt
        self.init_hoc(dt, tstop)

        h('objref cell')
        h.cell = self.create_cell()
        self.attach_clamp()
        self.attach_stim(stim)
        hoc_vectors = self.attach_recordings(ntimepts)

        self.log.debug("Running simulation for {} ms with dt = {}".format(h.tstop, h.dt))
        self.log.debug("({} total timesteps)".format(ntimepts))

        h.run()

        self.log.debug("Time to simulate: {}".format(datetime.now() - _start))

        return {k: np.array(v) for (k, v) in hoc_vectors.items()}


class Izhi(BaseModel):
    PARAM_NAMES = ('a', 'b', 'c', 'd')
    DEFAULT_PARAMS = (0.01, 0.2, -65., 2.)
    PARAM_RANGES = ( (0.01, 0.1), (0.1, 0.4), (-80, -50), (0.5, 5) )

    @property
    def stim_variable_str(self):
        return "cell.Iin"

    def create_cell(self):
        self.dummy = h.Section()
        cell = h.Izhi2003a(0.5,sec=self.dummy)

        for var in self.PARAM_NAMES:
            setattr(cell, var, getattr(self, var))

        return cell

    def attach_clamp(self):
        self.log.debug("Izhi cell, not using IClamp")

    def attach_recordings(self, ntimepts):
        vec = h.Vector(ntimepts)
        vec.record(h.cell._ref_V) # Capital V because it's not the real membrane voltage
        return {'v': vec}


class HHPoint5Param(BaseModel):
    PARAM_NAMES = ('gnabar', 'gkbar', 'gcabar', 'gl', 'cm')
    DEFAULT_PARAMS = (500, 10, 1.5, .0005, 0.5)
    PARAM_RANGES = ( (200, 800), (8, 15), (1, 2), (0.0004, 0.00055), (0.3, 0.7) )

    def create_cell(self):
        cell = h.Section()
        cell.insert('na')
        cell.insert('kv')
        cell.insert('ca')
        cell.insert('pas')

        cell(0.5).na.gbar = self.gnabar
        cell(0.5).kv.gbar = self.gkbar
        cell(0.5).ca.gbar = self.gcabar
        cell(0.5).pas.g = self.gl
        cell.cm = self.cm

        return cell

class HHBallStick7Param(BaseModel):
    PARAM_NAMES = (
        'gnabar_soma',
        'gnabar_dend',
        'gkbar_soma',
        'gkbar_dend',
        'gcabar_soma',
        'gl_soma',
        'cm'
    )
    DEFAULT_PARAMS = (500, 500, 10, 10, 1.5, .0005, 0.5)
    PARAM_RANGES = (
        (200, 800),
        (200, 800),
        (8, 15),
        (8, 15),
        (1, 2),
        (0.0004, 0.00055),
        (0.3, 0.7)
    )

    DEFAULT_SOMA_DIAM = 21 # source: https://synapseweb.clm.utexas.edu/dimensions-dendrites and Fiala and Harris, 1999, table 1.1

    def __init__(self, *args, **kwargs):
        self.soma_diam = kwargs.pop('soma_diam', self.DEFAULT_SOMA_DIAM)
        self.dend_diam = kwargs.pop('dend_diam', self.DEFAULT_SOMA_DIAM / 10)
        self.dend_length = kwargs.pop('dend_length', self.DEFAULT_SOMA_DIAM * 10)

        super(HHBallStick7Param, self).__init__(*args, **kwargs)
    
    def create_cell(self):
        soma = h.Section()
        soma.L = soma.diam = self.soma_diam
        soma.insert('na')
        soma.insert('kv')
        soma.insert('ca')
        soma.insert('pas')

        dend = h.Section()
        dend.L = self.dend_length
        dend.diam = self.dend_diam
        dend.insert('na')
        dend.insert('kv')

        dend.connect(soma(1))

        # Persist it
        self.dend = dend
        
        for sec in h.allsec():
            sec.cm = self.cm
        for seg in soma:
            seg.na.gbar = self.gnabar_soma
            seg.kv.gbar = self.gkbar_soma
            seg.ca.gbar = self.gcabar_soma
            seg.pas.g = self.gl_soma
        for seg in dend:
            seg.na.gbar = self.gnabar_dend
            seg.kv.gbar = self.gkbar_dend

        return soma
