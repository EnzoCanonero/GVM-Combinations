import sys, os
base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, base)
from gvm_toolkit import GVMCombination

here = os.path.dirname(__file__)
os.chdir(here)
comb = GVMCombination('input_files/LHC_comb.yaml')
info = comb.input_data()
for k in info['syst']:
    info['syst'][k]['error-on-error']['value'] = 0.0
info['syst']['LHCbJES']['error-on-error']['value'] = 0.1
comb.update_data(info)
print('mu', comb.fit_results['mu'])
print('ci', comb.confidence_interval())
