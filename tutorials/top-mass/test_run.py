import sys, os
base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, base)
from gvm.gvm_toolkit import GVMCombination
from gvm.config import build_input_data

here = os.path.dirname(__file__)
os.chdir(here)
data = build_input_data('input_files/LHC_comb.yaml')
comb = GVMCombination(data)
# Work with the input_data object directly
info = comb.get_input_data(copy=True)
# Clear EoE for all systematics by removing entries
for k in list(info.uncertain_systematics.keys()):
    info.uncertain_systematics.pop(k, None)
# Set LHCbJES EoE to 0.1 (dependent type assumed)
info.uncertain_systematics['LHCbJES'] = 0.1
comb.set_input_data(info)
print('mu', comb.fit_results.mu)
print('ci', comb.confidence_interval())

data = build_input_data('input_files/LHC_comb_fictitious_meas.yaml')
comb_fict = GVMCombination(data)
info = comb_fict.get_input_data(copy=True)
for k in list(info.uncertain_systematics.keys()):
    info.uncertain_systematics.pop(k, None)
info.uncertain_systematics['LHCbJES'] = 0.1
comb_fict.set_input_data(info)
print('mu fict', comb_fict.fit_results.mu)
print('ci fict', comb_fict.confidence_interval())
