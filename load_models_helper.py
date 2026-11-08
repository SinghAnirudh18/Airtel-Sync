import sys
import pickle
import joblib
import numpy as np
import numpy.random._pickle as nrp
import joblib.numpy_pickle as jnp

# Patch __bit_generator_ctor for cross-numpy compatibility
orig_ctor = nrp.__bit_generator_ctor

def safe_bit_generator_ctor(bit_generator_name='MT19937'):
    s = str(bit_generator_name)
    for k in nrp.BitGenerators:
        if k in s:
            return nrp.BitGenerators[k]()
    try:
        return orig_ctor(bit_generator_name)
    except Exception:
        return nrp.BitGenerators['PCG64']()

nrp.__bit_generator_ctor = safe_bit_generator_ctor

# Patch load_build so BitGenerator state dict mismatch doesn't raise TypeError
orig_load_build = jnp.NumpyUnpickler.load_build

def safe_load_build(self):
    inst = self.stack[-2]
    state = self.stack[-1]
    name = str(type(inst))
    if any(bg in name for bg in ['PCG64', 'MT19937', 'Philox', 'SFC64', 'BitGenerator']):
        if not isinstance(state, dict):
            # Fallback valid state dictionary for numpy 1.26
            if 'PCG64' in name:
                self.stack[-1] = {'bit_generator': 'PCG64', 'state': {'state': 1, 'inc': 1}, 'has_uint32': 0, 'uinteger': 0}
            else:
                self.stack[-1] = {'bit_generator': 'MT19937', 'state': {'key': np.zeros(624, dtype=np.uint32), 'pos': 0}, 'has_uint32': 0, 'uinteger': 0}
    orig_load_build(self)

jnp.NumpyUnpickler.load_build = safe_load_build

def load_all_models():
    models = {}
    for name in ['bundle_recommender_model.pkl', 'mrr_regressor_model.pkl', 'propensity_ensemble_model.pkl']:
        models[name] = joblib.load(f'models/{name}')
        print(f"[SUCCESS] Loaded models/{name} -> {type(models[name])}")
    return models

if __name__ == '__main__':
    loaded = load_all_models()
    print("All 3 models loaded successfully!")

