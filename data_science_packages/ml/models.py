__author__ = 'Brett Allen (brettallen777@gmail.com)'

import os
import pickle
from glob import glob
from typing import Any, List, Tuple

class ModelTracker(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name, model in kwargs.get('models', []):
            self[name] = model
        if 'models' in self:
            self.pop('models')

    def __setitem__(self, key, value):
        super().__setitem__(key, value)

    def append(self, entry: Tuple[str, Any]):
        # Prevent duplicates and keep latest model entries
        name, model = entry
        self[name] = model

    def get_names(self) -> List[str]:
        return list(self.keys())

    def get_models(self) -> List[Any]:
        return list(self.values())

    def save(self, model_dir: str):
        print('Saving Models:')
        for idx, entry in enumerate(self.items()):
            name, model = entry
            outpath = os.path.join(model_dir, f'{name}.pkl')
        
            # Save the model
            with open(outpath, 'wb') as f:
                pickle.dump(model, f, protocol=5)
        
            print(f'  {idx+1:>2}) {name:<45} | Saved to "{outpath}"')

    def load(self, model_dir: str):
        for path in glob(os.path.join(model_dir, '*.pkl')):
            print(path)
            name = os.path.splitext(os.path.basename(path))[0]
            with open(path, 'rb') as f:
                model = pickle.load(f)
            self[name] = model

    def __repr__(self):
        return [(name, model) for name, model in self.items()]
        