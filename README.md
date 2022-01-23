# Interpretability for GNN in Chemistry

This repository contains implementation of GNN intepretability method basing on atom substitution as well as some gradient based methods.

---

## GNNInterpreter.py

The main module containing source code. Contains GNNInterpreter class that should be used to get results.

### Public methods
 
 ```
 get_original_pred(self, return_tensor=False)
 ```
 Return model prediction of the model on the last interpreted molecule.

 #### Params
 - return_tensor

Bool. When True method will return torch.Tensor. When False method returns float.

---

```
get_importance_map_svg(self, 
                       mol,
                       method,
                       replace_atoms_with,
                       replace_atom_alg=,
                       calculate_atom_weight_alg,
                       backward_func
                       )
```

#### Params

- `mol`

Molecule to interpret the model for. Can be `rdkit.Chem.Mol` or smiles string.

- `method`

String. Interpretability method to use. Allowed values - **'substitution', 'gradcam', 'saliency'**. Default - 'substitution'. When the value is 'substitution' backward_func parameter is ignored. When the value is 'gradcam' or 'saliency' all parameters except of backward_func are ignored.

- `replace_atoms_with`

String. Describes with what the method will try to replace atoms in mol. Allowed values - **'all', 'zero', 'Br', 'C', 'Cl', 'F', 'H', 'I', 'N', 'O', 'P', 'S'**. Default - 'all'. When the value is 'all' replacing with all allowed atoms will be tried and the results will be aggregated to produce final result. When the value is 'zero' removing information about atom type will be performed and results of the removal will be returned as a result. When the value is other than 'all' and 'zero' replacement with specific atom will be performed and results of the replacement will be returned.

- `replace_atom_alg`

String. Describes what type of atom replacement will be used in the method. Allowed values - **'number', 'atom'**. Default - 'number'. When the value is 'number' replacement only of the atom number will be performed with preserving of the atom properties. When the value is 'atom' the replacement of the whole Atom object will be performed, changes of the atom properties are possible.

- `calculate_atom_weight_alg`

String. Describes how to calculate atom weight. Allowed values - **'signed', 'absolute'**. Default - 'signed'. When the value is 'signed' both positive and negative atom influence are calculated. When the value is 'absolute' only absolute value influence is calculated.

- `backward_func`

Callable. Callable object that performs backwad pass of the model on `mol` on call. Default - `None`.

#### Returns

- Tuple of two elements. First element is `IPython.display.SVG` object that can be displayed directly in Jupyter notebook. Second element is `svgutils.transform.SVGFigure` object that can be further modified.
---

#### Calling example

```
model = ...
featurizer = ...
mol = ...
interpreter = GNNInterpreter(model, featurizer)
svg, fig = interpreter.get_importance_map_svg(mol, 'substitution', 'all', 'number', 'absolute', None)
display(svg)
```

Please refer to [Usage exmaples](#usage-examples)

## Usage examples

Files `ESOL.jpynb` and `BACE.jpyng` contain example experiments for esol regression and bace classification tasks respectively.

Examine the notebooks for usage details.

Notebooks contain code to define and train a sample GNN model. It is possible to use interactive interface for interpretations generation as it is show in notebooks. The last cell of each notebook contains code to generate SVG and PNG images of selected interpretation methods. To be able to generate PNG you need to have `inkscape` installed. Otherwise feel free to modify the code and use your own tool to convert SVG to PNG.

Expressing gratitude to [umwpl2021](https://github.com/gmum/umwpl2021) repository authors. Code for experiments is based on their repo.