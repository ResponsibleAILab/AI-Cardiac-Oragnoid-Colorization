# AI Cardiac Organoid Colorization
Preprint version of the paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC11326121/

## Types of Models
- **Model 1**: Unet Generator with Patch Discriminator
- **Model 2**: CBAM + Unet Generator with Patch Discriminator
- **Model 3**: Generator Iterations technique implemented on Model 2, where the generator was trained 2 times in one epoch to make it stronger compared to the discriminator.

## Training the Models
These models can be configured by updating the parameters in `utils/params.py`:
- **Model 1**:
  ```python
  use_cbam = False
  generator_steps = 1
  discriminator_steps = 1
  ```
- **Model 2**:
  ```python
  use_cbam = True
  generator_steps = 1
  discriminator_steps = 1
  ```
  - **Model 3**:
  ```python
  use_cbam = True
  generator_steps = 2
  discriminator_steps = 1
  ```

After configuring the model, run `python3 organoid_train_main.py`
