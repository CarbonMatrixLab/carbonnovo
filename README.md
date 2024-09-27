
###  CarbonNovo
CarbonNovo: Joint Design of Protein Structure and Sequence Using a Unified Energy-based Model.
<https://proceedings.mlr.press/v235/ren24e.html> 

### Installation
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

pip install -r requirements.txt

```
**Model weights**
1. Download CarbonNovo model weights from <https://carbonnovo.s3.amazonaws.com/params.tar>, and place them in the ./params directory.
2. Download the ESM2 model weights from <https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt> and <https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt>, and place them in the `./params` directory. 


Please feel free to contact us if you have any problems with the installation (Haicang Zhang, zhanghaicang@ict.ac.cn).
### Usage
Example:
```bash
python predict.py sample_length=256 sample_number=4
```
Here, sample_length denotes the length of the proteins to be sampled, and sample_number denotes the number of samples. The sampled structures and sequences will be put in the directory./output.


### Citation
CarbonNovo: Joint Design of Protein Structure and Sequence Using a Unified Energy-based Model. M. Ren, T. Zhu, H. Zhang#. ICML 2024. https://openreview.net/attachment?id=FSxTEvuFa7&name=pdf

