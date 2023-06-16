# WCST_behavioral_model_and_analysis

Project co-developed with Jeong-Woo Kim and Yue Liu

## SSM package
### ssm package modified from original source here: https://github.com/lindermanlab/ssm

### Installation direction 

```
git clone https://github.com/lindermanlab/ssm
cd ssm
pip install numpy cython
pip install -e .
```

fitModel.py
  * fits HLM-GLM model to data in rawData/inputs/
  * writes saved model to rawData/modelSaves/

analyzeFits.py
  * runs all analysis (choice and transition probabilities, category analysis, inter-species performance comparison)
  * writes results to analysis/

predictObject.ipynb
  * reads files from rawData/forObjectPred/ and analysis/ (created by analyzeFits.py)
  * processes, statistically tests and visualizes results

Fig<1-7>.ipynb
  * reads files from analysis/
  * processes, statistically tests and visualizes results
  * saves figures in .eps format to figures/
