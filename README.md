<h1 align="center">SteeredMarigold: Steering Diffusion Towards Depth Completion of Largely Incomplete Depth Maps</h1>
<p align="center">
  <a href="https://steeredmarigold.github.io"><img alt="Homepage" src="https://img.shields.io/badge/Visit-Homepage-blue"></a>
  <a href="https://ieeexplore.ieee.org/abstract/document/11128449"><img alt="Paper (IEEE)" src="https://img.shields.io/badge/Read-Paper_(IEEE)-green"></a>
  <a href="https://arxiv.org/abs/2409.10202"><img alt="Paper (arXiv)" src="https://img.shields.io/badge/Read-Paper_(arXiv)-yellow"></a>
  <a href="https://steeredmarigold.github.io/static/images/poster8_small.pdf"><img alt="ICRA Poster" src="https://img.shields.io/badge/View-ICRA_Poster-red"></a>
</p>

This is the official implementation of "SteeredMarigold: Steering Diffusion Towards Depth Completion of Largely Incomplete Depth Maps" paper, presented at ICRA 2025.

# ⚙️ Environment

To run the code, create the following conda environment and set the environment variables.

Create the environment:

`conda env create -f environment.yml`

activate it:

`conda activate steered-marigold`

and install pip dependencies:

`pip install -r requirements.txt`

Set the environment variables pointing to your dataset directory and directory containing results:

`export DATASETS_DIR="/path/to/your/dataset/directory"`

`export MODELS_DIR="/path/to/directory/to/store/results"` 

# 🗄️ Data

Download labeled NYUv2 Depth dataset from: [cs.nyu.edu](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html). The downloaded .mat file is to be placed in "nyu-v2" sub-directory of your dataset directory.

# ⚡ Run

To reproduce the results from the paper, run the following scripts:

- `./validate_steeredmarigold_incomplete` for steering Marigold with largely incomplete depth maps
- `./validate_steeredmarigold_uniform` for steering with depth uniformly sampled from the available ground-truth
- `./validate_marigold` Marigold without steering

# 📚 Citation

```bibtex
@INPROCEEDINGS{steered-marigold,
      author={Gregorek, Jakub and Nalpantidis, Lazaros},
      booktitle={2025 IEEE International Conference on Robotics and Automation (ICRA)}, 
      title={SteeredMarigold: Steering Diffusion Towards Depth Completion of Largely Incomplete Depth Maps}, 
      year={2025},
      pages={13304-13311},
      doi={10.1109/ICRA55743.2025.11128449}
}
```

# 🖇️ License

Code in this repository is licensed under [Apache License, Version 2.0](https://github.com/DTU-PAS/steered-marigold/blob/main/LICENSE).