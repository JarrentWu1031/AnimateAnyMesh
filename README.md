<div align="center">
  <img src="https://github.com/JarrentWu1031/AnimateAnyMesh/blob/main/assets/logo.png" width="300px">
  

</div>

# AnimateAnyMesh: A Feed-Forward 4D Foundation Model for Text-Driven Universal Mesh Animation (ICCV 2025)

Zijie Wu<sup>1,2</sup>, Chaohui Yu<sup>2</sup>, Fan Wang<sup>2</sup>, Xiang Bai<sup>1</sup> <br>
<sup>1</sup>Huazhong University of Science and Technology (HUST), <sup>2</sup>DAMO Acadamy, Alibaba Group


<a href="https://animateanymesh.github.io/AnimateAnyMesh/"><img src='https://img.shields.io/badge/Project-AnimateAnyMesh-brightgreen?logo=github' alt='Project'></a>
<a href="https://arxiv.org/abs/2506.09982"><img src='https://img.shields.io/badge/arXiv-AnimateAnyMesh-B31B1B?logo=arxiv' alt='Paper PDF'></a>
<a href="https://huggingface.co/collections/JarrentWu/dymesh-dataset-68b903973ce18433bd75019a"><img src='https://img.shields.io/badge/HuggingFace-DyMesh Dataset-yellow?logo=huggingface' alt='Hugging Face Datasets'></a>
<a href="https://www.modelscope.cn/collections/DyMesh-Dataset-400fcdb3d60241"><img src="https://img.shields.io/badge/ModelScope-DyMesh Dataset-8A2BE2?logo=alibabacloud&logoColor=white&logoWidth=20" alt="ModelScope Collection"></a>
<a href="https://www.youtube.com/watch?v=q8xH9B0S4y0"><img src='https://img.shields.io/badge/Video-Demo-FF0000?logo=youtube' alt='Video'></a>
<a href="https://huggingface.co/JarrentWu/AnimateAnyMesh/tree/main"><img src='https://img.shields.io/badge/HuggingFace-Model Weights-yellow?logo=huggingface' alt='Hugging Face Weights'></a>
<a href="https://drive.google.com/file/d/1_ixt6pWlUpFvwFn6eV3xuOf1G7g6ijfo/view?usp=sharing"><img src='https://img.shields.io/badge/Google%20Drive-Model Weights-blue?logo=googledrive&logoColor=white' alt='Download from Google Drive'></a>


We present <b>AnimateAnyMesh</b>: the first feed-forward universal mesh animation framework that enables efficient motion generation for arbitrary 3D meshes. Given a static mesh and prompt, our method generates high-quality animations in only a few seconds.

![Demo GIF](https://github.com/animateanymesh/AnimateAnyMesh/blob/main/demo_source/github_demo.gif)

<!-- 

We present <b>AnimateAnyMesh</b>: the first feed-forward universal mesh animation framework that enables efficient motion generation for arbitrary 3D meshes. Given a static mesh and prompt, our method generates high-quality animations in only a few seconds.

<div align=center>
<img src="https://github.com/JarrentWu1031/AnimateAnyMesh/blob/main/assets/teaser.png" width=85%>
</div>

-->

## 🔥 Latest News

* Aug 29, 2025: 👋 The **DyMesh Dataset**([Huggingface](https://huggingface.co/collections/JarrentWu/dymesh-dataset-68b903973ce18433bd75019a), [Modelscope](https://www.modelscope.cn/collections/DyMesh-Dataset-400fcdb3d60241)) is released now, along with the extraction script! We have filtered out data sourced from [AMASS](https://amass.is.tue.mpg.de/) due to its license restriction!
* Aug 22, 2025: 👋 The model weights ([HuggingFace](https://huggingface.co/JarrentWu/AnimateAnyMesh/tree/main), [Google Drive](https://drive.google.com/file/d/1_ixt6pWlUpFvwFn6eV3xuOf1G7g6ijfo/view?usp=sharing)) of **AnimateAnyMesh** has been released! Thanks for the waiting! We also add FBX/ABC export code for a better usage. You can **Animate Your Static Mesh Now!!!**
* Aug 14, 2025: 👋 The inference code of **AnimateAnyMesh** has been released! Thanks for the waiting! The checkpoint will be released in a few days (Still training under the clean code).
* Jun 26, 2025: 👋 **AnimateAnyMesh** has been accepted by [ICCV2025](https://iccv.thecvf.com/)! We will release the code and the DyMesh Dataset mentioned in the paper asap. Please stay tuned for updates！
* Jun 11, 2025: 👋 The paper of **AnimateAnyMesh** is available at [Arxiv](https://arxiv.org/abs/2506.09982)! 

## 🧩 Dataset

The DyMesh dataset is now available, including two subsets with [16-frame](https://www.modelscope.cn/datasets/jarrentwu/DyMesh_16f) and [32-frame](https://www.modelscope.cn/datasets/jarrentwu/DyMesh_32f) sequences respectively. We have filtered out examples with more than 50k vertices. Please note that both subsets are quite large (approximately 1.7TB after uncompression), so you may choose to download only selected split sub-archives. For example: DyMesh_50000v_16f_0000_part_00 and DyMesh_50000v_16f_0000_part_01. After merging, these contain 50k examples in total. Use the following command to uncompress:
```
// Debian/Ubuntu:
sudo apt-get update
sudo apt-get install -y zstd tar -

cd ./tools
chmod 777 uncompress_dataset.sh
./uncompress_dataset.sh <DyMesh Dataset Dir> <Output Dir>
```

## 🔧 Preparation

The code is tested under Python 3.11, CUDA12.8 (CUDA 11.8+ recommended).
```
// Create conda env & Activate
conda create -n animateanymesh python=3.11
conda activate animateanymesh

// Install dependencies
pip install -r requirements.txt
```
You may have to install some dependencies when using bpy. You also have to download the model weights ([HuggingFace](https://huggingface.co/JarrentWu/AnimateAnyMesh/tree/main), [Google Drive](https://drive.google.com/file/d/1_ixt6pWlUpFvwFn6eV3xuOf1G7g6ijfo/view?usp=sharing)) and unzip them under the main folder.

## 📖 Usage

**Text-driven mesh animation.**

For the input mesh format, we only support .glb now. Please make sure the source mesh is static. You can find your desired ones at [Sketchfab](https://sketchfab.com/). 
To animate a static mesh, Use command like:
```
python test_drive.py --data_dir ./examples --vae_dir ./checkpoints --rf_model_dir ./checkpoints --json_dir ./checkpoints/dvae_factors --rf_exp rf_model --rf_epoch f --seed 666 --test_name dragon --prompt "The object is flying" --export_format fbx
```
Then, you are supposed to get a frontal rendered video of a flying dragon & the corresponding FBX export file.

**Tips to get better results with the current AnimateAnyMesh release.**

- **Try different seeds**: For the same prompt and object, outputs can vary significantly across seeds. Some seeds may produce results you find satisfactory and some may not.
- **Adjust guidance**: If a given prompt yields very small motion, try increasing --guidance_scale (e.g., to 5.0).
- **Adjust the number of sampled trajs**: Another way to amplify the motion is: setting --num_traj to be smaller than --max_length // 8. For example, for a mesh with 20,000 vertices, we default to sampling 20000 // 8 = 2500 trajectories; you can try setting --num_traj to 2000 or 1500. (Note: Sampling too few trajectories can degrade shape preservation. We train with an 8× downsampling; test-time mismatch may lead to shape distortion.)
- **If multiple seeds and trajectory counts still fail**: It likely indicates that captions similar to your prompt are rare or OOD problem. In that case, please try rephrasing or changing the prompt. All results shown in the paper and demo can be reproduced by the current model with suitable prompt/seed choices; in our tests we typically tried about three prompt/seed combinations per example.

## 🎬 Animation Example

![Demo GIF](https://github.com/animateanymesh/AnimateAnyMesh/blob/main/demo_source/dragon_demo.gif)

The exported animated dragon viewed in Blender.

## ⭐ Citation
If you find our work useful for your research, please star this repo and cite our paper. Thanks!
```bibtex
@article{wu2025animateanymesh,
    author = {Wu, Zijie and Yu, Chaohui and Wang, Fan and Bai, Xiang.},
    title  = {AnimateAnyMesh: A Feed-Forward 4D Foundation Model for Text-Driven Universal Mesh Animation},
    journal = {arXiv preprint arxiv:2506.09982},
    year   = {2025},
}
```

## ❤️ Acknowledgement 
Our code references some great repos, which are [3DShape2VecSet](https://github.com/1zb/3DShape2VecSet), [Point-E](https://github.com/openai/point-e), [rectified-flow-pytorch](https://github.com/lucidrains/rectified-flow-pytorch) and [CogVideoX](https://github.com/zai-org/CogVideo). We thank the authors for their excellent works! <br>
