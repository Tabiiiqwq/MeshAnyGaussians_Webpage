---
title: "MeshAnyGaussians"
# meta title
meta_title: ""
# meta description
description: "Final Report"
# save as draft
draft: false
---

<div style="display: flex; justify-content: space-around; flex-wrap: wrap; margin: 40px 0;">
  <div style="text-align: center; margin: 5px;">
    <h3>Siyuan Xie</h3>
    <p>UC Berkeley</p>
  </div>
  <div style="text-align: center; margin: 5px;">
    <h3>Alper Gel</h3>
    <p>UC Berkeley</p>
  </div>
  <div style="text-align: center; margin: 5px;">
    <h3>Ryan Arlett</h3>
    <p>UC Berkeley</p>
  </div>
  <div style="text-align: center; margin: 5px;">
    <h3>Sihan Ren</h3>
    <p>UC Berkeley</p>
  </div>
</div>

Should be a video iframe here; teaser  
Link to this webpage  
Link to slides version  

## Abstract

3D Gaussian Splatting (3DGS) is a classical method for accurate scene reconstruction. Since Gaussian splats serve as an implicit geometric primitive, prior works have attempted to convert them into explicit mesh representations to enable compatibility with modern industrial pipelines, such as SuGaR, GS2Mesh, and others. However, existing approaches often suffer from poor surface quality and undesirable object adhesion, which significantly limits their practicality in real-world applications.

To address these issues, we propose a pipeline with an adaptive mesh extraction process guided by local point density, coupled with a semantic-aware segmenting strategy. Specifically, we leverage a post-training process to augment an existing Gaussian scene with semantic information, resulting in a semantically enriched point cloud. Based on a text query, we render consistent multi-view depth maps and semantic masks. We then apply TSDF fusion in conjunction with an Iso-Octree structure to adaptively extract high-quality meshes from the masked depth regions.

In summary, our system enables high-quality mesh extraction of arbitrary objects specified by text input, given a video of a scene. Additionally, we implement a custom renderer that seamlessly supports both Gaussian splatting and the extracted mesh representation.

{{< image src="images/final_report/pipeline.png" caption="" alt="alter-text" height="" width="" position="center" command="fill" option="q100" class="img-fluid" title="image title"  webp="false" >}}

## Related Works

### Gaussian Splatting and Volume Rendering

Recent advances in neural rendering have proposed numerous techniques for efficient scene representation and novel-view synthesis. Volumetric approaches such as NeRF model the scene as a continuous radiance field, rendering images via volumetric ray marching. Specifically, the color $C$ of a pixel is computed as:

$$
C = \sum_{i=1}^{N} T_i \alpha_i c_i
$$

The transmittance $T_i$ is defined as $T_i = \prod_{j=1}^{i-1} (1 - \alpha_j)$, and the opacity is given by $\alpha_i = 1 - \exp(-\sigma_i \delta_i)$, where $\sigma_i$ and $\delta_i$ denote the density and interval length at the $i$-th sample, respectively.

While this formulation provides high fidelity, it requires dense and costly sampling per ray. In contrast, point-based and Gaussian splatting methods use discrete primitives projected directly onto the image plane. Each 3D Gaussian is defined by a position, an anisotropic covariance matrix, and a view-dependent color, often represented via spherical harmonics. When projected to 2D, these Gaussians are rasterized using alpha blending with visibility ordering, leading to the same image formation model as NeRF, but with $\alpha_i$ derived from the projected 2D Gaussian opacity. This enables real-time rendering and efficient training by avoiding expensive ray marching, while retaining the continuous nature and high quality of volumetric representations through a compact set of explicit 3D Gaussians.

### Spatial Language Embedding

Recent work has extended 3D Gaussian Splatting to encode semantic and language-aligned information into radiance fields. LangSplat introduces a mechanism to supervise each 3D Gaussian with CLIP-derived embeddings extracted from training views. To achieve this, high-dimensional CLIP features ($\mathbb{R}^{512}$) are first compressed into a lower-dimensional latent space ($\mathbb{R}^d$) using a scene-adaptive autoencoder. 

For each semantic level (subpart, part, whole), LangSplat learns view-specific latent targets via reconstruction. Each 3D Gaussian is assigned a learnable embedding $f_i^l \in \mathbb{R}^d$, rendered into views using alpha-composited splatting. The rendered semantic map is defined as:

$$
F_t^l(v) = \sum_{i \in \mathcal{N}(v)} f_i^l \cdot \alpha_i \cdot \prod_{j=1}^{i-1}(1 - \alpha_j)
$$

where $\alpha_i$ denotes the opacity of the $i$-th Gaussian and $\mathcal{N}(v)$ is the set of Gaussians overlapping pixel $v$. Supervision is provided by minimizing the discrepancy between projected features and the ground-truth latent embeddings:

$$
\mathcal{L_{\text{lang}}} = \sum_{t=1}^T \sum_{l \in \{s, p, w\}} \text{dist}(F_t^l(v), H_t^l(v))
$$

where the target feature $H_t^l(v)$ is the compressed CLIP embedding at that same pixel and level. Training aligns rendered and target features. This design enables per-Gaussian language conditioning and view-consistent semantic representations, allowing downstream applications such as natural language querying and part segmentation.

## Method

We decompose the overall pipeline into four subtasks: (1) 3DGS-based scene reconstruction, (2) semantic information extraction from the optimized Gaussian points, (3) depth estimation guided by text input, and (4) mesh extraction from masked depth maps using TSDF and iso-octree fusion.

{{< image src="images/final_report/Flowchart.png" caption="" alt="alter-text" height="" width="" position="center" command="fill" option="q100" class="img-fluid" title="image title"  webp="false" >}}

### Gaussian Splatting for Accurate Reconstruction

We begin by converting the input video into a dense Gaussian Splatting (GS) representation. To achieve this, we first sample the video at 2 fps, and only select the sharpest frames using a variety of OpenCV functions. Next, we run *SIFT-GPU* + *COLMAP* feature extraction followed by *COLMAP* matching [7][13]. Finally, to optimize for speed, we utilize *GLOMAP*'s mapper to complete the SFM process [8]. 

During training, we utilize the *Taming-3DGS* + Fused SSIM accelerated rasterization engine to achieve sub 20-minute training times on consumer GPU's (RTX 4070 mobile) [12].

### Language-Driven Semantic Query and Visualization

To support natural language interaction with 3D representations, we propose a two-stage pipeline centered around image-space query and visualization, with an optional extension for extracting semantic 3D Gaussians based on prompt alignment.

#### Stage 1: Per-View Query and Visualization

Our core pipeline enables language-based visualization by operating on latent semantic features extracted from each training view. Specifically, we first decode per-pixel features $F_t$ from each view $t$ into the CLIP-aligned embedding space using a trained decoder $\Psi$, obtaining:

$$
\hat{F}_t = \Psi(F_t) \in \mathbb{R}^{H \times W \times D}
$$

Given a set of textual prompts $\{p_k\}$, we compute a similarity score for each pixel:

$$
S_t^{(i,j,k)} = \text{sim}(\hat{F}_t^{(i,j)},\; \phi(p_k))
$$

Here, $\phi(p_k)$ denotes the CLIP embedding of prompt $p_k$. The similarity maps are smoothed via local averaging and normalized, followed by thresholding with $\tau$ to produce binary masks:

$$
M_t^{(k)} = \mathbb{1}[S_t^{(k)} > \tau]
$$

Among all CLIP attention heads, we select the one with the highest global response for each prompt to generate the final visualization. The system outputs per-prompt heatmaps, binary masks, and composite visual overlays. This stage is implemented via the `activate_stream()` and `text_query()` routines, and supports tasks such as semantic segmentation, text-guided editing, and interactive inspection of the 3D scene from 2D projections.

#### Stage 2 (Optional): Prompt-Guided 3D Gaussian Extraction

As an alternative or complementary step, we offer a method to extract subsets of 3D Gaussians that consistently correspond to a given prompt across multiple views. Each 3D point $x_i$ is projected into the image plane of view $t$ via:

$$
x_{i,t}^{2D} = \pi_t(x_i) = \mathbf{K}_t \left( \mathbf{R}_t x_i + \mathbf{t}_t \right)
$$

A point receives a binary vote if its projection falls inside the activated mask $M_t^{(k)}$ and passes a depth consistency check with rendered depth $D_t$:

$$
z_i < D_t(x_{i,t}^{2D}) \cdot (1 + \epsilon)
$$

We accumulate votes across all views and select points that are activated in at least $V$ images:

$$
\text{mask}\_k(i) = \mathbb{1} \left[ \sum\_{t=1}^{T} M_t^{(k)}(\pi_t(x_i)) \geq V \right]
$$

This yields a prompt-specific semantic subset of Gaussians, saved as a new PLY file. This extended module supports occlusion filtering, resolution control, and enables downstream applications like phrase-conditioned subcloud editing or text-guided mesh extraction.





### Stereo Matching

In parallel, we obtain high-quality depth estimates of the scene via stereo matching. For each training view, we synthesize a stereo pair by translating the camera slightly along the rightward (epipolar) direction and rendering a second image using the same 3DGS scene representation. Given that the two views are naturally aligned along the baseline, they can be directly fed into a stereo matching model to compute a dense depth map.

For stereo inference, we adopt NVIDIA's state-of-the-art *FoundationStereo* model, which yields accurate depth predictions from rectified stereo pairs [4]. Importantly, we choose stereo matching over monocular depth estimation methods such as Depth Anything, as the latter typically produce pseudo-depths bounded within $[0, 1]$ that are unsuitable for precise 3D reconstruction.

Additionally, due to the inherent noise in the Gaussian surfaces of 3DGS, we avoid computing depth directly from the expected volume density, which tends to produce unstable results. Instead, our stereo-based pipeline yields metrically accurate and globally consistent depth maps that can be reliably fused in the subsequent TSDF pipeline.

### Mesh Reconstruction

Finally, we extract a geometrically accurate mesh from the rendered heatmaps, depth maps, and associated camera parameters. We begin by masking the depth maps using the heatmaps to suppress irrelevant regions, setting those areas to zero. These masked depths are then fused into a continuous signed distance field using a TSDF integration method.

To achieve both high extraction efficiency and adaptive mesh resolution, we adopt the classic **IsoOctree** method [10]. This approach hierarchically partitions space using an octree structure, enabling efficient focus on regions where the signed distance function (SDF) exhibits sign changes. Furthermore, the method can condition on point cloud density to support spatially adaptive resolution control.

Specifically, we first construct a sparse octree where each node stores TSDF values at its voxel corners. For each voxel edge, we then build an associated **edge-tree**, a binary structure that encodes the multi-resolution sign-change status along that edge. If an edge contains a zero-crossing, we recursively traverse its edge-tree to identify the finest sub-edge containing the crossing, and interpolate the SDF values to obtain a well-defined **isovertex**.

{{< image src="images/final_report/image-20250503054122506.png" caption="" alt="alter-text" height="" width="" position="center" command="fill" option="q100" class="img-fluid" title="image title"  webp="false" >}}

For each leaf node in the octree, we extract iso-edges from its six faces using a marching squares–style algorithm. When a face borders a finer-resolution neighbor, we copy the precomputed iso-edges from the finer node to ensure boundary consistency. To prevent open surfaces, we check all isovertices with valence one and trace their symmetric counterparts through the edge-tree to form twin connections, closing any incomplete iso-contours.

Ultimately, every iso-edge is shared by exactly two faces, guaranteeing that the resulting mesh is both **watertight** and **manifold**. Each closed isocontour (isopolygon) is then triangulated using a minimal-area triangulation strategy to form the final triangle mesh.

This method enables consistent and high-fidelity mesh extraction from an unconstrained octree without requiring node refinement or vertex updates, achieving a balance between detail preservation and spatial sparsity.

### GS&Mesh Viewer

Finally, we developed a Windows application based on DirectX11 to visualize both intermediate Gaussian Splatting (GS) results and final mesh outputs [9]. The tool supports rendering `.PLY` files for GS data and `.OBJ` files for mesh geometry. The full renderer code is available on [Splat-Renderer](https://github.com/ryanfsa9/Splat-Renderer).

The renderer was built from scratch, with only a math helper file reused from a prior project. We first implemented a basic Windows GUI with file loading and camera control. On the CPU, we parsed `.PLY` files to extract Gaussian attributes (position, color, opacity, rotation, scale) and computed their covariance matrices. The initial rendering loop sorted Gaussians by depth and alpha-blended each onto the screen, but ignored rotation and ran slowly. We then transitioned to DirectX11 for GPU acceleration, implementing a full rendering pipeline: the vertex shader projects each Gaussian and computes its 2D covariance; the geometry shader builds a rotated quad per Gaussian; and the pixel shader computes Gaussian alpha blending per pixel. This GPU-based pipeline significantly improved performance and visual fidelity.

{{< image src="images/final_report/AD_4nXd8Sdc8KF3UR1Qmz5G3KXcl3anagW2tfA-1uMdtygYUhNSJfjVaHnzyIYC6d8w4Tl5bE6B3CA64laL8q3eDEVwBRGGz4YCI-pd5Y3sAj53hV-S9MdQ5BU7F2emQnT1u5AY.png" caption="" alt="alter-text" height="" width="" position="center" command="fill" option="q100" class="img-fluid" title="image title"  webp="false" >}}

> Example render result

Once the GS renderer was functional, implementing the mesh renderer was straightforward. We parsed `.OBJ` files into GPU buffers and used basic diffuse lighting in our shaders. DirectX handled triangle rasterization, and we simply rendered indexed geometry to the screen. With both modules integrated, our application can now efficiently render both Gaussian splats and traditional meshes for side-by-side analysis.



## Results

Table needed: 

input-image-frame	|	mesh_view_1	|	mesh_view_2	|	mesh_view_3




| Tables        |      Are      |  Cool |
| ------------- | :-----------: | ----: |
| col 3 is      | right-aligned | $1600 |
| col 2 is      |   centered    |   $12 |
| zebra stripes |   are neat    |    $1 |







### Publicly Released Code
1. **Splat Renderer**: https://github.com/ryanfsa9/Splat-Renderer 
2. **Complete Video to Mesh Training Pipeline**: https://github.com/alpergel/final-project.git

## References

1. **Depth Anything V2**  
   Yang, Lihe, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. "Depth Anything V2." *NeurIPS 2024 Poster*, 25 Sept. 2024, https://openreview.net/forum?id=cFTi3gLJ1X. :contentReference[oaicite:0]{index=0}

2. **Segment Anything (SAM)**  
   Kirillov, Alexander, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollar, and Ross Girshick. "Segment Anything." *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) 2023*, pp. 4015–4026. :contentReference[oaicite:1]{index=1}

3. **GS2Mesh**  
   Wolf, Yaniv, Amit Bracha, and Ron Kimmel. "GS2Mesh: Surface Reconstruction from Gaussian Splatting via Novel Stereo Views." *ECCV 2024 Poster*, 2024, https://gs2mesh.github.io/. :contentReference[oaicite:2]{index=2}

4. **FoundationStereo**  
   Wen, Bowen, et al. "FoundationStereo: Zero-Shot Stereo Matching." *arXiv preprint* arXiv:2501.09898, Jan. 2025, https://arxiv.org/abs/2501.09898. :contentReference[oaicite:3]{index=3}

5. **LangSplat**  
   Qin, Minghan, Wanhua Li, Jiawei Zhou, Haoqian Wang, and Hanspeter Pfister. "LangSplat: 3D Language Gaussian Splatting." *arXiv preprint* arXiv:2312.16084, Dec. 2023, https://arxiv.org/abs/2312.16084. :contentReference[oaicite:4]{index=4}

6. **Gaussian-Splatting-Lightning**  
   yzslab. *gaussian-splatting-lightning*. GitHub, 2024, https://github.com/yzslab/gaussian-splatting-lightning. :contentReference[oaicite:5]{index=5}

7. **COLMAP**  
   Schönberger, Johannes Lutz, and Jan-Michael Frahm. *COLMAP: Structure-from-Motion and Multi-View Stereo*. GitHub, 2024, https://github.com/colmap/colmap. :contentReference[oaicite:6]{index=6}

8. **GLOMAP**  
   Pan, Linfei, Dániel Baráth, Marc Pollefeys, and Johannes L. Schönberger. "Global Structure-from-Motion Revisited." *arXiv preprint* arXiv:2407.20219, Jul. 2024, https://arxiv.org/abs/2407.20219. :contentReference[oaicite:7]{index=7}

9. **Direct3D 11**  
   Microsoft. *Programming Guide for Direct3D 11*. Microsoft Learn, 2019, https://learn.microsoft.com/en-us/windows/win32/direct3d11/atoc-dx-graphics-direct3d-11. :contentReference[oaicite:8]{index=8}

10. **IsoOctree**  
    Kazhdan, Michael. "Unconstrained Isosurface Extraction on Arbitrary Octrees." *Symposium on Geometry Processing*, 2008, https://www.cs.jhu.edu/~misha/Code/IsoOctree/. :contentReference[oaicite:9]{index=9}

11. **OpenCLIP**  
    Ilharco, Gabriel, et al. "OpenCLIP." *Zenodo*, Jul. 2021, https://doi.org/10.5281/zenodo.5143773. :contentReference[oaicite:10]{index=10}

12. **Taming 3DGS**  
    Mallick, Saswat Subhajyoti, Rahul Goel, Bernhard Kerbl, Markus Steinberger, Francisco Vicente Carrasco, and Fernando De La Torre. "Taming 3DGS: High-Quality Radiance Fields with Limited Resources." *SIGGRAPH Asia 2024 Conference Papers*, Association for Computing Machinery, 2024, https://doi.org/10.1145/3680528.3687694. :contentReference[oaicite:11]{index=11}
13. **SiftGPU**  
    Wu, Changchang. "SiftGPU: A GPU Implementation of Scale Invariant Feature Transform (SIFT)." *GitHub*, 2007, https://github.com/pitzer/SiftGPU.git. :contentReference[oaicite:12]{index=12}


## Contributions

Automatic Video to SFM to Gaussian Splatting: Alper

Semantical Segmentation: Siyuan & Sihan

Stereo Matching: Alper

Mesh Recon: Siyuan

GS & Mesh Viewer: Ryan
