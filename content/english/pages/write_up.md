---
title: "MeshAnyGaussians Write Up"
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

We have developed a pipeline that extracts high-quality meshes of user-specified objects from an input video. Our approach builds upon 3D Gaussian Splatting (3DGS), a powerful method for accurate scene reconstruction. While Gaussian splats serve as an implicit geometric primitive, converting them into explicit mesh representations to enable compatibility with modern industrial pipelines remains challenging. Existing approaches like SuGaR and GS2Mesh often suffer from poor surface quality and undesirable object adhesion, which significantly limits their practicality.

Our pipeline works in several stages: First, we convert video into a Gaussian Splatting scene, then isolate specific objects based on text input through a semantic-aware segmentation strategy. For this, we leverage a post-training process to augment the Gaussian scene with semantic information, resulting in a semantically enriched point cloud. Based on the text query, we render consistent multi-view depth maps and semantic masks.

After the objects are separated, we utilize an improved mesh reconstruction algorithm guided by local point density via a state of the art Stereo-Matching model. We apply TSDF fusion in conjunction with an Iso-Octree structure to adaptively extract high-quality meshes from the masked depth regions, overcoming the limitations of previous approaches.

To visualize our results, we've implemented a custom DirectX11-based renderer capable of displaying both the extracted meshes and the intermediate Gaussian splatting scenes for comparison. Our implementation combines existing GS libraries (for video-to-GS conversion and object separation) with our own algorithms for mesh reconstruction and rendering.

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

Our pipeline begins by converting the input video into a Gaussian Splatting scene. To achieve this, we first sample the video at 2 fps, and only select the sharpest frames using a variety of OpenCV functions. Next, we run *SIFT-GPU* + *COLMAP* feature extraction followed by *COLMAP* matching [7][13]. Finally, to optimize for speed, we utilize *GLOMAP*'s mapper to complete the SFM process [8]. These functions combined allow for a 3D reconstruction of the camera poses, giving the SFM precursor that gaussian splatting requires. During training, we utilize the *Taming-3DGS* + Fused SSIM accelerated rasterization engine to achieve sub 20-minute training times on consumer GPU's (RTX 4070 mobile) [12]. 

Since we are building off of the Gaussian-Splatting-Lightning library, our work here entailed compiling seamless docker pipelines via docker-compose where the script calls the relevant docker container, runs the code required, and exits moving on to the subsequent task [6]. This allows any computer, with any OS to run our code, given they have an NVIDIA CUDA powered GPU. Overall, the goal of our work in this segment was to bring together existing libraries to create a seamless, robust, and quick video to gaussian splat pipeline, that can be easily utilized for subsequent pipeline stages.


### Language-Driven Semantic Query/Visualization

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

For this component, work also went towards pipelining it to work with our other components, in addition to fitting it within the Docker build infrastructure. Once this was complete, our group updated the CLIP model utilized from the OpenCLIP library, due to the out-of-date model choices of the LangSplat library [11]. As a further note for the reader, LangSplat utilizes the antiquated Segment-Anything-Model (SAM), which is significantly slower and produces worse segmentations than the recently released SAM 2.1 model series [2]. Due to this, a significant amount of our time for this stage was spent attempting to upgrade the SAM model, but due to a major difference in model inference infrastructure, and library configuration mismatches, we were unable to complete this upgrade. 

### Stereo Matching

In parallel with the segmentation process, we also obtain high-quality depth estimates of the scene via iterative stereo matching. In short, the stereo matcher uses the difference in object positions between two renders with slightly shifted camera positions to compute depth. For each training view, we artificially translate the gaussian view camera slightly rightword, and render a second image using the same 3DGS scene representation. Since we know each image from this process is aligned along the baseline, they can be reliably fed into a stereo matching model to compute a dense depth map. When picking the right model to perform the stereo-matching process, we opted to utilize NVIDIA’s state-of-the-art FoundationStereo model, given its recent release and significant advances. This model robustly yields accurate depth predictions from rectified stereo image pairs [4]. 

Initially we attempted to use DepthAnythingV2, which is a monocular (single-view) depth estimation model, but these models produce pseudo-depths bounded between $[0, 1]$ that are unsuitable for precise 3D reconstruction, proven through our initial noisy outputs. Further, due to the inherent noise in the Gaussian surfaces of 3DGS, we avoid computing depth directly from the expected volume density, since this also produced unstable results based on our preliminary testing. Instead, our stereo-based pipeline yields metrically accurate and globally consistent depth maps that can be reliably used by our mesh generation algorithm.


### Mesh Reconstruction

Finally, we extract a geometrically accurate mesh from the rendered heatmaps, depth maps, and associated camera parameters. We begin by masking the depth maps using the heatmaps to suppress irrelevant regions, setting those areas to zero. These masked depths are then fused into a continuous signed distance field using a Truncated Signed Distance Function (TSDF) integration scheme.

To build the TSDF field, we evaluate the signed distance between each voxel center and its projections in all training views. Let $x \in \mathbb{R}^3$ be a voxel center and let $D_t$ be the depth map of view $t$. The signed distance at $x$ is computed as:

$$
\text{TSDF}_t(x) = \frac{D_t(\pi_t(x)) - \| x - c_t \|}{\delta(x)}
$$

where $\pi_t(x)$ is the projection of $x$ onto the image plane of camera $t$, $c_t$ is the camera center, and $\delta(x)$ is the truncation distance—adaptively chosen based on projected depth or clamped to a global maximum $\delta_{\text{max}}$.

Only pixels with valid depth estimates contribute to the TSDF. The per-voxel TSDF value is computed by weighted averaging across views:

$$
\text{TSDF}(x) = \frac{\sum_t w_t(x) \cdot \text{TSDF}_t(x)}{\sum_t w_t(x)}
$$

Here, the weight $w_t(x)$ is inversely proportional to the projected depth, favoring closer observations:

$$
w_t(x) = \frac{1}{\| x - c_t \| + \epsilon}
$$

We discard large negative TSDF values in voxels that fall behind observed surfaces, as these typically correspond to occluded or non-visible regions. To prevent reconstruction holes due to insufficient valid observations, we assign a fallback value of $-1$ to voxels that are consistently behind surfaces in multiple views.

This fusion strategy avoids relying on surface normals, enabling compatibility with noisy or unstructured input depth maps. The resulting TSDF field is continuous, normalized, and suitable for iso-surface extraction.

To achieve both high extraction efficiency and adaptive mesh resolution, we adopt the classic **IsoOctree** method [10]. This approach hierarchically partitions space using an octree structure, enabling efficient focus on regions where the signed distance function (SDF) exhibits sign changes. Furthermore, the method can condition on point cloud density to support spatially adaptive resolution control.

Specifically, we first construct a sparse octree where each node stores TSDF values at its voxel corners. For each voxel edge, we then build an associated **edge-tree**, a binary structure that encodes the multi-resolution sign-change status along that edge. If an edge contains a zero-crossing, we recursively traverse its edge-tree to identify the finest sub-edge containing the crossing, and interpolate the SDF values to obtain a well-defined **isovertex**:

$$
\text{isovertex} = \text{lerp}(x_a, x_b;\, \frac{\text{TSDF}(x_a)}{\text{TSDF}(x_a) - \text{TSDF}(x_b)})
$$

For each leaf node in the octree, we extract iso-edges from its six faces using a marching squares–style algorithm. When a face borders a finer-resolution neighbor, we copy the precomputed iso-edges from the finer node to ensure boundary consistency. To prevent open surfaces, we check all isovertices with valence one and trace their symmetric counterparts through the edge-tree to form twin connections, closing any incomplete iso-contours.

Ultimately, every iso-edge is shared by exactly two faces, guaranteeing that the resulting mesh is both **watertight** and **manifold**. Each closed isocontour (isopolygon) is then triangulated using a minimal-area triangulation strategy to form the final triangle mesh.

This method enables consistent and high-fidelity mesh extraction from an unconstrained octree without requiring node refinement or vertex updates, achieving a balance between detail preservation and spatial sparsity.


### GS&Mesh Viewer
To view our results (both the intermediate Gaussian Splatting scenes and our output meshes), we created a DirectX11 based windows application from scratch that can render both `.PLY` (GS) and `.OBJ` (Mesh) files [9].


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
