---
title: "CS184 Project Milestone: MeshAnyGaussians"
# meta title
meta_title: ""
# meta description
description: "This is meta description"
# save as draft
draft: false
---
### Team36

**Team Members: Siyuan Xie, Alper Gel, Ryan Arlett, Sihan Ren**

> Extract any object’s mesh efficiently via text selection from a scene


### What we have accomplished
**Method**: We have the baseline code running now, which can generate a coarse mesh from a gaussian scene. Also, we have prepared a SOTA, industry-style mp4-to-gaussian script with minimal floaters and found a latest stereo-matching model, which will be both integrated into the baseline later. Finally, we are working on integrating the AdaptiveTSDF+IsoOctree method to extract mesh efficiently, while the gaussian segmentation is also under experimental.

**Renderer**: Our gaussian splatting renderer is close to being complete. So far we have the windows app and input collection system done. Our rendering algorithm involves the following steps:
1. Parse the input file (.ply) to extract the gaussians (locations, scales, rotations, colors)
2. Project each 3d gaussian into a 2d gaussian in clip space.
3. Sort gaussians by depth. We use the simple std::sort, but faster implementations use a radix sort on the GPU.
4. Alpha blend each gaussian onto the frame buffer, in order from back to front.
5. Handle input (camera movement) and repeat from step 2.

### Preliminary Results
- Current baseline code can generate coarse mesh. (Left two)
- Current baseline can segment the whole scene by language feature/sementics (right one)
- Current renderer can render approximately (without rotation of gaussians)

{{< image src="images/milestone/pipeline.png" caption="" alt="alter-text" height="" width="" position="center" command="fill" option="q100" class="img-fluid" title="image title"  webp="false" >}}

{{< image src="images/milestone/render.png" caption="" alt="alter-text" height="370" width="" position="center" command="" option="q100" class="img-fluid" title="image title"  webp="false" >}}
### Reflect on progress relative to plan

**Method**: Time is kind-of short, but we are in the integrating pipeline so that shouldn’t be so much difficult. We will replace some methods in the core .py file.
Renderer: Was planning to have the gaussian splatting renderer completely done by now, but it is pretty close so we shouldn’t have trouble finishing it and adding regular mesh rendering by the deadline.

**Renderer**: Was planning to have the gaussian splatting renderer completely done by now, but it is pretty close so we shouldn’t have trouble finishing it and adding regular mesh rendering by the deadline.


### Following work plan

**Method**:
- Integrate better depth module to baseline
- Integrate better gs-segment module to baseline
- Implement the Adaptive-TSDF/Iso-Octree module to baseline
- Finish the gaussian splatting renderer, and also add mesh rendering to it so we can display our meshed results next to the input gaussians.
- Separating GS with different semantics， handle user text query

**Renderer**:
The rendering algorithm is close but still has a few kinks to work out. 
As you can see below, it generates somewhat decent images, but not perfect since we don’t yet account for the rotations of the gaussians, treating them all as axis aligned to simplify the math. Currently all the rendering and sorting of the gaussians is done on the CPU for simplicity, so it runs pretty slow (about 1fps at 300x300 resolution for ~1 million gaussians).
