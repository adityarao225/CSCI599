# CSCI599

## How to use starter code
## Table of Contents
- [How to use](#how-to-use)
- [Assignment 1: Geometry Processing](#assignment-1-geometry-processing)
    - [Introduction](#introduction)
    - [Requirements / Rubric](#requirements--rubric)
- [Assignment 2: Structure From Motion](#assignment-2-structure-from-motion)
    - [Introduction](#introduction-1)
    - [Requirements / Rubric](#requirements--rubric-1)

## How to use
```shell
git clone https://github.com/jingyangcarl/CSCI599.git
cd CSCI599
ls ./ # you should see index.html and README.md showup in the terminal
code ./ # open this folder via vscode locally
# open and right click on the index.html
# select "Open With Live Server" to run the code over localhost.
```

## Project Details
This project explores advanced techniques for editing 3D models, focusing on two main methods: Loop Subdivision and Quadric Error Mesh Simplification, using a special way of organizing 3D model data called the half-edge data structure for better performance.

Loop Subdivision
Loop Subdivision is a technique that makes rough or blocky 3D models look smoother and more detailed. It works well with models made up of triangles. The idea is to take a simple model and make it more complex by adding more triangles in a smart way, based on the positions of the original points (vertices). The half-edge data structure is used here because it helps keep track of how points, lines (edges), and surfaces (faces) are connected, making it easier to figure out where to add new points and triangles.

When we start, we first organize the model's data using the half-edge structure. Then, we calculate where new points should go using a special formula. If we're working on the edge of the model, we simply put a new point in the middle. But if it's an interior part, we use the formula to decide the exact spot based on nearby points and edges. After placing new points, we connect them to form new, smaller triangles. Doing this several times makes the model smoother and more detailed with each step.

Quadric Error Mesh Simplification
This method aims to make a 3D model simpler by reducing the number of triangles it has, without losing its essential shape. It does this by measuring how much error would be introduced if we remove certain points and reorganize the remaining ones. Each point gets a score that tells us how much the model's shape would change if that point were removed. The process involves picking points with the lowest scores (meaning their removal would change the model the least) and removing them, then updating the model.

The half-edge data structure is crucial here too because it lets us quickly find which points and faces are connected, making it easier to calculate scores and decide which points to remove. For each point considered for removal, the method looks for the best new position that would introduce the least error, then updates the model by removing the point and adjusting the connections. This process is repeated until the model is as simple as needed.

In simpler terms, these techniques help in making 3D models look better or simpler, depending on what's needed, by adding or removing points and triangles in a smart way, with the help of a special way of organizing the model's data.

Time Complexity:
My analysis, based on the details of my implementation, suggests:

Loop Subdivision: Exponential in the number of iterations, roughly O(4^i * (V + E + F)), due to the quadrupling of faces with each iteration and the rebuilding of the half-edge structure.
Quadric Error Metrics Decimation: Dominated by edge collapse priority queue operations, O(E * log E), with significant overhead from mesh updates and error quadric computations.
These complexities reflect the nature of the operations I performed in my implementations, though actual performance can vary with the structure of the mesh and the specifics of my implementation.

## Decimation Plots

![Decimation with 3 iterations](./plots/decimation_3.png)

![Decimation with 5 iterations](./plots/decimation_5.png)

![Decimation with 7 iterations](./plots/decimation_7.png)

![Decimation with 9 iterations](./plots/decimation_9.png)

![Decimation with 11 iterations](./plots/decimation_11.png)

## Subdivision Plots

![Subdivision with 1 iteration](./plots/subdivision_1.png)

![Subdivision with 2 iterations](./plots/subdivision_2.png)

![Subdivision with 3 iterations](./plots/subdivision_3.png)

## Extra Credits

![Extra Credit with 11 iterations](./plots/extra_credit_11.png)

![Extra Credit with 9 iterations](./plots/extra_credit_9.png)

![Extra Credit with 7 iterations](./plots/extra_credit_7.png)

![Extra Credit with 5 iterations](./plots/extra_credit_5.png)

## Assignment 1: Geometry Processing
![Mesh Decimation](img/meshdecimation.png)

### Introduction
In this assignment, you will implement surface subdivision and simplification using **Loop Subdivision** and **Quadric Error Metrics**, respectively. The task requires the construction of a data structure with adjacency, such as half-edge or incidence matrices, to facilitate quick traversal of mesh regions for computations. You can find the algorithms in the class lectures. The outcome will be an upsampled or downsampled version of a given mesh.

The following files are used:
- `assignments/assignment1.py`
- `html/assignment1.html`
- `js/assignment1.js`

### Requirements / Rubric
* +40 pts: Implement loop subdivision.
* +40 pts: Implement Quadratic Error based mesh decimation.
* +20 pts: Write up your project, detials of data structure, algorithms, reporting runtime and visualiztions of results with different parameters.
* +10 pts: Extra credit (see below)
* -5*n pts: Lose 5 points for every time (after the first) you do not follow the instructions for the hand in format

**Forbidden** You are not allowed to call subdivision or simpilication functions directly. Reading, visualization and saving of meshes are provided in the start code.

**Extract Credit** You are free to complete any extra credit:

* up to 5 pts:Analyze corner cases (failure cases) and find solutions to prevent them.
* up to 10 pts: Using and compare two different data structures.
* up to 10 pts: Impelemnt another subdivision or simplication algorithm.
* up to 10 pts: Can we preserve the original vertices after decimation (the vertices of the new meshes are a subset of the original vertices) ? Give it a try.

For all extra credit, be sure to demonstrate in your write up cases where your extra credit.

## Assignment 2: Structure From Motion
![Mesh Decimation](img/sfm.png)

### Introduction
In this assignment, you will implement structure from motion in computer vision. Structure from motion (SFM) is a technique used to reconstruct the 3D structure of a scene from a sequence of 2D images or video frames. It involves estimating the camera poses and the 3D positions of the scene points.

The goal of SFM is to recover the 3D structure of the scene and the camera motion from a set of 2D image correspondences. This can be achieved by solving a bundle adjustment problem, which involves minimizing the reprojection error between the observed 2D points and the projected 3D points.

To implement SFM, you will need to perform the following steps:
1. Feature extraction: Extract distinctive features from the input images.
2. Feature matching: Match the features across different images to establish correspondences.
3. Camera pose estimation: Estimate the camera poses for each image.
4. Triangulation: Compute the 3D positions of the scene points using the camera poses and the corresponding image points.
5. Bundle adjustment: Refine the camera poses and the 3D points to minimize the reprojection error.

By implementing SFM, you will gain hands-on experience with fundamental computer vision techniques and learn how to reconstruct 3D scenes from 2D images. This assignment will provide you with a solid foundation for further studies in computer vision and related fields.

The following files are used:
- `assignments/assignment2/assignment2.py`
- `assignments/assignment2/feat_match.py`
- `assignments/assignment2/sfm.py`
- `assignments/assignment2/utils.py`
- `html/assignment2.html`
- `js/assignment2.js`

### Requirements / Rubric
* +80 pts: Implement the structure-from-motion algorithm with the start code.  
* +20 pts: Write up your project, algorithms, reporting results (reprojection error) and visualisations (point cloud and camera pose), compare your reconstruction with open source software Colmap.
* +10 pts: Extra credit (see below)
* -5*n pts: Lose 5 points for every time (after the first) you do not follow the instructions for the hand in format

**Extract Credit** You are free to complete any extra credit:

* up to 5 pts: Present results with your own captured data.
* up to 10 pts: Implement Bundle Adjustment in incremental SFM.
* up to 10 pts: Implement multi-view stereo (dense reconstruction).
* up to 20 pts: Create mobile apps to turn your SFM to a scanner.  
* up to 10 pts: Any extra efforts you build on top of basic SFM.

For all extra credit, be sure to demonstrate in your write up cases where your extra credit.
