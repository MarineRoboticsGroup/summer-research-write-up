# **Marine Robotics Group Documentation**

## **Introduction**

Our names are [Abigail Greenough](https://www.linkedin.com/in/abigail-greenough-7a56331a9) and [Calvin Cigna](https://www.linkedin.com/in/calvin-cigna-b4b6a6280), and the summer before our senior year, we interned with the [Marine Robotics Group](https://marinerobotics.mit.edu/) at [MIT CSAIL](https://www.csail.mit.edu/). 

This page is intended to function as a summary of our accomplishments over the course of this summer, while also serving as a springboard for future researchers. As you peruse the content, you might find it beneficial to consult our repository on [GitHub](https://github.com/MarineRoboticsGroup/visualizing-slam-optimization-problems). Feel free to explore it for a more in-depth understanding or if you're interested in some light reading.

## **Our Supervisor**

We had the privilege of working under [Alan Papalia](https://alanpapalia.github.io/), a 4th year grad student in the MIT-WHOI joint program. We would not have been able to come as far as we did in these three short months without Alan's support. He encouraged us daily to jump head first into unfamiliar topics, but was always nearby to work through problems or debug code. Alan has truly been a phenomenal supervisor and we owe all of our progress this summer to him.  

## **The Problem**

### **What is SLAM?**

Simultaneous Localization and Mapping (SLAM) is a problem in robotics, where an autonomous vehicle must create a map of an unknown environment while also keeping track of its location in the environment. 

### **SLAM Optimization**

Within the field of SLAM, we are concerned with the optimization problem: reducing the error between pose estimates (where the robot thinks it is/what it thinks the environment looks like) and the ground truth (where the robot actually is/what the environment actually looks like). However, this type of optimization is not simple. Since most, if not all, SLAM optimization problems are non-convex, the minimum that an optimization algorithm reaches is not guaranteed to be the global minimum. In other words, the solution may be a mathematical minimum, but not necessarily the most optimal solution. Additionally, since SLAM optimization problems are high-dimensional, they are extremely difficult to visualize. This difficulty makes it nearly impossible to distinguish the global minimum from other local minima since you cannot see the full picture.

This is where our work comes in. This summer we began preliminarly work on the visualization of non-convex SLAM optimization problems. Such a tool would aid SLAM roboticists in optimizing highly-dimensional, non-convex problems.

## **Preliminary Research**

### **Neural Net Optimization**

We began our research by exploring current techniques for visualizing optimization problems in neural networks. 

[Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/pdf/1712.09913.pdf) by Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, and Tom Goldstein provided insight into potential visualization softwares, such as matplotlib, yet most mathematical techniques were not applicable to SLAM, as aspects of network architecture greatly impact the convexity of neural net optimization problems. For instance, the use of skip connections creates smoother and more generalizable loss landscapes. However, since there is nothing in SLAM that can achieve the same simplifications, the usefulness of this paper is limited. It did, however, provide more insight into the difficulty of visualizing SLAM optimization problems.

The second paper we looked at was [Qualitatively Characterizing Neural Network Optimization Problems](https://arxiv.org/pdf/1412.6544.pdf) by Ian J. Goodfellow, Oriol Vinyals, and Andrew M. Saxe. While this paper also contained many neural-net-specific approaches, we did find a particularly useful/interesting that parameterizes the objective function in terms of $\alpha (t)$, the projection, and $\beta (t)$, the residual.

Before defining $\alpha (t)$ and $\beta (t)$, however, we must first define $\theta (t)$, $\theta _i$, and $\theta _f$. $\theta (t)$ is the stochastic gradient descent (SGD) trajectory at a time ${t}$, $\theta _i$ is the initialization point, and $\theta _f$ is the solution point.

Next, we can define unit vectors ${v} (t)$ and ${u}$.

$$
{v} (t) = \theta (t) - (\theta _i + \alpha (t){u})\\
{u} = \theta _f - \theta _i
$$

Now we are ready to define $\alpha (t)$ and $\beta (t)$.

$$
\alpha (t) = (\theta (t) - \theta _{i} ) ^T u\\
\beta (t) = (\theta (t) - \theta _{i} - \alpha (t) {u}) ^T v
$$

It is also important to note that $\alpha$ and $\beta$ are scalars that are given direction by unit vectors ${u}$ and ${v}$. 

To fully understand how this parameterization works, we first had to get comfortable with some topics in linear algebra.

### **Getting Familiar with Linear Algebra**

The level of traditional linear algebra needed to understand the above equations is low (thankfully!), so we got started with 3Blue1Brown's ["Essence of linear algebra"](https://www.youtube.com/watch?v=fNk_zzaMoSs&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) series on YouTube. It focuses on the mathematical meanings of things like vectors, matrices, and determinants, but doesn't go heavily into the computations. [ChatGPT](https://chat.openai.com) can also be super helpful when learning new topics, including linear algebra. 

### **Next Steps**

Once we were able to visualize the components (see diagram below), we decided to apply this math to SLAM optimization problems. Just as in neural networks, we can employ $\alpha(t)$ and $\beta(t)$ to parameterize our objective function, enabling us to visualize the loss landscape within a lower-dimensional space.

<div style = "text-align: center;">
    <img src="goodfellow-drawing.png" alt="Goodfellow Math Diagram" width="300">
</div>

## **Learning the Tools**

### **Python**

In order to code up the math from the Goodfellow paper, we first needed to learn Python. We both previously had exposure to C++ and are well-versed in Java, so learning Python was not overly difficult. The [MIT OCW Introduction To Computer Science and Programming in Python](https://ocw.mit.edu/courses/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/pages/syllabus/) was helpful, and we definitely recommend if you are also interested in learning Python. Apart from that, we played around with the assignments for the course and worked on small personal projects (Chess, TicTacToe, etc.) until we were comfortable with syntax, structure, and conventions in Python.

For this project we used [VSCode](https://code.visualstudio.com/) as our editor and [Anaconda](https://www.anaconda.com/) to set up our environment. Our repo on GitHub can be found [here](https://github.com/MarineRoboticsGroup/visualizing-slam-optimization-problems). We used Python version 3.10.12.

### **Matplotlib**

The next tool we needed was a visualization library. We chose to use [Matplotlib](https://matplotlib.org/) for its simplicity, popularity, and integration with [NumPy](https://numpy.org/). We didn't do much to get familiar with either library, as it is easy to learn as you go. 

We did however, make a fun bar graph shown below!

<div style = "text-align: center";>
    <img src="popular-prog-lang.png" alt="Bar Graph of Popular Programming Languages by Number of Developers" width = 400>
</div>


Either of the terminal commands below work to install Matplotlib.

```python
conda install matplotlib
```

```python
pip install matplotlib
```

Same goes for the commands to install NumPy.

```python
conda install numpy
```

```python
pip install numpy
```

## **SciPy Optimization**

Our next step was to begin visualizing simple optimization problems using [SciPy](https://scipy.org/). Either of the terminal commands below should work for installation.

```python
conda install scipy
```

```python
pip install scipy
```

### **2D Convex**

The first basic problem we practiced with was the optimization of a 2D parabola. This was a good first problem, since parabolas are convex and the global minimum is constant at the vertex.

The code is uninteresting, so we will just show the result below.

<div style="text-align: center;">
    <img src="2D-convex.png" alt="Graph of 2D Parabola with minimum" width = 400>
</div>

It should be no surprise that the red dot at the vertex of the parabola is also the minimum reached by [scipy.optimize.minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html).

### **3D Non-Convex**

To ramp up the difficulty, we moved on to visualize a non-convex function with an optimization trajectory in three dimensions. The objective function is defined below.

```python
def non_convex(x):
    return (x[0]**2 + x[1]**2) * np.cos(x[0])
```

The results are shown below. It is important to note that the red line is representative of the optimization trajectory.

<div style = "text-align: center";>
    <img src="3D-non-convex.png" alt="Non-Convex 3D plot with optimization trajectory" width = 400>
</div>

## **Vector Space vs Manifolds**

At this point, we had acquired all of the necessary tools to be able to put our ideas into code. However, we still had one major issue: The Goodfellow math (mentioned above), which we were planning to replicate in SLAM optimization problems, strictly applies to vector spaces, while SLAM optimization happens on manifolds. In order to implement the Goodfellow math, we needed a way to translate math from a manifold to math in a tangent vector space. 

### **So what is a Manifold?**
A manifold is a topological, multi-dimensional space where every point has a neighborhood that's homeomorphic to an open subset of Euclidean space. This means that while the space in its entirety might have a complex shape or curvature, focusing on a small section of it can make it appear flat at that point. Manifolds can have different levels of smoothness or differentiability, and there exist multiple different subsets, each with their own unique and specific properties.

The subset of manifolds relevant to our problem is a Lie group. A Lie group is a smooth manifold, meaning that they are differentiable across the entirety of their surface. Additionally, distance is preserved when projecting and retracting from vector spaces within a Lie group, and projections and retractions are inverse operations of one another. The relevance of these properties will be explained later.

### **What is a Vector Space?**
For our problem, a vector space is a ${d-1}$-dimensional space where ${d}$ is the dimensionality of the manifold. The vector space consists of vectors and operates in the same way a normal Euclidean space would. This allows us to circumvent mathematical issues and complexities that occur when computing on manifolds.

### **Geodesics**
Another useful term to know is a geodesic. A geodesic is the shortest path between two points along a curved surface or manifold. Geodesics can be thought of as the "straightest" paths on curved surfaces or within non-flat spaces. In our problem, the geodesic is used to show the optimal path for the optimization algorithm. While having a geodesic along the manifold is useful for visualization, we are unable to utilize it for the math in our problem.

## **Math On Manifolds**
### **Quick Acknowledgements**
The math required for the projections to and retractions from the tangent space is the work of [Nicolas Boumal](https://www.nicolasboumal.net/). Specifically his book, [An Introduction to Optimization on Smooth Manifolds](https://www.nicolasboumal.net/#book), was vital to our work. 

### **Tangent Vector Space**
To generate a vector space and work with Euclidean rules, we need to define a vector space tangent to our ${d}$-dimensional manifold. The vector space is defined by the point that begins the optimization trajectory, and the plane is made tangent to the manifold at that point. As stated in the math below, the ${d}$-dimensional tangent space is defined by the point ${x}$. The inner product of ${x}$ and any point ${v}$ that is on the manifold $\xi$ is zero, meaning the points are orthogonal.

$$
Proj_x : \xi \rightarrow T_x \mathcal{M}^{d-1} = \{v \in \xi : <x,v> = 0\}
$$

### **Projections**
To project a point from a ${d}$-dimensional manifold onto a ${d-1}$-dimensional tangent vector space, we needed to remove any similarity that the given point has with the point defining the tangent space. To do so, we take the inner product of the point we want to project, ${u}$, and the point that defines the vector space, ${x}$. We multiply the inner product by ${x}$ to point in the direction of the tangent space, and then finally subtract that from the original point ${u}$. This removes any similarity that ${u}$ shares in the direction of ${x}$, thereby projecting it onto the tangent space. The math describing this process is provided below.

$$
Proj_x : \xi \rightarrow T_x \mathcal{M}^{d-1} : u \mapsto Proj_x(u) = u \text{ }- <x,u>x
$$


### **Retractions**
Retractions are useful for being able to visualize the work done in the tangent space over the manifold, as opposed to keeping it in the tangent space where much of the manifold's complexity can be lost. To retract a point back onto the manifold, we need to decrease the magnitude of the vector defining the projected point. To do so, we divide the sum of the point ${v}$ and the point defining the tangent space ${x}$ by the two-norm of their sum. Doing so decreases the magnitude of the vector, retracting it back to the surface of the manifold. The math describing this is provided below.

$$
R_x(v) = {{x + v} \over \lVert x + v \rVert} = {{x + v} \over \sqrt {{1 + \lVert v \rVert}^2}}
$$

### **Funky Shifting with Spheres**
As we began whiteboarding projections and retractions, we quickly realized two significant issues when working on a sphere. Firstly, multiple points along the surface of the sphere can yield the same projection. For example, if you project two points from a sphere centered at the origin with the same X and Y values but opposite Z values, the points will be projected into the same position. Secondly, distances are not preserved when projected into the vector space. The distance following the geodesic on the manifold will always be greater than the same path in vector space. This is due to the geodesic working in an additional dimension and therefore having to move across additional space.

Fortunately, in Lie groups, both these issues are addressed due to the group's unique properties. In Lie groups, projections and retractions are inverse operations, meaning that each set of projections and retractions are reciprocals of one another. This prevents two points from being projected to the same location or two retractions from being at the same point on a manifold. Additionally, distance is always preserved, so the path along the geodesic is the same length as the path taken in the tangent space.

## **Simple Sphere**

### **Projecting Points**
To project points onto the tangent plane, we utilized the projection equation detailed in Boumal's work. As explained in the sections discussing Projections and Retractions, the process of projecting a point ${u}$ onto the tangent plane involves subtracting the cross product of ${u}$ and the vector defining the tangent plane—in alignment with the tangent plane's direction from ${u}$. This operation eliminates any directional similarities between ${u}$ and the defining vector, projecting the point onto the tangent plane.

```python
projected_point = (point - np.dot(self.tangent_plane_origin_pt, point) * self.tangent_plane_origin_pt)
```  
<p>&nbsp;</p>
However, when plotted, the projected point would appear within the sphere. This occurs because mathematically, the plane runs through the origin of the sphere. To fix this visualization issue and make the point appear to be projected onto the tangent plane, we add to the point the vector that defines the plane. This shifts the point in that direction, making it appear in the vector space.

```python
return projected_point + self.tangent_plane_origin_pt
```

<p>&nbsp;</p>

<div style ="text-align: center;">
<img src="SphereWithTanSpace.png" width="375" height="375" />
</div>

### **Sampling Points**
While it was easy to sample points in three dimensions, we nevertheless decided to sample using $\alpha$ and $\beta$. This approach would make scaling the problem into higher dimensions easier later on. To sample points, we walked along the trajectory and added additional points by shifting up and down the $\alpha$ and $\beta$ vectors that defined the point we were at. This allowed us to sample points not only on the trajectory but also around it, creating a landscape surrounding the trajectory. This gives us a better understanding of the path that the optimizer took and the obstacles it avoided.

```python
sampled_point_on_sphere = (
            self.alpha(index) + (shift * sign_alpha_shift))
            * self.u + (self.beta(index) + (shift * sign_beta_shift)) * self.v(index)
            sampled_point_on_plane = self.project_to_tan_plane(sampled_point_on_sphere)
```

### **Creating an Objective Function**
In order to optimize across a surface, there needs to be an objective function that informs the optimizer about how efficient or inefficient certain paths are. Unfortunately, on our 2-dimensional sphere, there was no inherent cost function. To compensate for this, we devised our own using the Karcher mean. The Karcher mean is defined as follows:

$$
{m} = arg\text{ }min \sum_{i=1}^N d^2 (p, x_i)
$$

* ${m}$ represents the Karcher mean, the point that minimizes the sum of squared distances<br>
* ${p}$ is the reference point.<br>
* ${x_i}$ represents the ${i}$-th point in the set<br>
* ${d^2(p, x_i)}$ represents the squared distance between ${p}$ and ${x_i}$<br>
* The ${arg\text{ }min}$ notation denotes the argument that minimizes the given expression<br>


The implementation of this in the program is below. It takes only one point at a time and squares the distances between the point we are trying to find the cost of and the three points we used to define the artificial optimization trajectory. The sum of these squared distances defines the cost of the function at that point.

```python
temp = (
            (np.linalg.norm(point - pt_origin)) ** 2
            + (np.linalg.norm(point - self.points_on_sphere[1])) ** 2
            + (np.linalg.norm(point - self.points_on_sphere[2])) ** 2
        )
```

### **Plotting the Objective Function**
After failing to visualize the objective function with a surface composed of projected points, we decided to invest some extra effort and visualize it as a surface. We created mesh grids spanning the X and Y coordinates of the sphere, which enabled us to access points and subsequently determine their cost functions. This cost is then assigned to the respective mesh grid containing the Z values for all XY coordinate pairs. These points are then plotted as a single surface, enabling us to visualize the cost as the trajectory moves across the surface of the manifold.

<div style="text-align: center;">
<img src="LossLandscapeMerged.png" width="70%" height="70%" />
</div>

## **Higher-Dimensional Sphere**

### **More Complex Data Sets** 

After successfully plotting loss landscapes over 2-dimensional spheres, our next steps involved increasing the dimensionality of the sphere. This increase in dimensionality brought our work closer to the datasets of SLAM optimization. Additionally, transitioning to a 7-dimensional sphere eliminated our ability to visualize it or the work in the vector space, making a deep understanding of the topics and issues even more important.

### **Pymanopt**

To make it easier to work on higher-dimensional surfaces, we began to implement additional tools, namely [Pymanopt](https://pymanopt.org/). Pymanopt is a tool for optimizing on selected manifolds and solving nonlinear problems. It contains functions that allowed us to simplify a lot of our code or eliminate parts entirely. Additionally, Pymanopt includes optimizers that allow us to truly optimize on a surface instead of creating our own artificial trajectory. It serves as a useful tool to make our work more efficient and bring us closer to what optimizing SLAM problems would be like.

Either of the following terminal commands below work to install Pymanopt.

```python
conda install pymanopt
```

```python
pip install pymanopt
```

### **Convex Objective Function**

The math from [Qualitatively Characterizing Neural Network Optimization Problems](https://arxiv.org/pdf/1412.6544.pdf) by Ian J. Goodfellow, Oriol Vinyals, and Andrew M. Saxe was now necessary to plot the objective function in terms of $\alpha (t)$ and $\beta (t)$. In this initial scenario with the 7-dimensional sphere, the Karcher Mean objective function was entirely convex. It is important to note, however, that even though the objective function is convex in this case, we are still working with a non-convex problem due to the geodesic convexity of the manifold.

It is important to define geodesic convexity and explain how it differs from traditional Euclidean convexity. Geodesic convexity relates to the notion of convexity on Riemannian manifolds. Due to the curvature of these manifolds, convexity cannot be defined in the same way as it is in Euclidean space. A set $\mathcal{M}$ on a Riemannian manifold is geodesically convex if the geodesic segment connecting any pair of points ${x}$ and ${y}$ in $\mathcal{M}$ lies entirely in $\mathcal{M}$. This concept is almost identical to how convexity is defined in Euclidean spaces, except it allows for the curvature of manifolds.

The graphical results are shown below.

<div style="text-align: center;">
<img src="convex-obj.png" alt="Convex Objective Plot" width = 400>
</div>

Since the objective function is convex, the optimization trajectory strictly follows the gradients of the loss function to the global minimum. This results in no variation of $\beta (t)$ values and a clearly uninteresting graph. Despite this still being a non-convex problem, the graph above is not an indicator of that due to the geodesic convexity of the manifold. 

### **Non-Convex Objective Function**

Our next step was to introduce non-convexities to the objective function to create a graph that is more representative of the non-convex nature of SLAM. After introducing non-convexities to the objective function, the optimization trajectory now has to navigate around obstacles, such as saddle points and local minima. The code that introduced non-convexities is shown below.

```python
    # generates a random matrix to add to karcher mean to try to make it non-convex
    mat_dim = self.manifold.random_point().shape[0]
    rand_mat = np.random.rand(mat_dim, mat_dim)
    rand_psd_mat = rand_mat @ rand_mat.T

    nonconvex_cost = (x.T @ rand_psd_mat @ x)
```

```nonconvex_cost``` is then added to the Karcher mean at a given point to introduce the aforementioned non-convexities. The graphical results are shown below.

<div style="text-align: center;">
<img src="non-convex-obj.png" alt="Non-Convex Objective Plot" width = 400>
</div>
 
It is easy to see that once non-convexities are introduced to an objective function, both optimization and visualization become much more challenging. This reinforces the difficulty of our work this summer and highlights the importance of further developing a visualization tool for SLAM roboticists.

## **Conclusion**
As we said before, none of this would have been possible without the support of Alan and the rest of Marine Robotics Group. Despite being high schoolers, we always felt like equals in the lab. For that, we are immensely grateful. If you take anything at all away from this, let it be that the most amazing things happen when you are out of your comfort zone; so, try something new!

