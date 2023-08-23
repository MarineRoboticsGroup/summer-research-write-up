# **Marine Robotics Group Documentation**

## **Introduction**

Our names are Abigail Greenough and Calvin Cigna and the summer before our senior year we interned with the [Marine Robotics Group](https://marinerobotics.mit.edu/). 

This page is intended to function as a summary of our accomplishments over the course of this summer, while also serving as a springboard for future researchers. As you peruse the content, you might find it beneficial to consult our repository on [GitHub](https://github.com/MarineRoboticsGroup/visualizing-slam-optimization-problems). Feel free to explore it for a more in-depth understanding or if you're seeking some light reading.

## **Our Supervisor**

We had the privilege of working under [Alan Papalia](https://alanpapalia.github.io/), a 4th year grad student in the MIT-WHOI joint program. We would not have been able to come as far as we did in these three short months without Alan's support. He encouraged us daily to jump head first into unfamiliar topics, but was always nearby to work through problems or debug code. Alan has truly been a phenomenal supervisor and we owe all of our progress this summer to him.  

## **The Problem**

### **SLAM Robotics**

Simultaneous Localization and Mapping (SLAM) is a problem in robotics where an autonomous vehicle must create a map of an unknown environment while also keeping track of their location in the environment. 

### **SLAM Optimization**

Within the field of SLAM, we are concerned with the optimization problem: reducing the error between pose estimates (where the robot thinks it is/what it thinks the environment looks like) and the ground truth (where the robot actually is/what the environment actually looks like). However, this type of optimization is not simple. Since most, if not all, SLAM optimization problems are non-convex, a minimum that an optimization algorithm reaches is not guaranteed to be the global minimum. In other words, the solution may be a mathematical minimum, but not necessarily the most optimal solution. Additionally, since SLAM optimization problems are highly-dimensional, they are extremely difficult to visualize. It therefore makes it nearly impossible to tell apart the global minimum from other local minima since you cannot see the full picture. 

This is where our work comes in. This summer we began preliminarly work on the visualization of non-convex SLAM optimization problems. Such a tool would aid SLAM roboticists in their optimization of highly-dimensional, non-convex problems.


## **Preliminary Research**

### **Neural Net Optimization**

We began our research by exploring current techniques for visualizing optimization problems in neural nets. 

[Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/pdf/1712.09913.pdf) by Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, and Tom Goldstein provided insight into potential visualization softwares, such as matplotlib, yet most mathematical techniques were not applicable to SLAM, as aspects of network architecture greatly impact the convexity of neural net optimization problems. For instance, the use of skip connections creates smoother and more generalizable loss landscapes. However, since there is nothing in SLAM that can achieve the same simplifications, the usefulness of this paper is limited. It did, however, provide more insight into the difficulty of visualizing SLAM optimization problems.

The second paper we looked at was [Qualitatively Characterizing Neural Network Optimization Problems](https://arxiv.org/pdf/1412.6544.pdf) by Ian J. Goodfellow, Oriol Vinyals, and Andrew M. Saxe. While this paper also contained many neural-net-specific approaches, we did find a particularly useful/interesting that parameterizes the objective function in terms of $\alpha (t)$, the projection, and $\beta (t)$, the residual.

Before defining $\alpha (t)$ and $\beta (t)$, however, we must first define $\theta (t)$, $\theta _i$, and $\theta _f$. $\theta (t)$ is the stochastic gradient descent (SGD) trajectory at a time ${t}$, $\theta _i$ is the initialization point, and $\theta _f$ is the solution point.

Next, we can define vectors ${v} (t)$ and ${u}$.

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

The level of traditional linear algebra needed to understand the above equations is low (thankfully!), so we got started with 3Blue1Brown's ["Essence of linear algebra"](https://www.youtube.com/watch?v=fNk_zzaMoSs&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) series on YouTube. It focuses on the mathematical meanings of things like vectors, matrices, and determinants, but doesn't go heavily into the computations. [ChatGPT](https://chat.openai.com) can also be super helpful when learning new topics, like linear algebra. 

### **Next Steps**

Once we were able to visualize the above components (see diagram below), we decided to apply this math to SLAM optimization problems. Similarly to how it worked in neural nets, we can use $\alpha (t)$ and $\beta (t)$ to parameterize our objective function in order to visualize the loss landscape in a lower-dimensional space. 

<div style = "text-align: center;">
    <img src="goodfellow-drawing.png" alt="Goodfellow Math Diagram" width="300">
</div>

## **Learning the Tools**

### **Python**

To be able to code up the math from the Goodfellow paper, we first needed to learn Python. We both previously had exposure to C++ and are well-versed in Java, so learning Python was not overly difficult. The [MIT OCW Introduction To Computer Science and Programming in Python](https://ocw.mit.edu/courses/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/pages/syllabus/) was helpful and we definitely recommend if you are also interested in learning Python. Other than that, we played around with the assignments for the course and worked on small personal projects (Chess, TicTacToe, etc) until we were comfortable with syntax, structure, and conventions in Python.

For this project we used [VSCode](https://code.visualstudio.com/) as our editor and [Anaconda](https://www.anaconda.com/) to set up our environment. Our repo on GitHub can be found [here](https://github.com/MarineRoboticsGroup/visualizing-slam-optimization-problems). We used version 3.10.12 of Python.

### **Matplotlib**

The next tool we needed was a visualization library. We chose to use [Matplotlib](https://matplotlib.org/) for its simplicity and popularity as well as its integration with [NumPy](https://numpy.org/). We didn't do terribly much to get familiar with either library, as it is easy to learn what you need as you go. 

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

The first basic problem we practiced with was the optimization of a 2D parabola. This was a good first problem since parabolas are convex and the global minimum is always the same as the vertex.

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

## **Working on Manifolds**

### **SLAM Manifolds**

### **What is a manifold?**

### **Geodesics**

### **Math On Manifolds**

### **Tangent Vector Space**

### **Projections & Retractions**


## **Simple Sphere**

### **Tangent Plane**

### **Projecting Points**
<div style ="text-align: center;">
<img src="SphereWithTanSpace.png" width="375" height="375" />
</div>

### **Sampling Points**

### **Plotting Objective Function**
After failing to visulize the objective function with a surface composed of projected points, we decided to put in some extra work and visulize it as a surface. We made a meshgrids spanning the X and Y coordinates of the sphere, allowing us to access points and then find their cost functions. That cost is then assigned to the respective meshgrid containing the Z values for all of the XY coordinate pairs. Those points are then all plotted as one surface allowing us to visulize the cost as the trajectory moves accross the surface of the manifold.

<div style="text-align: center;">
<img src="LossLandscapeMerged.png" width="100%" height="100%" />
</div>

### **Results**


## **Higher-Dimensional Sphere**

### **More Complex Data Sets** 
After successfully plotting loss landscapes over 2-dimensional spheres, our next endeavor involved elevating the dimensionality of the sphere. This escalation in dimensionality aimed to amplify the complexity of the problem, rendering it more akin to that of SLAM datasets. Moreover, transitioning to 3 dimensions eliminated our capacity to visually represent the sphere under consideration, thereby necessitating a robust grasp of the mathematical and logical aspects of the problem.

### **Pymanopt**
To make it easier to work on higher-dimensional surfaces, we began to implement additional tools, namely  [Pymanopt](https://pymanopt.org/). Pymanopt is a tool for optimizing on selected manifolds and solving nonlinear problems. It contains functions that allowed us to simplify lots of our code or eliminate parts entirely. Additionally, Pymanopt includes optimizers that allow us to truly optimize on a surface instead of creating our own artificial trajectory. It serves as a useful tool to make our work more efficient and bring us closer to what optimizing SLAM problems would be like.
 
## **Conclusion**

As we said before, none of this would have been possible with the support of Alan and the rest of Marine Robotics. Despite being high schoolers, we always felt like equals in the lab. For that, we are immensely grateful. If you take anything at all away from this post, let it be that the most amazing things happen when you are out of your comfort zone; so, try something new!
