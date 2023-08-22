# **Marine Robotics Group Documentation**

## **Introduction**

Our names are Abigail Greenough and Calvin Cigna and the summer before our senior year we interned with the [Marine Robotics Group] (https://marinerobotics.mit.edu/). 

This page is meant to serve as a summary of our work this summer as well as a jumping off point for future researchers.

## **Our Supervisor**

We had the privilege of working under [Alan Papalia] (https://alanpapalia.github.io/), a 4th year grad student in the MIT-WHOI joint program. We would not have been able to come as far as we did in these three short months without Alan's support. He encouraged us daily to jump head first into unfamiliar topics, but was always nearby to work through problems or debug code. Alan has truly been a phenomenal supervisor and we owe all of our progress this summer to him.  

## **The Problem**

### **SLAM Robotics**

Simultaneous Localization and Mapping (SLAM) is a problem in robotics where an autonomous vehicle must create a map of an unknown environment while also keeping track of their location in the environment. 

### **SLAM Optimization**

Within the field of SLAM, we are concerned with the optimization problem: reducing the error between pose estimates (where the robot thinks it is/what it thinks the environment looks like) and the ground truth. However, this optimization is not simple. Since most, if not all, SLAM optimization problems are non-convex, a minimum reached by an optimization algorithm is not guaranteed to be the global minimum. Additionally, since SLAM optimization problems are highly-dimensional, they are extremely difficult to visualize. It therefore makes it nearly impossible to tell apart the global minimum from other local minima since you cannot see the full picture. 

This is where our work comes in. This summer we began preliminarly work on the visualization of non-convex SLAM optimization problems. 


## **Preliminary Research**

### **Getting Familiar with Linear Algebra**

### **Neural Net Optimization**

## **Goodfellow Math**

## **Learning Python**

## **Matplotlib**

## **Working on Manifolds**

### **What is a manifold?**

### **Geodesics**

### **Projections & Retractions**

### **Tangent Vector Space**

## **Simple Sphere**

### **Tangent Plane**

### **Projecting Points**

### **Sampling Points**

### **Plotting Objective Function**

### **Results**

## **Higher-Dimensional Sphere**

### **Pymanopt**

## **Conclusion**

As we said before, none of this would have been possible with the support of Alan and the rest of Marine Robotics. Despite being high schoolers, we always felt like equals in the lab. For that, we are immensely grateful. If you take anything at all away from this post, let it be that the most amazing things happen when you are out of your comfort zone; so, try something new!
