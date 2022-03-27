# EdgeDetection
根据文章VE-PCN的补充材料

- We obtain the ground truth edges Pe with [1]. Specifically, object edges are identified by evaluating the query
point p from its k-nearest neighbors. We first find the k-nearest neighbors of each query point from the object and
denote c as the center of these neighbors. We then calculate the minimum distance v among all the neighboring points
to the query point. A query point p is classified as the edge point if ||c −p|| > λ ·v. We set λ = 5 and k = 100 for
our created dataset, and set λ = 1.8 and k = 150 for the Completion3D dataset.

   [1]Syeda Mariam Ahmed, Yan Zhi Tan, Chee Meng Chew, Ab-dullah Al Mamun, and Fook Seng Wong. Edge and corner
detection for unorganized 3d point clouds with application to robotic welding. In 2018 IEEE/RSJ International Confer-
ence on Intelligent Robots and Systems (IROS), pages 7350–7355. IEEE, 2018

判断点p是否属于edge的计算流程
1. 寻找p的k个邻近点，计算k个邻近点的中心点c
2. 计算k个邻近点中离p最近的点的距离
3. 如果|c-p|>λ·v，那么p就是edge

按照上述代码，在mvp数据集，2048点的complete点云上的结果

![chair](https://github.com/alfredtorres/EdgeDetection/blob/main/chair.png)
