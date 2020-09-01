# python-Robust_Elastic_Warping
Simple implementation of Robust Elastic Warping using opencv-python

*Parallax-Tolerant Image Stitching Based on Robust Elastic Warping* 这篇论文的Python实现，
仅实现了两幅拼接，需要的库：
```python
opencv-python==4.4.0.42
numpy
scipy
```

运行结果：
<table>
    <tr>
        <td ><center><img src="https://img-blog.csdnimg.cn/20200901212458335.png?x-oss-process" >mosaic_global</center></td>
        <td ><center><img src="https://img-blog.csdnimg.cn/20200901212738884.png?x-oss-process"  >mosaic_REW</center></td>
    </tr>
    <tr>
        <td ><center><img src="https://img-blog.csdnimg.cn/2020090121314848.jpg?x-oss-process" >mosaic_global</center></td>
        <td ><center><img src="https://img-blog.csdnimg.cn/20200901213147738.jpg?x-oss-process"  >mosaic_REW</center></td>
    </tr>
    <tr>
        <td ><center><img src="https://img-blog.csdnimg.cn/202009012129253.png?x-oss-process" >mosaic_global</center></td>
        <td ><center><img src="https://img-blog.csdnimg.cn/20200901212955734.png?x-oss-process"  >mosaic_REW</center></td>
    </tr>
    <tr>
        <td ><center><img src="https://img-blog.csdnimg.cn/20200901213324652.jpg?x-oss-process" >mosaic_global</center></td>
        <td ><center><img src="https://img-blog.csdnimg.cn/20200901213325812.jpg?x-oss-process"  >mosaic_REW</center></td>
    </tr>
</table>
