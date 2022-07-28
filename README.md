## AutoIntCL算法的线上效果分析

在之前我们已开源了[AutoIntCL算法](https://github.com/hmliangliang/AutoIntCL)、[GNNCL算法](https://github.com/hmliangliang/GNNCL)、[基于faiss的向量检索算法](https://github.com/hmliangliang/faissSearch)，相关的算法代码可以通过链接来访问。

我们将这些算法部署在某项目的线上服务中，并统计了近20多天的线上效果(2022.07.01--2022.07.25)，数据如下表所示：

表1  算法的点击通过率

| 算法 | AutoIntCL | GNNCL | faiss |battleclubv2 |
| ------- | ------- | ------- | ------- |------- |
|    点击通过率(%)     |   71.5178      |     62.0155    |    60.1324     |  56.4960      |


表2  算法的曝光成功率

| 算法 | AutoIntCL | GNNCL | faiss |battleclubv2 |
| ------- | ------- | ------- | ------- |------- |
|    曝光成功率(%)     |  2.7439       |      2.5544   |    2.5058     |   2.4217     |

综合线上效果来看，AutoIntCL> GNNCL> faiss> battleclubv2。在点击通过率上，AutoIntCL提升幅度最大，相对于battleclubv2提升了26.59%，GNNCL相对于battleclubv2提升了9.7697%，faiss相对于battleclubv2提升了6.4472%。在曝光成功率上，仍然是AutoIntCL提升幅度最大，相对于battleclubv2提升了13.3074%，GNNCL相对于battleclubv2提升了5.4821 %，faiss相对于battleclubv2提升了3.4748%。线上效果证明了AutoIntCL算法与GNNCL算法的有效性。
