## AutoIntCL算法的线上效果分析

在之前我们已开源了[AutoIntCL算法](https://github.com/hmliangliang/AutoIntCL)、[GNNCL算法](https://github.com/hmliangliang/GNNCL)、[基于faiss的向量检索算法](https://github.com/hmliangliang/faissSearch)，相关的算法代码可以通过链接来访问。

我们将这些算法部署在某项目的线上推荐服务中，并统计了近20多天的线上效果(2022.07.01--2022.07.25)，统计数据如下表所示：

表1  算法的点击通过率

| 算法 | AutoIntCL | GNNCL | faiss |battleclubv2 |
| ------- | ------- | ------- | ------- |------- |
|    点击通过率(%)     |   71.5178      |     62.0155    |    60.1324     |  56.4960      |


表2  算法的曝光成功率

| 算法 | AutoIntCL | GNNCL | faiss |battleclubv2 |
| ------- | ------- | ------- | ------- |------- |
|    曝光成功率(%)     |  2.7439       |      2.5544   |    2.5058     |   2.4217     |

综合线上效果来看，AutoIntCL> GNNCL> faiss> battleclubv2。在点击通过率上，AutoIntCL提升幅度最大，相对于battleclubv2提升了26.59%，GNNCL相对于battleclubv2提升了9.7697%，faiss相对于battleclubv2提升了6.4472%。在曝光成功率上，仍然是AutoIntCL提升幅度最大，相对于battleclubv2提升了13.3074%，GNNCL相对于battleclubv2提升了5.4821 %，faiss相对于battleclubv2提升了3.4748%。线上效果证明了AutoIntCL算法与GNNCL算法的有效性。

## 推荐算法效果汇总(2022.08.03-2022.08.07)
表1 已上线推荐算法(2022.08.03-2022.08.07)整体效果汇总(pair)
| 指标 |曝光总次数 |点击次数 |点击通过次数 |点击率 |点击通过率 |曝光成功率|
| :-----:| :----: | :----: | :----: | :----: | :----: | :----: |
|数据 |1417470|66729|44508|4.7076%|66.6996%|3.1400%|

表2 已上线推荐算法(2022.08.03-2022.08.07)整体效果汇总(pair)
| 指标 |曝光总人数 |点击人数 |点击通过人数 |点击率 |点击通过率 |曝光成功率|
| :-----:|: ----: | :----: | :----: | :----: | :----: | :----: |
|数据 |621305|56247|44226|9.0530%|78.6282%|7.1182%|

表3 randomv2推荐算法(2022.08.03-2022.08.07)的整体曝光平均通过人数
| 算法|曝光人数|曝光通过人数|平均曝光通过人数(曝光通过人数/曝光人数)|
| :-----:| :----: | :----: | :----: | 
randomv2|31508|2070|0.0657|
算法增量=点击通过人数-random_v2平均曝光通过人数*曝光总人数=44226-0.0657*621305=44226 -408193406
算法相对增长率=算法增量/点击通过人数*100% =3406/44226*100%=7.7014%

表4  各个算法在2022.08.03-2022.08.07期间的曝光成功率均值(pair)
| 算法| 曝光成功率(%)| 算法流量
| :-----:| :----: | :----: | 
| AutoIntCL| 2.9271| 5%->15%(2022.08.04后调整)| 
| battleclubv2| 2.5927| 5%| 
| battleclub-v2-autoint-v1| 2.7305| 5%| 
| battleclub_v2_autoint_v3| 2.3040| 10%->2022.08.04已下线| 
| chieffriendclub_battleclub_v2_autoint_v1| 2.6185| 5%| 
| faiss|2.7781| 5%| 
| GNNCLV2| 2.7049| 5%| 
| mulretrieval_v1_autoint_v1| 2.6471| 10%| 
| mulretrieval_v1_autoint_v3| 2.6488| 20%| 
| mulretrieval_v1_autoint_v4| 2.6453| 25%| 
| randomv2| 2.4043| 5%| 


表5  各个算法在2022.08.03-2022.08.07期间的曝光成功率均值(人数)
| 算法| 曝光成功率(%)| 算法流量|
| :-----:| :----: | :----: | 
|AutoIntCL|6.6401|5%->15%(2022.08.04后调整)|
||battleclubv2|6.0702|5%|
|battleclub-v2-autoint-v1|6.2217|5%|
|battleclub_v2_autoint_v3|4.2119|10%->2022.08.04已下线|
|chieffriendclub_battleclub_v2_autoint_v1|6.4038|5%|
|faiss|6.3700|5%|
|GNNCLV2|6.4090|5%|
|mulretrieval_v1_autoint_v1	6.5530|10%|
|mulretrieval_v1_autoint_v3	6.5816|20%|
|mulretrieval_v1_autoint_v4	6.6074|25%|
|randomv2|5.9628|5%|

表4与表5展示了各个已上线算法在2022.08.03-2022.08.07期间的曝光成功率均值，由算法的结果可以看出，无论是按pair统计口径上，还是按人数统计口径上，AutoIntCL算法效果最佳，相对于randomv2提升了21.74%(pair)与11.3%(人数)，结合battleclubv2算法可以看出，使用AutoIntCL算法(使用battleclubv2的结果作为候选集)对battleclubv2的结果进行精排，的确提升了算法的效果。同时结合AutoIntCL算法与battleclub-v2-autoint-v1算法可以看出，两者都是使用了autoint来融合特征，AutoIntCL算法比battleclub-v2-autoint-v1算法多了对比学习与多模态特征学习过程，AutoIntCL算法相对于battleclub-v2-autoint-v1算法提升了7.2001%(pair)与6.7249%(人数)，由此可见，引入对比学习与多模态特征学习过程，的确能提升算法的效果。

根据上周的算法效果，将调整算法流量，保留AutoIntCL、mulretrieval_v1_autoint_v3、mulretrieval_v1_autoint_v4与randomv2四个算法，其余的算法全部下线。其中新算法的流量切分如下：AutoIntCL：35%，mulretrieval_v1_autoint_v3：30%，mulretrieval_v1_autoint_v4：30%，randomv2：5%
