# FlexER: Flexible Entity Resolution for Multiple Intents
Implementation of our paper "FlexER: Flexible Entity Resolution for Multiple Intents" (currently under review for [VLDB22](https://vldb.org/2022/)).


## Abstract
Entity resolution (ER), a longstanding problem of data cleaning and integration, aims at identifying different data records that represent the same real-world entity. Existing approaches treat ER as a universal task, focusing only on finding perfectly matched records and separating the corresponding from non-corresponding ones. However, in real-world scenarios, where ER is part of a more general data project, downstream applications may not only require resolution of records that refer to the same entity but may also seek to match records that share different levels of commonality, relating, for example, to various granularity levels of the resolution. In what follows, we introduce the problem of multiple intents entity resolution (MIER), an extension to the universal (single intent) ER task. As a solution, we propose FlexER, utilizing contemporary solutions to universal ER tasks to solve multiple intents entity resolution. FlexER addresses the problem as multi-label classification and combines intent-based representations of record pairs using a graph convolutional network (GCN) to improve the outcome to multiple resolution problems.

In this work, we use [DITTO](https://github.com/megagonlabs/ditto), the state-of-the-art in universal ER. Specifically, we use the final embedding of the special token [cls](used for classification) as a record pair representation, and inject it into the FlexER system.

![mier_system](/images/mier_system_small.jpg)

## MIER: Multiple Intents Entity Resolution
We offer to extend the universal view of ER, pointedly reflected in the use of a single mapping from records to real-world entities, to include multiple intents.
To better understand the meaning of multiple intents, note that the universal view of ER implicitly assumes a single entity set by which the ER solution must abide.
Such an entity set is not explicitly known, yet typically referred to abstractly as a “real-world entity." We argue that an entity set of choice may vary according to user needs and different users may seek different solutions from the same dataset.

A one-size-fits-all resolution provides an adequate solution for universal ER, a standalone task with a single equivalence intent.
Yet, some data cleaning/integration challenges may involve multiple intents. Therefore, instead of performing a universal ER, we argue for enhancing ER to support multiple outcomes for multiple intents. 
A MIER involves a set of (possibly related) entity mappings for a set of intents E = {𝐸1, 𝐸2, · · · , 𝐸𝑃 }, offering multiple ways to divide 𝐷, each serving as a solution for a
respective intent.

For further details and official definitions, please refer to our paper (currently under review).
