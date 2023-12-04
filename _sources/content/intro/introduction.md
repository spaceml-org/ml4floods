```{image} ml4floods_banner.png
:name: banner
:align: center
```

<p style="text-align: center; font-style: italic;">ML4Floods: an ecosystem of data, models and code pipelines to tackle flooding with ML</p>

# Project rationale

## Introduction

Satellites image the whole globe, revisiting locations every few days or even hours to record the changing landscape. Their cameras see deep into the infrared and radar sensors see through clouds and at night. They capture information on many different observable parameters including dynamic and rapid processes and events like floods. Newer instruments can now provide contextual details at more and more precise scale, locating for instance individual buildings that are flooded, or even search and rescue people using airborne platforms. At the same time, ground-based sensors like rain gauges or river level stations record surface conditions unavailable from space. All of this data is stored - sometimes by commercial companies, but often by government agencies under open- or restricted-access policies. However, there remain significant barriers for non-specialists to access and use this information or to respond quickly in emergency situations.

```{image} flood_context.png
:alt: Floods devestate the world every year.
:width: 500px
:align: center
```

For flood disaster management, the real promise of this data is in translating it to actionable intelligence, directly answering queries like “Which areas will be flooded if rain continues to fall and rivers swell up in the next days?” or “Show me a map of the water level in streams wider than 1 meter.”


## Democratising AI-Enhanced Flooding Tools
Machine learning (ML) algorithms have the potential to offer significantly faster and more accurate flood mapping than traditional methods. Their adaptability means they can easily grow to accommodate more data over time, and expand to ingest a wide range of data types. Such 'data fusion' capabilities are potentially game-changing for a broad spectrum of users, from emergency workers to urban planners.


Training robust and reliable ML models is almost an art-form, requiring specialist knowledge of statistics, computing and data platforms. ML workflows have become much more accessible because of dedicated open-source libraries like PyTorch and TensorFlow. However, there are a myriad of subtle pitfalls associated with training and deploying ML models - these can produce deeply skewed results that still appear reasonable to the untrained eye. Democratising end-to-end integrated AI workflows avoids these pitfalls by creating a series of linked tools that non-ML expert users can trust to deploy machine learning. These tools incorporate data acquisition, preparation, calibration, enhancement and deployment steps, wrapped in an accessible interface. The full workflow is embedded in a visualisation tool that enables users to query, understand and use the insights and information generated. This unified suite of tools can get users 80% of the way and then allows straightforward customisation to a specific problem sub-domain.


```{figure} ../ml4ops/ts_albania.gif
---
name: ts_albania
width: 100%
align: center
---
Flood extent segmentation over a time series of satellite images from Albania. [Made with this tutorial](../ml4ops/HOWTO_inference_on_image_time_series.ipynb).
```


## MLOps tools: from data preparation to deployment


For machine learning practitioners, the operational tools supporting model development, from data-preparation to deployment, are still in infancy. Such ‘MLOps' tools are the equivalent of ‘DevOps' tools in software development. Analysing Earth-observation (EO) data also requires highly specialist knowledge, which doubly-complicates the workflows, and it is rare to find individual data scientists who are masters of both the ML and EO domains.


In this project, the four ML4CC teams have built a dedicated MLOps toolkit that makes applying machine-learning to satellite data easy, for the purpose of mapping flood extent. The toolkit is carefully designed to cater for different types of beneficiaries with a range of needs:


- **Flood scientists** - who want to understand and use actual earth observation data
- **Flood response teams** - who want rapid just-in-time access to a flood map to support emergency relief efforts.
- **Risk analysts** -  who want to analyse past flood events, or collate and visualise the data from a time-range.
- **ML-researchers** - who are investigating how to make a better ML model, or dataset.

## Understanding and using the tools

The figure below presents an overview of the ML4Floods toolkit alongside the users of each component (click for a larger version). The toolkit is structured as an end-to-end pipeline with components that 1) ingest, sort and organise satellite data, integrating ground-truth masks, 2) tile, augment and normalise the data, 3) train new models on the data, or run existing models on new data, and display uncertainty maps generated by the models, and 4) query and visualise the results via a web-based mapping application.

```{image} ml4cc_diagram_export.png
:alt: Diagram of the ML4CC pipeline
:width: 1000px
:align: center
```

Each of the components 1-3 can be accessed via an application programming interface (API) so that technical users can fine-tune their workflows, or adopt the components in their own tools. However, the teams also developed a graphical interface (4) that can run the toolkit through a simple point-and-click interface. This last component places the power of ML-enhanced flood segmentation models in the hands of ordinary users, like disaster relief coordinators and urban planners. Finally, the graphical tool is incredibly useful for machine-learning researchers, allowing them to quickly compare and contrast model results on the same interface, greatly speeding up the model development process.