## Project Title
Multi-Modal Analysis of Customer Support Tickets Using Text Analytics and YOLOv8n

##Overview

Customer support teams face thousands of incoming tickets acoss multiple product types, channels, and issue categories. Identifying recurring pain points, and issue categories. Identifying recurring pain points, tracking dissatisfaction drivers, and accurately routing issues remains challenging. The Customer Support Ticket Dataset provides structured ticket information such as product purchased, ticket type, priority, channel, customer satisfaction rating, and free-text ticket descriptions. 
This project applies text analytics, sentiment modeling, topic extraction, and predictive modeling to identify key trends in customer issues. A YOLOv8n image analytics component is added to explore how product images could help automate ticket routing and improve support workflows.

##Dataset
We use the Customer Support Ticket Dataset from Kaggle, which includes:
* Ticket Description (text of the issue)
* Ticket Type
* Product Purchased
* Ticket Priority
* Ticket Channel
* Time to Resolution
* Customer Satisfaction Rating
The dataset is relatively clean, allowing more focus on analytics rather than heavy preprocessing,

A small set of product iamges (10-20) will be gathered externally to support the YOLO component. 

##Methods
**Text Analytics**
* Text cleaning & normalization
* Sentiment analysis (VADER)
* Topic modeling (NMF or LDA)
* Keyword extraction and theme summaries

**Predictive Modeling**
* Predict Customer Satisfaction or Time to Resolution using metadata + text-derived features
* Linear/logistic regression or random forest

**Image Analytics** (YOLOv8n)
* Use pretrained YOLOv8n for object detection on a limited set of product images
* Demonstrate feasibility of automatic product identification in ticket workflows

##Why YOLOv8n? 
We use YOLOv8n because it provides the best balance of speed, simplicity, and practicality for this project. YOLO performs object detection in a single pass, enabling fast inference without GPU requirements. Although more advanced architectures can outperform YOLO on large, domain-specific datasets, they require far more training resources. Since our goal is a proof-of-concept on a small image set, pretrained YOLOv8n is the most appropriate choice. 

##Objectives 
1. Identify major themes in customer issues.
2. Analyze sentiment across ticket types, products, and priorities.
3. Predict satisfaction or resolution time from ticket features.
4. Demonstrate feasibility of integrating image analytics into support opperations.

##Limitations
* Synthetic/anonymized dataset
* Limited image set
* Pretrained YOLOv8n may not perfectly recognize product variants
* Topic models are exploratory and not definitive
  
