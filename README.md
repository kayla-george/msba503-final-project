## Project Title
Multi-Modal Analysis of Customer Support Tickets Using Text Analytics, YOLOv8n, and Mask R-CNN

## Overview

Customer support teams face thousands of incoming tickets acoss multiple product types, channels, and issue categories. Identifying recurring pain points, and issue categories. Identifying recurring pain points, tracking dissatisfaction drivers, and accurately routing issues remains challenging. The Customer Support Ticket Dataset provides structured ticket information such as product purchased, ticket type, priority, channel, customer satisfaction rating, and free-text ticket descriptions.

This project applies text analytics, sentiment modeling, topic extraction, and predictive modeling to identify key trends in customer issues. A YOLOv8n image analytics component and Mask R-CNN are added to explore how product images could help automate ticket routing and improve support workflows.

## Dataset
We use the Customer Support Ticket Dataset from Kaggle, which includes:
* Ticket Description (text of the issue)
* Ticket Type
* Product Purchased
* Ticket Priority
* Ticket Channel
* Time to Resolution
* Customer Satisfaction Rating

The dataset is relatively clean, allowing more focus on analytics rather than heavy preprocessing,
A small set of product images (10-20) will be gathered externally to support the YOLO and Mask R-CNN components. 

## Methods
**Text Analytics**
* Text cleaning & normalization
* Sentiment analysis (VADER)
* Topic modeling (NMF)
* Keyword extraction and theme summaries

**Predictive Modeling**
* Predict Customer Satisfaction using metadata + text-derived features
* Predict Ticket Priority based on ticket description text
* Logistic Regression with TF-IDF features

**Image Analytics** (YOLOv8n & Mask R-CNN)
* Use pretrained YOLOv8n for object detection on a limited set of product images
* Use pretrained Mask R-CNN for instance segmentation
* Demonstrate feasibility of automatic product identification in ticket workflows

## Why YOLOv8n? 
We use YOLOv8n because it provides the best balance of speed, simplicity, and practicality for this project. YOLO performs object detection in a single pass, enabling fast inference without heavy GPU requirements. Although more advanced architectures can outperform YOLO on large, domain-specific datasets, they require far more training resources. Since our goal is a proof-of-concept on a small image set, pretrained YOLOv8n is the most appropriate choice. 

## Why Mask R-CNN? 
Mask R-CNN extends object detection by adding instance segmentation — it not only identifies and localizes objects but also creates pixel-precise masks for each detected instance. This capability is particularly valuable in customer support scenarios where multiple products or product components might appear in a single image. For example, a customer might photograph a damaged device alongside its accessories, or submit images showing specific defective parts that need precise identification.

## Objectives 
1. Identify major themes in customer issues through topic modeling.
2. Analyze sentiment across ticket types, products, and priorities.
3. Predict satisfaction from ticket features and text.
4. Predict ticket priority based on ticket description text.
5. Demonstrate feasibility of integrating image analytics into support opperations.

## Project Structure
```
msba503-final-project/
├── Images                     # Product images for computer vision
    └── (image files)
├── 01_data_loading.ipynb          # Data loading, cleaning, initial EDA
├── 02_Sentiment_Analysis.ipynb    # Text preprocessing, sentiment, TF-IDF
├── 03_Topic_Modeling_and_EDA.ipynb # Topic modeling with NMF, EDA visualizations
├── 04_Modeling.ipynb               # Satisfaction & priority prediction models
├── 05_Image_Analytics_YOLO.ipynb   # YOLOv8n and Mask R-CNN demonstrations

├── customer_support_tickets.csv   
├── df_maskrcnn_detections.csv
├── df_model.csv
├── df_model_features.csv
├── df_yolo_detections.csv
├── yolov8n.pt
└── README.md
```

## Limitations
* Synthetic/anonymized dataset limits real-world comparison.
* Limited image set (10-20) images for computer vision demonstration.
* Pretrained YOLOv8n and Mask R-CNN may not perfectly recognize all product variants.
* Topic models are exploratory and interpretive rather than definitive.
  
## Key Findings
* **Topic Modeling**: Identified 6 distinct issue themes including "Missing Option", "Error Messages", "Software Updates", "Intermittent Behavior", "General Issues", and "Troubleshooting Attempts"
* **Sentiment Analysis**: Ticket sentiment tends to be neutral-to-positive across all priorities, reflecting polite language rather than frustration levels
* **Satisfaction Prediction**: Text alone achieves ~41% accuracy, indicating need for operational metadata
* **Priority Prediction**: Text-based priority classification shows limited separation between adjacent priority levels
* **Image Analytics**: Successfully demonstrates object detection and segmentation on product images

## Future Enhancements
* Incorporate operational features (response time, agent performance, customer history)
* Fine-tune computer vision models on specific product images
* Implement real-time ticket routing based on combined text + image analysis
* Develop interactive dashboards for support team insights
