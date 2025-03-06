# Advanced EDA Dashboard for News AI

An interactive dashboard for exploratory data analysis of the MIND dataset and model metrics.

## Features

- **News Content Analysis**: Explore distributions of categories, subcategories, and entities
- **User Behavior Analysis**: Analyze user reading patterns and preferences 
- **Entity Network Visualization**: See connections between entities mentioned in news
- **Metric Comparisons**: Compare different metrics across content types
- **Model Performance Analysis**: Visualize performance of different models

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
python run_streamlit.py
```

## Dashboard Sections

1. **Dataset Overview**: High-level statistics about the MIND dataset
2. **News Analysis**: Detailed analysis of news content
3. **User Analysis**: Exploration of user behavior patterns
4. **Entity Analysis**: Network graphs and entity relationships
5. **Model Metrics**: Performance visualization for different models

## Integration with News AI Pipeline

This dashboard connects directly with the medallion architecture:
- Bronze layer data for raw statistics
- Silver layer for feature exploration
- Silicon layer for metrics visualization
- Gold layer for model performance analysis

## Tech Stack

- Streamlit for interactive UI
- Plotly for advanced visualizations 
- NetworkX for relationship graphs
- Pandas/Numpy for data processing