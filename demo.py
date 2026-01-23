"""
Streamlit demo application for contrastive learning.

This demo allows users to upload images, visualize learned representations,
and explore the contrastive learning model interactively.
"""

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import io
import base64

from src.models.contrastive import SimCLR, MoCo, BYOL
from src.data.augmentation import get_augmentation_pipeline


# Page configuration
st.set_page_config(
    page_title="Contrastive Learning Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .upload-section {
        border: 2px dashed #cccccc;
        border-radius: 0.5rem;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_type: str, checkpoint_path: str = None) -> nn.Module:
    """
    Load a pre-trained contrastive learning model.
    
    Args:
        model_type: Type of model ('simclr', 'moco', 'byol')
        checkpoint_path: Path to model checkpoint
        
    Returns:
        Loaded model
    """
    # Create model based on type
    if model_type == "simclr":
        model = SimCLR(
            backbone="resnet50",
            pretrained=True,
            projection_dim=128,
            hidden_dim=2048,
            temperature=0.07,
        )
    elif model_type == "moco":
        model = MoCo(
            backbone="resnet50",
            pretrained=True,
            projection_dim=128,
            hidden_dim=2048,
            temperature=0.07,
            momentum=0.999,
            queue_size=65536,
        )
    elif model_type == "byol":
        model = BYOL(
            backbone="resnet50",
            pretrained=True,
            projection_dim=256,
            hidden_dim=4096,
            prediction_dim=256,
            momentum=0.996,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Load checkpoint if provided
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    return model


def preprocess_image(image: Image.Image, size: int = 224) -> torch.Tensor:
    """
    Preprocess image for model inference.
    
    Args:
        image: PIL Image
        size: Target size for resizing
        
    Returns:
        Preprocessed tensor
    """
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return transform(image).unsqueeze(0)


def extract_features(model: nn.Module, image: torch.Tensor) -> np.ndarray:
    """
    Extract features from the model.
    
    Args:
        model: Contrastive learning model
        image: Preprocessed image tensor
        
    Returns:
        Feature vector
    """
    with torch.no_grad():
        if hasattr(model, 'encode'):
            features = model.encode(image)
        else:
            features, _ = model(image)
    
    return features.squeeze().numpy()


def create_augmentation_pipeline(augmentation_type: str) -> transforms.Compose:
    """
    Create augmentation pipeline for visualization.
    
    Args:
        augmentation_type: Type of augmentation
        
    Returns:
        Augmentation pipeline
    """
    if augmentation_type == "simclr":
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomGaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif augmentation_type == "moco":
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomGaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def visualize_embeddings(features: np.ndarray, method: str = "tsne") -> go.Figure:
    """
    Visualize embeddings using dimensionality reduction.
    
    Args:
        features: Feature matrix
        method: Dimensionality reduction method
        
    Returns:
        Plotly figure
    """
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    elif method == "pca":
        reducer = PCA(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    embeddings = reducer.fit_transform(features)
    
    fig = px.scatter(
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        title=f"Feature Embeddings ({method.upper()})",
        labels={'x': f'{method.upper()} Component 1', 'y': f'{method.upper()} Component 2'},
    )
    
    return fig


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Contrastive Learning Demo</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Select Model",
        ["simclr", "moco", "byol"],
        help="Choose the contrastive learning model to use"
    )
    
    # Checkpoint upload
    checkpoint_file = st.sidebar.file_uploader(
        "Upload Model Checkpoint (Optional)",
        type=['pth', 'pt'],
        help="Upload a trained model checkpoint"
    )
    
    # Augmentation type
    augmentation_type = st.sidebar.selectbox(
        "Augmentation Type",
        ["simclr", "moco", "none"],
        help="Choose the augmentation strategy"
    )
    
    # Load model
    try:
        model = load_model(model_type)
        st.sidebar.success(f"✅ {model_type.upper()} model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {str(e)}")
        return
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📸 Image Upload", "🎨 Augmentation Demo", "📊 Feature Analysis", "📈 Model Comparison"])
    
    with tab1:
        st.header("Image Upload and Feature Extraction")
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image to extract features"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                st.subheader("Preprocessed Image")
                # Preprocess image
                processed_image = preprocess_image(image)
                
                # Denormalize for display
                denorm_image = processed_image.squeeze().permute(1, 2, 0)
                denorm_image = denorm_image * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
                denorm_image = torch.clamp(denorm_image, 0, 1)
                
                st.image(denorm_image.numpy(), caption="Preprocessed Image", use_column_width=True)
            
            # Extract features
            if st.button("Extract Features"):
                with st.spinner("Extracting features..."):
                    features = extract_features(model, processed_image)
                    
                    st.success("Features extracted successfully!")
                    
                    # Display feature statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Feature Dimension", len(features))
                    
                    with col2:
                        st.metric("Mean", f"{np.mean(features):.4f}")
                    
                    with col3:
                        st.metric("Std", f"{np.std(features):.4f}")
                    
                    with col4:
                        st.metric("Min", f"{np.min(features):.4f}")
                    
                    # Feature distribution
                    st.subheader("Feature Distribution")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(features, bins=50, alpha=0.7, edgecolor='black')
                    ax.set_xlabel("Feature Value")
                    ax.set_ylabel("Frequency")
                    ax.set_title("Distribution of Feature Values")
                    st.pyplot(fig)
    
    with tab2:
        st.header("Augmentation Visualization")
        
        # Sample images for augmentation demo
        sample_images = [
            "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Vd-Orig.png/256px-Vd-Orig.png",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Google_2015_logo.svg/272px-Google_2015_logo.svg.png",
        ]
        
        # Create augmentation pipeline
        aug_pipeline = create_augmentation_pipeline(augmentation_type)
        
        # Display augmented views
        st.subheader(f"Augmentation Strategy: {augmentation_type.upper()}")
        
        if augmentation_type != "none":
            st.info("This shows how the model sees different augmented views of the same image.")
            
            # Create a sample image
            sample_image = Image.new('RGB', (224, 224), color='red')
            
            # Generate augmented views
            augmented_views = []
            for i in range(6):
                aug_view = aug_pipeline(sample_image)
                augmented_views.append(aug_view)
            
            # Display augmented views
            cols = st.columns(3)
            for i, aug_view in enumerate(augmented_views):
                with cols[i % 3]:
                    # Denormalize for display
                    denorm_view = aug_view.permute(1, 2, 0)
                    denorm_view = denorm_view * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
                    denorm_view = torch.clamp(denorm_view, 0, 1)
                    
                    st.image(denorm_view.numpy(), caption=f"Augmented View {i+1}", use_column_width=True)
        else:
            st.info("No augmentation applied. The model will see the original image.")
    
    with tab3:
        st.header("Feature Analysis")
        
        st.info("Upload multiple images to analyze their feature representations.")
        
        # Multiple image upload
        uploaded_files = st.file_uploader(
            "Upload multiple images",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            accept_multiple_files=True,
            help="Upload multiple images to analyze their feature representations"
        )
        
        if uploaded_files:
            st.subheader(f"Analyzing {len(uploaded_files)} images...")
            
            # Extract features for all images
            all_features = []
            image_names = []
            
            progress_bar = st.progress(0)
            for i, uploaded_file in enumerate(uploaded_files):
                image = Image.open(uploaded_file).convert('RGB')
                processed_image = preprocess_image(image)
                features = extract_features(model, processed_image)
                
                all_features.append(features)
                image_names.append(uploaded_file.name)
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            all_features = np.array(all_features)
            
            # Visualization options
            viz_method = st.selectbox(
                "Visualization Method",
                ["tsne", "pca"],
                help="Choose dimensionality reduction method"
            )
            
            if st.button("Generate Visualization"):
                with st.spinner("Generating visualization..."):
                    fig = visualize_embeddings(all_features, viz_method)
                    
                    # Add image names as hover text
                    fig.update_traces(
                        hovertemplate=f"<b>%{{text}}</b><br>" +
                                    f"{viz_method.upper()} Component 1: %{{x}}<br>" +
                                    f"{viz_method.upper()} Component 2: %{{y}}<extra></extra>",
                        text=image_names
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Similarity matrix
            if st.button("Compute Similarity Matrix"):
                with st.spinner("Computing similarity matrix..."):
                    # Normalize features
                    normalized_features = all_features / np.linalg.norm(all_features, axis=1, keepdims=True)
                    
                    # Compute similarity matrix
                    similarity_matrix = np.dot(normalized_features, normalized_features.T)
                    
                    # Create heatmap
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(
                        similarity_matrix,
                        xticklabels=image_names,
                        yticklabels=image_names,
                        annot=True,
                        fmt='.2f',
                        cmap='viridis',
                        ax=ax
                    )
                    ax.set_title("Feature Similarity Matrix")
                    plt.xticks(rotation=45)
                    plt.yticks(rotation=0)
                    plt.tight_layout()
                    st.pyplot(fig)
    
    with tab4:
        st.header("Model Comparison")
        
        st.info("Compare different contrastive learning models.")
        
        # Model comparison
        models_to_compare = st.multiselect(
            "Select Models to Compare",
            ["simclr", "moco", "byol"],
            default=["simclr", "moco"]
        )
        
        if len(models_to_compare) >= 2:
            st.subheader("Model Architecture Comparison")
            
            # Create comparison table
            comparison_data = {
                "Model": [],
                "Backbone": [],
                "Projection Dim": [],
                "Temperature": [],
                "Special Features": []
            }
            
            for model_name in models_to_compare:
                if model_name == "simclr":
                    comparison_data["Model"].append("SimCLR")
                    comparison_data["Backbone"].append("ResNet-50")
                    comparison_data["Projection Dim"].append("128")
                    comparison_data["Temperature"].append("0.07")
                    comparison_data["Special Features"].append("NT-Xent Loss")
                elif model_name == "moco":
                    comparison_data["Model"].append("MoCo")
                    comparison_data["Backbone"].append("ResNet-50")
                    comparison_data["Projection Dim"].append("128")
                    comparison_data["Temperature"].append("0.07")
                    comparison_data["Special Features"].append("Momentum Queue")
                elif model_name == "byol":
                    comparison_data["Model"].append("BYOL")
                    comparison_data["Backbone"].append("ResNet-50")
                    comparison_data["Projection Dim"].append("256")
                    comparison_data["Temperature"].append("N/A")
                    comparison_data["Special Features"].append("Predictor Network")
            
            st.table(comparison_data)
            
            # Performance metrics (placeholder)
            st.subheader("Performance Metrics")
            st.info("These are example metrics. In practice, you would load actual evaluation results.")
            
            metrics_data = {
                "Model": models_to_compare,
                "Linear Probe Accuracy": [0.85, 0.87, 0.83],
                "k-NN Accuracy": [0.82, 0.84, 0.80],
                "Training Time (hours)": [12, 15, 18],
            }
            
            st.table(metrics_data)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>Contrastive Learning Demo | Built with Streamlit</p>
            <p>This demo showcases SimCLR, MoCo, and BYOL models for self-supervised learning.</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
