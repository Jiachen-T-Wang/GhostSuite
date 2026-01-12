"""
GPT-2 model family configurations.
This file contains the architecture specifications for different GPT-2 model sizes.
"""

# Model configurations for different GPT-2 sizes
GPT2_CONFIGS = {
    'GPT2-Small': {  # GPT-2 Small (124M)
        'n_layer': 12,
        'n_head': 12,
        'n_embd': 768,
        'block_size': 1024,
    },
    'GPT2-Medium': {  # GPT-2 Medium (355M)
        'n_layer': 24,
        'n_head': 16,
        'n_embd': 1024,
        'block_size': 1024,
    },
    'GPT2-Large': {  # GPT-2 Large (774M)
        'n_layer': 36,
        'n_head': 20,
        'n_embd': 1280,
        'block_size': 1024,
    },
    'GPT2-XL': {  # GPT-2 XL (1.5B)
        'n_layer': 48,
        'n_head': 25,
        'n_embd': 1600,
        'block_size': 1024,
    },
    'pythia70m': {
        'n_layer': 6,         # Corresponds to Pythia's 'num_hidden_layers'
        'n_head': 8,          # Corresponds to Pythia's 'num_attention_heads'
        'n_embd': 512         # Corresponds to Pythia's 'hidden_size'
    }
}

def get_model_config(model_size):
    """
    Get model configuration parameters for the specified model size.
    
    Args:
        model_size (str): One of 'small', 'medium', 'large', 'xl'
        
    Returns:
        dict: Dictionary containing model architecture parameters
    """
    if model_size not in GPT2_CONFIGS:
        raise ValueError(f"Unknown model size: {model_size}. Choose from: {list(GPT2_CONFIGS.keys())}")
    
    return GPT2_CONFIGS[model_size]