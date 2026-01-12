# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import math

import torch
from torch import nn
from torch.nn import init
from torchtitan.tools.logging import logger
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from .args import HFTransformerModelArgs


class SliceableModuleDict(nn.ModuleDict):
    """
    A ModuleDict that supports slicing like ModuleList.
    Keys are expected to be string representations of integers (e.g., "0", "1", "2").
    """

    def __getitem__(self, key):
        if isinstance(key, slice):
            # Handle slicing: convert slice to list of keys
            keys = sorted(
                self.keys(), key=lambda x: int(x) if x.isdigit() else float("inf")
            )
            sliced_keys = keys[key]
            # Return a new SliceableModuleDict with the sliced items
            return SliceableModuleDict({k: self[k] for k in sliced_keys})
        return super().__getitem__(key)

    def __iter__(self):
        # Iterate over values in sorted order by key (as integers)
        keys = sorted(
            self.keys(), key=lambda x: int(x) if x.isdigit() else float("inf")
        )
        for key in keys:
            yield self[key]

    def __len__(self):
        return len(self._modules)


class HFTransformerModel(nn.Module):
    def __init__(self, model_args: HFTransformerModelArgs):
        super().__init__()

        # NOTE(3outeille): This prevents Hugging Face modeling from initializing ROPE (inv_freq) buffers to NaN.
        # Needed when loading from seed checkpoint.
        if hasattr(model_args, "deterministic") and model_args.deterministic:
            torch.utils.deterministic.fill_uninitialized_memory = False

        # Try to import the model class dynamically from the transformers library if not found in globals
        model_class_name = model_args.architectures[0]
        model_cls = globals().get(model_class_name, None)
        if model_cls is None:
            try:
                transformers_mod = importlib.import_module("transformers")
                model_cls = getattr(transformers_mod, model_class_name)
            except (ImportError, AttributeError) as e:
                raise ImportError(
                    f"Could not find model class '{model_class_name}' in globals or transformers. "
                    f"Make sure the class is available. Original error: {e}"
                ) from e

        # Attempt to patch model weight initialization based on architecture type
        try:
            model_name_prefix = model_class_name.replace("ForCausalLM", "")
            model_module = importlib.import_module(model_cls.__module__)

            attention_cls = getattr(model_module, f"{model_name_prefix}Attention", None)
            mlp_cls = getattr(model_module, f"{model_name_prefix}MLP", None)
            decoder_layer_cls = getattr(
                model_module, f"{model_name_prefix}DecoderLayer", None
            )

            required_classes = {
                "Attention": attention_cls,
                "DecoderLayer": decoder_layer_cls,
            }

            if all(required_classes.values()):
                logger.info(f"Applying Llama-like patch for {model_name_prefix}")
                self._patch_hf_llama_like(
                    decoder_layer_cls=decoder_layer_cls,
                    attention_cls=attention_cls,
                    mlp_cls=mlp_cls,  # mlp_cls can be None
                )
            else:
                missing = [name for name, cls in required_classes.items() if not cls]
                logger.warning(
                    f"Could not find required classes ({', '.join(missing)}) for {model_name_prefix}. "
                    "Skipping Llama-like patch."
                )

        except Exception as e:
            logger.warning(
                f"Failed to apply agnostic patch for {model_class_name} due to: {e}. "
                "Weight initialization might not match TorchTitan."
            )

        self.model = model_cls(config=model_args)
        self.max_seq_len = model_args.max_seq_len
        self.cp_mesh = None

        # Resolve the backbone (container holding layers/embeddings) for different HF model structures.
        # Llama-like models expose `model`, GPTNeoX exposes `gpt_neox`, GPT-2 exposes `transformer`, etc.
        self.backbone = self._resolve_backbone(self.model)

        layers = getattr(self.backbone, "layers", None)
        # Convert ModuleList to ModuleDict to preserve original indices
        # This ensures state dict keys match checkpoint keys
        if isinstance(layers, nn.ModuleList):
            self.layers = SliceableModuleDict(
                {str(i): layer for i, layer in enumerate(layers)}
            )

        for layer in self.layers.values():
            layer.moe_enabled = False

    def set_cp_mesh(self, mesh):
        self.cp_mesh = mesh

    def _patch_hf_llama_like(self, decoder_layer_cls, attention_cls, mlp_cls=None):
        """
        This patch modifies a Hugging Face Llama-like model's weight initialization to match
        the initialization scheme used in TorchTitan. This is crucial for ensuring
        bit-for-bit reproducibility when converting checkpoints between the native
        TorchTitan format and the Hugging Face format.

        The patch targets the following aspects of the model:
        - `PreTrainedModel._initialize_weights`: Handles meta device initialization correctly.
        - `PreTrainedModel._init_weights`: Implements TorchTitan's specific initialization
          for attention, MLP, embedding, and layer norm layers. This includes depth-dependent
          initialization for attention and MLP layers.
        - `DecoderLayer.__init__`: Adds `layer_idx` to attention and MLP modules within
          each decoder layer, which is required for the depth-dependent initialization.
        """

        _original_decoder_layer_init = decoder_layer_cls.__init__

        def _decoder_layer_init_patched(self, config: PretrainedConfig, layer_idx: int):
            _original_decoder_layer_init(self, config, layer_idx)
            self.layer_idx = layer_idx
            # Ensure both attention and mlp modules have layer_idx for depth-based init
            attn = None
            if hasattr(self, "self_attn"):
                attn = self.self_attn
            elif hasattr(self, "attention"):  # GPTNeoX
                attn = self.attention
            if attn is not None:
                attn.layer_idx = layer_idx

            # some models might not have mlp in each layer
            if hasattr(self, "mlp") and self.mlp is not None:
                self.mlp.layer_idx = layer_idx
            elif hasattr(self, "feed_forward") and self.feed_forward is not None:
                self.feed_forward.layer_idx = layer_idx

        def _initialize_weights_patched(self, module):
            # NOTE(3outeille): monkey-patch PreTrainedModel to handle meta device initialization correctly
            # The default _initialize_weights sets _is_hf_initialized = True even on a meta device,
            # which prevents subsequent proper initialization.
            if getattr(module, "_is_hf_initialized", False):
                return

            for param in module.parameters(recurse=True):
                if param.device.type == "meta":
                    return

            # If not on a meta device, call the original weight initialization
            self._init_weights(module)
            module._is_hf_initialized = True

        def _init_weights_patched(self, module):
            """
            Patched version of _init_weights to match TorchTitan's initialization for Llama-like models.
            `self` is a PreTrainedModel instance.
            """
            config = self.config
            # Build tuple of classes to check for layer_idx-based init_std calculation
            layer_idx_classes = [attention_cls]
            if mlp_cls:
                layer_idx_classes.append(mlp_cls)
            layer_idx_classes = tuple(layer_idx_classes)

            if isinstance(module, layer_idx_classes):
                layer_idx = getattr(module, "layer_idx", 0)

                if hasattr(config, "depth_init") and config.depth_init:
                    init_std = 0.02 / (2 * (layer_idx + 1)) ** 0.5
                else:
                    init_std = 0.02 / (2 * config.num_hidden_layers) ** 0.5

            if isinstance(module, attention_cls):
                # GPTNeoX-style packed projections
                if hasattr(module, "query_key_value"):
                    qkv = module.query_key_value
                    nn.init.trunc_normal_(qkv.weight, mean=0.0, std=0.02)
                    if qkv.bias is not None:
                        fan_in, _ = init._calculate_fan_in_and_fan_out(qkv.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        init.uniform_(qkv.bias, -bound, bound)

                    out_proj = getattr(module, "dense", None)
                    if out_proj is not None:
                        nn.init.trunc_normal_(out_proj.weight, mean=0.0, std=init_std)
                        if out_proj.bias is not None:
                            fan_in, _ = init._calculate_fan_in_and_fan_out(
                                out_proj.weight
                            )
                            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                            init.uniform_(out_proj.bias, -bound, bound)
                else:
                    # Initialize weights and biases for q, k, v projections (Llama-like)
                    for proj_name in ["q_proj", "k_proj", "v_proj"]:
                        proj = getattr(module, proj_name)
                        nn.init.trunc_normal_(proj.weight, mean=0.0, std=0.02)
                        if proj.bias is not None:
                            fan_in, _ = init._calculate_fan_in_and_fan_out(
                                proj.weight
                            )
                            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                            init.uniform_(proj.bias, -bound, bound)

                    # Handle different names for the output projection layer
                    o_proj = getattr(module, "o_proj", getattr(module, "dense", None))
                    if o_proj is not None:
                        nn.init.trunc_normal_(o_proj.weight, mean=0.0, std=init_std)
                        if o_proj.bias is not None:
                            fan_in, _ = init._calculate_fan_in_and_fan_out(
                                o_proj.weight
                            )
                            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                            init.uniform_(o_proj.bias, -bound, bound)

            elif mlp_cls and isinstance(module, mlp_cls):
                # Handle different names for MLP layers
                gate_proj = getattr(module, "gate_proj", getattr(module, "fc1", None))
                up_proj = getattr(module, "up_proj", None)
                down_proj = getattr(module, "down_proj", getattr(module, "fc2", None))

                # gate_proj (or fc1) should always use std=0.02 for numerical stability.
                if gate_proj is not None:
                    nn.init.trunc_normal_(gate_proj.weight, mean=0.0, std=0.02)
                    if gate_proj.bias is not None:
                        fan_in, _ = init._calculate_fan_in_and_fan_out(gate_proj.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        init.uniform_(gate_proj.bias, -bound, bound)
                # up_proj and down_proj (or fc2) use the depth-dependent init_std.
                if up_proj is not None:
                    nn.init.trunc_normal_(up_proj.weight, mean=0.0, std=init_std)
                    if up_proj.bias is not None:
                        fan_in, _ = init._calculate_fan_in_and_fan_out(up_proj.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        init.uniform_(up_proj.bias, -bound, bound)
                if down_proj is not None:
                    nn.init.trunc_normal_(down_proj.weight, mean=0.0, std=init_std)
                    if down_proj.bias is not None:
                        fan_in, _ = init._calculate_fan_in_and_fan_out(down_proj.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        init.uniform_(down_proj.bias, -bound, bound)

            elif module is getattr(
                self, "lm_head", None
            ):  # TODO(3outeille): find a better way to detect lm_head
                final_out_std = config.hidden_size**-0.5
                cutoff_factor = 3
                nn.init.trunc_normal_(
                    module.weight,
                    mean=0.0,
                    std=final_out_std,
                    a=-cutoff_factor * final_out_std,
                    b=cutoff_factor * final_out_std,
                )
                if module.bias is not None:
                    module.bias.data.zero_()

            elif isinstance(module, nn.Embedding):
                # When tie_word_embeddings is True, use lm_head initialization
                if (
                    hasattr(config, "tie_word_embeddings")
                    and config.tie_word_embeddings
                ):
                    final_out_std = config.hidden_size**-0.5
                    cutoff_factor = 3
                    nn.init.trunc_normal_(
                        module.weight,
                        mean=0.0,
                        std=final_out_std,
                        a=-cutoff_factor * final_out_std,
                        b=cutoff_factor * final_out_std,
                    )
                else:
                    std = config.initializer_range
                    module.weight.data.normal_(mean=0.0, std=std)

                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

            elif (
                isinstance(
                    module,
                    (nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d),
                )
                or "LayerNorm" in module.__class__.__name__
                or "RMSNorm" in module.__class__.__name__
            ):
                # Norms can exist without weights (in which case they are None from torch primitives)
                if hasattr(module, "weight") and module.weight is not None:
                    module.weight.data.fill_(1.0)
                if hasattr(module, "bias") and module.bias is not None:
                    module.bias.data.zero_()

        decoder_layer_cls.__init__ = _decoder_layer_init_patched
        PreTrainedModel._init_weights = _init_weights_patched
        PreTrainedModel._initialize_weights = _initialize_weights_patched

    def _resolve_backbone(self, model):
        """Pick the container that holds layers/embeddings for different HF architectures."""
        for attr in ("model", "gpt_neox", "transformer"):
            if hasattr(model, attr):
                return getattr(model, attr)
        raise AttributeError(
            "Could not find a known backbone container (model/gpt_neox/transformer) on the HF model."
        )

    @property
    def tok_embeddings(self):
        """Returns the model's embed_tokens, handling different Hugging Face model structures."""
        if hasattr(self.backbone, "embed_tokens"):  # Llama-like
            return self.backbone.embed_tokens
        if hasattr(self.backbone, "wte"):  # GPT-2 like
            return self.backbone.wte
        if hasattr(self.backbone, "embed_in"):  # GPTNeoX
            return self.backbone.embed_in
        raise AttributeError(
            "Could not find embeddings in the model. Please check the model structure."
        )

    @tok_embeddings.setter
    def tok_embeddings(self, value):
        if hasattr(self.backbone, "embed_tokens"):
            self.backbone.embed_tokens = value
            return
        if hasattr(self.backbone, "wte"):
            self.backbone.wte = value
            return
        if hasattr(self.backbone, "embed_in"):
            self.backbone.embed_in = value
            return
        raise AttributeError(
            "Could not find embeddings in the model. Please check the model structure."
        )

    @property
    def layers(self):
        """Returns the model's layers, handling different Hugging Face model structures."""
        if hasattr(self.backbone, "layers"):
            return self.backbone.layers
        if hasattr(self.backbone, "h"):  # GPT-2 style
            return self.backbone.h
        raise AttributeError(
            "Could not find layers in the model. Please check the model structure."
        )

    @layers.setter
    def layers(self, value):
        if hasattr(self.backbone, "layers"):
            self.backbone.layers = value
            return
        if hasattr(self.backbone, "h"):
            self.backbone.h = value
            return
        raise AttributeError(
            "Could not find layers in the model. Please check the model structure."
        )

    @property
    def norm(self):
        """Returns the model's norm, handling different Hugging Face model structures."""
        if hasattr(self.backbone, "norm"):  # Llama-like
            return self.backbone.norm
        if hasattr(self.backbone, "final_layernorm"):  # Phi-like
            return self.backbone.final_layernorm
        if hasattr(self.backbone, "final_layer_norm"):  # GPTNeoX
            return self.backbone.final_layer_norm
        if hasattr(self.backbone, "ln_f"):  # GPT-2
            return self.backbone.ln_f
        raise AttributeError(
            "Could not find norm in the model. Please check the model structure."
        )

    @norm.setter
    def norm(self, value):
        if hasattr(self.backbone, "norm"):
            self.backbone.norm = value
            return
        if hasattr(self.backbone, "final_layernorm"):
            self.backbone.final_layernorm = value
            return
        if hasattr(self.backbone, "final_layer_norm"):
            self.backbone.final_layer_norm = value
            return
        if hasattr(self.backbone, "ln_f"):
            self.backbone.ln_f = value
            return
        raise AttributeError(
            "Could not find norm in the model. Please check the model structure."
        )

    @property
    def output(self):
        """Returns the model's output layer, handling different Hugging Face model structures."""
        if hasattr(self.model, "lm_head"):  # Llama-like
            return self.model.lm_head
        if hasattr(self.model, "embed_out"):  # GPTNeoX
            return self.model.embed_out
        raise AttributeError(
            "Could not find output (lm_head/embed_out) in the model. Please check the model structure."
        )

    @output.setter
    def output(self, value):
        if hasattr(self.model, "lm_head"):
            self.model.lm_head = value
            return
        if hasattr(self.model, "embed_out"):
            self.model.embed_out = value
            return
        raise AttributeError(
            "Could not find output (lm_head/embed_out) in the model. Please check the model structure."
        )

    @property
    def rotary_emb(self):
        """Returns the model's rotary_emb, handling different Hugging Face model structures."""
        if hasattr(self.backbone, "rotary_emb"):
            return self.backbone.rotary_emb
        raise AttributeError(
            "Could not find rotary_emb in the model. Please check the model structure."
        )

    @rotary_emb.setter
    def rotary_emb(self, value):
        if hasattr(self.backbone, "rotary_emb"):
            self.backbone.rotary_emb = value
            return
        raise AttributeError(
            "Could not find rotary_emb in the model. Please check the model structure."
        )

    def forward(self, *args, **kwargs):
        local_seq_len = self.max_seq_len
        local_seq_len //= (
            self.cp_mesh.size()
            if self.cp_mesh is not None and self.cp_mesh.size() > 1
            else 1
        )
        kwargs["position_ids"] = torch.arange(
            local_seq_len, device=args[0].device
        ).unsqueeze(0)
        output = self.backbone(*args, **kwargs)
        hidden = (
            output.last_hidden_state
            if hasattr(output, "last_hidden_state")
            else output[0]
            if isinstance(output, tuple)
            else output
        )
        return self.output(hidden)

    def init_weights(self, *args, **kwargs):
        # This method replicates the behavior of the original PreTrainedModel.init_weights,
        # but with a custom weight initialization function that skips nn.Identity modules (when PP is enabled)

        if self.model.config.pruned_heads:
            logger.info("Pruning heads as per model configuration.")
            self.model.prune_heads(self.model.config.pruned_heads)

        original_init_weights_fn = self.model._init_weights

        def selective_init(module):
            # For pipeline parallel, we need to skip nn.Identity modules
            if not isinstance(module, nn.Identity):
                original_init_weights_fn(module)
            else:
                logger.info("Skipping nn.Identity module during weight initialization.")

        self.model.apply(selective_init)

        # TODO(3outeille): For pipeline parallel, only tie weights if both input and output embeddings are on the same device
        # Maybe better way of handling this?
        if not isinstance(self.tok_embeddings, nn.Identity) and not isinstance(
            self.output, nn.Identity
        ):
            self.model.tie_weights()

    def named_children(self):
        """
        Provides a flattened view of the model's main components,
        making it compatible with TorchTitan's expectations.
        """
        yield "tok_embeddings", self.tok_embeddings
        yield "layers", self.layers
        yield "norm", self.norm
        yield "output", self.output
        yield "rotary_emb", self.rotary_emb

    def __setattr__(self, name, value):
        # If a property with a setter exists for this name, use it.
        # This is to bypass the nn.Module.__setattr__ logic that
        # directly registers modules and skips property setters.
        cls = self.__class__
        if hasattr(cls, name):
            prop = getattr(cls, name)
            if isinstance(prop, property) and prop.fset is not None:
                prop.fset(self, value)
                return

        # Otherwise, fall back to the default nn.Module behavior.
        super().__setattr__(name, value)
