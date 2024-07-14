import copy
import torch
from torch import nn

from .modeling_mt5 import MT5PreTrainedModel, MT5Stack
from .configuration_mt5 import MT5Config


class MT5ForMetricX(MT5PreTrainedModel):
  """MT5 model for regression."""

  def __init__(self, config: MT5Config):
    super().__init__(config)
    self.model_dim = config.d_model
    
    encoder_config = copy.deepcopy(config)
    encoder_config.is_decoder = False
    encoder_config.use_cache = False
    encoder_config.is_encoder_decoder = False

    decoder_config = copy.deepcopy(config)
    decoder_config.is_decoder = True
    decoder_config.is_encoder_decoder = False
    decoder_config.num_layers = config.num_decoder_layers

    self.shared = nn.Embedding(config.vocab_size, config.d_model)
    self.encoder = MT5Stack(encoder_config, self.shared)
    self.decoder = MT5Stack(decoder_config, self.shared)
    self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    # Initialize weights and apply final processing
    self.post_init()

    # Model parallel
    self.model_parallel = False
    self.device_map = None

  def forward(self, input_ids: torch.LongTensor, attention_mask: torch.FloatTensor):
    use_cache = self.config.use_cache

    # Encode
    encoder_outputs = self.encoder(
      input_ids=input_ids,
      attention_mask=attention_mask,
    )
    last_hidden_state, _, _, _, _ = encoder_outputs

    # Create 1 step of dummy input for the decoder.
    batch_size = input_ids.size(0)
    decoder_input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)

    # Decode
    decoder_outputs = self.decoder(
      input_ids=decoder_input_ids,
      encoder_hidden_states=last_hidden_state,
      encoder_attention_mask=attention_mask,
      use_cache=use_cache,
    )
    sequence_output, _, _, _, _ = decoder_outputs

    lm_logits = self.lm_head(sequence_output)

    # 250089 = <extra_id_10>
    predictions = lm_logits[:, 0, 250089]

    # Clip to 0 to 25
    predictions = torch.clamp(predictions, 0, 25)

    return predictions
