# This file is modified from Punica Project
# Check ref: https://github.com/punica-ai/punica

from quest.utils.utils import TensorLayout
import torch
from typing import Optional, List

class KvPool:

  def __init__(
      self,
      num_layers: int,
      num_heads: int,
      head_dim: int,
      capacity: int | list[int], # number of pages of each layer
      block_len: int,
      dtype: torch.dtype,
      device: torch.device,
  ):
    self._layout = TensorLayout.NHD

    if isinstance(capacity, int):
      capacity = [capacity] * num_layers
    assert len(capacity) == num_layers, "capacity must be a list of length num_layers"
    assert all(c > 0 for c in capacity), "capacity must be positive"

    # Allocate separate buffers for each layer
    self._bufs = []
    self._free = []

    for layer_idx in range(num_layers):
      layer_capacity = capacity[layer_idx]
      buf = torch.empty(
          (layer_capacity, 2, block_len, num_heads, head_dim),
          dtype=dtype,
          device=device)
      self._bufs.append(buf)
      self._free.append(set(range(layer_capacity)))

    self._num_layers = num_layers
    self._num_heads = num_heads
    self._head_dim = head_dim
    self._capacity = capacity
    self._block_len = block_len

  @property
  def layout(self):
    return self._layout

  @property
  def buf(self, layer_idx=None):
    if layer_idx is None:
      return self._bufs
    assert 0 <= layer_idx < self._num_layers
    return self._bufs[layer_idx]

  @property
  def num_layers(self):
    return self._num_layers

  @property
  def block_len(self):
    return self._block_len

  @property
  def num_free_blocks(self, layer_idx=None):
    if layer_idx is None:
      return [len(free_set) for free_set in self._free]
    assert 0 <= layer_idx < self._num_layers
    return len(self._free[layer_idx])

  @property
  def capacity(self, layer_idx=None):
    if layer_idx is None:
      return self._capacity
    assert 0 <= layer_idx < self._num_layers
    return self._capacity[layer_idx]

  def alloc_block(self, layer_idx: int) -> int:
    assert 0 <= layer_idx < self._num_layers
    if not self._free[layer_idx]:
      raise RuntimeError(f"No free blocks in layer {layer_idx}")
    idx = self._free[layer_idx].pop()
    return idx

  def free_block(self, layer_idx: int, idx: int):
    assert 0 <= layer_idx < self._num_layers
    assert 0 <= idx < self._capacity[layer_idx]
    assert idx not in self._free[layer_idx]
    self._free[layer_idx].add(idx)

class KvCache:
  """Key-value cache for one sequence."""

  def __init__(
      self,
      num_layers,
      num_heads,
      head_dim,
      max_seq_len: int | list[int], # number of tokens in each layer
      page_size,
      dtype: torch.dtype,
      device: torch.device
    ):
    
    if isinstance(max_seq_len, int):
      max_seq_len = [max_seq_len] * num_layers
    assert len(max_seq_len) == num_layers, "max_seq_len must be a list of length num_layers"
    assert all(m > 0 for m in max_seq_len), "max_seq_len must be positive"
    assert page_size > 0, "page_size must be positive"
    capacity = [(max_seq_len + page_size - 1) // page_size for max_seq_len in max_seq_len]

    self._pool = KvPool(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        capacity=capacity,
        block_len=page_size,
        dtype=dtype,
        device=device
    )
  
    self._indicies = [[] for _ in range(num_layers)]
    self._seqlen = 0
    self._num_layers = num_layers

  @property
  def pool(self) -> KvPool:
    return self._pool

  @property
  def seqlen(self) -> int:
    return self._seqlen

  @property
  def last_page_len(self) -> int:
    return (self.seqlen - 1) % self._pool.block_len + 1

  # @property
  def indicies(self, layer_idx = None) -> list[int]:
    if layer_idx is None:
      return self._indicies
    assert 0 <= layer_idx < self._num_layers
    return self._indicies[layer_idx]
  
  def buf_layer(self, layer_idx: int):
    assert 0 <= layer_idx < self._num_layers
    return self._pool.buf[layer_idx]

  def append_seq(self, seq_len: int, active_layers: Optional[List[int]] = None) -> list[int]:
    """Reserve space for tokens and return number of new pages
    Args:
      seq_len: number of tokens to append
      active_layers: list of layers to append. If None, all layers are appended.
    Returns:
      number of new pages appended
    """
    if seq_len <= 0:
        return 0
    
    if active_layers is None:
      active_layers = list(range(self._num_layers))

    appended_page_count = [0 for _ in range(self._num_layers)]
    for _ in range(seq_len):
      for layer_idx in active_layers:
        if layer_idx >= self._num_layers:
          raise RuntimeError(f"Layer index {layer_idx} out of range")
        last_page_offset = self.last_page_len
        block_len = self.pool.block_len
        if last_page_offset == block_len:
          self._indicies[layer_idx].append(self._pool.alloc_block(layer_idx))
          appended_page_count[layer_idx] += 1
      self._seqlen += 1
    return appended_page_count

  def release(self):
    """Release all blocks"""
    self._seqlen = 0
    for layer_idx in range(self.pool.num_layers):
      for block_idx in self._indicies[layer_idx]:
        self.pool.free_block(layer_idx, block_idx)
      self._indicies[layer_idx].clear()
