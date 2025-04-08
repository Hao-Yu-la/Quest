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
      return -1 # No free blocks
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
    if seq_len > 1: # Prefill
      for _ in range(seq_len):
        for layer_idx in active_layers:
          if layer_idx >= self._num_layers:
            raise RuntimeError(f"Layer index {layer_idx} out of range")
          last_page_offset = self.last_page_len
          block_len = self.pool.block_len
          if last_page_offset == block_len:
            new_page_idx = self._pool.alloc_block(layer_idx)
            if new_page_idx != -1:
              self._indicies[layer_idx].append(new_page_idx)
              appended_page_count[layer_idx] += 1
            else:
              appended_page_count[layer_idx] += 1 # No new page allocated, but still count it for metadata allocation
              # print(f"During prefill stage, Layer {layer_idx} is full, no new page allocated") # No error. Unload KV Cache that cannot be stored directly
        self._seqlen += 1
    elif seq_len == 1: # Decoding
      for layer_idx in active_layers:
        if layer_idx >= self._num_layers:
          raise RuntimeError(f"Layer index {layer_idx} out of range")
        last_page_offset = self.last_page_len
        block_len = self.pool.block_len
        if last_page_offset == block_len:
          new_page_idx = self._pool.alloc_block(layer_idx)
          if new_page_idx != -1:
            self._indicies[layer_idx].append(new_page_idx)
            appended_page_count[layer_idx] += 1
          else:
            raise RuntimeError(f"During decoding stage, Layer {layer_idx} is full, no new page allocated")
      self._seqlen += 1
    else:
      raise RuntimeError(f"Invalid seq_len {seq_len} for append_seq")
    return appended_page_count
  
  def allocate_page(self, layer_idx: int) -> int:
    """Allocate a page for a layer.
    Args:
      layer_idx: layer index
    Returns:
      page index
    """
    assert 0 <= layer_idx < self._num_layers
    page_idx = self._pool.alloc_block(layer_idx)
    if page_idx != -1:
      self._indicies[layer_idx].append(page_idx)
    return page_idx

  def evict_page(self, layer_idx: int, page_idx: int):
    """Evict a page from the GPU.
    Args:
      layer_idx: layer index
      page_idx: page index
    """
    assert 0 <= layer_idx < self._num_layers
    assert 0 <= page_idx < self._pool.capacity[layer_idx]
    if page_idx in self._indicies[layer_idx]:
      self._indicies[layer_idx].remove(page_idx)
    self.pool.free_block(layer_idx, page_idx)
    # print(f"Layer {layer_idx} page idx {page_idx} is evicted from GPU kv cache")

  def release(self):
    """Release all blocks"""
    self._seqlen = 0
    for layer_idx in range(self.pool.num_layers):
      for block_idx in self._indicies[layer_idx]:
        self.pool.free_block(layer_idx, block_idx)
      self._indicies[layer_idx].clear()

class KvMetadataCache(KvCache):
  """Cache for metadata of key-value cache blocks.
  This is used to store the metadata of full key-value cache blocks even if some blocks are offloaded to CPU.
  """
  def __init__(
      self,
      num_layers,
      num_heads,
      head_dim,
      max_seq_len: int, # number of blocks in each layer
      page_size,
      dtype: torch.dtype,
      device: torch.device
    ):
    super().__init__(num_layers, num_heads, head_dim, max_seq_len, page_size, dtype, device)
    self._capacity = (max_seq_len + page_size - 1) // page_size
    
    # Allocate importance scores
    # All heads share the same importance score
    self._importance_scores = torch.empty(
      (num_layers, self._capacity * page_size),
      dtype=torch.float32,
      device=device)

    # Allocate score indexes
    # Specify the index position of the KV Cache block corresponding to each meta information, while -1 means that the block is empty or is offloaded to CPU.
    self._store_indexes = torch.empty(
      (num_layers, self._capacity * page_size),
      dtype=torch.int32,
      device=device
    )
    self._store_indexes.fill_(-1)

  def update_page_importance(self, layer_idx: int, page_idx: int, score: float):
    """Update the importance score of a page.
    Args:
      layer_idx: layer index
      page_idx: page index
      score: importance score
    """
    assert 0 <= layer_idx < self._num_layers
    assert 0 <= page_idx < self.seqlen
    self._importance_scores[layer_idx, page_idx] = score

  def update_page_importance_layer(self, layer_idx: int, scores: torch.Tensor):
    """Update the importance score of a layer.
    Args:
      layer_idx: layer index
      scores: importance scores
    """
    assert 0 <= layer_idx < self._num_layers
    self._importance_scores[layer_idx, :scores.shape[0]] = scores
  
  def update_page_store_index(self, layer_idx: int, page_idx: int, store_index: int):
    """Update the store index of a page.
    Args:
      layer_idx: layer index
      page_idx: page index
      store_index: store index
    """
    assert 0 <= layer_idx < self._num_layers
    assert 0 <= page_idx < self.seqlen
    self._store_indexes[layer_idx, page_idx] = store_index

  def get_page_importance(self, layer_idx: int, page_idx: int) -> float:
    """Get the importance score of a page.
    Args:
      layer_idx: layer index
      page_idx: page index
      offset: offset of the page
    Returns:
      importance score
    """
    assert 0 <= layer_idx < self._num_layers
    assert 0 <= page_idx < self.seqlen
    return self._importance_scores[layer_idx, page_idx].item()
  
  def get_page_importance_layer(self, layer_idx: int) -> torch.Tensor:
    """Get the importance score of a layer.
    Args:
      layer_idx: layer index
    Returns:
      importance scores
    """
    assert 0 <= layer_idx < self._num_layers
    return self._importance_scores[layer_idx, :]
  
  def get_page_store_index(self, layer_idx: int, page_idx: int) -> int:
    """Get the store index of a page.
    Args:
      layer_idx: layer index
      page_idx: page index
    Returns:
      store index
    """
    assert 0 <= layer_idx < self._num_layers
    assert 0 <= page_idx < self.seqlen
    return self._store_indexes[layer_idx, page_idx].item()
  
  def find_least_important_page(self, layer_idx: int, num_pages=1, except_pages=None):
    """Find the least important page on the GPU in a layer.
    Args:
      layer_idx: layer index
    Returns:
      min_page: index of the least important page
      min_kvcache_index: index of the least important page in kv cache
    """
    assert 0 <= layer_idx < self._num_layers
    # Find the least important page on the GPU
    current_gpu_pages = self._store_indexes[layer_idx] != -1 # -1 means offloaded to CPU
    current_gpu_pages = current_gpu_pages.nonzero(as_tuple=True)[0][:-1] # The last page can not be offloaded
    if except_pages is not None:
      current_gpu_pages = current_gpu_pages[~torch.isin(current_gpu_pages, except_pages)]
    if current_gpu_pages.shape[0] <= num_pages:
      return current_gpu_pages, self._store_indexes[layer_idx][current_gpu_pages]
    current_gpu_pages_scores = self._importance_scores[layer_idx][current_gpu_pages]
    min_page = current_gpu_pages[current_gpu_pages_scores.argsort()[:num_pages]]
    min_kvcache_index = self._store_indexes[layer_idx][min_page]
    return min_page, min_kvcache_index
  
  def evict_page(self, layer_idx: int, page_idx: int):
    """Evict a page from the GPU.
    Args:
      layer_idx: layer index
      page_idx: page index
    """
    assert 0 <= layer_idx < self._num_layers
    assert 0 <= page_idx < self.seqlen
    self._store_indexes[layer_idx][page_idx] = -1
    self._importance_scores[layer_idx][page_idx] = 0
    # print(f"Layer {layer_idx} page {page_idx} is evicted from GPU metadata")
  
  def release(self):
    """Release all blocks"""
    self._seqlen = 0
    for layer_idx in range(self.pool.num_layers):
      for block_idx in self._indicies[layer_idx]:
        self.pool.free_block(layer_idx, block_idx)
      self._indicies[layer_idx].clear()
    self._importance_scores.fill_(0)
    self._store_indexes.fill_(-1)
    
class KvCacheCPU:
  """Key-value cache for one sequence on CPU.
  This is used to store the key-value cache blocks that are offloaded to CPU.
  """
  def __init__(
    self,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    max_seq_len: int, # number of blocks in each layer
    page_size,
    dtype: torch.dtype,
  ):
    self._num_layers = num_layers
    self._num_heads = num_heads
    self._head_dim = head_dim
    self._capacity = (max_seq_len + page_size - 1) // page_size
    self._page_size = page_size
    
    self._layout = TensorLayout.NHD
    self._bufs = torch.empty(
      (num_layers, self._capacity, 2, page_size, num_heads, head_dim), # The index of the storage location corresponds to the token location, no need for _free
      dtype=dtype,
      device=torch.device('cpu')
    )

    self._streams = {
      "h2d": torch.cuda.Stream(),  # CPU->GPU
      "d2h": torch.cuda.Stream()   # GPU->CPU
    }

    self._pending_transfers = {
      "h2d": [],  # cpu_idx -> gpu_idx
      "d2h": []   # gpu_idx -> cpu_idx
    }
    
  def __len__(self):
    return self._capacity

