from quest.utils.decode_wrapper import BatchDecodeWithPagedKVCacheWrapper
from quest.utils.kv_cache import KvCache, KvMetadataCache, KvCacheCPU
from quest.utils.utils import TensorLayout

import torch

class InferenceController:
    def __init__(
        self,
        num_layers,
        num_heads,
        num_key_value_heads,
        head_dim,
        page_size,
        page_budget, # Real page budget including the last page
        max_seq_len: int | list[int], # Real max for allocating kv / metadata of each layer
        dtype,
        device,
        quest_skip_layer = 2, # Skip first two layers
        topp=None, # For top-p filtering, select KV pages with the sum of attention score ratio > p
        max_seq_len_cpu: int = 0, # For allocating kv / metadata of each layer on CPU
        max_kvmetadata_len: int = 0, # For allocating metadata of each layer
    ):
        if isinstance(max_seq_len, int):
            max_seq_len = [max_seq_len] * num_layers
        assert len(max_seq_len) == num_layers, "max_seq_len should be a list with length equal to num_layers"
        assert all(m > 0 for m in max_seq_len), "max_seq_len should be a list with positive integers"
        assert page_size > 0, "page_size should be a positive integer"
        assert page_budget > 0, "page_budget should be a positive integer"
        if max_seq_len_cpu > 0:
            assert max_seq_len_cpu >= max(max_seq_len), "max_seq_len_cpu should be greater than max_seq_len"
        if max_kvmetadata_len <= 0:
            max_kvmetadata_len = max(max(max_seq_len), max_seq_len_cpu)
        
        self.kv_cache = KvCache(
            num_layers=num_layers,
            num_heads=num_key_value_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            page_size=page_size,
            dtype=dtype,
            device=device
        )
        self.metadata_cache = KvMetadataCache(
            num_layers=num_layers,
            num_heads=num_key_value_heads,
            head_dim=head_dim,
            max_seq_len=max_kvmetadata_len,
            page_size=page_size,
            dtype=dtype,
            device=device
        )
        if max_seq_len_cpu > 0:
            self.kv_cache_cpu = KvCacheCPU(
                num_layers=num_layers,
                num_heads=num_key_value_heads,
                head_dim=head_dim,
                max_seq_len=max_seq_len_cpu,
                page_size=page_size,
                dtype=dtype,
            )
        else:
            self.kv_cache_cpu = None

        self.layout = TensorLayout.NHD # Arbitrarily choose NHD. 
        self.device = device
        self.dtype = dtype

        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.num_layers = num_layers

        self._page_budget = page_budget
        self._decode_handler = BatchDecodeWithPagedKVCacheWrapper(kv_layout="NHD")
        self.quest_skip_layer = quest_skip_layer
        self.topp = topp # For top-p filtering, select KV pages with the sum of attention score ratio > p

        self.max_seq_len = max_seq_len
        self.max_seq_len_cpu = max_seq_len_cpu
        self.max_kvmetadata_len = max_kvmetadata_len

        self.kv_indices_with_last = None
        self.kv_indices_without_last = None
        self.metadata_indices = None
        self.kv_last_page_idx = None # For decoding self-attention
        self.metadata_last_page_idx = None

        self.kv_indptr_for_append = None
        self.metadata_indptr_for_append = None
        self.kv_indptr_for_approx_decode = None

        self.inference_page_budget = None

        self.topk_dout_buffer = None
        self.topk_dindices_buffer = None
        self.topp_num = None # For top-p filtering, the number of pages selected
        self.topk_buf = None
    
    # Used for controlling the number of pages
    # Here we skip first two layers by manipulating this.
    def set_page_budget(self, page_budget: int):
        self._page_budget = page_budget

    # Called once per forwarding in all layers
    # Adjust the metadata for paged_kv
    def prepare_metadata(self, seq_len: int):
        # Allocate entry for tokens
        appended_new_pages = self.kv_cache.append_seq(seq_len)
        # Allocate entry for metadata
        # Check if all values in appended_new_pages are the same
        if len(appended_new_pages) > 0:
            first_value = appended_new_pages[0]
            assert all(x == first_value for x in appended_new_pages), "All values in appended_new_pages should be the same"
        _ = self.metadata_cache.append_seq(appended_new_pages[0])
    
    # Prepare metadata used for inference under certain PAGE_BUDGET
    # Called multiple times for layer sensitivity
    def begin_forward(self, seq_len: int, updateTensor: bool = True):
        # Allocate tensor in advance
        # This is used for append kernels, which need original indices
        if updateTensor:
            self.kv_indptr_for_append = torch.tensor([[0, len(self.kv_cache.indicies(layer_idx))] for layer_idx in range(self.num_layers)], dtype=torch.int32, device=self.device)
            self.metadata_indptr_for_append = torch.tensor([[0, len(self.metadata_cache.indicies(layer_idx))] for layer_idx in range(self.num_layers)], dtype=torch.int32, device=self.device)
            self.kv_last_page_idx = [self.kv_cache.indicies(layer_idx)[-1] for layer_idx in range(self.num_layers)] # The last page index is always the newly generated page
            self.metadata_last_page_idx = [self.metadata_cache.indicies(layer_idx)[-1] for layer_idx in range(self.num_layers)]

        if seq_len > 1:
            # prefill requests
            # append_kv_cache_prefill and prefill_with_paged_kv_cache
            if updateTensor:
                self.kv_indices_with_last = [torch.tensor(self.kv_cache.indicies(layer_idx), dtype=torch.int32, device=self.device) for layer_idx in range(self.num_layers)]
                self.metadata_indices = [torch.tensor(self.metadata_cache.indicies(layer_idx), dtype=torch.int32, device=self.device) for layer_idx in range(self.num_layers)]
        else:
            # decode requests
            # append_kv_cache_decode, estimate_attn_score, topk_filtering
            cur_page_nums = [len(self.kv_cache.indicies(layer_idx)) for layer_idx in range(self.num_layers)] # 当前 GPU 上 KV Cache 的 page 数量
            assert all(x > 1 for x in cur_page_nums), "The number of pages in KV Cache should be greater than 1 for decoding" # at least two pages for excluding last page

            if updateTensor:
                # used for appending
                self.kv_indices_with_last = [torch.tensor(self.kv_cache.indicies(layer_idx), dtype=torch.int32, device=self.device) for layer_idx in range(self.num_layers)]

                # Only used for top-k filtering (because we manully exclude the last page) as input index
                self.kv_indices_without_last = [torch.tensor(self.kv_cache.indicies(layer_idx)[:-1], dtype=torch.int32, device=self.device).repeat(self.num_heads, 1) for layer_idx in range(self.num_layers)]

                # used for estimate
                self.metadata_indices = [torch.tensor(self.metadata_cache.indicies(layer_idx), dtype=torch.int32, device=self.device) for layer_idx in range(self.num_layers)]

            # used as page_budget for topk and approx kernel
            self.inference_page_budget = min(self._page_budget, min(cur_page_nums))

            # Exclude the last page for decoding
            self.kv_indptr_for_approx_decode = torch.tensor([0, self.inference_page_budget - 1], dtype=torch.int32, device=self.device)

            # Allocate buffer for top-k filtering
            self.topk_dout_buffer = torch.zeros((self.num_heads, self.inference_page_budget - 1), dtype=self.dtype, device=self.device)
            self.topk_dindices_buffer = torch.zeros((self.num_heads, self.inference_page_budget - 1), dtype=torch.int32, device=self.device)
            self.topp_num = torch.zeros((self.num_heads,), dtype=torch.int32, device=self.device)
            self.topk_buf = torch.zeros((self.num_heads, 8192 * 2 * (2+4) // 2 // 48), dtype=self.dtype, device=self.device)
            self.kv_cache_indices_for_topk = torch.tensor(range(self.metadata_cache.seqlen - 1), dtype=torch.int32, device=self.device).repeat(self.num_heads, 1)

            self._decode_handler.begin_forward(
                self.kv_indptr_for_approx_decode,
                self.num_heads,
                self.num_key_value_heads,
                self.head_dim,
                self.page_size,
                self.dtype
            )
    
    # Used for releasing resources
    # Free memory in CUDA side
    # called multiple times for layer sensitivity
    def end_forward(self):
        self._decode_handler.end_forward()
    
    def need_estimate(self, layer_idx:int) -> bool:
        if self.inference_page_budget is None:
            return False
        
        if self.topp is not None and layer_idx >= self.quest_skip_layer:
            return True

        cur_page_nums = len(self.kv_cache.indicies(layer_idx))
        return cur_page_nums > self.inference_page_budget
    
    def using_topp(self) -> bool:
        # if self.topp is not None, we need to do top-p filtering
        if self.topp is not None and self.topp > 0 and self.topp < 1:
            return True
        return False
    
    def clean_states(self):
        self.kv_cache.release()
        self.metadata_cache.release()

    def evict_pages(self, layer_idx: int, num_pages: int, except_pages: torch.Tensor=None) -> int:
        # Evict pages from the cache
        self.sync_d2h()
        min_page, min_kvcache_index = self.metadata_cache.find_least_important_page(layer_idx, num_pages, except_pages)
        for i in range(min_page.size(0)):
            self.metadata_cache.evict_page(layer_idx, min_page[i])
            self.kv_cache.evict_page(layer_idx, min_kvcache_index[i])
        return min_page.size(0)

    def async_move_to_cpu(self, layer_idx: int, page_idx: int) -> int:
        """Move a page from GPU to CPU.
        Args:
        layer_idx: layer index
        page_idx: page index
        Returns:
        page index on CPU
        """
        assert 0 <= layer_idx < self.num_layers
        assert 0 <= page_idx < self.max_kvmetadata_len
        kv_cache_idx = self.metadata_cache.get_page_store_index(layer_idx, page_idx)
        assert kv_cache_idx >= 0, "The page index is not valid"

        self.kv_cache_cpu._pending_transfers["d2h"].append((layer_idx, page_idx, kv_cache_idx))
        # Move the page from GPU to CPU
        with torch.cuda.stream(self.kv_cache_cpu._streams["d2h"]):
            self.kv_cache_cpu._bufs[layer_idx][page_idx].copy_(
                self.kv_cache.pool.buf[layer_idx][kv_cache_idx],
                non_blocking=True
            )
        return page_idx
    
    def async_move_to_gpu(self, layer_idx: int, page_idx: int) -> int:
        """Move a page from CPU to GPU.
        Args:
        layer_idx: layer index
        page_idx: page index
        Returns:
        page index on GPU
        """ 
        assert 0 <= layer_idx < self.num_layers
        assert 0 <= page_idx < self.max_kvmetadata_len
        kv_cache_idx = self.metadata_cache.get_page_store_index(layer_idx, page_idx)
        assert kv_cache_idx >= 0, "The page index is not valid"

        self.kv_cache_cpu._pending_transfers["h2d"].append((layer_idx, page_idx, kv_cache_idx))
        # Move the page from CPU to GPU
        with torch.cuda.stream(self.kv_cache_cpu._streams["h2d"]):
            self.kv_cache.pool.buf[layer_idx][kv_cache_idx].copy_(
                self.kv_cache_cpu._bufs[layer_idx][page_idx],
                non_blocking=True
            )
        return page_idx
    
    def sync_h2d(self):
        """Synchronize the transfer from CPU to GPU."""
        self.kv_cache_cpu._streams["h2d"].synchronize()
        self.kv_cache_cpu._pending_transfers["h2d"].clear()

    def sync_d2h(self):
        """Synchronize the transfer from GPU to CPU."""
        self.kv_cache_cpu._streams["d2h"].synchronize()
        self.kv_cache_cpu._pending_transfers["d2h"].clear()

    def is_transfer_complete(self, direction: str) -> bool:
        """Check if the transfer is complete."""
        if direction == "h2d":
            stream = self.kv_cache_cpu._streams["h2d"]
        elif direction == "d2h":
            stream = self.kv_cache_cpu._streams["d2h"]
        else:
            raise ValueError("Invalid direction. Use 'h2d' or 'd2h'.")
        return stream.query() == 0
        