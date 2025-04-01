#ifndef DECODE_SELECT_K_CUH
#define DECODE_SELECT_K_CUH

#include "raft/matrix/detail/select_k-inl.cuh"
#include <cuda_runtime.h>
#include <cub/cub.cuh> 

// We select raft::matrix::SelectAlgo by manually profiling on RTX4090.
// Note that seq_len lies in [1024, 8192] (which effectively means [16k, 128k] seq_len)
// K lies in [64, 256] (which is [1k, 4k] token_budget)

// Check: https://docs.rapids.ai/api/raft/nightly/cpp_api/matrix_ordering/#select-k

using namespace raft::matrix::detail::select::radix::impl;

/*!
   * \brief Select Top-k value in a batched tensor
   * \tparam T The data type
   * \tparam idxT The index type
   * \tparam num_heads batch size
   * \param in [batch_size, len] data of tensor
   * \param in_idx [batch_size, len] index of tensor
   * \param len column width
   * \param k number of top-k elements to select
   * \param out [batch_size, k] output data
   * \param out_idx [batch_size, k] output index
   * \param greater whether to select top-k or bottom-k
   */
template <typename T, typename IdxT>
void decode_select_k(const T* in,
					 const IdxT* in_idx,
                int batch_size, // batch_size = num_heads
                char* bufs,
					 IdxT len,
					 IdxT k,
					 T* out,
					 IdxT* out_idx,
					 bool greater = true,
					 raft::matrix::SelectAlgo _algo = raft::matrix::SelectAlgo::kRadix8bits) {
   // Parameters from kRadix8Bits
   constexpr int BitsPerPass = 8;
   constexpr int BlockSize = 512;
   auto kernel = radix_topk_one_block_kernel<T, IdxT, BitsPerPass, BlockSize>;

   int sm_cnt;
   {
     int dev;
      (cudaGetDevice(&dev));
      (cudaDeviceGetAttribute(&sm_cnt, cudaDevAttrMultiProcessorCount, dev));
   }

   const size_t max_chunk_size = calc_chunk_size<T, IdxT, BlockSize>(batch_size, len, sm_cnt, kernel, true);
   // const size_t buf_size = max_chunk_size * buf_len * 2 * (sizeof(T) + sizeof(IdxT));
   // const IdxT buf_len = calc_buf_len<T, IdxT, unsigned>(len);

   for (size_t offset = 0; offset < static_cast<size_t>(batch_size); offset += max_chunk_size) {
      int chunk_size = std::min(max_chunk_size, batch_size - offset);
      kernel<<<chunk_size, BlockSize, 0, nullptr>>>(in + offset * len,
                                                   in_idx ? (in_idx + offset * len) : nullptr,
                                                   len,
                                                   k,
                                                   out + offset * k,
                                                   out_idx + offset * k,
                                                   !greater,
                                                   bufs);
   }
}

// This kernel computes the top-k elements and their indices from the input tensor
// It uses shared memory to store intermediate results and performs a reduction
// to find the top-k elements efficiently
template <typename T, typename IdxT>
__global__ __forceinline__ void compute_top_p_kernel(const T* in,
                              IdxT len,
                              IdxT max_k,
                              const float top_p,
                              T* out,
                              IdxT* out_k) {
   // Parameters from kRadix8Bits
   extern __shared__ char shared_mem[];
   T* s_values = reinterpret_cast<T*>(shared_mem);

   int batch_id = blockIdx.x;
   
   // One block for one batch
   const T* current_in = in + batch_id * len;
   T* current_out = out + batch_id * max_k;
   IdxT* current_out_k = out_k + batch_id;

   // Initialize shared memory
   for (int i = threadIdx.x; i < max_k; i += blockDim.x) {
      s_values[i] = static_cast<T>(current_out[i]);
   }
   __syncthreads();

   // Compute the sum
   typedef cub::BlockReduce<T, 512> BlockReduce;
   __shared__ typename BlockReduce::TempStorage temp_storage;
   T thread_sum = 0;
   for (int i = threadIdx.x; i < len; i += blockDim.x) {
      thread_sum += current_in[i];
   }
   T block_sum = BlockReduce(temp_storage).Sum(thread_sum);
   __syncthreads();

   // Compute the prefix sum
   if (threadIdx.x == 0) {
      T cumsum = 0;
      T threshold = static_cast<T>(top_p) * block_sum;
      IdxT k = max_k; // Default to max_k
      for (IdxT i = 0; i < max_k; ++i) {
         cumsum += s_values[i];
         printf("i=%d, cumsum=%f, threshold=%f\n", i, (float)cumsum, (float)threshold);
         if (cumsum >= threshold) {
            k = i + 1;
            break;
         }
      }
      *current_out_k = k;
   }
}

/*!
   * \brief Select Top-k value in a batched tensor
   * \tparam T The data type
   * \tparam idxT The index type
   * \tparam num_heads batch size
   * \param in [batch_size, len] data of tensor
   * \param in_idx [batch_size, len] index of tensor
   * \param len column width
   * \param bufs Temporary buffer for top-k computation
   * \param k number of top-k elements to select
   * \param top_p the threshold of top-p
   * \param out [batch_size, k] output data
   * \param out_idx [batch_size, k] output index
   * \param out_k [batch_size] Number of selected elements for each batch
   * \param greater whether to select top-k or bottom-k
   */
template <typename T, typename IdxT> 
void decode_select_p(const T* in,
					 const IdxT* in_idx,
                int batch_size, // batch_size = num_heads
                char* bufs,
					 IdxT len,
					 IdxT k,
                const float top_p,
					 T* out,
					 IdxT* out_idx,
                IdxT* out_k,
					 bool greater = true,
					 raft::matrix::SelectAlgo _algo = raft::matrix::SelectAlgo::kRadix8bits) {
   // Parameters from kRadix8Bits
   constexpr int BitsPerPass = 8;
   constexpr int BlockSize = 512;
   auto kernel = radix_topk_one_block_kernel<T, IdxT, BitsPerPass, BlockSize>;
   auto top_p_kernel = compute_top_p_kernel<T, IdxT>;

   int sm_cnt;
   {
     int dev;
      (cudaGetDevice(&dev));
      (cudaDeviceGetAttribute(&sm_cnt, cudaDevAttrMultiProcessorCount, dev));
   }

   const size_t max_chunk_size = calc_chunk_size<T, IdxT, BlockSize>(batch_size, len, sm_cnt, kernel, true);
   // const size_t buf_size = max_chunk_size * buf_len * 2 * (sizeof(T) + sizeof(IdxT));
   // const IdxT buf_len = calc_buf_len<T, IdxT, unsigned>(len);

   size_t shared_mem_size = k * sizeof(T); // Shared memory size
   for (size_t offset = 0; offset < static_cast<size_t>(batch_size); offset += max_chunk_size) {
      int chunk_size = std::min(max_chunk_size, batch_size - offset);
      kernel<<<chunk_size, BlockSize, 0, nullptr>>>(in + offset * len,
                                                   in_idx ? (in_idx + offset * len) : nullptr,
                                                   len,
                                                   k,
                                                   out + offset * k,
                                                   out_idx + offset * k,
                                                   !greater,
                                                   bufs);
      
      // Compute top-p
      // The top-p kernel will be launched with a single block for each batch
      top_p_kernel<<<chunk_size, BlockSize, shared_mem_size, nullptr>>>(
         in + offset * len,
         len,
         k,
         top_p,
         out + offset * k,
         out_k + offset);
   }
   
}
#endif // DECODE_SELECT_K_CUH