// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:08:30 on Sun, Aug 27, 2023
//
// Description: tester

#ifndef __FLASH_ATTENTION_INFERENCE_TESTER_H__
#define __FLASH_ATTENTION_INFERENCE_TESTER_H__

#include "cuda_timer.h"
#include "tensor.h"

class Tester {
public:
    explicit Tester(size_t batch = 2, size_t seq_q = 256, size_t seq_k = 256, size_t head_q = 32, size_t head_k = 32,
                    size_t dim = 128, bool is_causal = true, int num_splits = 0, cudaStream_t stream = nullptr,
                    cudaDeviceProp *dev_prop = nullptr, bool is_alibi = false, bool is_hybrid = false,
                    size_t prefill_fraction = 0, size_t warmup_iterations = 1, size_t profiling_iterations = 10,
                    size_t sleep_duration = 100, bool enable_check = false)
        : m_batch(batch),
          m_seq_q(seq_q),
          m_seq_k(seq_k),
          m_head_q(head_q),
          m_head_k(head_k),
          m_dim(dim),
          m_is_causal(is_causal),
          m_num_splits(num_splits),
          m_stream(stream),
          m_dev_prop(dev_prop),
          m_is_alibi(is_alibi),
          m_is_hybrid(is_hybrid),
          m_prefill_fraction(prefill_fraction),
          m_warmup_iterations(warmup_iterations),
          m_profiling_iterations(profiling_iterations),
          m_sleep_duration(sleep_duration),
          m_enable_check(enable_check) {
        FAI_CHECK_GT(m_batch, 0);
        FAI_CHECK_GT(m_seq_q, 0);
        FAI_CHECK_GT(m_seq_k, 0);
        FAI_CHECK_GT(m_head_q, 0);
        FAI_CHECK_GT(m_head_k, 0);
        FAI_CHECK_EQ(m_head_q % m_head_k, 0);
        FAI_CHECK_GT(m_dim, 0);
        FAI_CHECK_LE(m_dim, 256);
        FAI_CHECK(m_dev_prop);
        if (m_is_hybrid) {
            FAI_CHECK_LE(m_prefill_fraction, 100);
            FAI_CHECK_EQ((m_batch * m_prefill_fraction) % 100, 0);
        }
        FAI_CHECK_GT(m_warmup_iterations, 0);
        FAI_CHECK_GT(m_profiling_iterations, 0);
        FAI_CHECK_GT(m_sleep_duration, 0);

        if (m_is_hybrid) {
            m_prefill_batch = (m_batch * m_prefill_fraction) / 100;
            m_decoding_batch = (m_batch * (100 - m_prefill_fraction)) / 100;
            FAI_CHECK_EQ(m_batch, m_prefill_batch + m_decoding_batch);
            m_total_q = m_prefill_batch * m_seq_q + m_decoding_batch;
        } else {
            m_total_q = m_batch * m_seq_q;
        }
        m_total_k = m_batch * m_seq_k;

        m_Q = new Tensor<cutlass::half_t>({m_total_q, m_head_q, m_dim}, "Tensor Q");
        FAI_CHECK(m_Q);
        m_K = new Tensor<cutlass::half_t>({m_total_k, m_head_k, m_dim}, "Tensor K");
        FAI_CHECK(m_K);
        m_V = new Tensor<cutlass::half_t>({m_total_k, m_head_k, m_dim}, "Tensor V");
        FAI_CHECK(m_V);
        m_O = new Tensor<cutlass::half_t>({m_total_q, m_head_q, m_dim}, "Tensor O");
        FAI_CHECK(m_O);
        m_base = new Tensor<cutlass::half_t>({m_total_q, m_head_q, m_dim}, "Tensor Base");
        FAI_CHECK(m_base);

        get_cu_seq(m_cu_seq_q, m_batch, m_seq_q, true, m_is_hybrid, m_prefill_batch, m_decoding_batch);
        // print_vector(m_cu_seq_q, "cu_seq_q");
        FAI_CHECK_CUDART_ERROR(cudaMalloc((void **)&m_cu_seq_q_dev, m_cu_seq_q.size() * sizeof(int)));
        FAI_CHECK(m_cu_seq_q_dev);
        FAI_CHECK_CUDART_ERROR(
            cudaMemcpy(m_cu_seq_q_dev, m_cu_seq_q.data(), m_cu_seq_q.size() * sizeof(int), cudaMemcpyHostToDevice));

        get_cu_seq(m_cu_seq_k, m_batch, m_seq_k);
        // print_vector(m_cu_seq_k, "cu_seq_k");
        FAI_CHECK_CUDART_ERROR(cudaMalloc((void **)&m_cu_seq_k_dev, m_cu_seq_k.size() * sizeof(int)));
        FAI_CHECK(m_cu_seq_k_dev);
        FAI_CHECK_CUDART_ERROR(
            cudaMemcpy(m_cu_seq_k_dev, m_cu_seq_k.data(), m_cu_seq_k.size() * sizeof(int), cudaMemcpyHostToDevice));

        if (m_enable_check) {
            clock_t start = clock();
            mha_cpu(m_Q, m_K, m_V, m_base, m_cu_seq_q.data(), m_cu_seq_k.data(), m_batch, m_seq_q, m_seq_k, m_is_causal,
                    m_is_alibi);
            clock_t end = clock();
            FLOG("MHA CPU use: %.3f ms", static_cast<double>(end - start) / (CLOCKS_PER_SEC * 1e-3));
        }
    }

    ~Tester() {
        if (m_Q) {
            delete m_Q;
            m_Q = nullptr;
        }

        if (m_K) {
            delete m_K;
            m_K = nullptr;
        }

        if (m_V) {
            delete m_V;
            m_V = nullptr;
        }

        if (m_O) {
            delete m_O;
            m_O = nullptr;
        }

        if (m_base) {
            delete m_base;
            m_base = nullptr;
        }

        if (m_cu_seq_q_dev) {
            FAI_CHECK_CUDART_ERROR(cudaFree((void *)m_cu_seq_q_dev));
            m_cu_seq_q_dev = nullptr;
        }

        if (m_cu_seq_k_dev) {
            FAI_CHECK_CUDART_ERROR(cudaFree((void *)m_cu_seq_k_dev));
            m_cu_seq_k_dev = nullptr;
        }
    }

    template <typename Func>
    void evaluate(Func &&mha, const std::string &name) {
        FLOG("----------------- Evaluating %s -----------------", name.c_str());
        usleep(m_sleep_duration * 1000);
        m_O->tearUp(m_base);

        // warm up
        m_cuda_timer.start();
        for (size_t i = 0; i < m_warmup_iterations; ++i) {
            mha(m_Q, m_K, m_V, m_O, m_cu_seq_q_dev, m_cu_seq_k_dev, m_batch, m_seq_q, m_seq_k, m_is_causal,
                m_num_splits, m_stream, m_dev_prop, m_is_alibi);
        }
        m_warmup_time = static_cast<double>(m_cuda_timer.end()) / static_cast<double>(m_warmup_iterations);
        FLOG("Warm up time: %.3f ms", m_warmup_time);

        if (m_enable_check) {
            m_O->moveToHost();
            m_O->checkValue(m_base);
        }

        profile(std::forward<Func>(mha), name);
    }

private:
    void get_cu_seq(std::vector<int> &cu_seq, size_t batch, size_t seq, bool is_seq_q = false, bool is_hybrid = false,
                    size_t prefill_batch = 0, size_t decoding_batch = 0) {
        cu_seq.resize(batch + 1);
        if (is_hybrid && is_seq_q) {
            FAI_CHECK_EQ(batch, prefill_batch + decoding_batch);
            for (size_t i = 0; i < decoding_batch + 1; ++i) {
                cu_seq[i] = i;
            }
            for (size_t i = 1; i < prefill_batch + 1; ++i) {
                cu_seq[decoding_batch + i] = cu_seq[decoding_batch] + i * seq;
            }
        } else {
            for (size_t i = 0; i < cu_seq.size(); ++i) {
                cu_seq[i] = i * seq;
            }
        }
    }

    void mha_cpu(Tensor<cutlass::half_t> *Q, Tensor<cutlass::half_t> *K, Tensor<cutlass::half_t> *V,
                 Tensor<cutlass::half_t> *O, int *cu_seq_q, int *cu_seq_k, size_t batch, size_t max_seq_q,
                 size_t max_seq_k, bool is_causal, bool is_alibi) {
        size_t total_q = Q->getShape()[0];
        size_t head_q = Q->getShape()[1];
        size_t dim = Q->getShape()[2];
        size_t head_k = K->getShape()[1];

        FAI_CHECK_EQ(head_q % head_k, 0);
        const size_t head_ratio = head_q / head_k;

        cutlass::half_t *q_ptr = Q->getHostPtr();
        cutlass::half_t *k_ptr = K->getHostPtr();
        cutlass::half_t *v_ptr = V->getHostPtr();
        cutlass::half_t *o_ptr = O->getHostPtr();

        // S = Q * K^T
        Tensor<float> *S = new Tensor<float>({total_q, head_q, max_seq_k}, "Tensor S");
        FAI_CHECK(S);
        float *s_ptr = S->getHostPtr();
        for (size_t b = 0; b < batch; ++b) {
            size_t seq_q = cu_seq_q[b + 1] - cu_seq_q[b];
            size_t seq_k = cu_seq_k[b + 1] - cu_seq_k[b];
            for (size_t h = 0; h < head_q; ++h) {
                for (size_t sq = 0; sq < seq_q; ++sq) {
                    for (size_t sk = 0; sk < seq_k; ++sk) {
                        float tmp = 0.0;
                        for (size_t d = 0; d < dim; ++d) {
                            tmp +=
                                static_cast<cutlass::half_t>(
                                    q_ptr[cu_seq_q[b] * (head_q * dim) + sq * (head_q * dim) + h * dim + d]) *
                                static_cast<cutlass::half_t>(k_ptr[cu_seq_k[b] * (head_k * dim) + sk * (head_k * dim) +
                                                                   (h / head_ratio) * dim + d]);
                        }
                        s_ptr[cu_seq_q[b] * (head_q * seq_k) + sq * (head_q * seq_k) + h * seq_k + sk] = tmp;
                    }
                }
            }
        }

        // P = Softmax(S)
        Tensor<cutlass::half_t> *P = new Tensor<cutlass::half_t>({total_q, head_q, max_seq_k}, "Tensor P");
        FAI_CHECK(P);
        cutlass::half_t *p_ptr = P->getHostPtr();
        float scale = 1.0 / std::sqrt(dim);
        for (size_t b = 0; b < batch; ++b) {
            size_t seq_q = cu_seq_q[b + 1] - cu_seq_q[b];
            size_t seq_k = cu_seq_k[b + 1] - cu_seq_k[b];
            size_t row_shift = seq_k - seq_q;
            for (size_t h = 0; h < head_q; ++h) {
                float h_slope = is_alibi ? (1.0 / exp2(8.0 * (h + 1) / head_q)) : 0.0;
                for (size_t sq = 0; sq < seq_q; ++sq) {
                    size_t col_limit = is_causal ? std::min(seq_k, sq + row_shift + 1) : seq_k;

                    // Max(S)
                    std::vector<float> tmp_s(seq_k, 0.0);
                    float max_s = -std::numeric_limits<float>::max();
                    for (size_t sk = 0; sk < col_limit; ++sk) {
                        tmp_s[sk] =
                            s_ptr[cu_seq_q[b] * (head_q * seq_k) + sq * (head_q * seq_k) + h * seq_k + sk] * scale;
                        if (is_alibi && sk < sq + row_shift) {
                            tmp_s[sk] +=
                                (h_slope * (static_cast<int>(sk) - static_cast<int>(sq) - static_cast<int>(row_shift)));
                        }
                        max_s = std::max(max_s, tmp_s[sk]);
                    }

                    // Sum(S)
                    float sum_s = 0.0;
                    for (size_t sk = 0; sk < col_limit; ++sk) {
                        tmp_s[sk] = std::exp(tmp_s[sk] - max_s);
                        sum_s += tmp_s[sk];
                    }

                    // Softmax(S)
                    for (size_t sk = 0; sk < col_limit; ++sk) {
                        p_ptr[cu_seq_q[b] * (head_q * seq_k) + sq * (head_q * seq_k) + h * seq_k + sk] =
                            static_cast<cutlass::half_t>(tmp_s[sk] / sum_s);
                    }

                    // Causal(S)
                    if (is_causal) {
                        for (size_t sk = col_limit; sk < seq_k; ++sk) {
                            p_ptr[cu_seq_q[b] * (head_q * seq_k) + sq * (head_q * seq_k) + h * seq_k + sk] = 0_hf;
                        }
                    }
                }
            }
        }

        // O = P * V
        for (size_t b = 0; b < batch; ++b) {
            size_t seq_q = cu_seq_q[b + 1] - cu_seq_q[b];
            size_t seq_k = cu_seq_k[b + 1] - cu_seq_k[b];
            for (size_t h = 0; h < head_q; ++h) {
                for (size_t sq = 0; sq < seq_q; ++sq) {
                    for (size_t d = 0; d < dim; ++d) {
                        float tmp = 0.0;
                        for (size_t sk = 0; sk < seq_k; ++sk) {
                            tmp +=
                                static_cast<cutlass::half_t>(
                                    p_ptr[cu_seq_q[b] * (head_q * seq_k) + sq * (head_q * seq_k) + h * seq_k + sk]) *
                                static_cast<cutlass::half_t>(v_ptr[cu_seq_k[b] * (head_k * dim) + sk * (head_k * dim) +
                                                                   (h / head_ratio) * dim + d]);
                        }
                        o_ptr[cu_seq_q[b] * (head_q * dim) + sq * (head_q * dim) + h * dim + d] =
                            static_cast<cutlass::half_t>(tmp);
                    }
                }
            }
        }

        if (S) {
            delete S;
            S = nullptr;
        }

        if (P) {
            delete P;
            P = nullptr;
        }
    }

    template <typename Func>
    void profile(Func &&mha, const std::string &name) {
        m_cuda_timer.start();
        for (size_t i = 0; i < m_profiling_iterations; ++i) {
            mha(m_Q, m_K, m_V, m_O, m_cu_seq_q_dev, m_cu_seq_k_dev, m_batch, m_seq_q, m_seq_k, m_is_causal,
                m_num_splits, m_stream, m_dev_prop, m_is_alibi);
        }
        m_profiling_time = static_cast<double>(m_cuda_timer.end()) / static_cast<double>(m_profiling_iterations);

        if (m_is_hybrid) {
            m_throughput = static_cast<double>((m_prefill_batch * m_seq_q * 2 + m_decoding_batch * 4) * m_seq_k *
                                               m_head_q * m_dim) *
                           1e-12 / (static_cast<double>(m_profiling_time) * 1e-3);
        } else {
            m_throughput = static_cast<double>(m_total_q * m_seq_k * m_head_q * m_dim * 4) * 1e-12 /
                           (static_cast<double>(m_profiling_time) * 1e-3);

            if (m_is_causal) {
                m_throughput /= 2;
            }
        }

        if ((std::abs(m_base_time) <= 1e-6) && (std::abs(m_base_throughput) <= 1e-6)) {
            m_base_time = m_profiling_time;
            m_base_throughput = m_throughput;
        }

        FLOG("%s exit, profiling time: %.3f ms (%.2f%%), throughput: %.3f TFLOPS (%.2f%%)", name.c_str(),
             m_profiling_time, m_profiling_time / m_base_time * 100, m_throughput,
             m_throughput / m_base_throughput * 100);
    }

    const size_t m_batch = 2;
    const size_t m_seq_q = 256;
    const size_t m_seq_k = 256;
    const size_t m_head_q = 32;
    const size_t m_head_k = 32;
    const size_t m_dim = 128;
    const bool m_is_causal = true;
    const int m_num_splits = 0;
    const cudaStream_t m_stream = nullptr;
    cudaDeviceProp *m_dev_prop = nullptr;
    const bool m_is_alibi = false;
    const bool m_is_hybrid = false;
    const size_t m_prefill_fraction = 0;
    const size_t m_warmup_iterations = 1;
    const size_t m_profiling_iterations = 10;
    const size_t m_sleep_duration = 100;
    const bool m_enable_check = false;

    // for hybrid
    size_t m_prefill_batch = 0;
    size_t m_decoding_batch = 0;

    size_t m_total_q = 0;
    size_t m_total_k = 0;

    Tensor<cutlass::half_t> *m_Q = nullptr;  // total_q * head_q * dim
    Tensor<cutlass::half_t> *m_K = nullptr;  // total_k * head_k * dim
    Tensor<cutlass::half_t> *m_V = nullptr;  // total_k * head_k * dim
    Tensor<cutlass::half_t> *m_O = nullptr;  // total_q * head_q * dim
    Tensor<cutlass::half_t> *m_base =
        nullptr;  // total_q * head_q * dim, base result, init tensor O before each attention

    std::vector<int> m_cu_seq_q;
    int *m_cu_seq_q_dev = nullptr;
    std::vector<int> m_cu_seq_k;
    int *m_cu_seq_k_dev = nullptr;

    CudaTimer m_cuda_timer;

    double m_warmup_time = 0.0;
    double m_profiling_time = 0.0;
    double m_throughput = 0.0;
    double m_base_time = 0.0;        // flash attn op
    double m_base_throughput = 0.0;  // flash attn op

    FAI_DISALLOW_COPY_AND_ASSIGN(Tester);
};

#endif  // __FLASH_ATTENTION_INFERENCE_TESTER_H__
