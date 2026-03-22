// C wrapper for llama.cpp API — avoids unsafe struct layout assumptions in Rust FFI.

#include "llama.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

// Set OpenMP thread pool size. Must be called before llama_backend_init().
void llama_ffi_set_omp_threads(int n_threads) {
    omp_set_num_threads(n_threads);
}

static int greedy_argmax(const float *logits, int n_vocab) {
    int best = 0;
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > logits[best]) best = i;
    }
    return best;
}

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// Load model with specified GPU layers.
struct llama_model * llama_ffi_load_model(const char *path, int n_gpu_layers) {
    struct llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = n_gpu_layers;
    return llama_model_load_from_file(path, params);
}

// Create context with specified parameters.
struct llama_context * llama_ffi_create_context(
    struct llama_model *model,
    int n_ctx,
    int n_batch,
    int n_threads
) {
    struct llama_context_params params = llama_context_default_params();
    params.n_ctx = n_ctx;
    params.n_batch = n_batch;
    params.n_ubatch = n_batch;
    params.n_threads = n_threads;
    params.n_threads_batch = n_threads;
    params.no_perf = false;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    return llama_init_from_model(model, params);
}

// Simple decode: feed tokens, get logits for last token.
// Returns 0 on success, non-zero on error.
int llama_ffi_decode(
    struct llama_context *ctx,
    const int *tokens,
    int n_tokens,
    float *logits_out,   // [n_vocab] output buffer
    int n_vocab
) {
    // Use batch_get_one for simple sequential decode
    struct llama_batch batch = llama_batch_get_one((llama_token *)tokens, n_tokens);
    int rc = llama_decode(ctx, batch);
    if (rc != 0) return rc;

    // Copy last token's logits
    float *logits = llama_get_logits_ith(ctx, -1);
    if (!logits) return -100;
    memcpy(logits_out, logits, n_vocab * sizeof(float));
    return 0;
}

// Clear KV cache
void llama_ffi_clear_cache(struct llama_context *ctx) {
    llama_memory_t mem = llama_get_memory(ctx);
    if (mem) llama_memory_clear(mem, true);
}

// Generate tokens: prefill + decode loop entirely in C.
// Returns number of tokens generated. Output tokens written to out_tokens[].
// out_logits (if non-NULL) receives logits for the LAST generated token [n_vocab].
int llama_ffi_generate(
    struct llama_context *ctx,
    const struct llama_vocab *vocab,
    const int *prompt_tokens,
    int n_prompt,
    int *out_tokens,      // [max_new] output buffer
    float *out_logits,    // [n_vocab] logits for last token (or NULL)
    int max_new,
    int n_vocab
) {
    // Prefill: feed all prompt tokens
    struct llama_batch batch = llama_batch_get_one((llama_token *)prompt_tokens, n_prompt);
    int rc = llama_decode(ctx, batch);
    if (rc != 0) return -1;

    // Get first output token
    float *logits = llama_get_logits_ith(ctx, -1);
    if (!logits) return -1;

    int best = greedy_argmax(logits, n_vocab);
    out_tokens[0] = best;
    int n_generated = 1;

    // Decode loop
    for (int step = 1; step < max_new; step++) {
        // Check EOS
        if (llama_vocab_is_eog(vocab, best)) break;

        // Single token decode
        llama_token tok = best;
        batch = llama_batch_get_one(&tok, 1);
        rc = llama_decode(ctx, batch);
        if (rc != 0) break;

        logits = llama_get_logits_ith(ctx, -1);
        if (!logits) break;

        best = greedy_argmax(logits, n_vocab);
        out_tokens[n_generated++] = best;
    }

    // Copy last logits if requested
    if (out_logits && logits) {
        memcpy(out_logits, logits, n_vocab * sizeof(float));
    }

    return n_generated;
}

// Profiling version: prints llama.cpp internal timings + per-step wall clock.
void llama_ffi_profile(
    struct llama_context *ctx,
    const struct llama_vocab *vocab,
    const int *prompt_tokens,
    int n_prompt,
    int max_new,
    int n_vocab
) {
    fprintf(stderr, "\n=== llama_ffi_profile: n_prompt=%d, max_new=%d, n_threads=%d ===\n",
            n_prompt, max_new, (int)llama_n_threads(ctx));

    llama_perf_context_reset(ctx);

    // Prefill
    double t0 = get_time_ms();
    struct llama_batch batch = llama_batch_get_one((llama_token *)prompt_tokens, n_prompt);
    int rc = llama_decode(ctx, batch);
    double t_prefill = get_time_ms() - t0;
    fprintf(stderr, "[profile] prefill: %d tokens in %.2f ms (%.1f t/s)\n",
            n_prompt, t_prefill, n_prompt / (t_prefill / 1000.0));
    if (rc != 0) { fprintf(stderr, "[profile] prefill failed: %d\n", rc); return; }

    float *logits = llama_get_logits_ith(ctx, -1);
    if (!logits) return;
    int best = 0;
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > logits[best]) best = i;
    }

    // Decode loop with per-step timing
    double decode_total = 0;
    double decode_min = 1e9, decode_max = 0;
    int n_decoded = 0;

    for (int step = 0; step < max_new - 1; step++) {
        if (llama_vocab_is_eog(vocab, best)) break;

        llama_token tok = best;
        batch = llama_batch_get_one(&tok, 1);

        double t_step_start = get_time_ms();
        rc = llama_decode(ctx, batch);
        double t_step = get_time_ms() - t_step_start;

        if (rc != 0) break;
        decode_total += t_step;
        if (t_step < decode_min) decode_min = t_step;
        if (t_step > decode_max) decode_max = t_step;
        n_decoded++;

        // Print first few steps and periodically
        if (step < 5 || step % 10 == 0) {
            fprintf(stderr, "[profile] decode step %d: %.2f ms\n", step, t_step);
        }

        logits = llama_get_logits_ith(ctx, -1);
        if (!logits) break;
        best = greedy_argmax(logits, n_vocab);
    }

    fprintf(stderr, "[profile] decode: %d tokens in %.2f ms\n", n_decoded, decode_total);
    if (n_decoded > 0) {
        fprintf(stderr, "[profile]   avg=%.2f ms/tok, min=%.2f ms, max=%.2f ms (%.1f t/s)\n",
                decode_total / n_decoded, decode_min, decode_max,
                n_decoded / (decode_total / 1000.0));
    }

    // llama.cpp internal perf counters
    fprintf(stderr, "\n[profile] llama.cpp internal timings:\n");
    llama_perf_context_print(ctx);

    // Clear for next run
    llama_memory_t mem = llama_get_memory(ctx);
    if (mem) llama_memory_clear(mem, true);
}
