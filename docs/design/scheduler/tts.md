# TTS Streaming Scheduler (`scheduler/tts.rs`)

Back to [main design doc](README.md).


Multi-stage pipeline: Thinker (hidden states) → Talker (codec layer-0) → Code Predictor
(remaining RVQ layers) → Code2Wav (waveform). Stages run concurrently with streaming buffers
between them.

### Pipeline Architecture

```rust
// scheduler/tts.rs

struct TtsPipelineScheduler {
    /// Stage 0: Thinker (LLM, produces hidden states)
    thinker: ArScheduler,
    /// Stage 1: Talker (small AR model, produces layer-0 codec codes)
    talker: ArScheduler,
    /// Stage 2: Code Predictor (small AR model, fills remaining RVQ layers)
    code_predictor: ArScheduler,
    /// Stage 3: Code2Wav (non-AR vocoder, produces waveform chunks)
    vocoder_queue: VecDeque<VocoderChunk>,

    /// Streaming buffers between stages
    thinker_to_talker: StreamBuffer,
    talker_to_predictor: StreamBuffer,
    predictor_to_vocoder: StreamBuffer,
}

struct StreamBuffer {
    chunks: VecDeque<Tensor>,
    chunk_size: usize,           // tokens per chunk (e.g., 10)
}
```

### Pipeline Flow

```rust
// scheduler/tts.rs

impl TtsPipelineScheduler {
    /// One pipeline tick. Each stage processes available data.
    fn step(&mut self) -> Option<AudioChunk> {
        // Stage 0: Thinker generates hidden states
        let thinker_plan = self.thinker.step();
        let thinker_result = self.thinker_runner.execute(&thinker_plan);
        self.thinker.update(&thinker_result, &thinker_plan);
        if let Some(hidden) = thinker_result.hidden_states() {
            self.thinker_to_talker.push(hidden);
        }

        // Stage 1: Talker consumes hidden states, produces codec layer-0
        if let Some(hidden_chunk) = self.thinker_to_talker.pop_chunk() {
            let talker_plan = self.talker.step_with_input(&hidden_chunk);
            let talker_result = self.talker_runner.execute(&talker_plan);
            self.talker.update(&talker_result, &talker_plan);
            if let Some(codes) = talker_result.codec_codes() {
                self.talker_to_predictor.push(codes);
            }
        }

        // Stage 2: Code Predictor fills remaining RVQ layers
        if let Some(layer0_codes) = self.talker_to_predictor.pop_chunk() {
            let predictor_plan = self.code_predictor.step_with_input(&layer0_codes);
            let predictor_result = self.predictor_runner.execute(&predictor_plan);
            self.code_predictor.update(&predictor_result, &predictor_plan);
            if let Some(full_codes) = predictor_result.full_codes() {
                self.predictor_to_vocoder.push(full_codes);
            }
        }

        // Stage 3: Vocoder (non-AR, just runs conv1d pipeline)
        if let Some(codes_chunk) = self.predictor_to_vocoder.pop_chunk() {
            let audio = self.vocoder_runner.decode_chunk(&codes_chunk);
            return Some(audio);
        }

        None
    }
}
```

Key points:
- **Each AR stage reuses `ArScheduler`.** Thinker and talker are AR models with KV cache —
  they use the same scheduler, just with different models.
- **Code2Wav is not AR.** It's a pure convolutional pipeline, no KV cache, no scheduling decisions.
  Just feed in codec codes, get waveform out.
- **Streaming via chunk buffers.** Each stage produces output in chunks. The next stage consumes
  chunks when available. This enables first-audio latency in ~200ms (thinker emits first hidden
  states → talker produces first codes → vocoder generates first waveform chunk).
- **Pipeline parallelism is temporal.** While vocoder decodes chunk N, talker produces chunk N+1,
  thinker produces chunk N+2. Different stages work on different time slices simultaneously.

