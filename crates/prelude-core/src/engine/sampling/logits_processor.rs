//! Token sampling from logits.
//!
//! `LogitsProcessor` converts a raw logits tensor into a sampled token ID.
//! Supports greedy (argmax), top-k, top-p, and temperature-scaled sampling.

use crate::tensor::{DType, Error, Result, Tensor, D};
use rand::{distr::Distribution, SeedableRng};

#[derive(Clone, PartialEq, Debug)]
pub enum Sampling {
    ArgMax,
    All { temperature: f64 },
    TopK { k: usize, temperature: f64 },
    TopP { p: f64, temperature: f64 },
    TopKThenTopP { k: usize, p: f64, temperature: f64 },
}

pub struct LogitsProcessor {
    rng: rand::rngs::StdRng,
    sampling: Sampling,
    /// Seed kept alongside the live `rng` so `Clone` can reconstruct a
    /// fresh StdRng — `rand::rngs::StdRng` doesn't implement `Clone` in
    /// rand 0.10, but storing the seed lets us hand prefill its own copy
    /// while the sequence state keeps another for decode. The clones
    /// don't share advanced state, so after prefill the decode-side RNG
    /// restarts from the seed — acceptable because determinism was
    /// already per-request, and it's strictly better than silently
    /// breaking non-greedy sampling post-prefill.
    seed: u64,
}

impl Clone for LogitsProcessor {
    fn clone(&self) -> Self {
        Self {
            rng: rand::rngs::StdRng::seed_from_u64(self.seed),
            sampling: self.sampling.clone(),
            seed: self.seed,
        }
    }
}

impl LogitsProcessor {
    pub fn from_sampling(seed: u64, sampling: Sampling) -> Self {
        let rng = rand::rngs::StdRng::seed_from_u64(seed);
        Self { rng, sampling, seed }
    }

    pub fn new(seed: u64, temperature: Option<f64>, top_p: Option<f64>) -> Self {
        let temperature =
            temperature.and_then(|v| if v < 1e-7 { None } else { Some(v) });
        let sampling = match temperature {
            None => Sampling::ArgMax,
            Some(temperature) => match top_p {
                None => Sampling::All { temperature },
                Some(p) => Sampling::TopP { p, temperature },
            },
        };
        Self::from_sampling(seed, sampling)
    }

    fn sample_argmax(&mut self, logits: Tensor) -> Result<u32> {
        logits.argmax(D::Minus1)?.to_scalar::<u32>()
    }

    fn sample_multinomial(&mut self, prs: &Vec<f32>) -> Result<u32> {
        let distr =
            rand::distr::weighted::WeightedIndex::new(prs).map_err(Error::wrap)?;
        let next_token = distr.sample(&mut self.rng) as u32;
        Ok(next_token)
    }

    fn sample_topp(&mut self, prs: &mut Vec<f32>, top_p: f32) -> Result<u32> {
        let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();
        argsort_indices.sort_by(|&i, &j| prs[j].total_cmp(&prs[i]));
        let mut cumsum = 0.;
        for index in &argsort_indices {
            if cumsum >= top_p {
                prs[*index] = 0.0;
            } else {
                cumsum += prs[*index];
            }
        }
        self.sample_multinomial(prs)
    }

    fn sample_topk(&mut self, prs: &mut Vec<f32>, top_k: usize) -> Result<u32> {
        if top_k >= prs.len() {
            self.sample_multinomial(prs)
        } else {
            let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();
            let (indices, _, _) = argsort_indices
                .select_nth_unstable_by(top_k, |&i, &j| prs[j].total_cmp(&prs[i]));
            let prs = indices.iter().map(|&i| prs[i]).collect::<Vec<_>>();
            let index = self.sample_multinomial(&prs)?;
            Ok(indices[index as usize] as u32)
        }
    }

    fn sample_topk_topp(
        &mut self,
        prs: &mut Vec<f32>,
        top_k: usize,
        top_p: f32,
    ) -> Result<u32> {
        if top_k >= prs.len() {
            self.sample_topp(prs, top_p)
        } else {
            let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();
            let (indices, _, _) = argsort_indices
                .select_nth_unstable_by(top_k, |&i, &j| prs[j].total_cmp(&prs[i]));
            let mut prs = indices.iter().map(|&i| prs[i]).collect::<Vec<_>>();
            let sum_p = prs.iter().sum::<f32>();
            let index = if top_p <= 0.0 || top_p >= sum_p {
                self.sample_multinomial(&prs)?
            } else {
                self.sample_topp(&mut prs, top_p)?
            };
            Ok(indices[index as usize] as u32)
        }
    }

    pub fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        self.sample_f(logits, |_| {})
    }

    pub fn sample_f(
        &mut self,
        logits: &Tensor,
        f: impl FnOnce(&mut [f32]),
    ) -> Result<u32> {
        let logits = logits.to_dtype(DType::F32)?;
        let prs = |temperature: f64| -> Result<Vec<f32>> {
            let logits = (&logits / temperature)?;
            let last_dim = logits.rank() - 1;
            let prs = crate::ops::ops_for(logits.device()).softmax(&logits, last_dim)?;
            let mut prs = prs.to_vec1()?;
            f(&mut prs);
            Ok(prs)
        };

        let next_token = match &self.sampling {
            Sampling::ArgMax => self.sample_argmax(logits)?,
            Sampling::All { temperature } => {
                let prs = prs(*temperature)?;
                self.sample_multinomial(&prs)?
            }
            Sampling::TopP { p, temperature } => {
                let mut prs = prs(*temperature)?;
                if *p <= 0.0 || *p >= 1.0 {
                    self.sample_multinomial(&prs)?
                } else {
                    self.sample_topp(&mut prs, *p as f32)?
                }
            }
            Sampling::TopK { k, temperature } => {
                let mut prs = prs(*temperature)?;
                self.sample_topk(&mut prs, *k)?
            }
            Sampling::TopKThenTopP { k, p, temperature } => {
                let mut prs = prs(*temperature)?;
                self.sample_topk_topp(&mut prs, *k, *p as f32)?
            }
        };
        Ok(next_token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Device, Tensor};

    #[test]
    fn argmax_selects_highest() {
        let mut proc = LogitsProcessor::from_sampling(42, Sampling::ArgMax);
        let logits = Tensor::from_vec(
            vec![0.1f32, 0.2, 0.9, 0.3],
            (4,),
            &Device::Cpu,
        ).unwrap();
        let token = proc.sample(&logits).unwrap();
        assert_eq!(token, 2);
    }

    #[test]
    fn argmax_deterministic() {
        let logits = Tensor::from_vec(
            vec![0.5f32, 0.1, 0.8, 0.3],
            (4,),
            &Device::Cpu,
        ).unwrap();

        let t1 = LogitsProcessor::from_sampling(1, Sampling::ArgMax)
            .sample(&logits).unwrap();
        let t2 = LogitsProcessor::from_sampling(99, Sampling::ArgMax)
            .sample(&logits).unwrap();
        assert_eq!(t1, t2); // argmax is seed-independent
    }

    #[test]
    fn from_sampling_constructors() {
        let proc1 = LogitsProcessor::new(42, None, None);
        assert_eq!(proc1.sampling, Sampling::ArgMax);

        let proc2 = LogitsProcessor::new(42, Some(0.8), None);
        assert_eq!(proc2.sampling, Sampling::All { temperature: 0.8 });

        let proc3 = LogitsProcessor::new(42, Some(0.8), Some(0.9));
        assert_eq!(proc3.sampling, Sampling::TopP { p: 0.9, temperature: 0.8 });

        // near-zero temperature → argmax
        let proc4 = LogitsProcessor::new(42, Some(1e-10), None);
        assert_eq!(proc4.sampling, Sampling::ArgMax);
    }

    #[test]
    fn topk_stays_in_range() {
        let mut proc = LogitsProcessor::from_sampling(42, Sampling::TopK {
            k: 2, temperature: 1.0,
        });
        let logits = Tensor::from_vec(
            vec![1.0f32, 0.5, 0.1, 0.01],
            (4,),
            &Device::Cpu,
        ).unwrap();
        // Sample many times — result should always be 0 or 1 (top-2)
        for _ in 0..20 {
            let token = proc.sample(&logits).unwrap();
            assert!(token <= 1, "top-2 should only select token 0 or 1, got {token}");
        }
    }
}
