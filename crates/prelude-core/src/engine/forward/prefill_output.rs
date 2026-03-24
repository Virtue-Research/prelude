use super::prefill::PrefillForwardResult;
use super::super::candle_err;
use crate::EngineError;

pub(crate) struct PrefillOutputRows {
    pub item_seq_counts: Vec<usize>,
    rows: std::vec::IntoIter<Vec<f32>>,
}

impl PrefillOutputRows {
    pub fn next_row(&mut self) -> Result<Vec<f32>, EngineError> {
        self.rows
            .next()
            .ok_or_else(|| EngineError::Internal("prefill output rows exhausted early".into()))
    }
}

pub(crate) fn materialize_prefill_output_rows(
    forward_result: Option<PrefillForwardResult>,
) -> Result<Option<PrefillOutputRows>, EngineError> {
    let Some(result) = forward_result else {
        return Ok(None);
    };

    let item_seq_counts = result.item_seq_counts;
    let output_f32 = result
        .output
        .to_dtype(candle_core::DType::F32)
        .map_err(candle_err)?;
    let output_rows = output_f32.to_vec2().map_err(candle_err)?;

    Ok(Some(PrefillOutputRows {
        item_seq_counts,
        rows: output_rows.into_iter(),
    }))
}
