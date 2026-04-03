use super::*;

fn tokenize_text(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>, EngineError> {
    tokenizer
        .encode_with_special_tokens(text, true)
        .map_err(|e| EngineError::Internal(format!("tokenization failed: {e}")))
}

pub(crate) fn tokenize_prompt_input(
    tokenizer: &Tokenizer,
    input: &PromptInput,
) -> Result<Vec<u32>, EngineError> {
    match input {
        PromptInput::Text(text) => tokenize_text(tokenizer, text),
        PromptInput::TokenIds(ids) => Ok(ids.clone()),
    }
}

impl Engine {
    pub fn tokenize_batch(
        &self,
        inputs: &ClassificationInputs,
    ) -> Result<(Vec<Vec<u32>>, u32), EngineError> {
        tokenize_batch_inputs(&self.tokenizer, inputs)
    }
}

pub(crate) fn tokenize_batch_inputs(
    tokenizer: &Tokenizer,
    inputs: &ClassificationInputs,
) -> Result<(Vec<Vec<u32>>, u32), EngineError> {
    match inputs {
        ClassificationInputs::Texts(texts) => {
            let mut all_ids = Vec::with_capacity(texts.len());
            let mut total_tokens = 0u32;
            for text in texts {
                let ids = tokenize_text(tokenizer, text)?;
                total_tokens += ids.len() as u32;
                all_ids.push(ids);
            }
            Ok((all_ids, total_tokens))
        }
        ClassificationInputs::TokenIds(ids) => {
            let total: u32 = ids.iter().map(|t| t.len() as u32).sum();
            Ok((ids.clone(), total))
        }
    }
}
