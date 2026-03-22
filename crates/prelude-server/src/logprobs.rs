use std::collections::HashMap;

use prelude_core::{
    ChatCompletionLogprobContent, ChatCompletionLogprobs, ChatCompletionTopLogprob,
    CompletionLogprobs, GenerateResult, PromptLogprobEntry, TokenLogprobInfo,
};

/// Convert engine logprobs → OpenAI `/v1/completions` flat format.
pub fn to_completion_logprobs(
    infos: &[TokenLogprobInfo],
    _output_text: &str,
) -> CompletionLogprobs {
    let mut tokens = Vec::with_capacity(infos.len());
    let mut token_logprobs = Vec::with_capacity(infos.len());
    let mut text_offset = Vec::with_capacity(infos.len());
    let mut top_logprobs = Vec::with_capacity(infos.len());

    let mut offset = 0u32;
    for info in infos {
        tokens.push(info.token.clone());
        token_logprobs.push(info.logprob);
        text_offset.push(offset);
        offset += info.token.len() as u32;

        let mut map = std::collections::HashMap::new();
        for (_, tok_str, lp) in &info.top_logprobs {
            map.insert(tok_str.clone(), *lp);
        }
        top_logprobs.push(map);
    }

    CompletionLogprobs {
        tokens,
        token_logprobs,
        text_offset,
        top_logprobs,
    }
}

/// Convert a single engine logprob info → OpenAI `/v1/chat/completions` structured format entry.
pub fn to_chat_logprob_content(info: &TokenLogprobInfo) -> ChatCompletionLogprobContent {
    ChatCompletionLogprobContent {
        token: info.token.clone(),
        logprob: info.logprob,
        bytes: Some(info.token.as_bytes().to_vec()),
        top_logprobs: info
            .top_logprobs
            .iter()
            .map(|(_, tok_str, lp)| ChatCompletionTopLogprob {
                token: tok_str.clone(),
                logprob: *lp,
                bytes: Some(tok_str.as_bytes().to_vec()),
            })
            .collect(),
    }
}

/// Convert engine logprobs → OpenAI `/v1/chat/completions` structured format.
pub fn to_chat_logprobs(infos: &[TokenLogprobInfo]) -> ChatCompletionLogprobs {
    ChatCompletionLogprobs {
        content: infos.iter().map(to_chat_logprob_content).collect(),
    }
}

/// Convert prompt token logprobs from GenerateResult into vLLM-compatible response format.
/// Returns (prompt_logprobs, prompt_token_ids) or (None, None) if not requested.
///
/// vLLM format: `prompt_logprobs[0] = None` (no prediction before first token),
/// `prompt_logprobs[i] = {token_id: {logprob, rank, decoded_token}}` for i >= 1.
pub fn to_prompt_logprobs_response(
    result: &GenerateResult,
) -> (
    Option<Vec<Option<HashMap<u32, PromptLogprobEntry>>>>,
    Option<Vec<u32>>,
) {
    let infos = match &result.prompt_token_logprobs {
        Some(infos) if !infos.is_empty() => infos,
        _ => return (None, None),
    };

    // Build prompt_logprobs: [None, {token_id: entry}, {token_id: entry}, ...]
    let mut prompt_lps = Vec::with_capacity(infos.len() + 1);
    prompt_lps.push(None); // First position has no prediction

    let mut prompt_token_ids = Vec::with_capacity(infos.len() + 1);
    // We don't have the first token ID in the logprob infos (it starts from position 1)
    // The first token is the one being predicted at position 0
    // For now, collect from the infos
    for info in infos {
        let mut entry_map = HashMap::new();
        entry_map.insert(
            info.token_id,
            PromptLogprobEntry {
                logprob: info.logprob,
                rank: Some(1), // The actual token's rank
                decoded_token: Some(info.token.clone()),
            },
        );
        // Add top-k entries
        for (tid, tok_str, lp) in &info.top_logprobs {
            if *tid != info.token_id {
                entry_map.insert(
                    *tid,
                    PromptLogprobEntry {
                        logprob: *lp,
                        rank: None,
                        decoded_token: Some(tok_str.clone()),
                    },
                );
            }
        }
        prompt_lps.push(Some(entry_map));
        prompt_token_ids.push(info.token_id);
    }

    (Some(prompt_lps), Some(prompt_token_ids))
}
