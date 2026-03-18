use prelude_core::{
    ChatCompletionLogprobContent, ChatCompletionLogprobs, ChatCompletionTopLogprob,
    CompletionLogprobs, TokenLogprobInfo,
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
