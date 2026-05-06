use super::*;

fn make_seq(id: &str, prompt_len: usize, max_new: u32) -> Sequence {
    Sequence::new(
        id.to_string(),
        vec![1; prompt_len],
        SamplingParams::default(),
        max_new,
        vec![],
        vec![],
        None,
    )
}

fn make_seq_with_prefix_key(
    id: &str,
    prompt_len: usize,
    max_new: u32,
    prefix_cache_key: u64,
) -> Sequence {
    let mut seq = make_seq(id, prompt_len, max_new);
    seq.prefix_cache_key = Some(prefix_cache_key);
    seq
}

fn make_seq_with_tokens_and_prefix_key(
    id: &str,
    tokens: Vec<u32>,
    max_new: u32,
    prefix_cache_key: u64,
) -> Sequence {
    let mut seq = Sequence::new(
        id.to_string(),
        tokens,
        SamplingParams::default(),
        max_new,
        vec![],
        vec![],
        None,
    );
    seq.prefix_cache_key = Some(prefix_cache_key);
    seq
}

fn make_atomic_prefill_seq(id: &str, prompt_len: usize, max_new: u32) -> Sequence {
    let mut seq = make_seq(id, prompt_len, max_new);
    seq.prefill_must_be_atomic = true;
    seq
}

#[test]
fn test_basic_prefill_then_decode() {
    let config = SchedulerConfig {
        max_running_requests: 4,
        max_num_batched_tokens: 1024,
        max_total_tokens: 4096,
        ..Default::default()
    };
    let mut sched = Scheduler::new(config);

    sched.add_request(make_seq("r1", 10, 20));
    assert_eq!(sched.num_waiting(), 1);

    let step = sched.schedule_step().unwrap();
    assert_eq!(step.forward_mode, ForwardMode::Prefill);
    assert_eq!(step.prefill_request_ids, vec!["r1"]);
    assert_eq!(step.prefill_chunk_lens, vec![10]);
    assert!(step.decode_request_ids.is_empty());
    assert_eq!(sched.num_running(), 1);
    assert_eq!(sched.num_waiting(), 0);

    let step = sched.schedule_step().unwrap();
    assert_eq!(step.forward_mode, ForwardMode::Decode);
    assert_eq!(step.decode_request_ids, vec!["r1"]);
    assert!(step.prefill_request_ids.is_empty());
}

#[test]
fn test_max_running_requests_limit() {
    let config = SchedulerConfig {
        max_running_requests: 2,
        max_num_batched_tokens: 4096,
        max_total_tokens: 65536,
        ..Default::default()
    };
    let mut sched = Scheduler::new(config);

    sched.add_request(make_seq("r1", 10, 20));
    sched.add_request(make_seq("r2", 10, 20));
    sched.add_request(make_seq("r3", 10, 20));

    let step = sched.schedule_step().unwrap();
    assert_eq!(step.forward_mode, ForwardMode::Prefill);
    assert_eq!(step.prefill_request_ids.len(), 2);
    assert_eq!(sched.num_waiting(), 1);
    assert_eq!(sched.num_running(), 2);

    let step = sched.schedule_step().unwrap();
    assert_eq!(step.forward_mode, ForwardMode::Decode);
}

#[test]
fn test_token_budget_limits_prefill() {
    let config = SchedulerConfig {
        max_running_requests: 10,
        max_num_batched_tokens: 15,
        max_total_tokens: 65536,
        ..Default::default()
    };
    let mut sched = Scheduler::new(config);

    sched.add_request(make_seq("r1", 10, 20));
    sched.add_request(make_seq("r2", 10, 20));

    let step = sched.schedule_step().unwrap();
    assert_eq!(step.forward_mode, ForwardMode::Prefill);
    // Budget=15, first request takes 10, only 5 left → can't fit second (10)
    assert_eq!(step.prefill_request_ids.len(), 1);
    assert_eq!(sched.num_waiting(), 1);
}

#[test]
fn test_finish_and_drain() {
    let mut sched = Scheduler::new(SchedulerConfig::default());

    sched.add_request(make_seq("r1", 5, 10));
    let _ = sched.schedule_step();

    sched.finish_request("r1", SeqFinishReason::Eos);
    let _ = sched.schedule_step();

    let done = sched.take_finished();
    assert_eq!(done.len(), 1);
    assert_eq!(done[0].request_id, "r1");
    assert_eq!(sched.num_running(), 0);
}

#[test]
fn test_abort() {
    let mut sched = Scheduler::new(SchedulerConfig::default());

    sched.add_request(make_seq("r1", 5, 10));
    let aborted = sched.abort_request("r1");
    assert!(aborted.is_some());
    assert_eq!(sched.num_waiting(), 0);

    assert!(sched.abort_request("r999").is_none());
}

#[test]
fn test_preemption_under_memory_pressure() {
    let config = SchedulerConfig {
        max_running_requests: 10,
        max_num_batched_tokens: 4096,
        max_total_tokens: 50,
        new_token_ratio: 0.0,
        ..Default::default()
    };
    let mut sched = Scheduler::new(config);

    sched.add_request(make_seq("r1", 20, 20));
    let _ = sched.schedule_step();

    for _ in 0..10 {
        sched.on_token_generated("r1", 42);
    }

    sched.add_request(make_seq("r2", 25, 20));
    let step = sched.schedule_step().unwrap();

    assert_eq!(step.forward_mode, ForwardMode::Prefill);
    assert!(step.prefill_request_ids.contains(&"r2".to_string()));
    assert!(
        sched
            .waiting_queue
            .iter()
            .any(|sequence| sequence.request_id == "r1")
    );
}

#[test]
fn test_idle_returns_none() {
    let mut sched = Scheduler::new(SchedulerConfig::default());
    assert!(sched.schedule_step().is_none());
    assert!(!sched.has_work());
}

#[test]
fn test_decode_reservation_cap_is_configurable() {
    let mut sched = Scheduler::new(SchedulerConfig {
        decode_reservation_cap: 16,
        ..Default::default()
    });
    sched.running.push(make_seq("r1", 8, 128));
    sched.running.push(make_seq("r2", 8, 6));

    assert_eq!(sched.estimate_running_future_tokens(), 22);
}

#[test]
fn test_chunked_prefill_emits_mixed_step() {
    let config = SchedulerConfig {
        chunked_prefill: true,
        ..Default::default()
    };
    let mut sched = Scheduler::new(config);
    sched.add_request(make_seq("r1", 8, 20));

    let first = sched.schedule_step().unwrap();
    assert_eq!(first.forward_mode, ForwardMode::Prefill);
    assert_eq!(first.prefill_request_ids, vec!["r1"]);
    assert_eq!(first.prefill_chunk_lens, vec![8]);

    sched.add_request(make_seq("r2", 6, 20));
    let second = sched.schedule_step().unwrap();
    assert_eq!(second.forward_mode, ForwardMode::Mixed);
    assert_eq!(second.prefill_request_ids, vec!["r2"]);
    assert_eq!(second.prefill_chunk_lens, vec![6]);
    assert_eq!(second.decode_request_ids, vec!["r1"]);
}

#[test]
fn test_chunked_prefill_defers_duplicate_uncached_prefix_key() {
    let config = SchedulerConfig {
        chunked_prefill: true,
        max_running_requests: 4,
        max_num_batched_tokens: 1024,
        max_total_tokens: 65536,
        ..Default::default()
    };
    let mut sched = Scheduler::new(config);

    sched.add_request(make_seq_with_prefix_key("r1", 100, 1, 42));
    sched.add_request(make_seq_with_prefix_key("r2", 100, 1, 42));
    sched.add_request(make_seq_with_prefix_key("r3", 100, 1, 7));

    let step = sched.schedule_step().unwrap();
    assert_eq!(step.forward_mode, ForwardMode::Prefill);
    assert_eq!(step.prefill_request_ids, vec!["r1", "r3"]);
    assert_eq!(sched.num_waiting(), 1);
    assert_eq!(
        sched
            .waiting_queue
            .front()
            .map(|seq| seq.request_id.as_str()),
        Some("r2")
    );
}

#[test]
fn test_running_prefill_defers_waiting_duplicate_prefix_key() {
    let config = SchedulerConfig {
        chunked_prefill: true,
        max_running_requests: 4,
        max_num_batched_tokens: 150,
        max_total_tokens: 65536,
        ..Default::default()
    };
    let mut sched = Scheduler::new(config);

    sched.add_request(make_seq_with_prefix_key("r1", 200, 1, 42));

    let first = sched.schedule_step().unwrap();
    assert_eq!(first.prefill_request_ids, vec!["r1"]);
    assert_eq!(first.prefill_chunk_lens, vec![150]);
    assert_eq!(sched.running[0].status, SequenceStatus::Prefilling);

    sched.add_request(make_seq_with_prefix_key("r2", 20, 1, 42));
    sched.add_request(make_seq_with_prefix_key("r3", 20, 1, 7));

    let second = sched.schedule_step().unwrap();
    assert_eq!(second.forward_mode, ForwardMode::Prefill);
    assert_eq!(second.prefill_request_ids, vec!["r1", "r3"]);
    assert_eq!(second.prefill_chunk_lens, vec![50, 20]);
    assert_eq!(sched.num_waiting(), 1);
    assert_eq!(
        sched
            .waiting_queue
            .front()
            .map(|seq| seq.request_id.as_str()),
        Some("r2")
    );
}

#[test]
fn test_shared_prefix_planner_targets_deepest_waiting_boundary() {
    let config = SchedulerConfig {
        chunked_prefill: true,
        block_size: 4,
        ..Default::default()
    };
    let mut sched = Scheduler::new(config);

    let a: Vec<u32> = (0..17).collect();
    let mut b: Vec<u32> = (0..16).collect();
    b.push(100);

    sched.add_request(make_seq_with_tokens_and_prefix_key("r1", a, 1, 1));
    sched.add_request(make_seq_with_tokens_and_prefix_key("r2", b, 1, 2));

    assert_eq!(sched.plan_waiting_shared_prefixes(), 2);
    assert_eq!(sched.waiting_queue[0].prefix_cache_target_len, Some(16));
    assert_eq!(sched.waiting_queue[1].prefix_cache_target_len, Some(16));
    assert_eq!(
        sched.waiting_queue[0].prefix_cache_key,
        sched.waiting_queue[1].prefix_cache_key
    );
}

#[test]
fn test_shared_prefix_target_controls_hybrid_bootstrap_chunk() {
    let config = SchedulerConfig {
        chunked_prefill: true,
        max_running_requests: 4,
        max_num_batched_tokens: 64,
        max_total_tokens: 65536,
        block_size: 4,
        ..Default::default()
    };
    let mut sched = Scheduler::new(config);

    let a: Vec<u32> = (0..17).collect();
    let mut b: Vec<u32> = (0..16).collect();
    b.push(100);
    let mut r1 = make_seq_with_tokens_and_prefix_key("r1", a, 1, 1);
    let mut r2 = make_seq_with_tokens_and_prefix_key("r2", b, 1, 2);
    r1.deltanet_slot = Some(0);
    r2.deltanet_slot = Some(1);
    sched.add_request(r1);
    sched.add_request(r2);

    sched.plan_waiting_shared_prefixes();
    let step = sched.schedule_step().unwrap();

    assert_eq!(step.prefill_request_ids, vec!["r1"]);
    assert_eq!(step.prefill_chunk_lens, vec![16]);
    assert_eq!(sched.num_waiting(), 1);
    assert_eq!(sched.waiting_queue[0].request_id, "r2");
}

#[test]
fn test_shared_prefix_planner_uses_running_leader() {
    let config = SchedulerConfig {
        chunked_prefill: true,
        block_size: 4,
        ..Default::default()
    };
    let mut sched = Scheduler::new(config);

    let leader_tokens: Vec<u32> = (0..17).collect();
    let mut follower_tokens: Vec<u32> = (0..16).collect();
    follower_tokens.push(100);

    let mut leader = make_seq_with_tokens_and_prefix_key("leader", leader_tokens, 1, 1);
    leader.status = SequenceStatus::Prefilling;
    leader.kv_computed_len = 4;
    sched.running.push(leader);
    sched.add_request(make_seq_with_tokens_and_prefix_key(
        "follower",
        follower_tokens,
        1,
        2,
    ));

    assert_eq!(sched.plan_waiting_shared_prefixes(), 1);
    assert_eq!(sched.running[0].prefix_cache_target_len, Some(16));
    assert_eq!(sched.waiting_queue[0].prefix_cache_target_len, Some(16));
}

#[test]
fn test_shared_prefix_planner_extends_partial_waiting_hit() {
    let config = SchedulerConfig {
        chunked_prefill: true,
        block_size: 4,
        ..Default::default()
    };
    let mut sched = Scheduler::new(config);

    let a: Vec<u32> = (0..17).collect();
    let mut b: Vec<u32> = (0..16).collect();
    b.push(100);

    let mut r1 = make_seq_with_tokens_and_prefix_key("r1", a, 1, 1);
    let mut r2 = make_seq_with_tokens_and_prefix_key("r2", b, 1, 2);
    r1.kv_computed_len = 8;
    r2.kv_computed_len = 8;
    r1.block_table = vec![10, 11];
    r2.block_table = vec![10, 11];

    sched.add_request(r1);
    sched.add_request(r2);

    assert_eq!(sched.plan_waiting_shared_prefixes(), 2);
    assert_eq!(sched.waiting_queue[0].prefix_cache_target_len, Some(16));
    assert_eq!(sched.waiting_queue[1].prefix_cache_target_len, Some(16));
}

#[test]
fn test_partial_shared_prefix_target_admits_one_leader() {
    let config = SchedulerConfig {
        chunked_prefill: true,
        max_running_requests: 4,
        max_num_batched_tokens: 64,
        max_total_tokens: 65536,
        block_size: 4,
        ..Default::default()
    };
    let mut sched = Scheduler::new(config);

    let a: Vec<u32> = (0..17).collect();
    let mut b: Vec<u32> = (0..16).collect();
    b.push(100);

    let mut r1 = make_seq_with_tokens_and_prefix_key("r1", a, 1, 1);
    let mut r2 = make_seq_with_tokens_and_prefix_key("r2", b, 1, 2);
    r1.kv_computed_len = 8;
    r2.kv_computed_len = 8;
    r1.block_table = vec![10, 11];
    r2.block_table = vec![10, 11];
    r1.deltanet_slot = Some(0);
    r2.deltanet_slot = Some(1);

    sched.add_request(r1);
    sched.add_request(r2);
    sched.plan_waiting_shared_prefixes();

    let step = sched.schedule_step().unwrap();
    assert_eq!(step.prefill_request_ids, vec!["r1"]);
    assert_eq!(step.prefill_chunk_lens, vec![8]);
    assert_eq!(sched.num_waiting(), 1);
    assert_eq!(sched.waiting_queue[0].request_id, "r2");
}

#[test]
fn test_late_waiting_shared_prefix_updates_running_leader_key() {
    let config = SchedulerConfig {
        chunked_prefill: true,
        max_running_requests: 4,
        max_num_batched_tokens: 64,
        max_total_tokens: 65536,
        block_size: 4,
        ..Default::default()
    };
    let mut sched = Scheduler::new(config);

    let leader_tokens: Vec<u32> = (0..17).collect();
    let mut follower_tokens: Vec<u32> = (0..16).collect();
    follower_tokens.push(100);

    let mut leader = make_seq_with_tokens_and_prefix_key("leader", leader_tokens, 1, 1);
    leader.status = SequenceStatus::Prefilling;
    leader.kv_computed_len = 4;
    leader.deltanet_slot = Some(0);
    sched.running.push(leader);
    let mut follower = make_seq_with_tokens_and_prefix_key("follower", follower_tokens, 1, 2);
    follower.deltanet_slot = Some(1);
    sched.add_request(follower);

    sched.plan_waiting_shared_prefixes();
    assert_eq!(
        sched.running[0].prefix_cache_key,
        sched.waiting_queue[0].prefix_cache_key
    );

    let step = sched.schedule_step().unwrap();
    assert_eq!(step.prefill_request_ids, vec!["leader"]);
    assert_eq!(step.prefill_chunk_lens, vec![12]);
    assert_eq!(sched.num_waiting(), 1);
}

#[test]
fn test_shared_prefix_planner_clears_stale_target() {
    let config = SchedulerConfig {
        chunked_prefill: true,
        block_size: 4,
        ..Default::default()
    };
    let mut sched = Scheduler::new(config);

    let tokens: Vec<u32> = (0..17).collect();
    let mut seq = make_seq_with_tokens_and_prefix_key("r1", tokens, 1, 1);
    seq.prefix_cache_target_len = Some(16);
    sched.add_request(seq);

    assert_eq!(sched.plan_waiting_shared_prefixes(), 0);
    assert_eq!(sched.waiting_queue[0].prefix_cache_target_len, None);
}

// ── Chunked prefill specific tests ───────────────────────────────

#[test]
fn test_large_prefill_is_chunked() {
    let config = SchedulerConfig {
        chunked_prefill: true,
        max_num_batched_tokens: 100,
        max_total_tokens: 65536,
        ..Default::default()
    };
    let mut sched = Scheduler::new(config);
    sched.add_request(make_seq("r1", 250, 20));

    // First step: chunk of 100 tokens
    let step1 = sched.schedule_step().unwrap();
    assert_eq!(step1.forward_mode, ForwardMode::Prefill);
    assert_eq!(step1.prefill_request_ids, vec!["r1"]);
    assert_eq!(step1.prefill_chunk_lens, vec![100]);
    // Request is in running but still Prefilling
    assert_eq!(sched.num_running(), 1);
    assert_eq!(sched.running[0].kv_computed_len, 100);
    assert_eq!(sched.running[0].status, SequenceStatus::Prefilling);

    // Second step: another chunk of 100
    let step2 = sched.schedule_step().unwrap();
    assert_eq!(step2.prefill_chunk_lens, vec![100]);
    assert_eq!(sched.running[0].kv_computed_len, 200);
    assert_eq!(sched.running[0].status, SequenceStatus::Prefilling);

    // Third step: remaining 50
    let step3 = sched.schedule_step().unwrap();
    assert_eq!(step3.prefill_chunk_lens, vec![50]);
    assert_eq!(sched.running[0].kv_computed_len, 250);
    assert_eq!(sched.running[0].status, SequenceStatus::Decoding);

    // Fourth step: decode
    let step4 = sched.schedule_step().unwrap();
    assert_eq!(step4.forward_mode, ForwardMode::Decode);
    assert_eq!(step4.decode_request_ids, vec!["r1"]);
}

#[test]
fn test_chunked_prefill_with_concurrent_decode() {
    let config = SchedulerConfig {
        chunked_prefill: true,
        max_num_batched_tokens: 50,
        max_total_tokens: 65536,
        ..Default::default()
    };
    let mut sched = Scheduler::new(config);

    // First request: short prefill, completes immediately
    sched.add_request(make_seq("r1", 10, 20));
    let step = sched.schedule_step().unwrap();
    assert_eq!(step.prefill_chunk_lens, vec![10]);

    // r1 is now decoding. Add r2 with long prefill.
    sched.add_request(make_seq("r2", 200, 20));

    // Next step: r1 decodes (1 token), r2 gets chunk (budget=50, decode=1, prefill=49)
    let step = sched.schedule_step().unwrap();
    assert_eq!(step.forward_mode, ForwardMode::Mixed);
    assert_eq!(step.decode_request_ids, vec!["r1"]);
    assert_eq!(step.prefill_request_ids, vec!["r2"]);
    assert_eq!(step.prefill_chunk_lens, vec![49]); // 50 - 1 decode
}

#[test]
fn test_deadlock_prevention() {
    // Budget is 10, but request has 100 tokens. Should still make progress.
    let config = SchedulerConfig {
        chunked_prefill: true,
        max_num_batched_tokens: 10,
        max_total_tokens: 65536,
        ..Default::default()
    };
    let mut sched = Scheduler::new(config);
    sched.add_request(make_seq("r1", 100, 20));

    let step = sched.schedule_step().unwrap();
    assert_eq!(step.prefill_chunk_lens, vec![10]);
    assert_eq!(sched.running[0].kv_computed_len, 10);
    // Should NOT deadlock — request makes progress
}

#[test]
fn test_long_prefill_token_threshold() {
    let config = SchedulerConfig {
        chunked_prefill: true,
        max_num_batched_tokens: 8192,
        long_prefill_token_threshold: 512,
        max_total_tokens: 65536,
        ..Default::default()
    };
    let mut sched = Scheduler::new(config);
    sched.add_request(make_seq("r1", 2048, 20));

    // Per-request cap limits chunk to 512 even though budget allows more
    let step = sched.schedule_step().unwrap();
    assert_eq!(step.prefill_chunk_lens, vec![512]);
}

#[test]
fn test_multiple_partial_prefills_share_budget() {
    // Two large prefills. Budget=100.
    // Step 1: r1 admitted (chunk=100), r2 stays waiting (budget exhausted).
    // Step 2: r1 (running, 100 remaining) takes full budget. r2 still waiting.
    // Step 3: r1 done → decode (1 tok), r2 admitted with remaining budget (99 tokens).
    let config = SchedulerConfig {
        chunked_prefill: true,
        max_num_batched_tokens: 100,
        max_total_tokens: 65536,
        ..Default::default()
    };
    let mut sched = Scheduler::new(config);

    sched.add_request(make_seq("r1", 200, 20));
    sched.add_request(make_seq("r2", 200, 20));

    // Step 1: only r1 admitted (budget exhausted after 100)
    let step1 = sched.schedule_step().unwrap();
    assert_eq!(step1.prefill_request_ids, vec!["r1"]);
    assert_eq!(step1.prefill_chunk_lens, vec![100]);
    assert_eq!(sched.num_running(), 1);
    assert_eq!(sched.num_waiting(), 1);

    // Step 2: r1 still prefilling (100 remaining), takes full budget
    let step2 = sched.schedule_step().unwrap();
    assert_eq!(step2.prefill_request_ids, vec!["r1"]);
    assert_eq!(step2.prefill_chunk_lens, vec![100]);
    assert_eq!(sched.running[0].status, SequenceStatus::Decoding);

    // Step 3: r1 decoding (1 tok) + r2 admitted (chunk=99)
    let step3 = sched.schedule_step().unwrap();
    assert_eq!(step3.forward_mode, ForwardMode::Mixed);
    assert_eq!(step3.decode_request_ids, vec!["r1"]);
    assert_eq!(step3.prefill_request_ids, vec!["r2"]);
    assert_eq!(step3.prefill_chunk_lens, vec![99]); // budget 100 - 1 decode
}

#[test]
fn test_last_chunk_transitions_to_decode() {
    // A request with exactly budget-sized prefill should complete in one step
    // and be decodable in the next.
    let config = SchedulerConfig {
        chunked_prefill: true,
        max_num_batched_tokens: 50,
        max_total_tokens: 65536,
        ..Default::default()
    };
    let mut sched = Scheduler::new(config);
    sched.add_request(make_seq("r1", 50, 20));

    let step = sched.schedule_step().unwrap();
    assert_eq!(step.prefill_chunk_lens, vec![50]);
    assert_eq!(sched.running[0].status, SequenceStatus::Decoding);
    assert_eq!(sched.running[0].kv_computed_len, 50);

    // Next step must be decode
    let step2 = sched.schedule_step().unwrap();
    assert_eq!(step2.forward_mode, ForwardMode::Decode);
    assert_eq!(step2.decode_request_ids, vec!["r1"]);
}

#[test]
fn test_preemption_resets_partial_prefill() {
    // Test that rollback_prefill (executor failure) properly resets
    // kv_computed_len for partially-prefilled requests.
    let config = SchedulerConfig {
        chunked_prefill: true,
        max_num_batched_tokens: 50,
        max_total_tokens: 65536,
        ..Default::default()
    };
    let mut sched = Scheduler::new(config);

    sched.add_request(make_seq("r1", 200, 20));
    let step = sched.schedule_step().unwrap();
    assert_eq!(step.prefill_chunk_lens, vec![50]);
    assert_eq!(sched.running[0].kv_computed_len, 50);

    // Simulate executor failure → rollback
    sched.rollback_prefill(&step.prefill_request_ids);

    // r1 should be back in waiting with kv_computed_len=0
    assert_eq!(sched.num_running(), 0);
    assert_eq!(sched.num_waiting(), 1);
    let r1 = &sched.waiting_queue[0];
    assert_eq!(r1.kv_computed_len, 0, "rollback must reset kv_computed_len");
    assert_eq!(r1.status, SequenceStatus::Waiting);

    // r1 can be re-scheduled from scratch
    let step2 = sched.schedule_step().unwrap();
    assert_eq!(step2.prefill_chunk_lens, vec![50]);
    assert_eq!(sched.running[0].kv_computed_len, 50);
}

#[test]
fn test_budget_exactly_equals_prefill_no_chunk() {
    // prefill_len == budget → should complete in one step, no chunking
    let config = SchedulerConfig {
        chunked_prefill: true,
        max_num_batched_tokens: 128,
        max_total_tokens: 65536,
        ..Default::default()
    };
    let mut sched = Scheduler::new(config);
    sched.add_request(make_seq("r1", 128, 20));

    let step = sched.schedule_step().unwrap();
    assert_eq!(step.prefill_chunk_lens, vec![128]);
    assert_eq!(sched.running[0].status, SequenceStatus::Decoding);
}

#[test]
fn test_only_decode_requests_gets_full_budget() {
    let config = SchedulerConfig {
        chunked_prefill: true,
        max_num_batched_tokens: 10,
        max_total_tokens: 65536,
        ..Default::default()
    };
    let mut sched = Scheduler::new(config);

    // Prefill 3 short requests
    sched.add_request(make_seq("r1", 3, 20));
    sched.add_request(make_seq("r2", 3, 20));
    sched.add_request(make_seq("r3", 3, 20));
    let _ = sched.schedule_step(); // prefills all 3 (9 tokens < budget 10)

    // All should be decoding now
    assert!(
        sched
            .running
            .iter()
            .all(|s| s.status == SequenceStatus::Decoding)
    );

    // Decode step: 3 tokens, well within budget
    let step = sched.schedule_step().unwrap();
    assert_eq!(step.forward_mode, ForwardMode::Decode);
    assert_eq!(step.decode_request_ids.len(), 3);
}

#[test]
fn test_new_request_while_partial_prefill_in_running() {
    // r1 is partially prefilled (100 of 200). r2 arrives with 30 tokens.
    // Budget=150: r1 gets remaining 100 first (running priority),
    // r2 gets 30 from leftover budget. Both are prefill (no decode yet).
    let config = SchedulerConfig {
        chunked_prefill: true,
        max_num_batched_tokens: 150,
        max_total_tokens: 65536,
        ..Default::default()
    };
    let mut sched = Scheduler::new(config);

    sched.add_request(make_seq("r1", 200, 20));
    let _ = sched.schedule_step(); // r1 gets 150-token chunk

    // Add r2 while r1 is partially prefilled
    sched.add_request(make_seq("r2", 30, 20));

    let step = sched.schedule_step().unwrap();
    // r1 (running, 50 remaining) + r2 (waiting, 30) = 80 < budget 150
    assert_eq!(step.forward_mode, ForwardMode::Prefill); // both are prefill, no decode
    assert!(step.prefill_request_ids.contains(&"r1".to_string()));
    assert!(step.prefill_request_ids.contains(&"r2".to_string()));

    let r1_idx = step
        .prefill_request_ids
        .iter()
        .position(|id| id == "r1")
        .unwrap();
    let r2_idx = step
        .prefill_request_ids
        .iter()
        .position(|id| id == "r2")
        .unwrap();
    assert_eq!(step.prefill_chunk_lens[r1_idx], 50); // r1's remaining
    assert_eq!(step.prefill_chunk_lens[r2_idx], 30); // r2 fully admitted

    // After this step: r1 is Decoding, r2 is Decoding
    assert!(
        sched
            .running
            .iter()
            .all(|s| s.status == SequenceStatus::Decoding)
    );
}

#[test]
fn test_abort_partial_prefill() {
    let config = SchedulerConfig {
        chunked_prefill: true,
        max_num_batched_tokens: 50,
        max_total_tokens: 65536,
        ..Default::default()
    };
    let mut sched = Scheduler::new(config);

    sched.add_request(make_seq("r1", 200, 20));
    let _ = sched.schedule_step(); // r1 gets 50-token chunk
    assert_eq!(sched.num_running(), 1);
    assert_eq!(sched.running[0].kv_computed_len, 50);

    // Abort partial prefill
    let aborted = sched.abort_request("r1");
    assert!(aborted.is_some());
    assert_eq!(sched.num_running(), 0);
    assert_eq!(sched.num_waiting(), 0);
}

#[test]
fn test_decode_budget_exhaustion_blocks_prefill() {
    // Many decode requests consume all budget, leaving nothing for prefill.
    let config = SchedulerConfig {
        chunked_prefill: true,
        max_num_batched_tokens: 5,
        max_total_tokens: 65536,
        ..Default::default()
    };
    let mut sched = Scheduler::new(config);

    // Create 5 small requests that complete prefill immediately
    for i in 0..5 {
        sched.add_request(make_seq(&format!("r{i}"), 1, 100));
    }
    let _ = sched.schedule_step(); // prefill all 5 (5 tokens = budget)

    // Now 5 requests are decoding. Add r5 to waiting.
    sched.add_request(make_seq("r5", 10, 20));

    let step = sched.schedule_step().unwrap();
    // 5 decode tokens consume entire budget=5 → no room for r5 prefill
    assert_eq!(step.forward_mode, ForwardMode::Decode);
    assert_eq!(step.decode_request_ids.len(), 5);
    assert!(step.prefill_request_ids.is_empty());
    assert_eq!(sched.num_waiting(), 1); // r5 still waiting
}

#[test]
fn test_second_waiting_request_gets_chunked() {
    // Budget=100. First request needs 60 tokens, second needs 80.
    // Second should get chunked to 40 (remaining budget).
    let config = SchedulerConfig {
        chunked_prefill: true,
        max_num_batched_tokens: 100,
        max_total_tokens: 65536,
        ..Default::default()
    };
    let mut sched = Scheduler::new(config);

    sched.add_request(make_seq("r1", 60, 20));
    sched.add_request(make_seq("r2", 80, 20));

    let step = sched.schedule_step().unwrap();
    assert_eq!(step.prefill_request_ids, vec!["r1", "r2"]);
    assert_eq!(step.prefill_chunk_lens[0], 60); // r1 fits fully
    assert_eq!(step.prefill_chunk_lens[1], 40); // r2 gets remaining budget

    // r1 completed prefill, r2 still prefilling
    let r1 = sched.running.iter().find(|s| s.request_id == "r1").unwrap();
    assert_eq!(r1.status, SequenceStatus::Decoding);
    let r2 = sched.running.iter().find(|s| s.request_id == "r2").unwrap();
    assert_eq!(r2.status, SequenceStatus::Prefilling);
    assert_eq!(r2.kv_computed_len, 40);
}

#[test]
fn test_atomic_prefill_is_not_chunked_to_fill_remaining_budget() {
    let config = SchedulerConfig {
        chunked_prefill: true,
        max_num_batched_tokens: 100,
        max_total_tokens: 65536,
        ..Default::default()
    };
    let mut sched = Scheduler::new(config);

    sched.add_request(make_seq("r1", 60, 20));
    sched.add_request(make_atomic_prefill_seq("r2", 80, 0));

    let step = sched.schedule_step().unwrap();
    assert_eq!(step.prefill_request_ids, vec!["r1"]);
    assert_eq!(step.prefill_chunk_lens, vec![60]);
    assert_eq!(sched.num_waiting(), 1);
    assert_eq!(sched.waiting_queue[0].request_id, "r2");
    assert_eq!(sched.waiting_queue[0].kv_computed_len, 0);
}

#[test]
fn test_rollback_prefill_resets_partial() {
    let config = SchedulerConfig {
        chunked_prefill: true,
        max_num_batched_tokens: 50,
        max_total_tokens: 65536,
        ..Default::default()
    };
    let mut sched = Scheduler::new(config);

    sched.add_request(make_seq("r1", 200, 20));
    let step = sched.schedule_step().unwrap();
    assert_eq!(step.prefill_chunk_lens, vec![50]);
    assert_eq!(sched.running[0].kv_computed_len, 50);

    // Simulate executor failure → rollback
    sched.rollback_prefill(&step.prefill_request_ids);
    assert_eq!(sched.num_running(), 0);
    assert_eq!(sched.num_waiting(), 1);
    assert_eq!(sched.waiting_queue[0].kv_computed_len, 0);
    assert_eq!(sched.waiting_queue[0].status, SequenceStatus::Waiting);
}

#[test]
fn test_complete_lifecycle_with_chunking() {
    // End-to-end: prefill chunks → decode → finish → drain
    let config = SchedulerConfig {
        chunked_prefill: true,
        max_num_batched_tokens: 30,
        max_total_tokens: 65536,
        ..Default::default()
    };
    let mut sched = Scheduler::new(config);
    sched.add_request(make_seq("r1", 50, 3));

    // Chunk 1
    let s1 = sched.schedule_step().unwrap();
    assert_eq!(s1.prefill_chunk_lens, vec![30]);

    // Chunk 2 (remaining 20)
    let s2 = sched.schedule_step().unwrap();
    assert_eq!(s2.prefill_chunk_lens, vec![20]);
    assert_eq!(sched.running[0].status, SequenceStatus::Decoding);

    // 3 decode steps
    for i in 0..3 {
        let sd = sched.schedule_step().unwrap();
        assert_eq!(sd.forward_mode, ForwardMode::Decode);
        sched.on_token_generated("r1", 100 + i);
    }

    // Finish
    sched.finish_request("r1", SeqFinishReason::Eos);
    let _ = sched.schedule_step(); // drains finished
    let done = sched.take_finished();
    assert_eq!(done.len(), 1);
    assert_eq!(done[0].output_ids.len(), 3);
    assert_eq!(sched.num_running(), 0);
}

// ── Block-level admission tests ──────────────────────────────────

#[test]
fn test_block_admission_rejects_when_no_blocks() {
    // block_size=16, available_blocks=0 → can't admit any request
    let config = SchedulerConfig {
        chunked_prefill: true,
        max_num_batched_tokens: 8192,
        block_size: 16,
        ..Default::default()
    };
    let mut sched = Scheduler::new(config);
    sched.available_blocks = 0;
    sched.add_request(make_seq("r1", 10, 20));

    // No blocks → nothing scheduled (waiting stays)
    assert!(sched.schedule_step().is_none());
    assert_eq!(sched.num_waiting(), 1);
}

#[test]
fn test_block_admission_respects_block_granularity() {
    // block_size=16. Request needs 20 prompt + 20 decode = 40 tokens = 3 blocks.
    // Give exactly 3 blocks → fits. Give 2 → doesn't fit.
    let config = SchedulerConfig {
        chunked_prefill: true,
        max_num_batched_tokens: 8192,
        block_size: 16,
        decode_reservation_cap: 20,
        ..Default::default()
    };

    // 2 blocks: not enough (need ceil(40/16)=3)
    let mut sched = Scheduler::new(config.clone());
    sched.available_blocks = 2;
    sched.add_request(make_seq("r1", 20, 20));
    assert!(sched.schedule_step().is_none());

    // 3 blocks: just enough
    let mut sched = Scheduler::new(config);
    sched.available_blocks = 3;
    sched.add_request(make_seq("r1", 20, 20));
    let step = sched.schedule_step().unwrap();
    assert_eq!(step.prefill_request_ids, vec!["r1"]);
}

#[test]
fn test_block_admission_decrements_for_multiple_requests() {
    // block_size=16, 10 blocks available.
    // r1: 10 prompt + 10 decode = 20 tokens → 2 blocks
    // r2: 10 prompt + 10 decode = 20 tokens → 2 blocks
    // r3: 10 prompt + 10 decode = 20 tokens → 2 blocks
    // Total: 6 blocks, fits in 10.
    // r4 same → 8 blocks total, fits.
    // r5 same → 10 blocks total, fits.
    // r6 same → 12 blocks, doesn't fit.
    let config = SchedulerConfig {
        chunked_prefill: true,
        max_num_batched_tokens: 8192,
        block_size: 16,
        decode_reservation_cap: 10,
        ..Default::default()
    };
    let mut sched = Scheduler::new(config);
    sched.available_blocks = 10;

    for i in 0..6 {
        sched.add_request(make_seq(&format!("r{i}"), 10, 10));
    }

    let step = sched.schedule_step().unwrap();
    // Should admit 5 (5*2=10 blocks) but not the 6th (would need 12)
    assert_eq!(step.prefill_request_ids.len(), 5);
    assert_eq!(sched.num_waiting(), 1);
}

#[test]
fn test_block_admission_large_request_rejected() {
    // block_size=16. Request needs 100 prompt + 20 decode = 120 tokens = 8 blocks.
    // Only 5 blocks available → rejected.
    let config = SchedulerConfig {
        chunked_prefill: true,
        max_num_batched_tokens: 8192,
        block_size: 16,
        decode_reservation_cap: 20,
        ..Default::default()
    };
    let mut sched = Scheduler::new(config);
    sched.available_blocks = 5;
    sched.add_request(make_seq("r1", 100, 20));
    assert!(sched.schedule_step().is_none());
    assert_eq!(sched.num_waiting(), 1);
}

#[test]
fn test_available_blocks_default() {
    let sched = Scheduler::new(SchedulerConfig::default());
    // Without block_manager, available_blocks stays at MAX (no limit).
    assert_eq!(sched.available_blocks, usize::MAX);
}
