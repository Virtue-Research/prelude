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

#[test]
fn test_basic_prefill_then_decode() {
    let config = SchedulerConfig {
        max_running_requests: 4,
        max_prefill_tokens: 1024,
        max_total_tokens: 4096,
        ..Default::default()
    };
    let mut sched = Scheduler::new(config);

    sched.add_request(make_seq("r1", 10, 20));
    assert_eq!(sched.num_waiting(), 1);

    let step = sched.schedule_step().unwrap();
    assert_eq!(step.forward_mode, ForwardMode::Prefill);
    assert_eq!(step.prefill_request_ids, vec!["r1"]);
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
        max_prefill_tokens: 4096,
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
fn test_prefill_token_budget() {
    let config = SchedulerConfig {
        max_running_requests: 10,
        max_prefill_tokens: 15,
        max_total_tokens: 65536,
        ..Default::default()
    };
    let mut sched = Scheduler::new(config);

    sched.add_request(make_seq("r1", 10, 20));
    sched.add_request(make_seq("r2", 10, 20));

    let step = sched.schedule_step().unwrap();
    assert_eq!(step.forward_mode, ForwardMode::Prefill);
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
        max_prefill_tokens: 4096,
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
fn test_mixed_chunked_emits_mixed_step() {
    let config = SchedulerConfig {
        mixed_chunked: true,
        ..Default::default()
    };
    let mut sched = Scheduler::new(config);
    sched.add_request(make_seq("r1", 8, 20));

    let first = sched.schedule_step().unwrap();
    assert_eq!(first.forward_mode, ForwardMode::Prefill);
    assert_eq!(first.prefill_request_ids, vec!["r1"]);

    sched.add_request(make_seq("r2", 6, 20));
    let second = sched.schedule_step().unwrap();
    assert_eq!(second.forward_mode, ForwardMode::Mixed);
    assert_eq!(second.prefill_request_ids, vec!["r2"]);
    assert_eq!(second.decode_request_ids, vec!["r1"]);
}
