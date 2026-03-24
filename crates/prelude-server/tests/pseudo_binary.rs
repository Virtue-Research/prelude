mod common;

use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use reqwest::blocking::Client;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE};
use serde_json::json;

use common::find_free_port;

struct TestServer {
    child: Child,
    base_url: String,
}

impl TestServer {
    fn start(extra_args: &[&str]) -> Self {
        let port = find_free_port();
        let binary = option_env!("CARGO_BIN_EXE_prelude-server")
            .map(ToOwned::to_owned)
            .or_else(|| std::env::var("CARGO_BIN_EXE_prelude-server").ok())
            .unwrap_or_else(|| {
                let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
                manifest_dir
                    .join("../../target/debug/prelude-server")
                    .display()
                    .to_string()
            });

        let mut command = Command::new(binary);
        command
            .args([
                "--pseudo",
                "--model",
                "test-model",
                "--host",
                "127.0.0.1",
                "--port",
                &port.to_string(),
            ])
            .args(extra_args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let child = command.spawn().expect("failed to spawn prelude-server");
        let mut server = Self {
            child,
            base_url: format!("http://127.0.0.1:{port}"),
        };
        server.wait_until_ready();
        server
    }

    fn wait_until_ready(&mut self) {
        let client = Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(30);
        while Instant::now() < deadline {
            if let Some(status) = self.child.try_wait().unwrap() {
                panic!("prelude-server exited before becoming healthy: {status}");
            }
            if let Ok(resp) = client.get(format!("{}/health", self.base_url)).send()
                && resp.status().is_success()
            {
                return;
            }
            thread::sleep(Duration::from_millis(200));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
        panic!("prelude-server did not become healthy in time");
    }

    fn client() -> Client {
        Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap()
    }

    fn shutdown(mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

#[test]
fn pseudo_binary_serves_completion_and_classification() {
    let server = TestServer::start(&[]);
    let client = TestServer::client();

    let health = client
        .get(format!("{}/health", server.base_url))
        .send()
        .unwrap();
    assert!(health.status().is_success());

    let models = client
        .get(format!("{}/v1/models", server.base_url))
        .send()
        .unwrap()
        .json::<serde_json::Value>()
        .unwrap();
    assert_eq!(models["data"][0]["id"], "test-model");

    let completion = client
        .post(format!("{}/v1/completions", server.base_url))
        .header(CONTENT_TYPE, "application/json")
        .json(&json!({
            "model": "test-model",
            "prompt": "hello world",
            "max_tokens": 2,
            "seed": 7
        }))
        .send()
        .unwrap()
        .json::<serde_json::Value>()
        .unwrap();
    assert!(
        !completion["choices"][0]["text"]
            .as_str()
            .unwrap()
            .is_empty()
    );

    let classify = client
        .post(format!("{}/v1/classify", server.base_url))
        .header(CONTENT_TYPE, "application/json")
        .json(&json!({
            "model": "test-model",
            "input": ["safe", "unsafe"]
        }))
        .send()
        .unwrap()
        .json::<serde_json::Value>()
        .unwrap();
    assert_eq!(classify["data"].as_array().unwrap().len(), 2);

    server.shutdown();
}

#[test]
fn pseudo_binary_enforces_api_keys_on_v1_routes() {
    let server = TestServer::start(&["--api-key", "sk-secret"]);
    let client = TestServer::client();

    let health = client
        .get(format!("{}/health", server.base_url))
        .send()
        .unwrap();
    assert_eq!(health.status(), reqwest::StatusCode::OK);

    let unauthenticated = client
        .get(format!("{}/v1/models", server.base_url))
        .send()
        .unwrap();
    assert_eq!(unauthenticated.status(), reqwest::StatusCode::UNAUTHORIZED);

    let authenticated = client
        .get(format!("{}/v1/models", server.base_url))
        .header(AUTHORIZATION, "Bearer sk-secret")
        .send()
        .unwrap();
    assert_eq!(authenticated.status(), reqwest::StatusCode::OK);

    server.shutdown();
}
