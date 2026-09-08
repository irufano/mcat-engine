# mcat-engine

Pure, DB-free Multidimensional Computer Adaptive Testing (MCAT) engine —
item selection, ability estimation, exposure control, and stopping rules for
three simultaneous latent traits (**Verbal**, **Numeric**, **Reasoning**, or
any other trait set a caller configures).

Extracted from `mcat-api/src/engines/mcat` so the algorithm core can be
reused by other apps beyond `mcat-api`, independent of any host's database
layer or web framework. Callers convert their own item/settings DTOs into
[`types::EngineItem`](src/types.rs) / [`types::EngineSettings`](src/types.rs)
at the call boundary — see `mcat-api`'s `From<&Item> for EngineItem` and
`From<&TestSettings> for EngineSettings` adapters
(`mcat-api/src/modules/item/dto.rs`, `mcat-api/src/modules/settings/dto.rs`)
for a worked example.

## Source layout

- `engine.rs` — `McatEngine`: the four orchestration entry points (select
  next item, estimate theta, compute SE, check stopping rule)
- `mirt.rs` — MIRT math (M3PL probability, Fisher information, log-likelihood)
- `estimation/`, `selection/`, `stopping/`, `exposure/` — pluggable method
  implementations (MLE/MAP/EAP, D-optimal/A-optimal/KL-information,
  fixed-length/SE-threshold/convergence/hybrid, Sympson-Hetter)
- `dimensions.rs` — item-eligibility masking + a-vector projection for
  heterogeneous item banks
- `debug.rs` — structured debug/trace payloads for selection/estimation/stopping
- `playground.rs` — DB-free step/run orchestration for algorithm experimentation
- `error.rs`, `types.rs` — the crate's boundary types (see above)
- `examples/` — prototype/reference implementations that predate (and are
  cross-checked against) the production engine above: `item_selection/{d_optimal,a_optimal,kl_information}`,
  `estimation/{mle,map,eap}`, plus `.md` notes per method and `overview.md`.
  Run one with `cargo run --example <name>`, e.g. `cargo run --example estimation_eap`
  (see `Cargo.toml`'s `[[example]]` entries for the full name list).

## Status: local path dependency

This crate isn't pushed to its own git remote yet. `mcat-api/Cargo.toml`
currently depends on it as a local path:

```toml
mcat-engine = { path = "../mcat-engine" }
```

**Once this directory is pushed to its own private git remote and tagged**
(e.g. `v0.1.0`), switch that to a pinned git dependency instead:

```toml
mcat-engine = { git = "ssh://git@<host>/<you>/mcat-engine.git", tag = "v0.1.0" }
```

That's what lets other apps depend on this crate too, each pinned to
whichever tag they've picked up — no infra beyond a private git remote and
per-app SSH/deploy-key access. See the "reuse across apps" discussion this
crate came out of for the full reasoning (git dependency now vs. a private
Cargo registry later, if/when the number of consuming apps grows enough to
justify it).


psrecord $(pgrep -f "target/release/mcat-api") --interval 1 --log mcat-usage.csv 

cat mcat-usage.csv
