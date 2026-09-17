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

## Status: git dependency (branch, unpinned)

This crate lives at `https://github.com/irufano/mcat-engine` (private).
`mcat-api/Cargo.toml` depends on it over SSH, tracking `master`:

```toml
mcat-engine = { git = "ssh://git@github.com/irufano/mcat-engine.git", branch = "master" }
```

**Once mcat-engine cuts a release** (e.g. `v0.1.0`), switch that to a
pinned tag instead so `mcat-api` doesn't move with every push to `master`:

```toml
mcat-engine = { git = "ssh://git@github.com/irufano/mcat-engine.git", tag = "v0.1.0" }
```

That's what lets other apps depend on this crate too, each pinned to
whichever tag they've picked up — no infra beyond the private git remote and
per-app SSH/deploy-key access. See the "reuse across apps" discussion this
crate came out of for the full reasoning (git dependency now vs. a private
Cargo registry later, if/when the number of consuming apps grows enough to
justify it).

Consumers need an SSH key registered on the GitHub account/org that owns
this repo (`git@github.com` must authenticate — `ssh -T git@github.com`).

## Installing via SSH

One-time setup for any machine that needs to `cargo build`/`cargo check` a
project depending on this private repo (e.g. `mcat-api`).

1. **Check for an existing GitHub-capable key.**
   ```bash
   ssh -T git@github.com
   ```
   `Hi <username>! You've successfully authenticated...` means you're already
   set up — skip to step 5.

2. **Generate a key** if step 1 failed (`Permission denied (publickey)`):
   ```bash
   ssh-keygen -t ed25519 -C "<your-email>" -f ~/.ssh/id_ed25519_github -N ""
   ```

3. **Load it into the agent** (macOS also stores it in the login keychain
   so it survives reboots):
   ```bash
   ssh-add --apple-use-keychain ~/.ssh/id_ed25519_github   # macOS
   ssh-add ~/.ssh/id_ed25519_github                        # other OSes
   ```

4. **Point `github.com` at this key** — add to `~/.ssh/config`:
   ```
   Host github.com
     HostName github.com
     User git
     IdentityFile ~/.ssh/id_ed25519_github
     IdentitiesOnly yes
   ```
   Then add the *public* key (`~/.ssh/id_ed25519_github.pub`) at
   [github.com/settings/ssh/new](https://github.com/settings/ssh/new), and
   confirm with `ssh -T git@github.com` (repeat step 1).

5. **Add the dependency** in the consuming crate's `Cargo.toml`:
   ```toml
   mcat-engine = { git = "ssh://git@github.com/irufano/mcat-engine.git", branch = "master" }
   ```

6. **Run `cargo check`.** If it fails with `no authentication methods
   succeeded` / `attempted ssh-agent authentication, but no usernames
   succeeded`, cargo's bundled libgit2 isn't picking up the ssh-agent +
   per-host config from step 4 — tell cargo to shell out to your system
   `git` (which already works per step 1) instead. Add a project-local
   `.cargo/config.toml` next to that crate's `Cargo.toml`:
   ```toml
   [net]
   git-fetch-with-cli = true
   ```
   Then `cargo check` again.


psrecord $(pgrep -f "target/release/mcat-api") --interval 1 --log mcat-usage.csv 

cat mcat-usage.csv
