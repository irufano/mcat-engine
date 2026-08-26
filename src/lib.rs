//! Pure, DB-free Multidimensional Computer Adaptive Testing (MCAT) engine —
//! item selection, ability estimation, exposure control, and stopping rules.
//!
//! Extracted from `mcat-api/src/engines/mcat`. Consumes only plain,
//! engine-owned types ([`types::EngineItem`], [`types::EngineSettings`]) so
//! it carries no dependency on any host application's database layer or web
//! framework — see `error` and `types` for the boundary types callers
//! convert into/from their own DTOs.

pub mod debug;
pub mod dimensions;
pub mod engine;
pub mod error;
pub mod estimation;
pub mod exposure;
pub mod mirt;
pub mod playground;
pub mod selection;
pub mod stopping;
pub mod types;
