//! Minimal, engine-owned item/settings shapes. These carry only the fields
//! [`crate::engine`] and [`crate::playground`] actually read — no DB
//! (`sqlx`), timestamp (`chrono`), or persistence-only fields. A host
//! application converts its own item/settings DTOs into these at the call
//! boundary (see `mcat-api`'s `From<&Item> for EngineItem` /
//! `From<&TestSettings> for EngineSettings` adapters).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineItem {
    pub id: String,
    /// Discrimination vector, positional to the tested-dimension order the
    /// caller is using for this session (see `dimensions::project_item` for
    /// heterogeneous-bank masking, done by the caller before this point).
    pub a_params: Vec<f64>,
    pub d_param: f64,
    pub c_param: f64,
    /// Sympson-Hetter control parameter `K_i` = `P(A|S)`.
    pub sh_r_param: f64,
    pub is_active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineSettings {
    pub selection_method: String,
    pub estimation_method: String,
    pub stopping_rule: String,
    pub exposure_method: String,
    pub n_min: i32,
    pub n_max: i32,
    pub se_threshold: f64,
    pub convergence_delta: f64,
    pub prior_mean: Vec<f64>,
    pub prior_cov_diag: Vec<f64>,
    pub eap_quadrature_points: i32,
}
