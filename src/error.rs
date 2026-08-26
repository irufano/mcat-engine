//! Engine-owned error type. Replaces the host application's HTTP-flavored
//! error type (e.g. `mcat-api`'s `AppError`) at the crate boundary — callers
//! map [`EngineError`] back into their own error type with a `From` impl.

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    /// A settings/config value the engine couldn't make sense of (e.g. an
    /// unrecognized selection/estimation/stopping method string). Maps to
    /// `AppError::Internal` in `mcat-api` — these strings are validated at
    /// write time, so seeing one here indicates stored/caller data drifted
    /// out of sync with the engine's supported method set.
    #[error("{0}")]
    InvalidConfig(String),

    /// A request-shaped problem the caller can fix (e.g. an infeasible EAP
    /// quadrature grid, a referenced item that doesn't exist). Maps to
    /// `AppError::BadRequest` in `mcat-api`.
    #[error("{0}")]
    InvalidRequest(String),
}

pub type EngineResult<T> = Result<T, EngineError>;
