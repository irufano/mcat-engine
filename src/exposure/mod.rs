#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::str::FromStr;

pub mod sympson_hetter;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ExposureMethod {
    None,
    SympsonHetter,
}

impl FromStr for ExposureMethod {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "none" => Ok(Self::None),
            "sympson_hetter" => Ok(Self::SympsonHetter),
            other => Err(format!("Unknown exposure method: {other}")),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum BankExhaustedAction {
    Error,
    ForceStop,
    Extend,
}

impl FromStr for BankExhaustedAction {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "error" => Ok(Self::Error),
            "force_stop" => Ok(Self::ForceStop),
            "extend" => Ok(Self::Extend),
            other => Err(format!("Unknown bank exhausted action: {other}")),
        }
    }
}

pub fn is_eligible(method: &ExposureMethod, sh_r_param: f64) -> bool {
    match method {
        ExposureMethod::None => true,
        ExposureMethod::SympsonHetter => sympson_hetter::is_eligible(sh_r_param),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_all_known_exposure_methods() {
        assert_eq!(
            "none".parse::<ExposureMethod>().unwrap(),
            ExposureMethod::None
        );
        assert_eq!(
            "sympson_hetter".parse::<ExposureMethod>().unwrap(),
            ExposureMethod::SympsonHetter
        );
    }

    #[test]
    fn rejects_unknown_exposure_method() {
        assert!("bogus".parse::<ExposureMethod>().is_err());
    }

    #[test]
    fn parses_all_known_bank_exhausted_actions() {
        assert_eq!(
            "error".parse::<BankExhaustedAction>().unwrap(),
            BankExhaustedAction::Error
        );
        assert_eq!(
            "force_stop".parse::<BankExhaustedAction>().unwrap(),
            BankExhaustedAction::ForceStop
        );
        assert_eq!(
            "extend".parse::<BankExhaustedAction>().unwrap(),
            BankExhaustedAction::Extend
        );
    }

    #[test]
    fn rejects_unknown_bank_exhausted_action() {
        assert!("bogus".parse::<BankExhaustedAction>().is_err());
    }

    #[test]
    fn none_method_is_always_eligible_regardless_of_r_param() {
        assert!(is_eligible(&ExposureMethod::None, 0.0));
        assert!(is_eligible(&ExposureMethod::None, 1.0));
    }

    #[test]
    fn sympson_hetter_dispatch_respects_r_param_zero() {
        for _ in 0..200 {
            assert!(!is_eligible(&ExposureMethod::SympsonHetter, 0.0));
        }
    }
}
