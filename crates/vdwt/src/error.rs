//! Errors.

use std::fmt;

/// Errors from configuring the encoder or decoding a stream.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// The stream is malformed or truncated.
    InvalidData(&'static str),
    /// The stream uses a feature this decoder does not implement.
    Unsupported(&'static str),
    /// The encoder configuration is invalid.
    InvalidConfig(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidData(what) => write!(f, "invalid stream: {what}"),
            Error::Unsupported(what) => write!(f, "unsupported stream: {what}"),
            Error::InvalidConfig(what) => write!(f, "invalid configuration: {what}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T, E = Error> = std::result::Result<T, E>;
