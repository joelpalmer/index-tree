//! # Arena based tree structure
//!
//! [Based on this lib](https://github.com/saschagrunert/indextree/blob/master/src/lib.rs)
//!

use failure::{bail, Fail, Fallible};
#[cfg(feature = "par_iter")]
use rayon::prelude::*;
use std::{
    fmt, mem,
    num::NonZeroUsize,
    ops::{Index, IndexMut},
};

#[derive(PartialEq, Eq, PartialOrd, Ord, Copy, Clone, Debug, Hash)]
#[cfg_attr(feature = "deser", derive(Deserialize, Serialize))]
/// A node identifier within a particular `Arena`
pub struct NodeId {
    // One-based index
    index1: NonZeroUsize,
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.index1)
    }
}

#[derive(Debug, Fail)]
/// Possible node failures
pub enum NodeError {
    #[fail(display = "Can not append a node to itself")]
    AppendSelf,

    #[fail(display = "Can not prepend a node to itself")]
    PrependSelf,

    #[fail(display = "Can not insert a node before itself")]
    InsertBeforeSelf,

    #[fail(display = "Can not insert a node after itself")]
    InsertAfterSelf,

    #[fail(display = "First child is already set")]
    FirstChildAlreadySet,

    #[fail(display = "Previous sibling is already set")]
    PreviousSiblingAlreadySet,

    #[fail(display = "Next sibling is already set")]
    NextSiblingAlreadySet,

    #[fail(display = "Previous sibling not equal current node")]
    PreviousSiblingNotSelf,

    #[fail(display = "Next sibling not equal current node")]
    NextSiblingNotSelf,

    #[fail(display = "First child not equal current node")]
    FirstChildNotSelf,

    #[fail(display = "Last child not equal current node")]
    LastChildNotSelf,

    #[fail(display = "Previous sibling is not set")]
    PreviousSiblingNotSet,

    #[fail(display = "Next sibling is not set")]
    NextSiblingNotSet,

    #[fail(display = "First child is not set")]
    FirstChildNotSet,

    #[fail(display = "Last child is not set")]
    LastChildNotSet,
}

#[derive(PartialEq, Clone, Debug)]
#[cfg_attr(feature = "deser", derive(Deserialize, Serialize))]
#[cfg_attr(feature = "derive-eq", derive(Eq))]
/// A node within a particular `Arena`
pub struct Node<T> {
    // Keep these fields private (w/ read-only accessors so that we can keep
    // them consistent. E.g. the parent of a node's child is that node.
    parent: Option<NodeId>,
    previous_sibling: Option<NodeId>,
    next_sibling: Option<NodeId>,
    first_child: Option<NodeId>,
    last_child: Option<NodeId>,
    removed: bool,

    /// The actual data which will be stored within the tree
    pub data: T,
}
