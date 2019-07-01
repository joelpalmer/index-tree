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

impl<T> fmt::Display for Node<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(parent) = self.parent {
            write!(f, "parent: {}; ", parent)?;
        } else {
            write!(f, "no parent; ")?;
        }
        if let Some(previous_sibling) = self.previous_sibling {
            write!(f, "previous sibling: {}; ", previous_sibling)?;
        } else {
            write!(f, "no previous sibling; ")?;
        }
        if let Some(next_sibling) = self.next_sibling {
            write!(f, "next sibling: {}; ", next_sibling)?;
        } else {
            write!(f, "no next sibling; ")?;
        }
        if let Some(first_child) = self.first_child {
            write!(f, "first child: {}; ", first_child)?;
        } else {
            write!(f, "no first child; ")?;
        }
        if let Some(last_child) = self.last_child {
            write!(f, "last child: {}; ", last_child)?;
        } else {
            write!(f, "no last child; ")?;
        }
        Ok(())
    }
}

#[derive(PartialEq, Clone, Debug)]
#[cfg_attr(feature = "deser", derive(Deserialize, Serialize))]
#[cfg_attr(feature = "derive-eq", derive(Eq))]
/// An `Arena` structure containing certain Nodes
pub struct Arena<T> {
    nodes: Vec<Node<T>>,
}

impl<T> Arena<T> {
    /// Create a new empty `Arena`
    pub fn new() -> Arena<T> {
        Arena { nodes: Vec::new() }
    }

    /// Creates a new node from its associated data.
    ///
    /// #Panics
    ///
    /// Panics if the arena already has `usize::max_value()` nodes.
    pub fn new_node(&mut self, data: T) -> NodeId {
        let next_index1 = NonZeroUsize::new(self.nodes.len().wrapping_add(1))
            .expect("Too many nodes in the arena");
        self.nodes.push(Node {
            parent: None,
            first_child: None,
            last_child: None,
            previous_sibling: None,
            next_sibling: None,
            removed: false,
            data,
        });
        NodeId::from_non_zero_usize(next_index1)
    }

    /// Count nodes in the arena
    pub fn count(&self) -> usize {
        self.nodes.len()
    }

    /// Return true if arena has no nodes, false otherwise
    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }

    /// Get a reference to the node with the given id if in the arena,
    /// None otherwise.
    pub fn get(&self, id: NodeId) -> Option<&Node<T>> {
        self.nodes.get(id.index0())
    }

    /// Get a mutable reference to the node with the given id if in the arena,
    /// None otherwise
    pub fn get_mut(&mut self, id: NodeId) -> Option<&mut Node<T>> {
        self.nodes.get_mut(id.index0())
    }

    /// Iterate over all the nodes in the arena in storage-order.
    ///
    /// Note that this iterator also contains removed elements which can be
    /// tested with the `is_removed()` method on the node.
    pub fn iter(&self) -> std::slice::Iter<Node<T>> {
        self.nodes.iter()
    }
}

#[cfg(feature = "par_iter")]
impl<T: Sync> Arena<T> {
    /// Return an parallel iterator over the whole arena.
    ///
    /// Note that this iterator also contains removed elements, which can be
    /// tested with the `is_removed()` method on the node.
    pub fn par_iter(&self) -> rayon::slice::Iter<Node<T>> {
        self.nodes.par_iter()
    }
}

trait GetPairMut<T> {
    /// Get mutable references to two distinct nodes
    fn get_tuple_mut(&mut self, a: usize, b: usize) -> Option<(&mut T, &mut T)>;
}

impl<T> GetPairMut<T> for Vec<T> {
    fn get_tuple_mut(&mut self, a: usize, b: usize) -> Option<(&mut T, &mut T)> {
        if a == b {
            return None;
        }
        let (xs, ys) = self.split_at_mut(std::cmp::max(a, b));
        if a < b {
            Some((&mut xs[a], &mut ys[0]))
        } else {
            Some((&mut ys[0], &mut xs[b]))
        }
    }
}

impl<T> Index<NodeId> for Arena<T> {
    type Output = Node<T>;
    fn index(&self, node: NodeId) -> &Node<T> {
        &self.nodes[node.index0()]
    }
}

impl<T> IndexMut<NodeId> for Arena<T> {
    fn index_mut(&mut self, node: NodeId) -> &mut Node<T> {
        &mut self.nodes[node.index0()]
    }
}

impl<T> Node<T> {
    /// Return the ID of the parent node unless this node if the root
    pub fn parent(&self) -> Option<NodeId> {
        self.parent
    }

    /// Return the ID of the first child of this node, unless it has no child.
    pub fn first_child(&self) -> Option<NodeId> {
        self.first_child
    }

    /// Return the ID of the last child of this node, unless it has no child.
    pub fn last_child(&self) -> Option<NodeId> {
        self.last_child
    }

    /// Return the ID of the previous sibling of this node, unless it is a
    /// first child.
    pub fn previous_sibling(&self) -> Option<NodeId> {
        self.previous_sibling
    }

    /// Return the ID of the next sibling of this node, unless it is a
    /// last child.
    pub fn next_sibling(&self) -> Option<NodeId> {
        self.next_sibling
    }

    /// Check if the node is marked as removed
    pub fn is_removed(&self) -> bool {
        self.removed
    }
}

impl NodeId {
    /// Returns zero-based index.
    fn index0(self) -> usize {
        // Safe because `self.index1 >= 1` is guaranteed by
        // `NonZeroUsize` type
        self.index1.get() - 1
    }

    /// Create a `NodeId` used for attempting to get `Node`s references from an
    /// `Arena`.
    ///
    /// Note that a zero-based index should be given.
    ///
    /// # Panics
    ///
    /// Panics if the value is `usize::max_value()`.
    pub fn new(index0: usize) -> Self {
        let index1 = NonZeroUsize::new(index0.wrapping_add(1))
            .expect("Attempt to create `NodeId` from `usize::max_value()`");
        NodeId { index1 }
    }

    /// Creates a new `NodeId` from the given one-based index.
    pub fn from_non_zero_usize(index1: NonZeroUsize) -> Self {
        NodeId { index1 }
    }

    /// Return an iterator of references to this node & its ancestors.
    ///
    /// Call `.next().unwrap()` once on the iterator to skip the node itself.
    pub fn ancestors<T>(self, arena: &Arena<T>) -> Ancestors<T> {
        Ancestors {
            arena,
            node: Some(self),
        }
    }

    /// Return an iterator of references to this node & the siblings before it.
    ///
    /// Call `.next().unwrap()` once on the iterator to skip the node itself.
    pub fn preceding_siblings<T>(self, arena: &Arena<T>) -> PrecedingSiblings<T> {
        PrecedingSiblings {
            arena,
            node: Some(self),
        }
    }

    // need other types of relatives first!
}

macro_rules! impl_node_iterator {
    ($name:ident, $next:expr) => {
        impl<'a, T> Iterator for $name<'a, T> {
            type Item = NodeId;

            fn next(&mut self) -> Option<NodeId> {
                match self.node.take() {
                    Some(node) => {
                        self.node = $next(&self.arena[node]);
                        Some(node)
                    }
                    None => None,
                }
            }
        }
    };
}

/// An iterator of references to the ancestors for a given node.
pub struct Ancestors<'a, T: 'a> {
    arena: &'a Arena<T>,
    node: Option<NodeId>,
}

// impl_node_iterator!

/// An iterator of references to the siblings before a given node.
pub struct PrecedingSiblings<'a, T: 'a> {
    arena: &'a Arena<T>,
    node: Option<NodeId>,
}

// impl_node_iterator!
// ...
